#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_gdino_deeplab_leaf_lesion_end2end.py

End-to-end evaluation for:
  GroundingDINO leaf/lesion detection
  + leaf-level DeepLabV3+ ROI segmentation
  + lesion-local DeepLabV3+ ROI segmentation
  + probability fusion pasted back to the full image

This is the metric that should be compared with full-image DeepLabV3+.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.utils._pytree as pytree

if not hasattr(pytree, "register_pytree_node"):
    def _compat_register_pytree_node(nodetype, flatten_fn, unflatten_fn, **kwargs):
        return pytree._register_pytree_node(nodetype, flatten_fn, unflatten_fn)
    pytree.register_pytree_node = _compat_register_pytree_node

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    import segmentation_models_pytorch as smp
except Exception as e:
    raise RuntimeError("Install segmentation-models-pytorch before running this script.") from e

try:
    from torchvision.ops import nms as tv_nms
    HAS_TV_NMS = True
except Exception:
    HAS_TV_NMS = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--gdino-model-dir", required=True)
    p.add_argument("--leaf-deeplab-ckpt", required=True)
    p.add_argument("--lesion-deeplab-ckpt", required=True)
    p.add_argument("--save-dir", required=True)

    p.add_argument("--leaf-prompt", default="leaf .")
    p.add_argument("--lesion-prompt", default="leaf lesion .")
    p.add_argument("--leaf-box-threshold", type=float, default=0.25)
    p.add_argument("--leaf-text-threshold", type=float, default=0.20)
    p.add_argument("--leaf-nms-iou", type=float, default=0.50)
    p.add_argument("--lesion-box-threshold", type=float, default=0.15)
    p.add_argument("--lesion-text-threshold", type=float, default=0.15)
    p.add_argument("--lesion-nms-iou", type=float, default=0.30)
    p.add_argument("--max-leaf-per-image", type=int, default=8)
    p.add_argument("--max-lesion-per-image", type=int, default=100)
    p.add_argument("--leaf-duplicate-iou", type=float, default=0.75)
    p.add_argument("--leaf-lesion-inter-ratio", type=float, default=0.30)
    p.add_argument("--keep-unrelated-leaf-topk", type=int, default=0)
    p.add_argument("--fallback-leaf-from-lesion", action="store_true")
    p.add_argument("--fallback-leaf-expand", type=float, default=2.0)
    p.add_argument("--disable-center-rule", action="store_true")
    p.add_argument("--leaf-padding", type=float, default=0.02)
    p.add_argument("--lesion-padding", type=float, default=0.08)

    p.add_argument("--leaf-encoder-name", default=None)
    p.add_argument("--lesion-encoder-name", default=None)
    p.add_argument("--image-size", nargs=2, type=int, default=None,
                   help="H W. If omitted, read from checkpoint config.")
    p.add_argument("--leaf-prob-threshold", type=float, default=None,
                   help="If omitted, read best_threshold from checkpoint or use 0.5.")
    p.add_argument("--lesion-prob-threshold", type=float, default=None)
    p.add_argument("--fusion-thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    p.add_argument("--leaf-weight", type=float, default=0.75)
    p.add_argument("--lesion-weight", type=float, default=1.00)
    p.add_argument("--fusion-mode", choices=["max", "mean", "lesion_or_leaf"], default="max")
    p.add_argument("--min-component-area", type=int, default=0)

    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-vis", type=int, default=80)
    p.add_argument("--save-architecture-steps", action="store_true",
                   help="Save blue architecture-step images: input, boxes, ROIs, branch masks, pasted maps, final mask.")
    p.add_argument("--save-pred-masks", action="store_true",
                   help="Save final binary prediction masks to save-dir/pred_masks.")
    p.add_argument("--pred-mask-threshold", type=float, default=None,
                   help="Threshold used when saving pred masks. If omitted, uses per-image best Dice threshold.")
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_float_list(x: str) -> List[float]:
    return [float(v.strip()) for v in x.split(",") if v.strip()]


def read_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"))
    return (m > 0).astype(np.uint8)


def area_xyxy(b: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in b]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / (area_xyxy(a) + area_xyxy(b) - inter + 1e-9)


def inter_area_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def center_inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    x1, y1, x2, y2 = [float(v) for v in inner]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)


def clip_box(b: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in b]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(x1 + 1, min(w, round(x2))))
    y2 = int(max(y1 + 1, min(h, round(y2))))
    return x1, y1, x2, y2


def expand_box(b: Sequence[float], w: int, h: int, ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in b]
    bw, bh = x2 - x1, y2 - y1
    return clip_box((x1 - bw * ratio, y1 - bh * ratio, x2 + bw * ratio, y2 + bh * ratio), w, h)


def expand_box_scale(b: Sequence[float], w: int, h: int, scale: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in b]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    return clip_box((cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0), w, h)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if len(boxes) == 0:
        return []
    if HAS_TV_NMS:
        keep = tv_nms(torch.as_tensor(boxes, dtype=torch.float32),
                      torch.as_tensor(scores, dtype=torch.float32), float(iou_thr))
        return keep.cpu().tolist()
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        rest = []
        for j in order[1:]:
            if iou_xyxy(boxes[i], boxes[int(j)]) <= iou_thr:
                rest.append(int(j))
        order = np.array(rest, dtype=np.int64)
    return keep


@torch.no_grad()
def run_prompt(model, processor, image: Image.Image, prompt: str, box_thr: float, text_thr: float, device, amp: bool):
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
        outputs = model(**inputs)
    target_sizes = torch.tensor([[image.height, image.width]], device=device)
    fn = processor.post_process_grounded_object_detection
    sig = inspect.signature(fn)
    kwargs = {"outputs": outputs, "target_sizes": target_sizes}
    if "box_threshold" in sig.parameters:
        kwargs["box_threshold"] = box_thr
    elif "threshold" in sig.parameters:
        kwargs["threshold"] = box_thr
    if "text_threshold" in sig.parameters:
        kwargs["text_threshold"] = text_thr
    res = fn(**kwargs)[0]
    boxes = res["boxes"].detach().float().cpu().numpy()
    scores = res["scores"].detach().float().cpu().numpy()
    return boxes, scores


def predict_class(model, processor, image, prompt, box_thr, text_thr, nms_iou, max_det, device, amp):
    W, H = image.size
    boxes, scores = run_prompt(model, processor, image, prompt, box_thr, text_thr, device, amp)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    boxes = boxes.astype(np.float32)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, W - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, H - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, W)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, H)
    boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
    boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)
    keep = nms_numpy(boxes, scores, nms_iou)
    boxes = boxes[keep]
    scores = scores[keep]
    order = np.argsort(-scores)
    if max_det > 0:
        order = order[:max_det]
    return boxes[order], scores[order]


@dataclass
class LeafCandidate:
    box: Tuple[int, int, int, int]
    score: float
    lesion_indices: List[int]
    quality: float
    fallback: bool = False


def leaf_related_to_lesion(leaf_box, lesion_box, center_rule=True, inter_ratio=0.30) -> bool:
    if center_rule and center_inside(lesion_box, leaf_box):
        return True
    inter = inter_area_xyxy(leaf_box, lesion_box)
    return inter / max(1e-9, area_xyxy(lesion_box)) >= inter_ratio


def filter_leaf_boxes(leaf_boxes, leaf_scores, lesion_boxes, W, H, duplicate_iou, max_leaf_per_image,
                      keep_unrelated_topk, center_rule, inter_ratio, fallback_leaf_from_lesion, fallback_expand):
    candidates: List[LeafCandidate] = []
    for i, b in enumerate(leaf_boxes):
        lb = clip_box(b, W, H)
        rel = []
        for j, les in enumerate(lesion_boxes):
            if leaf_related_to_lesion(lb, les, center_rule, inter_ratio):
                rel.append(j)
        is_diseased = 1 if len(rel) > 0 else 0
        area_ratio = area_xyxy(lb) / max(1.0, W * H)
        area_penalty = 0.20 if (area_ratio < 0.02 or area_ratio > 0.95) else 0.0
        quality = float(is_diseased * 10.0 + len(rel) + float(leaf_scores[i]) - area_penalty)
        candidates.append(LeafCandidate(lb, float(leaf_scores[i]), rel, quality))

    if fallback_leaf_from_lesion:
        covered = set()
        for c in candidates:
            covered.update(c.lesion_indices)
        for j, les in enumerate(lesion_boxes):
            if j not in covered:
                candidates.append(LeafCandidate(expand_box_scale(les, W, H, fallback_expand), 0.001, [j], 9.0, True))

    order = sorted(range(len(candidates)), key=lambda k: candidates[k].quality, reverse=True)
    kept: List[LeafCandidate] = []
    for idx in order:
        c = candidates[idx]
        duplicate = False
        for kc in kept:
            if iou_xyxy(c.box, kc.box) >= duplicate_iou and not (set(c.lesion_indices) - set(kc.lesion_indices)):
                duplicate = True
                break
        if not duplicate:
            kept.append(c)

    diseased = [c for c in kept if len(c.lesion_indices) > 0]
    unrelated = sorted([c for c in kept if len(c.lesion_indices) == 0], key=lambda x: x.score, reverse=True)
    final = sorted(diseased + unrelated[:max(0, keep_unrelated_topk)], key=lambda x: x.quality, reverse=True)
    if max_leaf_per_image > 0 and len(final) > max_leaf_per_image:
        diseased_final = [c for c in final if len(c.lesion_indices) > 0]
        unrelated_final = [c for c in final if len(c.lesion_indices) == 0]
        final = diseased_final + unrelated_final[:max(0, max_leaf_per_image - len(diseased_final))]
    return final


def build_deeplab(encoder_name: str):
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def load_deeplab(ckpt_path: Path, encoder_override: str | None, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    encoder = encoder_override or cfg.get("encoder_name", "resnet34")
    image_size = tuple(cfg.get("image_size", [512, 512]))
    threshold = float(ckpt.get("best_threshold", cfg.get("prob_threshold", 0.5)))
    model = build_deeplab(encoder)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, cfg, image_size, threshold


def letterbox_image(img: Image.Image, target_h: int, target_w: int):
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    meta = {"src_w": src_w, "src_h": src_h, "new_w": new_w, "new_h": new_h, "pad_x": pad_x, "pad_y": pad_y}
    return canvas, meta


def preprocess(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()[None, ...]


@torch.no_grad()
def predict_roi_prob(model, image: Image.Image, box: Sequence[int], image_size: Tuple[int, int], device, amp: bool) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    roi = image.crop((x1, y1, x2, y2))
    target_h, target_w = int(image_size[0]), int(image_size[1])
    lb, meta = letterbox_image(roi, target_h, target_w)
    inp = preprocess(lb).to(device)
    with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
        prob = torch.sigmoid(model(inp))[0, 0].detach().float().cpu().numpy()
    py, px = meta["pad_y"], meta["pad_x"]
    nh, nw = meta["new_h"], meta["new_w"]
    prob = prob[py:py + nh, px:px + nw]
    prob_img = Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8))
    prob_img = prob_img.resize((meta["src_w"], meta["src_h"]), Image.BILINEAR)
    return np.asarray(prob_img).astype(np.float32) / 255.0


def paste_prob(full_prob: np.ndarray, roi_prob: np.ndarray, box: Sequence[int]):
    x1, y1, x2, y2 = [int(v) for v in box]
    full_prob[y1:y2, x1:x2] = np.maximum(full_prob[y1:y2, x1:x2], roi_prob)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    try:
        import cv2
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        out = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, num):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                out[labels == i] = 1
        return out
    except Exception:
        return mask


def fuse_probs(leaf_prob: np.ndarray, lesion_prob: np.ndarray, mode: str, leaf_weight: float, lesion_weight: float) -> np.ndarray:
    if mode == "mean":
        return (leaf_prob * leaf_weight + lesion_prob * lesion_weight) / max(1e-6, leaf_weight + lesion_weight)
    if mode == "lesion_or_leaf":
        return np.maximum(leaf_prob * leaf_weight, lesion_prob)
    return np.maximum(leaf_prob * leaf_weight, lesion_prob * lesion_weight)


def metric_counts(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int, eps=1e-7):
    return {
        "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "pixel_acc": float((tp + tn + eps) / (tp + fp + fn + tn + eps)),
    }


def save_vis(image: Image.Image, gt: np.ndarray, pred: np.ndarray, leaf_boxes, lesion_boxes, out_path: Path):
    im = image.convert("RGB")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    arr = np.zeros((im.height, im.width, 4), dtype=np.uint8)
    arr[(gt > 0) & (pred == 0)] = [255, 0, 0, 120]
    arr[(gt == 0) & (pred > 0)] = [0, 80, 255, 120]
    arr[(gt > 0) & (pred > 0)] = [0, 255, 0, 120]
    overlay = Image.fromarray(arr, mode="RGBA")
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(im)
    for b in leaf_boxes:
        draw.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])], outline=(0, 255, 255), width=2)
    for b in lesion_boxes:
        draw.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])], outline=(255, 255, 0), width=2)
    ensure_dir(out_path.parent)
    im.save(out_path)


BLUE = (0, 102, 255)
BLUE_RGBA = (0, 102, 255, 135)


def _clip_box_int(box: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def overlay_blue_mask(image: Image.Image, mask: np.ndarray, alpha: int = 135) -> Image.Image:
    im = image.convert("RGBA")
    m = mask.astype(bool)
    if m.shape != (im.height, im.width):
        m_img = Image.fromarray((m.astype(np.uint8) * 255), mode="L").resize(im.size, Image.NEAREST)
        m = np.asarray(m_img) > 0
    arr = np.zeros((im.height, im.width, 4), dtype=np.uint8)
    arr[m] = [BLUE_RGBA[0], BLUE_RGBA[1], BLUE_RGBA[2], alpha]
    return Image.alpha_composite(im, Image.fromarray(arr, mode="RGBA")).convert("RGB")


def blue_mask_on_white(mask: np.ndarray, size: Tuple[int, int] | None = None) -> Image.Image:
    m = mask.astype(bool)
    if size is not None and m.shape != (size[1], size[0]):
        m_img = Image.fromarray((m.astype(np.uint8) * 255), mode="L").resize(size, Image.NEAREST)
        m = np.asarray(m_img) > 0
        h, w = size[1], size[0]
    else:
        h, w = m.shape
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[m] = [BLUE[0], BLUE[1], BLUE[2]]
    return Image.fromarray(arr, mode="RGB")


def save_boxes_blue(image: Image.Image, boxes: Sequence[Sequence[float]], out_path: Path, label: str = "lesion") -> None:
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)
    width = max(2, int(min(im.size) * 0.004))
    for i, b in enumerate(boxes, 1):
        draw.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])], outline=BLUE, width=width)
        draw.text((float(b[0]) + 3, float(b[1]) + 3), f"{label}{i}", fill=BLUE)
    ensure_dir(out_path.parent)
    im.save(out_path)


def make_roi_grid(
    image: Image.Image,
    roi_items: Sequence[Tuple[Sequence[float], np.ndarray]],
    out_path: Path,
    title: str,
    prob_threshold: float | None = None,
    mask_only: bool = False,
    max_items: int = 8,
    thumb_size: Tuple[int, int] = (180, 180),
) -> None:
    items = list(roi_items)[:max_items]
    tw, th = thumb_size
    label_h = 34
    if not items:
        canvas = Image.new("RGB", (tw * 2, th + label_h), "white")
        ImageDraw.Draw(canvas).text((10, 10), f"{title}: no ROI", fill=BLUE)
        ensure_dir(out_path.parent)
        canvas.save(out_path)
        return
    cols = min(4, len(items))
    rows = int(math.ceil(len(items) / cols))
    canvas = Image.new("RGB", (cols * tw, rows * (th + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    W, H = image.size
    for idx, (box, prob) in enumerate(items):
        x1, y1, x2, y2 = _clip_box_int(box, W, H)
        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        if prob is not None and prob_threshold is not None:
            roi_mask = (prob >= float(prob_threshold)).astype(np.uint8)
            if mask_only:
                crop = blue_mask_on_white(roi_mask, size=crop.size)
            else:
                crop = overlay_blue_mask(crop, roi_mask, alpha=145)
        crop.thumbnail((tw, th), Image.BILINEAR)
        tile = Image.new("RGB", (tw, th + label_h), "white")
        tile.paste(crop, ((tw - crop.width) // 2, 0))
        ImageDraw.Draw(tile).text((6, th + 8), f"{title} #{idx + 1}", fill=BLUE)
        cx = (idx % cols) * tw
        cy = (idx // cols) * (th + label_h)
        canvas.paste(tile, (cx, cy))
    ensure_dir(out_path.parent)
    canvas.save(out_path)


def save_architecture_steps(
    image: Image.Image,
    stem: str,
    out_root: Path,
    lesion_boxes: Sequence[Sequence[float]],
    used_lesion_items: Sequence[Tuple[Sequence[float], np.ndarray]],
    used_leaf_items: Sequence[Tuple[Sequence[float], np.ndarray]],
    lesion_prob: np.ndarray,
    leaf_prob: np.ndarray,
    fused_prob: np.ndarray,
    final_threshold: float,
    leaf_thr: float,
    lesion_thr: float,
    min_component_area: int,
) -> None:
    sample_dir = out_root / stem
    ensure_dir(sample_dir)
    image.convert("RGB").save(sample_dir / "01_input_leaf_image.jpg")
    save_boxes_blue(image, lesion_boxes, sample_dir / "02_disease_candidate_boxes.jpg", label="lesion")
    make_roi_grid(image, used_lesion_items, sample_dir / "03_cropped_disease_roi_patches.jpg", "disease ROI")
    make_roi_grid(image, used_leaf_items, sample_dir / "04_cropped_leaf_roi_patches.jpg", "leaf ROI")
    make_roi_grid(
        image, used_lesion_items, sample_dir / "05_lesion_local_roi_output_masks.jpg",
        "lesion-local mask", prob_threshold=lesion_thr, mask_only=True,
    )
    lesion_pasted_mask = (lesion_prob >= max(1e-6, lesion_thr)).astype(np.uint8)
    overlay_blue_mask(image, lesion_pasted_mask).save(sample_dir / "06_lesion_local_pasted_to_full_image.jpg")
    make_roi_grid(
        image, used_leaf_items, sample_dir / "07_leaf_global_roi_output_masks.jpg",
        "leaf-global mask", prob_threshold=leaf_thr, mask_only=True,
    )
    leaf_pasted_mask = (leaf_prob >= max(1e-6, leaf_thr)).astype(np.uint8)
    overlay_blue_mask(image, leaf_pasted_mask).save(sample_dir / "08_leaf_global_pasted_to_full_image.jpg")
    final_mask = (fused_prob >= float(final_threshold)).astype(np.uint8)
    final_mask = remove_small_components(final_mask, min_component_area)
    blue_mask_on_white(final_mask, size=image.size).save(sample_dir / "09a_final_output_mask_blue_only.jpg")
    overlay_blue_mask(image, final_mask, alpha=155).save(sample_dir / "09_final_output_mask_blue.jpg")


def main():
    args = parse_args()
    prepared_root = Path(args.prepared_root)
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(args.amp and device.type == "cuda")

    processor = AutoProcessor.from_pretrained(args.gdino_model_dir)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model_dir).to(device).eval()
    leaf_model, leaf_cfg, leaf_size, leaf_thr = load_deeplab(Path(args.leaf_deeplab_ckpt), args.leaf_encoder_name, device)
    lesion_model, lesion_cfg, lesion_size, lesion_thr = load_deeplab(Path(args.lesion_deeplab_ckpt), args.lesion_encoder_name, device)
    if args.image_size is not None:
        leaf_size = tuple(args.image_size)
        lesion_size = tuple(args.image_size)
    if args.leaf_prob_threshold is not None:
        leaf_thr = float(args.leaf_prob_threshold)
    if args.lesion_prob_threshold is not None:
        lesion_thr = float(args.lesion_prob_threshold)

    ann_path = prepared_root / "dino_coco" / "annotations" / f"instances_{args.split}.json"
    image_root = prepared_root / "dino_coco" / "images" / args.split
    mask_root = prepared_root / "deeplab_full" / "masks" / args.split
    coco = load_json(ann_path)
    fusion_thresholds = parse_float_list(args.fusion_thresholds)

    per_image_rows = []
    counts_by_thr = {float(t): [0, 0, 0, 0] for t in fusion_thresholds}
    mean_parts_by_thr = {float(t): [] for t in fusion_thresholds}
    vis_count = 0
    pred_mask_dir = save_dir / "pred_masks"
    pred_mask_blue_dir = save_dir / "pred_masks_blue"
    if args.save_pred_masks:
        ensure_dir(pred_mask_dir)
        ensure_dir(pred_mask_blue_dir)

    for img_info in tqdm(coco["images"], desc="End2End eval"):
        img_name = img_info["file_name"]
        img_path = image_root / img_name
        mask_path = mask_root / (Path(img_name).stem + ".png")
        if not img_path.exists() or not mask_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        gt = read_mask(mask_path)
        W, H = image.size
        leaf_prob = np.zeros((H, W), dtype=np.float32)
        lesion_prob = np.zeros((H, W), dtype=np.float32)

        lesion_boxes, lesion_scores = predict_class(
            gdino, processor, image, args.lesion_prompt,
            args.lesion_box_threshold, args.lesion_text_threshold,
            args.lesion_nms_iou, args.max_lesion_per_image, device, amp,
        )
        leaf_boxes, leaf_scores = predict_class(
            gdino, processor, image, args.leaf_prompt,
            args.leaf_box_threshold, args.leaf_text_threshold,
            args.leaf_nms_iou, 0, device, amp,
        )
        leaf_candidates = filter_leaf_boxes(
            leaf_boxes, leaf_scores, lesion_boxes, W, H,
            duplicate_iou=args.leaf_duplicate_iou,
            max_leaf_per_image=args.max_leaf_per_image,
            keep_unrelated_topk=args.keep_unrelated_leaf_topk,
            center_rule=not args.disable_center_rule,
            inter_ratio=args.leaf_lesion_inter_ratio,
            fallback_leaf_from_lesion=args.fallback_leaf_from_lesion,
            fallback_expand=args.fallback_leaf_expand,
        )

        used_leaf_boxes = []
        used_leaf_items = []
        for c in leaf_candidates:
            box = expand_box(c.box, W, H, args.leaf_padding)
            roi_prob = predict_roi_prob(leaf_model, image, box, leaf_size, device, amp)
            paste_prob(leaf_prob, roi_prob, box)
            used_leaf_boxes.append(box)
            used_leaf_items.append((box, roi_prob))

        used_lesion_boxes = []
        used_lesion_items = []
        for b in lesion_boxes:
            box = expand_box(b, W, H, args.lesion_padding)
            roi_prob = predict_roi_prob(lesion_model, image, box, lesion_size, device, amp)
            paste_prob(lesion_prob, roi_prob, box)
            used_lesion_boxes.append(box)
            used_lesion_items.append((box, roi_prob))

        leaf_prob = np.where(leaf_prob >= leaf_thr, leaf_prob, 0.0).astype(np.float32)
        lesion_prob = np.where(lesion_prob >= lesion_thr, lesion_prob, 0.0).astype(np.float32)
        fused_prob = fuse_probs(leaf_prob, lesion_prob, args.fusion_mode, args.leaf_weight, args.lesion_weight)

        best_img = None
        for thr in fusion_thresholds:
            pred = (fused_prob >= thr).astype(np.uint8)
            pred = remove_small_components(pred, args.min_component_area)
            tp, fp, fn, tn = metric_counts(pred, gt)
            c = counts_by_thr[thr]
            c[0] += tp; c[1] += fp; c[2] += fn; c[3] += tn
            m = metrics_from_counts(tp, fp, fn, tn)
            mean_parts_by_thr[thr].append(m)
            if best_img is None or m["dice"] > best_img["dice"]:
                best_img = {"threshold": thr, **m}

        per_image_rows.append({
            "image": img_name,
            "num_leaf_roi": len(used_leaf_boxes),
            "num_lesion_roi": len(used_lesion_boxes),
            **{f"best_{k}": v for k, v in (best_img or {}).items()},
        })

        if vis_count < args.num_vis and best_img is not None:
            pred_vis = (fused_prob >= float(best_img["threshold"])).astype(np.uint8)
            pred_vis = remove_small_components(pred_vis, args.min_component_area)
            save_vis(image, gt, pred_vis, used_leaf_boxes, used_lesion_boxes,
                     save_dir / "visualizations" / f"{Path(img_name).stem}.jpg")
            if args.save_architecture_steps:
                save_architecture_steps(
                    image=image,
                    stem=Path(img_name).stem,
                    out_root=save_dir / "architecture_steps",
                    lesion_boxes=used_lesion_boxes,
                    used_lesion_items=used_lesion_items,
                    used_leaf_items=used_leaf_items,
                    lesion_prob=lesion_prob,
                    leaf_prob=leaf_prob,
                    fused_prob=fused_prob,
                    final_threshold=float(best_img["threshold"]),
                    leaf_thr=leaf_thr,
                    lesion_thr=lesion_thr,
                    min_component_area=args.min_component_area,
                )
            vis_count += 1

        if args.save_pred_masks and best_img is not None:
            save_thr = float(args.pred_mask_threshold) if args.pred_mask_threshold is not None else float(best_img["threshold"])
            pred_mask = (fused_prob >= save_thr).astype(np.uint8)
            pred_mask = remove_small_components(pred_mask, args.min_component_area)
            Image.fromarray((pred_mask * 255).astype(np.uint8)).save(
                pred_mask_dir / f"{Path(img_name).stem}.png"
            )
            blue_mask_on_white(pred_mask, size=image.size).save(
                pred_mask_blue_dir / f"{Path(img_name).stem}.png"
            )

    summary = {
        "settings": vars(args),
        "leaf_deeplab_config": leaf_cfg,
        "lesion_deeplab_config": lesion_cfg,
        "leaf_roi_threshold": leaf_thr,
        "lesion_roi_threshold": lesion_thr,
        "thresholds": {},
    }
    best_global_thr = None
    best_global_dice = -1.0
    for thr in fusion_thresholds:
        tp, fp, fn, tn = counts_by_thr[thr]
        global_m = metrics_from_counts(tp, fp, fn, tn)
        mean_m = {}
        if mean_parts_by_thr[thr]:
            for k in ["dice", "iou", "precision", "recall", "pixel_acc"]:
                mean_m[f"mean_{k}"] = float(np.mean([x[k] for x in mean_parts_by_thr[thr]]))
        entry = {**global_m, **mean_m, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        summary["thresholds"][f"{thr:.3f}"] = entry
        if global_m["dice"] > best_global_dice:
            best_global_dice = global_m["dice"]
            best_global_thr = thr
    summary["best_threshold"] = best_global_thr
    summary["best_global_dice"] = best_global_dice

    save_json(save_dir / "summary_metrics.json", summary)
    csv_path = save_dir / "per_image_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = sorted(set().union(*(r.keys() for r in per_image_rows))) if per_image_rows else ["image"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_image_rows:
            w.writerow(r)

    print("=" * 90)
    print("End-to-end evaluation finished")
    print(f"best_threshold : {best_global_thr}")
    print(f"best_global_dice: {best_global_dice:.6f}")
    if best_global_thr is not None:
        best = summary["thresholds"][f"{best_global_thr:.3f}"]
        print(f"precision      : {best['precision']:.6f}")
        print(f"recall         : {best['recall']:.6f}")
        print(f"mean_dice      : {best.get('mean_dice', 0.0):.6f}")
    print(f"summary        : {save_dir / 'summary_metrics.json'}")
    print(f"per-image csv  : {csv_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
