#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_pred_roi_leaf_lesion.py

用训练好的 GroundingDINO 生成 Pred ROI，用于后续 DeepLabV3+ 的 GT ROI + Pred ROI 混合训练。

核心逻辑：
1) 分别预测 leaf 与 leaf lesion；
2) lesion 框只做 lesion 类内部 NMS，不因 leaf 过滤被删除；
3) leaf 框先 NMS，再与 lesion 框关联；高度重叠 leaf 框只保留质量更好的；
4) 保存 leaf_roi_pred、lesion_roi_pred；
5) 同时生成 leaf_roi_mixed、lesion_roi_mixed manifest，方便后续训练。

推荐：训练 DeepLabV3+ 时使用 letterbox 版本训练脚本。
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils._pytree as pytree

if not hasattr(pytree, "register_pytree_node"):
    def _compat_register_pytree_node(nodetype, flatten_fn, unflatten_fn, **kwargs):
        return pytree._register_pytree_node(nodetype, flatten_fn, unflatten_fn)
    pytree.register_pytree_node = _compat_register_pytree_node

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

try:
    from torchvision.ops import nms as tv_nms
    HAS_TV_NMS = True
except Exception:
    HAS_TV_NMS = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", required=True, help="prepared_leaf_disease_leaf_level")
    p.add_argument("--model-dir", required=True, help="runs/gdino_leaf_lesion_joint/best")
    p.add_argument("--splits", default="train,val,test")

    p.add_argument("--leaf-prompt", default="leaf .")
    p.add_argument("--lesion-prompt", default="leaf lesion .")

    p.add_argument("--leaf-box-threshold", type=float, default=0.20)
    p.add_argument("--leaf-text-threshold", type=float, default=0.20)
    p.add_argument("--leaf-nms-iou", type=float, default=0.50)

    p.add_argument("--lesion-box-threshold", type=float, default=0.15)
    p.add_argument("--lesion-text-threshold", type=float, default=0.15)
    p.add_argument("--lesion-nms-iou", type=float, default=0.30)

    p.add_argument("--max-leaf-per-image", type=int, default=8,
                   help="leaf 框最多保留数量。与 lesion 关联的 leaf 框优先保留。<=0 表示不限制。")
    p.add_argument("--max-lesion-per-image", type=int, default=100)

    p.add_argument("--leaf-duplicate-iou", type=float, default=0.75,
                   help="两个 leaf 框 IoU 大于该值时认为高度重叠，删除质量较差的。")
    p.add_argument("--leaf-lesion-inter-ratio", type=float, default=0.30,
                   help="leaf 与 lesion 交集面积 / lesion 面积 >= 该值，则认为关联。")
    p.add_argument("--disable-center-rule", action="store_true",
                   help="默认 lesion 中心点落入 leaf 即认为关联；加该参数则禁用。")

    p.add_argument("--keep-unrelated-leaf-topk", type=int, default=0,
                   help="额外保留多少个未关联 lesion 的 leaf 框作为负样本。")
    p.add_argument("--fallback-leaf-from-lesion", action="store_true",
                   help="如果某个 lesion 找不到关联 leaf，则用 lesion 扩大框生成 fallback leaf ROI。")
    p.add_argument("--fallback-leaf-expand", type=float, default=2.0)

    p.add_argument("--min-positive-pixels-lesion", type=int, default=5)
    p.add_argument("--min-positive-pixels-leaf", type=int, default=5)
    p.add_argument("--keep-negative-leaf-topk", type=int, default=0)
    p.add_argument("--keep-negative-lesion", action="store_true")

    p.add_argument("--lesion-padding", type=float, default=0.08)
    p.add_argument("--leaf-padding", type=float, default=0.02)

    p.add_argument("--leaf-gt-repeat", type=int, default=2,
                   help="leaf_roi_mixed 中 GT leaf ROI 重复次数，2≈GT:Pred 2:1。")
    p.add_argument("--lesion-gt-repeat", type=int, default=3,
                   help="lesion_roi_mixed 中 GT lesion ROI 重复次数，3≈GT:Pred 3:1。")
    p.add_argument("--leaf-gt-jitter", type=int, default=2,
                   help="train mixed 中每个 GT leaf ROI 额外生成多少个随机扰动 ROI。")
    p.add_argument("--lesion-gt-jitter", type=int, default=2,
                   help="train mixed 中每个 GT lesion ROI 额外生成多少个随机扰动 ROI。")
    p.add_argument("--jitter-shift", type=float, default=0.10,
                   help="GT ROI jitter 的中心随机偏移比例，相对原框宽高。")
    p.add_argument("--jitter-scale-min", type=float, default=0.85)
    p.add_argument("--jitter-scale-max", type=float, default=1.25)
    p.add_argument("--jitter-seed", type=int, default=42)
    p.add_argument("--val-mixed-source", choices=["pred", "gt_pred"], default="pred",
                   help="mixed 的 val manifest 使用 pred-only，还是 gt+pred。推荐 pred。")

    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-vis", type=int, default=80)
    return p.parse_args()


def split_list(x: str) -> List[str]:
    return [s.strip() for s in x.split(",") if s.strip()]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj: Any):
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"))
    return (m > 0).astype(np.uint8)


def save_mask(path: Path, mask: np.ndarray):
    ensure_dir(path.parent)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


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
        quality = float(is_diseased * 10.0 + len(rel) * 1.0 + float(leaf_scores[i]) - area_penalty)
        candidates.append(LeafCandidate(lb, float(leaf_scores[i]), rel, quality))

    fallback_count = 0
    if fallback_leaf_from_lesion:
        covered = set()
        for c in candidates:
            covered.update(c.lesion_indices)
        for j, les in enumerate(lesion_boxes):
            if j not in covered:
                fb = expand_box_scale(les, W, H, fallback_expand)
                candidates.append(LeafCandidate(fb, 0.001, [j], 9.0, True))
                fallback_count += 1

    order = sorted(range(len(candidates)), key=lambda k: candidates[k].quality, reverse=True)
    kept: List[LeafCandidate] = []
    for idx in order:
        c = candidates[idx]
        duplicate = False
        for kc in kept:
            if iou_xyxy(c.box, kc.box) >= duplicate_iou:
                # 如果当前框含有 kept 框没有覆盖的 lesion，则保留，避免删掉带病害的叶片框。
                if set(c.lesion_indices) - set(kc.lesion_indices):
                    continue
                duplicate = True
                break
        if not duplicate:
            kept.append(c)

    diseased = [c for c in kept if len(c.lesion_indices) > 0]
    unrelated = [c for c in kept if len(c.lesion_indices) == 0]
    unrelated = sorted(unrelated, key=lambda x: x.score, reverse=True)[:max(0, keep_unrelated_topk)]
    final = sorted(diseased + unrelated, key=lambda x: x.quality, reverse=True)

    if max_leaf_per_image > 0 and len(final) > max_leaf_per_image:
        diseased_final = [c for c in final if len(c.lesion_indices) > 0]
        unrelated_final = [c for c in final if len(c.lesion_indices) == 0]
        remain = max(0, max_leaf_per_image - len(diseased_final))
        final = diseased_final + unrelated_final[:remain]

    stats = {
        "raw_leaf": int(len(leaf_boxes)),
        "filtered_leaf": int(len(final)),
        "fallback_leaf": int(fallback_count),
        "diseased_leaf": int(sum(1 for c in final if len(c.lesion_indices) > 0)),
    }
    return final, stats


MANIFEST_FIELDS = [
    "image", "mask", "split", "roi_type", "source", "score", "source_image",
    "x1", "y1", "x2", "y2", "width", "height",
    "positive_pixels", "num_linked_lesions", "fallback",
]


def crop_and_save(image: Image.Image, mask: np.ndarray, box: Sequence[int], image_out: Path, mask_out: Path):
    x1, y1, x2, y2 = [int(v) for v in box]
    crop_img = image.crop((x1, y1, x2, y2))
    crop_mask = mask[y1:y2, x1:x2]
    ensure_dir(image_out.parent)
    ensure_dir(mask_out.parent)
    crop_img.save(image_out)
    save_mask(mask_out, crop_mask)
    return int((crop_mask > 0).sum())


def jitter_box(box: Sequence[float], w: int, h: int, rng: random.Random,
               shift_ratio: float, scale_min: float, scale_max: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0 + rng.uniform(-shift_ratio, shift_ratio) * bw
    cy = (y1 + y2) / 2.0 + rng.uniform(-shift_ratio, shift_ratio) * bh
    sw = bw * rng.uniform(scale_min, scale_max)
    sh = bh * rng.uniform(scale_min, scale_max)
    return clip_box((cx - sw / 2.0, cy - sh / 2.0, cx + sw / 2.0, cy + sh / 2.0), w, h)


def write_manifest(path: Path, rows: List[Dict[str, Any]]):
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


def read_gt_manifest(path: Path, repeat: int, source_name: str) -> List[Dict[str, Any]]:
    if not path.exists() or repeat <= 0:
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for _ in range(repeat):
                rr = dict(r)
                rr.setdefault("score", "1.0")
                rr["source"] = source_name
                rr.setdefault("positive_pixels", "")
                rr.setdefault("num_linked_lesions", "")
                rr.setdefault("fallback", "0")
                rows.append(rr)
    return rows


def make_gt_jitter_rows(prepared_root: Path, gt_type: str, split: str, count: int,
                        source_name: str, args) -> List[Dict[str, Any]]:
    if count <= 0 or split != "train":
        return []

    gt_manifest = prepared_root / "deeplab_roi" / gt_type / f"manifest_{split}.csv"
    if not gt_manifest.exists():
        return []

    rng = random.Random(int(args.jitter_seed) + (17 if gt_type == "leaf_roi" else 31))
    full_image_root = prepared_root / "deeplab_full" / "images" / split
    full_mask_root = prepared_root / "deeplab_full" / "masks" / split
    rows: List[Dict[str, Any]] = []

    with open(gt_manifest, "r", encoding="utf-8") as f:
        for idx, r in enumerate(csv.DictReader(f)):
            src_name = r.get("source_image") or Path(r.get("image", "")).name
            img_path = full_image_root / src_name
            mask_path = full_mask_root / (Path(src_name).stem + ".png")
            if not img_path.exists() or not mask_path.exists():
                continue
            image = Image.open(img_path).convert("RGB")
            mask = read_mask(mask_path)
            W, H = image.size
            try:
                base_box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
            except Exception:
                continue

            for j in range(count):
                box = jitter_box(
                    base_box, W, H, rng,
                    shift_ratio=float(args.jitter_shift),
                    scale_min=float(args.jitter_scale_min),
                    scale_max=float(args.jitter_scale_max),
                )
                roi_name = f"{Path(src_name).stem}_{gt_type}_jitter_{idx:06d}_{j:02d}.jpg"
                roi_mask_name = roi_name.replace(".jpg", ".png")
                img_out = prepared_root / "deeplab_roi" / f"{gt_type}_jitter" / "images" / split / roi_name
                mask_out = prepared_root / "deeplab_roi" / f"{gt_type}_jitter" / "masks" / split / roi_mask_name
                pos_pixels = crop_and_save(image, mask, box, img_out, mask_out)
                rows.append({
                    "image": str(img_out.relative_to(prepared_root)),
                    "mask": str(mask_out.relative_to(prepared_root)),
                    "split": split,
                    "roi_type": f"{gt_type}_jitter",
                    "source": source_name,
                    "score": "1.0",
                    "source_image": src_name,
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    "width": box[2] - box[0],
                    "height": box[3] - box[1],
                    "positive_pixels": pos_pixels,
                    "num_linked_lesions": r.get("num_linked_lesions", ""),
                    "fallback": 0,
                })

    write_manifest(prepared_root / "deeplab_roi" / f"{gt_type}_jitter" / f"manifest_{split}.csv", rows)
    return rows


def make_mixed_manifests(prepared_root: Path, splits: List[str], args):
    roi_pairs = [
        ("leaf_roi", "leaf_roi_pred", "leaf_roi_mixed", args.leaf_gt_repeat, args.leaf_gt_jitter, "gt_leaf"),
        ("lesion_roi", "lesion_roi_pred", "lesion_roi_mixed", args.lesion_gt_repeat, args.lesion_gt_jitter, "gt_lesion"),
    ]
    for gt_type, pred_type, mixed_type, repeat, jitter_count, gt_source in roi_pairs:
        for split in splits:
            gt_manifest = prepared_root / "deeplab_roi" / gt_type / f"manifest_{split}.csv"
            pred_manifest = prepared_root / "deeplab_roi" / pred_type / f"manifest_{split}.csv"
            mixed_rows = []
            if split == "train":
                mixed_rows += read_gt_manifest(gt_manifest, repeat=repeat, source_name=gt_source)
                mixed_rows += make_gt_jitter_rows(
                    prepared_root=prepared_root,
                    gt_type=gt_type,
                    split=split,
                    count=jitter_count,
                    source_name=f"{gt_source}_jitter",
                    args=args,
                )
                if pred_manifest.exists():
                    with open(pred_manifest, "r", encoding="utf-8") as f:
                        mixed_rows += list(csv.DictReader(f))
            else:
                if args.val_mixed_source == "gt_pred":
                    mixed_rows += read_gt_manifest(gt_manifest, repeat=1, source_name=gt_source)
                if pred_manifest.exists():
                    with open(pred_manifest, "r", encoding="utf-8") as f:
                        mixed_rows += list(csv.DictReader(f))
            write_manifest(prepared_root / "deeplab_roi" / mixed_type / f"manifest_{split}.csv", mixed_rows)


def draw_vis(image: Image.Image, leaf_candidates: List[LeafCandidate], lesion_boxes, lesion_scores, out_path: Path):
    from PIL import ImageDraw, ImageFont
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    for i, b in enumerate(lesion_boxes):
        x1, y1, x2, y2 = [float(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 2, max(0, y1 - 18)), f"les {float(lesion_scores[i]):.2f}", fill=(255, 0, 0), font=font)

    for c in leaf_candidates:
        x1, y1, x2, y2 = c.box
        color = (0, 255, 0) if len(c.lesion_indices) > 0 else (0, 180, 255)
        if c.fallback:
            color = (255, 200, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = "leaf* " if len(c.lesion_indices) > 0 else "leaf "
        if c.fallback:
            tag = "fallback "
        draw.text((x1 + 2, y1 + 2), f"{tag}{c.score:.2f}", fill=color, font=font)

    ensure_dir(out_path.parent)
    im.save(out_path)


def main():
    args = parse_args()
    prepared_root = Path(args.prepared_root)
    splits = split_list(args.splits)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(args.amp and device.type == "cuda")

    print(f"[INFO] device={device}, amp={amp}")
    print(f"[INFO] loading GroundingDINO: {args.model_dir}")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_dir).to(device)
    model.eval()

    summary = {"settings": vars(args), "splits": {}}

    for split in splits:
        ann_path = prepared_root / "dino_coco" / "annotations" / f"instances_{split}.json"
        image_root = prepared_root / "dino_coco" / "images" / split
        mask_root = prepared_root / "deeplab_full" / "masks" / split
        coco = load_json(ann_path)

        pred_leaf_rows, pred_lesion_rows = [], []
        split_stats = {
            "images": 0,
            "raw_leaf_boxes": 0,
            "filtered_leaf_boxes": 0,
            "raw_lesion_boxes": 0,
            "saved_leaf_roi": 0,
            "saved_lesion_roi": 0,
            "skipped_leaf_no_positive": 0,
            "skipped_lesion_no_positive": 0,
            "fallback_leaf": 0,
        }
        vis_count = 0

        for img_info in tqdm(coco["images"], desc=f"Pred ROI {split}"):
            img_name = img_info["file_name"]
            img_path = image_root / img_name
            mask_path = mask_root / (Path(img_name).stem + ".png")
            if not img_path.exists() or not mask_path.exists():
                continue

            image = Image.open(img_path).convert("RGB")
            W, H = image.size
            mask = read_mask(mask_path)

            lesion_boxes, lesion_scores = predict_class(
                model, processor, image,
                prompt=args.lesion_prompt,
                box_thr=args.lesion_box_threshold,
                text_thr=args.lesion_text_threshold,
                nms_iou=args.lesion_nms_iou,
                max_det=args.max_lesion_per_image,
                device=device,
                amp=amp,
            )
            leaf_boxes, leaf_scores = predict_class(
                model, processor, image,
                prompt=args.leaf_prompt,
                box_thr=args.leaf_box_threshold,
                text_thr=args.leaf_text_threshold,
                nms_iou=args.leaf_nms_iou,
                max_det=0,
                device=device,
                amp=amp,
            )

            leaf_candidates, leaf_stats = filter_leaf_boxes(
                leaf_boxes=leaf_boxes,
                leaf_scores=leaf_scores,
                lesion_boxes=lesion_boxes,
                W=W,
                H=H,
                duplicate_iou=args.leaf_duplicate_iou,
                max_leaf_per_image=args.max_leaf_per_image,
                keep_unrelated_topk=args.keep_unrelated_leaf_topk,
                center_rule=not args.disable_center_rule,
                inter_ratio=args.leaf_lesion_inter_ratio,
                fallback_leaf_from_lesion=args.fallback_leaf_from_lesion,
                fallback_expand=args.fallback_leaf_expand,
            )

            split_stats["images"] += 1
            split_stats["raw_leaf_boxes"] += int(len(leaf_boxes))
            split_stats["filtered_leaf_boxes"] += int(len(leaf_candidates))
            split_stats["raw_lesion_boxes"] += int(len(lesion_boxes))
            split_stats["fallback_leaf"] += int(leaf_stats["fallback_leaf"])

            for li, (b, s) in enumerate(zip(lesion_boxes, lesion_scores)):
                roi_box = expand_box(b, W, H, args.lesion_padding)
                pos_pixels = int(mask[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]].sum())
                if pos_pixels < args.min_positive_pixels_lesion and not args.keep_negative_lesion:
                    split_stats["skipped_lesion_no_positive"] += 1
                    continue
                roi_name = f"{Path(img_name).stem}_predlesion_{li:03d}.jpg"
                roi_mask_name = roi_name.replace(".jpg", ".png")
                img_out = prepared_root / "deeplab_roi" / "lesion_roi_pred" / "images" / split / roi_name
                mask_out = prepared_root / "deeplab_roi" / "lesion_roi_pred" / "masks" / split / roi_mask_name
                pos_pixels = crop_and_save(image, mask, roi_box, img_out, mask_out)
                split_stats["saved_lesion_roi"] += 1
                pred_lesion_rows.append({
                    "image": str(img_out.relative_to(prepared_root)),
                    "mask": str(mask_out.relative_to(prepared_root)),
                    "split": split,
                    "roi_type": "lesion_roi_pred",
                    "source": "pred_lesion",
                    "score": float(s),
                    "source_image": img_name,
                    "x1": roi_box[0], "y1": roi_box[1], "x2": roi_box[2], "y2": roi_box[3],
                    "width": roi_box[2] - roi_box[0],
                    "height": roi_box[3] - roi_box[1],
                    "positive_pixels": pos_pixels,
                    "num_linked_lesions": 1,
                    "fallback": 0,
                })

            negative_leaf_saved = 0
            for li, c in enumerate(leaf_candidates):
                roi_box = expand_box(c.box, W, H, args.leaf_padding)
                pos_pixels = int(mask[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]].sum())
                if pos_pixels < args.min_positive_pixels_leaf:
                    if negative_leaf_saved >= args.keep_negative_leaf_topk:
                        split_stats["skipped_leaf_no_positive"] += 1
                        continue
                    negative_leaf_saved += 1
                roi_name = f"{Path(img_name).stem}_predleaf_{li:03d}.jpg"
                roi_mask_name = roi_name.replace(".jpg", ".png")
                img_out = prepared_root / "deeplab_roi" / "leaf_roi_pred" / "images" / split / roi_name
                mask_out = prepared_root / "deeplab_roi" / "leaf_roi_pred" / "masks" / split / roi_mask_name
                pos_pixels = crop_and_save(image, mask, roi_box, img_out, mask_out)
                split_stats["saved_leaf_roi"] += 1
                pred_leaf_rows.append({
                    "image": str(img_out.relative_to(prepared_root)),
                    "mask": str(mask_out.relative_to(prepared_root)),
                    "split": split,
                    "roi_type": "leaf_roi_pred",
                    "source": "pred_leaf",
                    "score": float(c.score),
                    "source_image": img_name,
                    "x1": roi_box[0], "y1": roi_box[1], "x2": roi_box[2], "y2": roi_box[3],
                    "width": roi_box[2] - roi_box[0],
                    "height": roi_box[3] - roi_box[1],
                    "positive_pixels": pos_pixels,
                    "num_linked_lesions": len(c.lesion_indices),
                    "fallback": int(c.fallback),
                })

            if vis_count < args.num_vis:
                draw_vis(
                    image,
                    leaf_candidates=leaf_candidates,
                    lesion_boxes=lesion_boxes,
                    lesion_scores=lesion_scores,
                    out_path=prepared_root / "deeplab_roi" / "pred_roi_visualizations" / split / f"{Path(img_name).stem}.jpg",
                )
                vis_count += 1

        write_manifest(prepared_root / "deeplab_roi" / "leaf_roi_pred" / f"manifest_{split}.csv", pred_leaf_rows)
        write_manifest(prepared_root / "deeplab_roi" / "lesion_roi_pred" / f"manifest_{split}.csv", pred_lesion_rows)
        summary["splits"][split] = split_stats
        print(f"[{split}] {split_stats}")

    make_mixed_manifests(prepared_root, splits, args)
    save_json(prepared_root / "deeplab_roi" / "pred_roi_summary.json", summary)

    print("=" * 90)
    print("Pred ROI generation finished")
    print(f"prepared_root : {prepared_root}")
    print(f"summary       : {prepared_root / 'deeplab_roi' / 'pred_roi_summary.json'}")
    print("Generated:")
    print("  deeplab_roi/leaf_roi_pred")
    print("  deeplab_roi/lesion_roi_pred")
    print("  deeplab_roi/leaf_roi_mixed")
    print("  deeplab_roi/lesion_roi_mixed")
    print("=" * 90)


if __name__ == "__main__":
    main()
