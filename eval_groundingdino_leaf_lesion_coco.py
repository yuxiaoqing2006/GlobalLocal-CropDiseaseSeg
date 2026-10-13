#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_groundingdino_leaf_lesion_coco.py

Evaluate a jointly fine-tuned GroundingDINO detector on COCO annotations containing:
  - leaf
  - leaf lesion

Features
--------
1. Supports separate prompts for different classes:
     leaf .
     leaf lesion .
2. Supports separate thresholds for different classes.
3. Reports detection metrics per class:
     precision@IoU, recall@IoU, F1@IoU, matched_mean_iou,
     mean_gt_coverage_recall, mean_best_box_gt_recall,
     mean_union_box_precision, mean_component_hit_rate.
4. If pycocotools is installed, also reports COCO AP/AP50/AP75 per class and all classes.
5. Saves detections.json, summary_metrics.json, per_image_metrics.csv and visualizations.

Recommended command
-------------------
python eval_groundingdino_leaf_lesion_coco.py \
  --ann prepared_leaf_disease_leaf_level/dino_coco/annotations/instances_test.json \
  --image-root prepared_leaf_disease_leaf_level/dino_coco/images/test \
  --model-dir runs/gdino_leaf_lesion_joint/best \
  --save-dir runs/eval_gdino_leaf_lesion_joint_test \
  --category-names "leaf,leaf lesion" \
  --prompts "leaf .|leaf lesion ." \
  --box-thresholds "0.20,0.15" \
  --text-thresholds "0.20,0.15" \
  --nms-ious "0.50,0.30" \
  --iou-threshold 0.50 \
  --num-vis 100 \
  --amp
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO = True
except Exception:
    HAS_COCO = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--save-dir", required=True)

    p.add_argument("--category-names", default="leaf,leaf lesion",
                   help="COCO category names to evaluate, comma-separated.")
    p.add_argument("--prompts", default="leaf .|leaf lesion .",
                   help="Prompts for each category, separated by '|'.")
    p.add_argument("--box-thresholds", default="0.20,0.15")
    p.add_argument("--text-thresholds", default="0.20,0.15")
    p.add_argument("--nms-ious", default="0.50,0.30")
    p.add_argument("--iou-threshold", type=float, default=0.50)

    p.add_argument("--max-dets-per-class", type=int, default=100)
    p.add_argument("--num-vis", type=int, default=100)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--eval-coco", action="store_true",
                   help="Force COCO eval. If omitted, COCO eval is run when pycocotools is available.")
    return p.parse_args()


def split_csv(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def split_pipe(x: str) -> List[str]:
    return [v.strip() for v in x.split("|") if v.strip()]


def parse_float_list(x: str, n: int, default: float) -> List[float]:
    vals = [float(v.strip()) for v in x.split(",") if v.strip()]
    if len(vals) == 1 and n > 1:
        vals = vals * n
    if len(vals) != n:
        raise ValueError(f"Expected {n} values but got {len(vals)}: {x}")
    return vals


def load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (aa + ba - inter + 1e-9))


def xywh_to_xyxy(b):
    x, y, w, h = map(float, b)
    return [x, y, x + w, y + h]


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = map(float, b)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clip_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    boxes = boxes.astype(np.float32).copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, max(0, w - 1))
    boxes[:, 1] = np.clip(boxes[:, 1], 0, max(0, h - 1))
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
    boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)
    return boxes


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if len(boxes) == 0:
        return []
    if HAS_TV_NMS:
        keep = tv_nms(torch.as_tensor(boxes, dtype=torch.float32),
                      torch.as_tensor(scores, dtype=torch.float32),
                      float(iou_thr))
        return keep.cpu().tolist()

    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        rest = []
        for j in order[1:]:
            if box_iou(boxes[i], boxes[int(j)]) <= iou_thr:
                rest.append(int(j))
        order = np.array(rest, dtype=np.int64)
    return keep


@torch.no_grad()
def run_one_prompt(model, processor, image, prompt, box_thr, text_thr, device, amp):
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
    boxes = res["boxes"].detach().cpu().float().numpy()
    scores = res["scores"].detach().cpu().float().numpy()
    return boxes, scores


def build_gt(coco: Dict[str, Any], category_names: List[str]):
    cat_name_to_id = {str(c["name"]).lower(): int(c["id"]) for c in coco["categories"]}
    eval_cat_ids = {}
    for name in category_names:
        key = name.lower()
        if key not in cat_name_to_id:
            raise RuntimeError(f"Category {name} not found. COCO categories: {cat_name_to_id}")
        eval_cat_ids[name] = cat_name_to_id[key]

    img_id_to_info = {int(img["id"]): img for img in coco["images"]}
    gt_by_img_cat = defaultdict(list)
    for ann in coco["annotations"]:
        cid = int(ann["category_id"])
        for name, eid in eval_cat_ids.items():
            if cid == eid:
                gt_by_img_cat[(int(ann["image_id"]), name)].append(xywh_to_xyxy(ann["bbox"]))
    return img_id_to_info, eval_cat_ids, gt_by_img_cat


def greedy_match_metrics(gt_boxes, pred_boxes, scores, iou_thr=0.5):
    pred_order = np.argsort(-np.asarray(scores)).tolist() if len(scores) else []
    matched_gt = set()
    tp = 0
    matched_ious = []

    for pi in pred_order:
        best_iou, best_gi = 0.0, -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = box_iou(pred_boxes[pi], gt)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_iou >= iou_thr and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)
            matched_ious.append(best_iou)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "matched_mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def coverage_metrics(gt_boxes, pred_boxes, image_w, image_h):
    if len(gt_boxes) == 0:
        return {
            "gt_coverage_recall": 1.0 if len(pred_boxes) == 0 else 0.0,
            "best_box_gt_recall": 1.0 if len(pred_boxes) == 0 else 0.0,
            "union_box_precision": 1.0 if len(pred_boxes) == 0 else 0.0,
            "component_hit_rate": 1.0 if len(pred_boxes) == 0 else 0.0,
        }
    if len(pred_boxes) == 0:
        return {
            "gt_coverage_recall": 0.0,
            "best_box_gt_recall": 0.0,
            "union_box_precision": 0.0,
            "component_hit_rate": 0.0,
        }

    H, W = int(image_h), int(image_w)
    gt_mask = np.zeros((H, W), dtype=bool)
    pred_mask = np.zeros((H, W), dtype=bool)

    gt_cover_values = []
    best_cover_values = []
    hits = 0
    for gt in gt_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in gt]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        gt_mask[y1:y2, x1:x2] = True

        area_gt = (x2 - x1) * (y2 - y1)
        inter_union = 0
        best_inter = 0
        for pb in pred_boxes:
            px1, py1, px2, py2 = [int(round(v)) for v in pb]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(W, px2), min(H, py2)
            ix1, iy1 = max(x1, px1), max(y1, py1)
            ix2, iy2 = min(x2, px2), min(y2, py2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            best_inter = max(best_inter, inter)
        # union coverage: create local union mask
        local_pred = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        for pb in pred_boxes:
            px1, py1, px2, py2 = [int(round(v)) for v in pb]
            ix1, iy1 = max(x1, px1), max(y1, py1)
            ix2, iy2 = min(x2, px2), min(y2, py2)
            if ix2 > ix1 and iy2 > iy1:
                local_pred[iy1-y1:iy2-y1, ix1-x1:ix2-x1] = True
        inter_union = int(local_pred.sum())
        gt_cover_values.append(inter_union / max(1, area_gt))
        best_cover_values.append(best_inter / max(1, area_gt))
        if best_inter > 0:
            hits += 1

    for pb in pred_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in pb]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            pred_mask[y1:y2, x1:x2] = True

    union_area = int(pred_mask.sum())
    union_prec = int((pred_mask & gt_mask).sum()) / max(1, union_area)
    return {
        "gt_coverage_recall": float(np.mean(gt_cover_values)) if gt_cover_values else 0.0,
        "best_box_gt_recall": float(np.mean(best_cover_values)) if best_cover_values else 0.0,
        "union_box_precision": float(union_prec),
        "component_hit_rate": hits / max(1, len(gt_boxes)),
    }


def draw_vis(image, preds_by_name, gt_by_name, out_path):
    im = image.convert("RGB")
    draw = ImageDraw.Draw(im)
    colors = {
        "leaf": (0, 255, 0),
        "leaf lesion": (255, 0, 0),
    }
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # GT dashed-like thicker darker
    for name, boxes in gt_by_name.items():
        color = colors.get(name, (255, 255, 0))
        for b in boxes:
            x1, y1, x2, y2 = map(float, b)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1 + 2, y1 + 2), f"GT {name}", fill=color, font=font)

    # Pred
    for name, items in preds_by_name.items():
        color = colors.get(name, (0, 200, 255))
        for b, s in items:
            x1, y1, x2, y2 = map(float, b)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1 + 2, max(0, y1 - 18)), f"{name} {s:.2f}", fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def run_coco_eval(ann_path, detections, save_dir, category_ids):
    if not HAS_COCO:
        return {}
    if len(detections) == 0:
        return {}
    coco_gt = COCO(str(ann_path))
    det_path = save_dir / "detections_cocoeval.json"
    save_json(det_path, detections)
    coco_dt = coco_gt.loadRes(str(det_path))

    out = {}
    for name, cid in category_ids.items():
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [cid]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        out[name] = {
            "AP": float(ev.stats[0]),
            "AP50": float(ev.stats[1]),
            "AP75": float(ev.stats[2]),
        }

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.catIds = list(category_ids.values())
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    out["all"] = {
        "AP": float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
    }
    return out


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    category_names = split_csv(args.category_names)
    prompts = split_pipe(args.prompts)
    if len(category_names) != len(prompts):
        raise ValueError("--category-names and --prompts must have same length.")

    box_thrs = parse_float_list(args.box_thresholds, len(category_names), 0.15)
    text_thrs = parse_float_list(args.text_thresholds, len(category_names), 0.15)
    nms_ious = parse_float_list(args.nms_ious, len(category_names), 0.30)

    coco = load_json(Path(args.ann))
    img_id_to_info, eval_cat_ids, gt_by_img_cat = build_gt(coco, category_names)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(args.amp and device.type == "cuda")
    print(f"[INFO] device={device} amp={amp}")
    print(f"[INFO] loading model: {args.model_dir}")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_dir).to(device)
    model.eval()

    detections = []
    per_image_rows = []
    accum = {name: defaultdict(float) for name in category_names}
    counts = {name: 0 for name in category_names}

    vis_count = 0
    for img_info in tqdm(coco["images"], desc="Evaluating"):
        img_id = int(img_info["id"])
        img_path = Path(args.image_root) / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        preds_by_name = {}
        gts_by_name = {}
        for idx, name in enumerate(category_names):
            boxes, scores = run_one_prompt(
                model=model, processor=processor, image=image, prompt=prompts[idx],
                box_thr=box_thrs[idx], text_thr=text_thrs[idx],
                device=device, amp=amp
            )
            boxes = clip_boxes(boxes, W, H)
            if len(boxes) > 0:
                keep = nms_numpy(boxes, scores, nms_ious[idx])
                boxes = boxes[keep]
                scores = scores[keep]
                order = np.argsort(-scores)
                if args.max_dets_per_class > 0:
                    order = order[:args.max_dets_per_class]
                boxes = boxes[order]
                scores = scores[order]

            gt_boxes = gt_by_img_cat.get((img_id, name), [])
            gts_by_name[name] = gt_boxes
            preds_by_name[name] = list(zip(boxes.tolist(), scores.tolist()))

            for b, s in zip(boxes, scores):
                detections.append({
                    "image_id": int(img_id),
                    "category_id": int(eval_cat_ids[name]),
                    "bbox": xyxy_to_xywh(b),
                    "score": float(s),
                })

            m = greedy_match_metrics(gt_boxes, boxes.tolist(), scores.tolist(), args.iou_threshold)
            c = coverage_metrics(gt_boxes, boxes.tolist(), W, H)

            row = {
                "image_id": img_id,
                "file_name": img_info["file_name"],
                "category": name,
                "num_gt": len(gt_boxes),
                "num_pred": len(boxes),
                **m,
                **c,
            }
            per_image_rows.append(row)

            for k, v in row.items():
                if isinstance(v, (int, float)):
                    accum[name][k] += float(v)
            counts[name] += 1

        if vis_count < args.num_vis:
            draw_vis(image, preds_by_name, gts_by_name, save_dir / "visualizations" / f"{Path(img_info['file_name']).stem}.jpg")
            vis_count += 1

    summary = {"per_class": {}, "settings": vars(args)}
    for name in category_names:
        n = max(1, counts[name])
        summary["per_class"][name] = {k: float(v / n) for k, v in accum[name].items() if k not in ["image_id"]}

    if HAS_COCO or args.eval_coco:
        try:
            summary["coco_eval"] = run_coco_eval(Path(args.ann), detections, save_dir, eval_cat_ids)
        except Exception as e:
            summary["coco_eval_error"] = str(e)

    save_json(save_dir / "detections.json", detections)
    save_json(save_dir / "summary_metrics.json", summary)

    csv_path = save_dir / "per_image_metrics.csv"
    if per_image_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)

    print("=" * 90)
    print("GroundingDINO leaf + lesion evaluation finished")
    for name, metrics in summary["per_class"].items():
        print("-" * 90)
        print(f"[{name}]")
        for k in [
            "precision", "recall", "f1", "matched_mean_iou",
            "gt_coverage_recall", "best_box_gt_recall",
            "union_box_precision", "component_hit_rate",
            "num_gt", "num_pred"
        ]:
            if k in metrics:
                print(f"{k:26s}: {metrics[k]:.6f}")
        if "coco_eval" in summary and name in summary["coco_eval"]:
            ce = summary["coco_eval"][name]
            print(f"AP50                      : {ce.get('AP50', 0):.6f}")
            print(f"AP75                      : {ce.get('AP75', 0):.6f}")
            print(f"AP                        : {ce.get('AP', 0):.6f}")
    print("-" * 90)
    print(f"summary saved to    : {save_dir / 'summary_metrics.json'}")
    print(f"per-image csv saved : {csv_path}")
    print(f"detections saved    : {save_dir / 'detections.json'}")
    print(f"visualizations      : {save_dir / 'visualizations'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
