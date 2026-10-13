#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_leafdisease_with_leaf_roi.py

从 LeafDisease/aug_data + XML 叶片框标注，从头生成：

1) dino_coco
   - images/{train,val,test}
   - annotations/instances_{split}.json              # leaf + leaf lesion
   - annotations/instances_{split}_lesion_only.json  # 仅病斑，方便原流程评估
   - annotations/instances_{split}_leaf_only.json     # 仅叶片，方便单独训练/检查 leaf detector

2) deeplab_full
   - images/{train,val,test}
   - masks/{train,val,test}

3) deeplab_roi
   - lesion_roi/images/{train,val,test}
   - lesion_roi/masks/{train,val,test}
   - lesion_roi/manifest_{split}.csv
   - leaf_roi/images/{train,val,test}
   - leaf_roi/masks/{train,val,test}
   - leaf_roi/manifest_{split}.csv

说明：
- lesion_roi：根据病斑 mask 的连通域 bbox 裁剪，训练“病害局部分支”。
- leaf_roi：根据人工叶片 bbox 裁剪，mask 仍然是病斑 mask，训练“叶片级全局分支”。
- dino_coco 同时包含 leaf 和 leaf lesion 两类，便于后续 GroundingDINO 学 leaf + lesion。
- split 按文件名前缀分组，例如 00000_0.jpg, 00000_1.jpg ... 会进入同一个 split，避免增强样本泄漏。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--src-root", type=str, required=True,
                   help="LeafDisease 根目录，里面应有 aug_data/images, aug_data/masks, XML")
    p.add_argument("--out-root", type=str, required=True,
                   help="输出 prepared 数据集目录，例如 prepared_leaf_disease_leaf_level")
    p.add_argument("--images-subdir", type=str, default="aug_data/images")
    p.add_argument("--masks-subdir", type=str, default="aug_data/masks")
    p.add_argument("--leaf-ann-dir", type=str, default="XML")

    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--leaf-labels", type=str, default="leaf,叶子,叶片")
    p.add_argument("--accept-all-leaf-labels", action="store_true",
                   help="如果 XML/JSON 文件夹里只有叶片框，不管 label 名是什么都作为 leaf。")
    p.add_argument("--leaf-coord-format", choices=["center_xywh", "xywh"], default="center_xywh",
                   help="叶片 JSON 中 coordinates 的含义。center_xywh 表示 x/y 是框中心；xywh 表示 x/y 是左上角。默认 center_xywh。")

    p.add_argument("--lesion-category-id", type=int, default=1)
    p.add_argument("--leaf-category-id", type=int, default=2)
    p.add_argument("--lesion-category-name", type=str, default="leaf lesion")
    p.add_argument("--leaf-category-name", type=str, default="leaf")

    p.add_argument("--component-min-area", type=int, default=10,
                   help="病斑连通域最小面积，太小则忽略。")
    p.add_argument("--roi-padding", type=float, default=0.08,
                   help="病斑 ROI 外扩比例。0.08 表示宽高各外扩 8%。")
    p.add_argument("--leaf-padding", type=float, default=0.02,
                   help="叶片 ROI 外扩比例。")
    p.add_argument("--fallback-full-leaf", action="store_true",
                   help="某图没有 leaf 标注时，用整图作为 leaf ROI 和 leaf bbox。")

    p.add_argument("--copy-mode", choices=["copy", "symlink"], default="copy")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def family_id(stem: str) -> str:
    """
    让 00000_0, 00000_1, ..., 00000_4 归为同一组 00000。
    若最后一个下划线后的部分是数字，则去掉最后一段；否则直接返回 stem。
    """
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def find_mask(mask_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # 有些 mask 可能是 png
    for p in mask_dir.glob(stem + ".*"):
        if p.suffix.lower() in IMG_EXTS:
            return p
    return None


def copy_or_link(src: Path, dst: Path, mode: str):
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def load_mask_binary(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"))
    return (m > 0).astype(np.uint8)


def save_mask(path: Path, mask: np.ndarray):
    ensure_dir(path.parent)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def clip_xyxy(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(x1 + 1, min(w, round(x2))))
    y2 = int(max(y1 + 1, min(h, round(y2))))
    return x1, y1, x2, y2


def expand_box(box: Tuple[int, int, int, int], w: int, h: int, ratio: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * ratio, bh * ratio
    return clip_xyxy((x1 - dx, y1 - dy, x2 + dx, y2 + dy), w, h)


def xyxy_to_xywh(box: Tuple[int, int, int, int]) -> List[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def connected_components_boxes(mask: np.ndarray, min_area: int = 10) -> List[Tuple[int, int, int, int, int]]:
    """
    返回 [(x1,y1,x2,y2,area), ...]
    """
    try:
        import cv2
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        boxes = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if int(area) >= min_area:
                boxes.append((int(x), int(y), int(x + bw), int(y + bh), int(area)))
        return boxes
    except Exception:
        # 简单兜底：整张前景 bbox
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return []
        area = int((mask > 0).sum())
        if area < min_area:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1), area)]


def label_ok(label: str, label_set: set[str], accept_all: bool) -> bool:
    return accept_all or str(label).strip().lower() in label_set


def ann_obj_to_box(obj: Dict[str, Any], coord_format: str = "center_xywh") -> Optional[Tuple[float, float, float, float]]:
    # 当前格式：coordinates: {x,y,width,height}
    c = obj.get("coordinates")
    if isinstance(c, dict):
        if all(k in c for k in ["x", "y", "width", "height"]):
            x, y, bw, bh = float(c["x"]), float(c["y"]), float(c["width"]), float(c["height"])
            if coord_format == "center_xywh":
                return x - bw / 2.0, y - bh / 2.0, x + bw / 2.0, y + bh / 2.0
            return x, y, x + bw, y + bh
        if all(k in c for k in ["xmin", "ymin", "xmax", "ymax"]):
            return float(c["xmin"]), float(c["ymin"]), float(c["xmax"]), float(c["ymax"])

    if "points" in obj and isinstance(obj["points"], list):
        xs, ys = [], []
        for p in obj["points"]:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                xs.append(float(p[0])); ys.append(float(p[1]))
        if xs:
            return min(xs), min(ys), max(xs), max(ys)

    if "bbox" in obj and isinstance(obj["bbox"], list) and len(obj["bbox"]) >= 4:
        x, y, bw, bh = [float(v) for v in obj["bbox"][:4]]
        return x, y, x + bw, y + bh

    x = obj.get("x"); y = obj.get("y")
    bw = obj.get("width", obj.get("w")); bh = obj.get("height", obj.get("h"))
    if None not in [x, y, bw, bh]:
        x, y, bw, bh = float(x), float(y), float(bw), float(bh)
        return x, y, x + bw, y + bh

    return None


def read_leaf_json(path: Path, label_set: set[str], accept_all: bool, coord_format: str = "center_xywh") -> Tuple[Optional[str], List[Tuple[float, float, float, float]]]:
    data = load_json(path)
    image_name = None
    boxes: List[Tuple[float, float, float, float]] = []

    def handle_obj(obj: Dict[str, Any]):
        nonlocal image_name, boxes
        if image_name is None:
            image_name = obj.get("image", obj.get("imagePath", obj.get("filename", obj.get("file_name"))))

        if isinstance(obj.get("annotations"), list):
            for a in obj["annotations"]:
                if not isinstance(a, dict):
                    continue
                label = str(a.get("label", a.get("name", a.get("class", "")))).strip()
                if not label_ok(label, label_set, accept_all):
                    continue
                b = ann_obj_to_box(a, coord_format)
                if b is not None:
                    boxes.append(b)
            return

        if isinstance(obj.get("shapes"), list):
            for s in obj["shapes"]:
                if not isinstance(s, dict):
                    continue
                label = str(s.get("label", s.get("name", ""))).strip()
                if not label_ok(label, label_set, accept_all):
                    continue
                b = ann_obj_to_box(s, coord_format)
                if b is not None:
                    boxes.append(b)
            return

        label = str(obj.get("label", obj.get("name", obj.get("class", "")))).strip()
        if label_ok(label, label_set, accept_all):
            b = ann_obj_to_box(obj, coord_format)
            if b is not None:
                boxes.append(b)

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                handle_obj(obj)
    elif isinstance(data, dict):
        handle_obj(data)

    return image_name, boxes


def read_leaf_xml(path: Path, label_set: set[str], accept_all: bool) -> Tuple[Optional[str], List[Tuple[float, float, float, float]]]:
    root = ET.parse(path).getroot()
    fn_node = root.find("filename")
    image_name = fn_node.text.strip() if fn_node is not None and fn_node.text else None
    boxes = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        label = name_node.text.strip() if name_node is not None and name_node.text else ""
        if not label_ok(label, label_set, accept_all):
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        vals = {}
        ok = True
        for k in ["xmin", "ymin", "xmax", "ymax"]:
            n = bnd.find(k)
            if n is None or n.text is None:
                ok = False
                break
            vals[k] = float(n.text)
        if ok:
            boxes.append((vals["xmin"], vals["ymin"], vals["xmax"], vals["ymax"]))
    return image_name, boxes


def load_all_leaf_annotations(ann_dir: Path, label_set: set[str], accept_all: bool, coord_format: str = "center_xywh") -> Dict[str, List[Tuple[float, float, float, float]]]:
    out: Dict[str, List[Tuple[float, float, float, float]]] = defaultdict(list)
    files = sorted([
        p for p in ann_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in [".json", ".xml"] and "checkpoint" not in p.name.lower()
    ])
    for p in files:
        try:
            if p.suffix.lower() == ".json":
                image_name, boxes = read_leaf_json(p, label_set, accept_all, coord_format)
            else:
                image_name, boxes = read_leaf_xml(p, label_set, accept_all)
        except Exception as e:
            print(f"[WARN] failed to read leaf ann {p}: {e}")
            continue

        keys = set()
        keys.add(p.stem)
        if image_name:
            keys.add(Path(str(image_name)).stem)
        for k in keys:
            out[k].extend(boxes)
    return out


def split_by_family(images: List[Path], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, str]:
    fam_to_imgs = defaultdict(list)
    for p in images:
        fam_to_imgs[family_id(p.stem)].append(p)

    fams = sorted(fam_to_imgs.keys())
    rng = random.Random(seed)
    rng.shuffle(fams)

    n = len(fams)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)

    fam_split = {}
    for i, fam in enumerate(fams):
        if i < n_train:
            fam_split[fam] = "train"
        elif i < n_train + n_val:
            fam_split[fam] = "val"
        else:
            fam_split[fam] = "test"

    img_split = {}
    for fam, ps in fam_to_imgs.items():
        for p in ps:
            img_split[p.stem] = fam_split[fam]
    return img_split


def make_coco(categories: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"images": [], "annotations": [], "categories": categories}


def add_coco_image(coco: Dict[str, Any], image_id: int, file_name: str, w: int, h: int):
    coco["images"].append({"id": int(image_id), "file_name": file_name, "width": int(w), "height": int(h)})


def add_coco_ann(coco: Dict[str, Any], ann_id: int, image_id: int, cat_id: int, bbox_xyxy: Tuple[int, int, int, int], area: Optional[float] = None):
    bbox = xyxy_to_xywh(bbox_xyxy)
    if area is None:
        area = bbox[2] * bbox[3]
    coco["annotations"].append({
        "id": int(ann_id),
        "image_id": int(image_id),
        "category_id": int(cat_id),
        "bbox": [float(v) for v in bbox],
        "area": float(area),
        "iscrowd": 0,
    })


def crop_and_save(image: Image.Image, mask: np.ndarray, box: Tuple[int, int, int, int], img_out: Path, mask_out: Path):
    x1, y1, x2, y2 = box
    crop_img = image.crop((x1, y1, x2, y2))
    crop_mask = mask[y1:y2, x1:x2]
    ensure_dir(img_out.parent)
    crop_img.save(img_out)
    save_mask(mask_out, crop_mask)


def write_manifest(path: Path, rows: List[Dict[str, Any]]):
    ensure_dir(path.parent)
    fieldnames = [
        "image", "mask", "split", "roi_type", "source_image",
        "x1", "y1", "x2", "y2", "width", "height"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    image_dir = src_root / args.images_subdir
    mask_dir = src_root / args.masks_subdir
    leaf_ann_dir = src_root / args.leaf_ann_dir
    out_root = Path(args.out_root)

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    ensure_dir(out_root)

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.")

    images = list_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    image_to_mask = {}
    for img in images:
        m = find_mask(mask_dir, img.stem)
        if m is not None:
            image_to_mask[img.stem] = m

    valid_images = [p for p in images if p.stem in image_to_mask]
    if not valid_images:
        raise RuntimeError("No image-mask pairs found.")

    split_map = split_by_family(valid_images, args.train_ratio, args.val_ratio, args.seed)

    label_set = {x.strip().lower() for x in args.leaf_labels.split(",") if x.strip()}
    leaf_anns = load_all_leaf_annotations(leaf_ann_dir, label_set, args.accept_all_leaf_labels, args.leaf_coord_format)

    cats_both = [
        {"id": args.lesion_category_id, "name": args.lesion_category_name, "supercategory": "disease"},
        {"id": args.leaf_category_id, "name": args.leaf_category_name, "supercategory": "plant"},
    ]
    cats_lesion = [{"id": args.lesion_category_id, "name": args.lesion_category_name, "supercategory": "disease"}]
    cats_leaf = [{"id": args.leaf_category_id, "name": args.leaf_category_name, "supercategory": "plant"}]

    coco_both = {s: make_coco(cats_both) for s in ["train", "val", "test"]}
    coco_lesion = {s: make_coco(cats_lesion) for s in ["train", "val", "test"]}
    coco_leaf = {s: make_coco(cats_leaf) for s in ["train", "val", "test"]}

    manifests = {
        ("lesion_roi", "train"): [], ("lesion_roi", "val"): [], ("lesion_roi", "test"): [],
        ("leaf_roi", "train"): [], ("leaf_roi", "val"): [], ("leaf_roi", "test"): [],
    }

    image_id_counter = {s: 1 for s in ["train", "val", "test"]}
    ann_id_both = {s: 1 for s in ["train", "val", "test"]}
    ann_id_lesion = {s: 1 for s in ["train", "val", "test"]}
    ann_id_leaf = {s: 1 for s in ["train", "val", "test"]}

    stats = defaultdict(int)

    for img_path in valid_images:
        split = split_map[img_path.stem]
        mask_path = image_to_mask[img_path.stem]

        with Image.open(img_path).convert("RGB") as im:
            w, h = im.size
            mask = load_mask_binary(mask_path)

            # 1) copy full image/mask
            for root_name in ["dino_coco", "deeplab_full"]:
                dst_img = out_root / root_name / "images" / split / img_path.name
                copy_or_link(img_path, dst_img, args.copy_mode)

            full_mask_out = out_root / "deeplab_full" / "masks" / split / (img_path.stem + ".png")
            save_mask(full_mask_out, mask)

            # 2) coco image entries
            img_id = image_id_counter[split]
            image_id_counter[split] += 1
            add_coco_image(coco_both[split], img_id, img_path.name, w, h)
            add_coco_image(coco_lesion[split], img_id, img_path.name, w, h)
            add_coco_image(coco_leaf[split], img_id, img_path.name, w, h)

            # 3) lesion bboxes from disease mask
            lesion_components = connected_components_boxes(mask, args.component_min_area)
            for (x1, y1, x2, y2, area) in lesion_components:
                box = clip_xyxy((x1, y1, x2, y2), w, h)
                add_coco_ann(coco_both[split], ann_id_both[split], img_id, args.lesion_category_id, box, area)
                ann_id_both[split] += 1
                add_coco_ann(coco_lesion[split], ann_id_lesion[split], img_id, args.lesion_category_id, box, area)
                ann_id_lesion[split] += 1

                # lesion ROI crop
                roi_box = expand_box(box, w, h, args.roi_padding)
                roi_name = f"{img_path.stem}_lesion_{len(manifests[('lesion_roi', split)]):04d}.jpg"
                roi_mask_name = roi_name.replace(".jpg", ".png")
                roi_img_out = out_root / "deeplab_roi" / "lesion_roi" / "images" / split / roi_name
                roi_mask_out = out_root / "deeplab_roi" / "lesion_roi" / "masks" / split / roi_mask_name
                crop_and_save(im, mask, roi_box, roi_img_out, roi_mask_out)
                manifests[("lesion_roi", split)].append({
                    "image": str(roi_img_out.relative_to(out_root)),
                    "mask": str(roi_mask_out.relative_to(out_root)),
                    "split": split,
                    "roi_type": "lesion_roi",
                    "source_image": img_path.name,
                    "x1": roi_box[0], "y1": roi_box[1], "x2": roi_box[2], "y2": roi_box[3],
                    "width": roi_box[2] - roi_box[0],
                    "height": roi_box[3] - roi_box[1],
                })

            # 4) leaf boxes from annotations
            raw_leaf_boxes = leaf_anns.get(img_path.stem, [])
            leaf_boxes: List[Tuple[int, int, int, int]] = []
            for b in raw_leaf_boxes:
                leaf_boxes.append(clip_xyxy(b, w, h))

            if not leaf_boxes and args.fallback_full_leaf:
                leaf_boxes = [(0, 0, w, h)]

            if not leaf_boxes:
                stats["images_without_leaf_box"] += 1

            for li, lb in enumerate(leaf_boxes):
                # coco leaf
                add_coco_ann(coco_both[split], ann_id_both[split], img_id, args.leaf_category_id, lb)
                ann_id_both[split] += 1
                add_coco_ann(coco_leaf[split], ann_id_leaf[split], img_id, args.leaf_category_id, lb)
                ann_id_leaf[split] += 1

                # leaf ROI crop: mask is still disease mask
                leaf_roi_box = expand_box(lb, w, h, args.leaf_padding)
                leaf_roi_name = f"{img_path.stem}_leaf_{li:02d}.jpg"
                leaf_roi_mask_name = leaf_roi_name.replace(".jpg", ".png")
                leaf_img_out = out_root / "deeplab_roi" / "leaf_roi" / "images" / split / leaf_roi_name
                leaf_mask_out = out_root / "deeplab_roi" / "leaf_roi" / "masks" / split / leaf_roi_mask_name
                crop_and_save(im, mask, leaf_roi_box, leaf_img_out, leaf_mask_out)
                manifests[("leaf_roi", split)].append({
                    "image": str(leaf_img_out.relative_to(out_root)),
                    "mask": str(leaf_mask_out.relative_to(out_root)),
                    "split": split,
                    "roi_type": "leaf_roi",
                    "source_image": img_path.name,
                    "x1": leaf_roi_box[0], "y1": leaf_roi_box[1], "x2": leaf_roi_box[2], "y2": leaf_roi_box[3],
                    "width": leaf_roi_box[2] - leaf_roi_box[0],
                    "height": leaf_roi_box[3] - leaf_roi_box[1],
                })

            stats[f"{split}_images"] += 1
            stats[f"{split}_lesion_boxes"] += len(lesion_components)
            stats[f"{split}_leaf_boxes"] += len(leaf_boxes)

    # 5) save coco annotations
    ann_root = out_root / "dino_coco" / "annotations"
    for split in ["train", "val", "test"]:
        save_json(ann_root / f"instances_{split}.json", coco_both[split])
        save_json(ann_root / f"instances_{split}_lesion_only.json", coco_lesion[split])
        save_json(ann_root / f"instances_{split}_leaf_only.json", coco_leaf[split])

    # 6) save manifests
    for roi_type in ["lesion_roi", "leaf_roi"]:
        for split in ["train", "val", "test"]:
            write_manifest(out_root / "deeplab_roi" / roi_type / f"manifest_{split}.csv", manifests[(roi_type, split)])

    # 7) save split info and summary
    split_info = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "family_rule": "remove last underscore suffix if numeric, e.g. 00000_0 -> 00000",
        "splits": split_map,
    }
    save_json(out_root / "split_info.json", split_info)

    summary = dict(stats)
    summary.update({
        "src_root": str(src_root),
        "out_root": str(out_root),
        "num_all_images": len(images),
        "num_valid_image_mask_pairs": len(valid_images),
        "num_leaf_annotation_keys": len(leaf_anns),
        "leaf_coord_format": args.leaf_coord_format,
        "lesion_category_id": args.lesion_category_id,
        "leaf_category_id": args.leaf_category_id,
        "outputs": [
            "dino_coco/images/{train,val,test}",
            "dino_coco/annotations/instances_{split}.json",
            "dino_coco/annotations/instances_{split}_lesion_only.json",
            "dino_coco/annotations/instances_{split}_leaf_only.json",
            "deeplab_full/images/{train,val,test}",
            "deeplab_full/masks/{train,val,test}",
            "deeplab_roi/lesion_roi/images,masks,manifest",
            "deeplab_roi/leaf_roi/images,masks,manifest",
        ]
    })
    save_json(out_root / "summary.json", summary)

    print("=" * 90)
    print("Dataset preparation finished.")
    print(f"src_root                 : {src_root}")
    print(f"out_root                 : {out_root}")
    print(f"all images               : {len(images)}")
    print(f"valid image-mask pairs   : {len(valid_images)}")
    print(f"leaf annotation keys     : {len(leaf_anns)}")
    print("-" * 90)
    for split in ["train", "val", "test"]:
        print(f"{split:5s} images={summary.get(split+'_images',0)} "
              f"lesion_boxes={summary.get(split+'_lesion_boxes',0)} "
              f"leaf_boxes={summary.get(split+'_leaf_boxes',0)}")
    print(f"images without leaf box  : {summary.get('images_without_leaf_box',0)}")
    print(f"summary                  : {out_root / 'summary.json'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
