#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_groundingdino_leaf_lesion_coco.py

Train / fine-tune GroundingDINO on a prepared COCO dataset containing BOTH:
  1) leaf
  2) leaf lesion

This script is modified from your previous single-class lesion training script. The key change is:
- It reads COCO categories.
- It maps category names to prompt indices.
- It assigns class_labels according to category order, not all zeros.

Recommended COCO categories
---------------------------
prepared_leaf_disease_leaf_level/dino_coco/annotations/instances_train.json

categories:
  {"id": 1, "name": "leaf lesion"}
  {"id": 2, "name": "leaf"}

Recommended command
-------------------
python train_groundingdino_leaf_lesion_coco.py \
  --train-ann prepared_leaf_disease_leaf_level/dino_coco/annotations/instances_train.json \
  --val-ann prepared_leaf_disease_leaf_level/dino_coco/annotations/instances_val.json \
  --train-image-root prepared_leaf_disease_leaf_level/dino_coco/images/train \
  --val-image-root prepared_leaf_disease_leaf_level/dino_coco/images/val \
  --output-dir runs/gdino_leaf_lesion_joint \
  --model-id IDEA-Research/grounding-dino-tiny \
  --category-names "leaf,leaf lesion" \
  --category-prompts "leaf|leaf lesion" \
  --epochs 40 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 2e-5 \
  --weight-decay 1e-4 \
  --warmup-ratio 0.1 \
  --num-workers 4 \
  --patience 8 \
  --min-delta 1e-4 \
  --save-every 5 \
  --clip-grad-norm 1.0 \
  --amp \
  --freeze-text-encoder

Notes
-----
1) category-names and category-prompts must have the same order.
   Example:
     category-names   = leaf,leaf lesion
     category-prompts = leaf|leaf lesion
   Then:
     leaf        -> class_label 0
     leaf lesion -> class_label 1

2) The prompt used for every image will be:
     "leaf . leaf lesion ."
   unless you manually provide --fixed-prompt.

3) If OOM occurs, add:
     --shortest-edge 640 --longest-edge 960
   or:
     --shortest-edge 512 --longest-edge 768
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.utils._pytree as pytree

# transformers / pytorch compatibility patch for some environments
# MUST be applied before importing transformers
if not hasattr(pytree, "register_pytree_node"):
    def _compat_register_pytree_node(nodetype, flatten_fn, unflatten_fn, **kwargs):
        return pytree._register_pytree_node(nodetype, flatten_fn, unflatten_fn)
    pytree.register_pytree_node = _compat_register_pytree_node

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_list_csv(x: str) -> List[str]:
    return [v.strip() for v in x.split(",") if v.strip()]


def parse_list_pipe(x: str) -> List[str]:
    return [v.strip() for v in x.split("|") if v.strip()]


def build_prompt_from_phrases(phrases: Sequence[str]) -> str:
    parts = []
    for p in phrases:
        p = p.strip()
        if not p:
            continue
        p = p.rstrip(".").strip()
        parts.append(p)
    return " . ".join(parts) + " ."


class CocoGroundingMultiClassDataset(Dataset):
    def __init__(
        self,
        ann_path: str,
        image_root: str,
        category_names: Sequence[str],
        category_prompts: Sequence[str],
        fixed_prompt: Optional[str] = None,
        min_box_size: float = 1.0,
        keep_empty_images: bool = True,
    ) -> None:
        self.ann_path = Path(ann_path)
        self.image_root = Path(image_root)
        self.min_box_size = float(min_box_size)
        self.keep_empty_images = bool(keep_empty_images)

        self.category_names = [x.strip() for x in category_names]
        self.category_prompts = [x.strip() for x in category_prompts]
        if len(self.category_names) != len(self.category_prompts):
            raise ValueError("--category-names and --category-prompts must have the same length.")

        self.prompt = fixed_prompt.strip() if fixed_prompt else build_prompt_from_phrases(self.category_prompts)

        # category name -> class label index in prompt order
        self.cat_name_to_class_label = {
            name.lower(): i for i, name in enumerate(self.category_names)
        }

        coco = load_json(self.ann_path)
        self.images = coco.get("images", [])
        anns = coco.get("annotations", [])
        cats = coco.get("categories", [])

        self.coco_cat_id_to_name = {int(c["id"]): str(c["name"]) for c in cats}
        self.coco_cat_id_to_class_label: Dict[int, int] = {}
        for cid, cname in self.coco_cat_id_to_name.items():
            key = cname.strip().lower()
            if key in self.cat_name_to_class_label:
                self.coco_cat_id_to_class_label[cid] = self.cat_name_to_class_label[key]

        if not self.coco_cat_id_to_class_label:
            raise RuntimeError(
                f"No COCO categories match --category-names={self.category_names}. "
                f"COCO categories are: {self.coco_cat_id_to_name}"
            )

        self.image_id_to_anns: Dict[int, List[dict]] = defaultdict(list)
        for ann in anns:
            cid = int(ann.get("category_id", -1))
            if cid not in self.coco_cat_id_to_class_label:
                continue
            self.image_id_to_anns[int(ann["image_id"])].append(ann)

        # only keep images that actually exist on disk
        filtered_images = []
        missing = 0
        empty = 0
        for item in self.images:
            img_path = self.image_root / item["file_name"]
            if not img_path.exists():
                missing += 1
                continue
            if not self.keep_empty_images and len(self.image_id_to_anns.get(int(item["id"]), [])) == 0:
                empty += 1
                continue
            filtered_images.append(item)

        self.images = filtered_images
        if missing > 0:
            print(f"[WARN] {missing} images listed in {self.ann_path.name} are missing from disk.")
        if empty > 0:
            print(f"[WARN] {empty} empty images removed from dataset.")

        self.class_counts = defaultdict(int)
        self.valid_box_count = 0
        for item in self.images:
            image_id = int(item["id"])
            for ann in self.image_id_to_anns.get(image_id, []):
                cid = int(ann["category_id"])
                cl = self.coco_cat_id_to_class_label[cid]
                self.class_counts[int(cl)] += 1
                self.valid_box_count += 1

    def __len__(self) -> int:
        return len(self.images)

    def get_stats(self) -> dict:
        by_name = {}
        for i, name in enumerate(self.category_names):
            by_name[name] = int(self.class_counts.get(i, 0))
        return {
            "num_images": len(self.images),
            "num_boxes": int(self.valid_box_count),
            "class_counts": by_name,
            "prompt": self.prompt,
            "coco_cat_id_to_name": self.coco_cat_id_to_name,
            "coco_cat_id_to_class_label": self.coco_cat_id_to_class_label,
        }

    def __getitem__(self, idx: int) -> dict:
        item = self.images[idx]
        image_id = int(item["id"])
        image_path = self.image_root / item["file_name"]

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        class_labels = []
        for ann in self.image_id_to_anns.get(image_id, []):
            cid = int(ann["category_id"])
            if cid not in self.coco_cat_id_to_class_label:
                continue

            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w < self.min_box_size or h < self.min_box_size:
                continue

            cx = (x + w / 2.0) / width
            cy = (y + h / 2.0) / height
            nw = w / width
            nh = h / height

            # Clamp normalized boxes to avoid invalid labels.
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(1e-6, min(1.0, nw))
            nh = max(1e-6, min(1.0, nh))

            boxes.append([cx, cy, nw, nh])
            class_labels.append(int(self.coco_cat_id_to_class_label[cid]))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            class_labels_t = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            class_labels_t = torch.tensor(class_labels, dtype=torch.long)

        return {
            "image": image,
            "text": self.prompt,
            "labels": {
                "boxes": boxes_t,
                "class_labels": class_labels_t,
            },
            "image_path": str(image_path),
        }


class GroundingCollator:
    def __init__(self, processor, shortest_edge: Optional[int] = None, longest_edge: Optional[int] = None):
        self.processor = processor
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge

    def __call__(self, batch: Sequence[dict]) -> dict:
        images = [x["image"] for x in batch]
        texts = [x["text"] for x in batch]
        labels = [x["labels"] for x in batch]

        kwargs = dict(images=images, text=texts, return_tensors="pt", padding=True)
        if self.shortest_edge is not None or self.longest_edge is not None:
            size = {}
            if self.shortest_edge is not None:
                size["shortest_edge"] = int(self.shortest_edge)
            if self.longest_edge is not None:
                size["longest_edge"] = int(self.longest_edge)
            kwargs["size"] = size

        try:
            inputs = self.processor(**kwargs)
        except TypeError:
            # Some old processors do not accept size at call time.
            kwargs.pop("size", None)
            inputs = self.processor(**kwargs)

        inputs["labels"] = labels
        inputs["image_paths"] = [x["image_path"] for x in batch]
        return inputs


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if k == "labels":
            out[k] = [{kk: vv.to(device) for kk, vv in x.items()} for x in v]
        elif k == "image_paths":
            out[k] = v
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def freeze_text_encoder_if_needed(model, freeze: bool):
    if not freeze:
        return
    frozen = 0
    total = 0
    for name, param in model.named_parameters():
        total += 1
        lname = name.lower()
        if (
            "text_model" in lname
            or "bert" in lname
            or "language" in lname
            or "text_encoder" in lname
        ):
            param.requires_grad = False
            frozen += 1
    print(f"[INFO] freeze_text_encoder=True, frozen parameter tensors: {frozen}/{total}")


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def save_checkpoint(
    save_path: Path,
    model,
    processor,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val: float,
    args: dict,
):
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "args": args,
        },
        save_path / "trainer_state.pt",
    )


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled: bool) -> dict:
    model.eval()
    meter = AverageMeter()

    for batch in tqdm(loader, desc="[VAL]", leave=False):
        batch = move_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_mask=batch.get("pixel_mask"),
                labels=batch["labels"],
            )
            loss = outputs.loss
        bs = batch["pixel_values"].shape[0]
        meter.update(float(loss.item()), bs)

    return {"val_loss": meter.avg}


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    amp_enabled: bool,
    grad_accum_steps: int,
    clip_grad_norm: float,
) -> dict:
    model.train()
    loss_meter = AverageMeter()
    step_meter = AverageMeter()

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="[TRAIN]", leave=False)
    for step, batch in enumerate(pbar, start=1):
        t0 = time.time()
        batch = move_to_device(batch, device)
        bs = batch["pixel_values"].shape[0]

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_mask=batch.get("pixel_mask"),
                labels=batch["labels"],
            )
            raw_loss = outputs.loss
            loss = raw_loss / grad_accum_steps

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        do_step = (step % grad_accum_steps == 0) or (step == len(loader))
        if do_step:
            if scaler is not None and amp_enabled:
                scaler.unscale_(optimizer)
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        loss_meter.update(float(raw_loss.item()), bs)
        step_meter.update(time.time() - t0)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", step_s=f"{step_meter.avg:.2f}")

    return {"train_loss": loss_meter.avg}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train-ann", type=str, required=True)
    p.add_argument("--val-ann", type=str, required=True)
    p.add_argument("--train-image-root", type=str, required=True)
    p.add_argument("--val-image-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)

    p.add_argument("--model-id", type=str, default="IDEA-Research/grounding-dino-tiny")

    # Multi-class category configuration
    p.add_argument("--category-names", type=str, default="leaf,leaf lesion",
                   help="COCO category names to train, comma-separated. Order must match category-prompts.")
    p.add_argument("--category-prompts", type=str, default="leaf|leaf lesion",
                   help="Prompt phrases, separated by '|'. Order must match category-names.")
    p.add_argument("--fixed-prompt", type=str, default=None,
                   help='Optional manual prompt. If omitted, built from --category-prompts, e.g. "leaf . leaf lesion ."')

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--freeze-text-encoder", action="store_true")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint directory containing trainer_state.pt")
    p.add_argument("--min-box-size", type=float, default=1.0)
    p.add_argument("--drop-empty-images", action="store_true")

    # Optional resize to avoid OOM
    p.add_argument("--shortest-edge", type=int, default=None)
    p.add_argument("--longest-edge", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_from = args.resume if args.resume else args.model_id
    processor = AutoProcessor.from_pretrained(load_from)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(load_from)

    freeze_text_encoder_if_needed(model, args.freeze_text_encoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    category_names = parse_list_csv(args.category_names)
    category_prompts = parse_list_pipe(args.category_prompts)

    train_dataset = CocoGroundingMultiClassDataset(
        ann_path=args.train_ann,
        image_root=args.train_image_root,
        category_names=category_names,
        category_prompts=category_prompts,
        fixed_prompt=args.fixed_prompt,
        min_box_size=args.min_box_size,
        keep_empty_images=not args.drop_empty_images,
    )
    val_dataset = CocoGroundingMultiClassDataset(
        ann_path=args.val_ann,
        image_root=args.val_image_root,
        category_names=category_names,
        category_prompts=category_prompts,
        fixed_prompt=args.fixed_prompt,
        min_box_size=args.min_box_size,
        keep_empty_images=not args.drop_empty_images,
    )

    collator = GroundingCollator(
        processor,
        shortest_edge=args.shortest_edge,
        longest_edge=args.longest_edge,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
        drop_last=False,
    )

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    total_optimizer_steps = math.ceil(len(train_loader) / max(1, args.grad_accum_steps)) * args.epochs
    warmup_steps = int(round(total_optimizer_steps * args.warmup_ratio))

    def lr_lambda(step: int):
        if total_optimizer_steps <= 0:
            return 1.0
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    if args.resume:
        state_path = Path(args.resume) / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            if state.get("optimizer") is not None:
                optimizer.load_state_dict(state["optimizer"])
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            if state.get("scaler") is not None and scaler is not None:
                scaler.load_state_dict(state["scaler"])
            start_epoch = int(state.get("epoch", 0)) + 1
            best_val = float(state.get("best_val", float("inf")))
            print(f"[INFO] resumed from {args.resume} at epoch {start_epoch}, best_val={best_val:.6f}")
        else:
            print(f"[WARN] resume checkpoint has no trainer_state.pt: {state_path}. Loaded model weights only.")

    cfg = vars(args).copy()
    cfg["train_size"] = len(train_dataset)
    cfg["val_size"] = len(val_dataset)
    cfg["train_stats"] = train_dataset.get_stats()
    cfg["val_stats"] = val_dataset.get_stats()

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    history = []

    print("=" * 90)
    print("GroundingDINO leaf + lesion joint training")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Device        : {device}")
    print(f"AMP enabled   : {amp_enabled}")
    print(f"Prompt        : {train_dataset.prompt}")
    print(f"Category order: {category_names}")
    print(f"Train stats   : {json.dumps(train_dataset.get_stats()['class_counts'], ensure_ascii=False)}")
    print(f"Val stats     : {json.dumps(val_dataset.get_stats()['class_counts'], ensure_ascii=False)}")
    if args.shortest_edge or args.longest_edge:
        print(f"Processor size: shortest_edge={args.shortest_edge}, longest_edge={args.longest_edge}")
    print("=" * 90)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_log = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_accum_steps=max(1, args.grad_accum_steps),
            clip_grad_norm=args.clip_grad_norm,
        )
        val_log = evaluate(model, val_loader, device, amp_enabled)

        used_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "lr": current_lr,
            **train_log,
            **val_log,
            "time_sec": used_time,
        }
        history.append(record)
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_log['train_loss']:.6f} "
            f"val_loss={val_log['val_loss']:.6f} "
            f"lr={current_lr:.3e} time={used_time/60.0:.1f}min"
        )

        save_checkpoint(out_dir / "last", model, processor, optimizer, scheduler, scaler, epoch, best_val, cfg)

        improved = (best_val - val_log["val_loss"]) > args.min_delta
        if improved:
            best_val = val_log["val_loss"]
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(out_dir / "best", model, processor, optimizer, scheduler, scaler, epoch, best_val, cfg)
            print(f"[INFO] best model updated at epoch {epoch}, val_loss={best_val:.6f}")
        else:
            bad_epochs += 1

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(out_dir / f"checkpoint-epoch-{epoch}", model, processor, optimizer, scheduler, scaler, epoch, best_val, cfg)

        if bad_epochs >= args.patience:
            print(f"[INFO] early stopping triggered at epoch {epoch}. best_epoch={best_epoch}, best_val={best_val:.6f}")
            break

    print("=" * 90)
    print(f"Finished. best_epoch={best_epoch}, best_val={best_val:.6f}")
    print(f"Output dir: {out_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
