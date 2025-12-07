from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .dataset import YoloDataset, collate_yolo
from .model import DetectionTransformer


def compute_loss(
    outputs, target_boxes, target_classes, num_classes: int, criterion: nn.Module
) -> torch.Tensor:
    class_logits = outputs["class_logits"]  # (B, num_queries, num_classes)
    pred_boxes = outputs["boxes"]  # (B, num_queries, 4)

    # Match predictions to ground-truth by index; here we perform a naive positional match
    # which keeps the code lightweight. Production systems should use Hungarian matching.
    gt_classes = target_classes
    gt_boxes = target_boxes

    loss_ce = criterion(class_logits.flatten(0, 1), gt_classes.flatten())
    loss_bbox = nn.functional.smooth_l1_loss(pred_boxes, gt_boxes, reduction="mean")
    loss = loss_ce + loss_bbox
    return loss


def train_epoch(
    model: DetectionTransformer,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    for step, (images, boxes, classes) in enumerate(data_loader, start=1):
        images = images.to(device)
        boxes = boxes.to(device)
        classes = classes.to(device)

        outputs = model(images)
        loss = compute_loss(outputs, boxes, classes, num_classes=num_classes, criterion=criterion)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        if step % log_every == 0:
            print(f"step={step} loss={loss.item():.4f}")
    return total_loss / max(len(data_loader), 1)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train detection transformer on YOLO data")
    parser.add_argument("data_dir", type=Path, help="Root dataset directory containing images/ and labels/")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--num-queries", type=int, default=25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    return TrainingConfig(
        data_dir=args.data_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        log_every=args.log_every,
    )


def main() -> None:
    config = parse_args()
    device = torch.device(config.device)

    dataset = YoloDataset(data_dir=config.data_dir, image_size=config.image_size, max_detections=config.num_queries)
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_yolo,
    )

    model = DetectionTransformer(
        num_classes=config.num_classes,
        num_queries=config.num_queries,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(config.max_epochs):
        avg_loss = train_epoch(
            model,
            data_loader,
            optimizer,
            criterion,
            device,
            num_classes=config.num_classes,
            log_every=config.log_every,
        )
        print(f"epoch={epoch+1} avg_loss={avg_loss:.4f}")

    checkpoint_path = Path("checkpoints") / "det_transformer.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
