from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import torch
from .dataset import YoloDataset, collate_yolo
from .model import DetectionTransformer


def load_model(checkpoint: Path, device: torch.device, num_classes: int, num_queries: int) -> DetectionTransformer:
    state = torch.load(checkpoint, map_location=device)
    cfg = state.get("config", {})
    model = DetectionTransformer(
        num_classes=num_classes or cfg.get("num_classes", 80),
        num_queries=num_queries or cfg.get("num_queries", 25),
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_encoder_layers=cfg.get("num_encoder_layers", 4),
        dim_feedforward=cfg.get("dim_feedforward", 256),
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model


def benchmark(
    checkpoint: Path,
    data_dir: Path,
    device_str: str,
    warmup: int,
    iterations: int,
    num_classes: int,
    num_queries: int,
    quantize: bool,
) -> None:
    device = torch.device(device_str)
    dataset = YoloDataset(data_dir=data_dir, image_size=128, max_detections=num_queries)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_yolo)
    model = load_model(checkpoint, device, num_classes=num_classes, num_queries=num_queries)

    if quantize:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    # warmup
    with torch.inference_mode():
        for i, (images, _, _) in enumerate(loader):
            images = images.to(device)
            _ = model(images)
            if i + 1 >= warmup:
                break

    times: List[float] = []
    with torch.inference_mode():
        for i, (images, _, _) in enumerate(loader):
            if i >= iterations:
                break
            images = images.to(device)
            start = time.perf_counter()
            _ = model(images)
            torch.cuda.synchronize() if device.type == "cuda" else None
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    if not times:
        print("No samples available for benchmarking.")
        return
    print(f"Ran {len(times)} iterations")
    print(f"avg latency: {sum(times) / len(times):.3f} ms | min: {min(times):.3f} ms | max: {max(times):.3f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency benchmark for the detection transformer")
    parser.add_argument("checkpoint", type=Path, help="Path to trained checkpoint")
    parser.add_argument("data_dir", type=Path, help="Dataset root (for sample images)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--num-queries", type=int, default=25)
    parser.add_argument("--quantize", action="store_true", help="Enable dynamic int8 quantization for Linear layers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        device_str=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
