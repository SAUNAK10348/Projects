from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

from .config import InferenceConfig
from .model import DetectionTransformer
from .pipeline import FusionNetPipeline


def load_model(config: InferenceConfig) -> DetectionTransformer:
    checkpoint = torch.load(config.checkpoint, map_location=config.device)
    model_config = checkpoint.get("config", {})
    model = DetectionTransformer(
        num_classes=len(config.label_map),
        num_queries=model_config.get("num_queries", config.num_queries),
        d_model=model_config.get("d_model", 128),
        nhead=model_config.get("nhead", 4),
        num_encoder_layers=model_config.get("num_encoder_layers", 4),
        dim_feedforward=model_config.get("dim_feedforward", 256),
        dropout=model_config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.device)
    model.eval()
    return model


def prepare_image(image_path: Path, image_size: int) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return tfm(image).unsqueeze(0)


def run_inference(model: DetectionTransformer, image: torch.Tensor, label_map: List[str], memory_path: Path | None = None):
    pipeline = FusionNetPipeline(model=model, label_map=label_map, memory_path=memory_path)
    with torch.no_grad():
        outputs = pipeline.forward(image)
    return outputs


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(description="Run inference with detection transformer")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint saved by train.py")
    parser.add_argument("image", type=Path, help="Image to run inference on")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="Class names")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-queries", type=int, default=25)
    parser.add_argument("--memory-path", type=Path, default=None, help="Optional path to persist the knowledge graph memory")
    args = parser.parse_args()

    return InferenceConfig(
        checkpoint=args.checkpoint,
        image=args.image,
        label_map=args.labels,
        image_size=args.image_size,
        device=args.device,
        num_queries=args.num_queries,
        memory_path=args.memory_path,
    )


def main() -> None:
    config = parse_args()
    model = load_model(config)
    image_tensor = prepare_image(config.image, config.image_size)
    image_tensor = image_tensor.to(config.device)

    outputs = run_inference(model, image_tensor, config.label_map, memory_path=getattr(config, "memory_path", None))
    for pred in outputs["detections"]:
        print(pred)
    print("GNN metrics:", outputs["metrics"])


if __name__ == "__main__":
    main()
