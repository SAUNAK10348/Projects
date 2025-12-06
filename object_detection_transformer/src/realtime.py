from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from PIL import Image
from torchvision import transforms

from .inference import load_model
from .pipeline import FusionNetPipeline


class RealtimeConfig(argparse.Namespace):
    checkpoint: Path
    labels: List[str]
    image_size: int
    device: str
    num_queries: int
    camera: int | None
    video: Path | None
    confidence: float
    max_frames: int | None
    show_window: bool


def preprocess_frame(frame, image_size: int, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    tensor = tfm(pil_image).unsqueeze(0)
    return tensor.to(device)


def xywh_to_xyxy(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x_center, y_center, bw, bh = box
    x0 = int((x_center - bw / 2) * width)
    y0 = int((y_center - bh / 2) * height)
    x1 = int((x_center + bw / 2) * width)
    y1 = int((y_center + bh / 2) * height)
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def draw_predictions(frame, predictions, confidence: float) -> None:
    h, w, _ = frame.shape
    for pred in predictions:
        if pred["score"] < confidence:
            continue
        x0, y0, x1, y1 = xywh_to_xyxy(pred["box_xywh"], w, h)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{pred['class']}: {pred['score']:.2f}"
        cv2.putText(frame, label, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def parse_args() -> RealtimeConfig:
    parser = argparse.ArgumentParser(description="Run real-time inference from webcam or video file")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint saved by train.py")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="Class names")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-queries", type=int, default=25)
    parser.add_argument("--camera", type=int, help="Camera index (default 0)")
    parser.add_argument("--video", type=Path, help="Video file to stream")
    parser.add_argument("--confidence", type=float, default=0.3, help="Score threshold for drawing boxes")
    parser.add_argument("--max-frames", type=int, help="Limit frames processed (useful for benchmarks)")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV window for headless runs")
    args = parser.parse_args(namespace=RealtimeConfig())

    if args.camera is None and args.video is None:
        args.camera = 0
    return args


def main() -> None:
    config = parse_args()
    config.label_map = config.labels
    model = load_model(config)
    pipeline = FusionNetPipeline(model=model, label_map=config.label_map)

    if config.video is not None:
        source = str(config.video)
    else:
        # Use integer camera index directly to avoid OpenCV treating it as a filename
        source = config.camera if config.camera is not None else 0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    frame_count = 0
    window_name = "DetTransformer"
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        tensor = preprocess_frame(frame, config.image_size, config.device)
        outputs = pipeline.forward(tensor)
        predictions = outputs["detections"]
        draw_predictions(frame, predictions, config.confidence)

        if not config.no_window:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        now = time.time()
        if now - last_print >= 1.0:
            print(
                f"Processed {frame_count} frames | Last scores: {[p['score'] for p in predictions[:3]]} | RMS vel: {outputs['metrics'].rms_velocity_overall:.3f}"
            )
            last_print = now

        if config.max_frames and frame_count >= config.max_frames:
            break

    cap.release()
    if not config.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
