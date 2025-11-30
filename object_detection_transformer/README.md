# Tensor-Valued Object Detection Transformer

This folder contains a runnable skeleton for an object-detection transformer that ingests YOLO-style labels. It provides:

- A dataset loader that reads images plus `*.txt` YOLO labels.
- An encoder that flattens images to tensors and feeds a multi-head attention stack.
- A detection transformer with learned queries and explainability hooks (attention maps and weight export).
- Training, inference, and real-time demo scripts for **Windows/PowerShell**.

The code is deliberately lightweight so it can run on CPU for smoke tests while remaining compatible with GPU training.

## Project Layout
- `src/` contains the transformer model, dataset utilities, and training/inference entrypoints.
- `scripts/` contains PowerShell helpers for Windows.
- `requirements.txt` lists runtime dependencies.

## Quickstart (Windows PowerShell)
```pwsh
cd object_detection_transformer
./scripts/setup_workspace.ps1            # creates .venv and installs deps
./.venv/Scripts/Activate.ps1

# Train on a YOLO-formatted dataset (images/*.jpg, labels/*.txt)
./scripts/train.ps1 /path/to/dataset --epochs 1 --batch-size 4 --num-classes 80 --num-queries 25

# Run single-image inference
python -m src.inference checkpoints/det_transformer.pt /path/to/test.jpg --labels person car dog

# Stream real-time webcam/video inference with OpenCV overlay
./scripts/realtime.ps1 checkpoints/det_transformer.pt --labels person car dog --camera 0 --confidence 0.4
```

The dataset folder should contain `images/*.jpg` and `labels/*.txt` files in YOLO format (`class x_center y_center width height`).

## Training Guide
- **Inputs**: place resized training images under `dataset/images/` and matching YOLO label files under `dataset/labels/`.
- **Command**: `./scripts/train.ps1 dataset --epochs 50 --batch-size 16 --num-classes <classes> --num-queries <queries>`.
- **Outputs**: checkpoints are written to `checkpoints/det_transformer.pt` along with basic optimizer state for resuming.
- **GPU use**: add `--device cuda` if a CUDA-capable GPU and drivers are present.

## Testing & Evaluation Guide
- **Smoke test**: `python -m compileall src` verifies the module graph after edits.
- **Single image**: `python -m src.inference checkpoints/det_transformer.pt sample.jpg --labels ...`.
- **Batch eval**: run inference over a folder via a short PowerShell loop (e.g., `Get-ChildItem images/*.jpg | ForEach-Object { python -m src.inference ... $_ ... }`).
- **Explainability**: load a checkpoint and call `model.explain()` to retrieve encoder attention maps for visualization.

## Real-Time Deployment Notes
- The `src.realtime` entrypoint opens a webcam or video file, overlays bounding boxes, and prints lightweight telemetry once per second.
- Use `--no-window` for headless or remote deployments and pipe the logs to your monitoring system.
- Tune `--image-size` and `--num-queries` to balance accuracy vs. latency for 60 FPS pipelines.
