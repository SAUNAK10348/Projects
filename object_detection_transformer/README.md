# Tensor-Valued Object Detection Transformer

This folder contains a runnable skeleton for an object-detection transformer that ingests YOLO-style labels. It now ships helpers for **Windows (PowerShell)** and **Linux (Bash)**. It provides:

- A dataset loader that reads images plus `*.txt` YOLO labels.
- An encoder that flattens images to tensors and feeds a multi-head attention stack.
- A detection transformer with learned queries and explainability hooks (attention maps and weight export).
- Training, inference, and real-time demo scripts for **Windows/PowerShell**.

The code is deliberately lightweight so it can run on CPU for smoke tests while remaining compatible with GPU training.

## Project Layout
- `src/` contains the transformer model, dataset utilities, and training/inference entrypoints.
- `scripts/` contains PowerShell helpers for Windows (PowerShell Core works cross-platform) and Bash helpers for Linux/macOS.
- `requirements.txt` lists runtime dependencies.
- `use.md` provides a Codespaces walkthrough (works for other Linux hosts too).
- `how_to_use.txt` is a detailed step-by-step manual for training, testing, inference, and real-time runs on Windows and Linux.
- `technical_report.txt` is an in-depth architectural report covering data flow, transformer design, and deployment notes.
- `kaggle_fusionnet_notebook.py` is a notebook-style script to copy into Kaggle for TPU/GPU training with API placeholders.
- `scripts/dashboard.(sh|ps1)` launches the Streamlit dashboard for metrics, explainability, and live video.
- `scripts/benchmark.(sh|ps1)` runs a latency sweep with optional dynamic quantization to tune for 60 FPS.

## Quickstart (Windows PowerShell)
```pwsh
cd object_detection_transformer
./scripts/setup_workspace.ps1            # creates .venv and installs deps
./.venv/Scripts/Activate.ps1

# Train on a YOLO-formatted dataset (images/*.jpg, labels/*.txt)
./scripts/train.ps1 C:\\data\\yolo --epochs 1 --batch-size 4 --num-classes 80 --num-queries 25

# Run single-image inference
python -m src.inference checkpoints/det_transformer.pt C:\\data\\yolo\\images\\sample.jpg --labels person car dog

# Stream real-time webcam/video inference with OpenCV overlay
./scripts/realtime.ps1 checkpoints/det_transformer.pt --labels person car dog --camera 0 --confidence 0.4
```

## Quickstart (Linux Bash)
```bash
cd object_detection_transformer
./scripts/setup_workspace.sh             # creates .venv and installs deps
source .venv/bin/activate

# Train on a YOLO-formatted dataset (images/*.jpg, labels/*.txt)
./scripts/train.sh data/yolo --epochs 1 --batch-size 4 --num-classes 80 --num-queries 25

# Run single-image inference
python -m src.inference checkpoints/det_transformer.pt data/yolo/images/sample.jpg --labels person car dog

# Stream real-time webcam/video inference with OpenCV overlay
./scripts/realtime.sh checkpoints/det_transformer.pt --labels person car dog --camera 0 --confidence 0.4
```

## Quickstart (GitHub Codespaces)
Codespaces are Linux-based, so use the Bash helpers plus the standard `/workspaces/<repo>` path.
```bash
cd object_detection_transformer
./scripts/setup_workspace.sh
source .venv/bin/activate

# Train (CPU by default). Enable GPU Codespaces and switch to --device cuda if available.
./scripts/train.sh /workspaces/<repo>/data/yolo --epochs 10 --batch-size 8 --num-classes 80 --num-queries 25 --device cpu

# Inference on one image
python -m src.inference checkpoints/det_transformer.pt /workspaces/<repo>/data/yolo/images/sample.jpg --labels person car dog --confidence 0.4

# Real-time video (webcams are typically unavailable in Codespaces)
./scripts/realtime.sh checkpoints/det_transformer.pt --labels person car dog --video /workspaces/<repo>/data/demo.mp4 --confidence 0.4 --no-window true
```

For more Codespaces tips (data upload, batch loops, VS Code tasks), see `use.md`.

The dataset folder should contain `images/*.jpg` and `labels/*.txt` files in YOLO format (`class x_center y_center width height`). Ensure the two subfolders contain the same basenames (e.g., `images/cat.jpg` pairs with `labels/cat.txt`).

## Training Guide
- **Inputs**: place resized training images under `dataset/images/` and matching YOLO label files under `dataset/labels/` (filenames must match between the two folders).
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
- Cameras are opened with their integer index (0 by default) to avoid OpenCV treating them as filenames; pass `--video <path>` to stream a file instead.

## FusionNet extras (encoder-only + Vision RAG + GNN dashboard)
- The detection backbone is now encoder-only: YOLO-style patches plus learned queries are passed through the encoder; detection tokens are fused with a `KnowledgeGraphMemory` (Vision RAG style) before class/box heads.
- A lightweight GNN (`src.gnn.py`) aggregates detections into RMS velocity and count metrics for every frame. These metrics are emitted during inference/real-time runs and logged under `artifacts/metrics.jsonl`.
- Launch the Streamlit dashboard to visualize detections, metrics, and explainability JSONL files: `./scripts/dashboard.sh` (Linux/macOS) or `./scripts/dashboard.ps1` (Windows). The dashboard reads `artifacts/live.mp4` if you save video buffers.
- For TPU training on Kaggle, copy `kaggle_fusionnet_notebook.py` into a notebook, fill `API_ENDPOINT`/`API_TOKEN`, and sync checkpoints back to the Codespaces backend.
- To benchmark latency and experiment with quantization or patch sizing, run `./scripts/benchmark.sh checkpoints/det_transformer.pt data/yolo --device cuda --iterations 50 --quantize` (or the PowerShell equivalent). Adjust `--image-size` during training to trade accuracy for FPS.
- API notes: the Kaggle notebook contains `API_ENDPOINT` and `API_TOKEN` placeholders used for pushing checkpoints to your backend; keep these secrets in environment variables when executing on Kaggle.
