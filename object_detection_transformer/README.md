# Tensor-Valued Object Detection Transformer

This folder contains a runnable skeleton for an object-detection transformer that ingests YOLO-style labels. It is **Windows-first** and ships PowerShell helpers only. It provides:

- A dataset loader that reads images plus `*.txt` YOLO labels.
- An encoder that flattens images to tensors and feeds a multi-head attention stack.
- A detection transformer with learned queries and explainability hooks (attention maps and weight export).
- Training and inference scripts for Linux/macOS shells.

The code is deliberately lightweight so it can run on CPU for smoke tests while remaining compatible with GPU training.

## Project Layout
- `src/` contains the transformer model, dataset utilities, and training/inference entrypoints.
- `scripts/` contains portable Bash helpers for Linux/macOS shells.
- `requirements.txt` lists runtime dependencies.

## Quickstart (Linux/macOS)
```bash
cd object_detection_transformer
./scripts/setup_workspace.sh
source .venv/bin/activate
./scripts/train.sh /path/to/dataset --epochs 1 --batch-size 4 --num-classes 80 --num-queries 25
python -m src.inference checkpoints/det_transformer.pt /path/to/test.jpg --labels person car dog
```

The dataset folder should contain `images/*.jpg` and `labels/*.txt` files in YOLO format (`class x_center y_center width height`).

## Explainability
Call `model.explain()` on a loaded `DetectionTransformer` instance to retrieve encoder attention weights for downstream visualization or auditing.
