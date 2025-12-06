# Using the Object Detection Transformer in GitHub Codespaces

This guide focuses on Codespaces defaults (Ubuntu + Python 3.10+, CPU-only by default). Run commands in the VS Code integrated terminal unless noted otherwise.

## 1) Open the project folder
```bash
cd object_detection_transformer
```

## 2) Create and activate the virtual environment
```bash
./scripts/setup_workspace.sh      # installs Python deps into .venv
source .venv/bin/activate         # re-run in every new terminal tab
```

## 3) Put data in the workspace
- Create `/workspaces/<repo>/data/yolo/images` and `/workspaces/<repo>/data/yolo/labels`.
- Drag/drop files into the Explorer or use `curl`/`wget` inside the Codespace.
- Filenames must match across folders (e.g., `images/dog.jpg` with `labels/dog.txt`).

## 4) Train on YOLO-style data
```bash
./scripts/train.sh /workspaces/<repo>/data/yolo --epochs 10 --batch-size 8 --num-classes 80 --num-queries 25 --device cpu
```
- Checkpoints land in `checkpoints/det_transformer.pt`. Re-run training with `--resume checkpoints/det_transformer.pt` to continue.
- If you enable a GPU-backed Codespace, switch `--device cuda`.

## 5) Run single-image inference
```bash
python -m src.inference checkpoints/det_transformer.pt \
  /workspaces/<repo>/data/yolo/images/sample.jpg \
  --labels person car dog --confidence 0.4
```

## 6) Batch-evaluate a folder (CPU-friendly loop)
```bash
for img in /workspaces/<repo>/data/yolo/images/*.jpg; do
  python -m src.inference checkpoints/det_transformer.pt "$img" --labels person car dog --confidence 0.4
done | tee logs/batch_eval.txt
```

## 7) Real-time demo in Codespaces
- Codespaces generally cannot access a local webcam. Use `--video` with a hosted or uploaded video file instead of `--camera`.
- Use `--no-window true` to avoid GUI errors in headless shells.
```bash
./scripts/realtime.sh checkpoints/det_transformer.pt \
  --labels person car dog --video /workspaces/<repo>/data/demo.mp4 \
  --confidence 0.4 --no-window true
```

## 8) Tips for smoother workflows
- Add VS Code tasks in `.vscode/tasks.json` that call `./scripts/train.sh` or `./scripts/realtime.sh` for one-click runs.
- Persist artifacts by committing checkpoints/logs you want to keep or downloading them before deleting the Codespace.
- If imports fail after dependency tweaks, rerun `./scripts/setup_workspace.sh --recreate`.
