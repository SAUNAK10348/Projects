# Using the Object Detection Transformer in GitHub Codespaces

These steps assume a Codespace with Python 3.10+ and GPU access disabled by default (CPU works for smoke tests).

1. **Open the project folder**
   ```bash
   cd object_detection_transformer
   ```

2. **Create and activate the virtual environment**
   ```bash
   ./scripts/setup_workspace.sh
   source .venv/bin/activate
   ```

3. **Train on a YOLO-style dataset** (expects `images/` and `labels/` under the dataset root):
   ```bash
   ./scripts/train.sh /workspaces/<repo>/data/yolo --epochs 1 --batch-size 4 --num-classes 80 --num-queries 25
   ```

4. **Run single-image inference**
   ```bash
   python -m src.inference checkpoints/det_transformer.pt /workspaces/<repo>/data/yolo/images/sample.jpg --labels person car dog
   ```

5. **Launch the real-time demo** (webcam or video file):
   ```bash
   ./scripts/realtime.sh checkpoints/det_transformer.pt --labels person car dog --camera 0 --confidence 0.4
   ```

6. **Keep the environment active** by re-running `source .venv/bin/activate` in new terminals within the Codespace.

> Tip: Use the VS Code "Run and Debug" panel to create tasks that invoke `./scripts/train.sh` and `./scripts/realtime.sh` so you can trigger experiments with one click.
