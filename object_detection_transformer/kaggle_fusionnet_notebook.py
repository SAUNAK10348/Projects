"""
Notebook-style script for Kaggle TPU/GPUs.
Copy/paste cells into a Kaggle notebook. Fill the API settings to sync
artifacts with the Codespaces backend/front-end.

Minimal launch inside the Kaggle terminal (after adding the notebook file):
```
python kaggle_fusionnet_notebook.py
```
"""

# %% Imports
import json
import os
import subprocess
import zipfile
from pathlib import Path

import requests
import torch
from torchvision import transforms

from src.dataset import YoloDataset, collate_yolo
from src.model import DetectionTransformer
from src.train import compute_loss

BACKEND_API = os.getenv("FUSIONNET_BACKEND_URL", "")  # e.g., https://<codespace>-8000.app/github.dev
FRONTEND_API = os.getenv("FUSIONNET_FRONTEND_URL", "")  # can reuse BACKEND_API if combined
API_TOKEN = os.getenv("FUSIONNET_API_TOKEN", "")  # bearer token matching backend launch
DATASET_SLUG = os.getenv("FUSIONNET_KAGGLE_DATASET", "")  # Kaggle dataset slug, requires kaggle.json API key
DATASET_ZIP = Path(os.getenv("FUSIONNET_DATASET_ZIP", "/kaggle/input/dataset.zip"))
DATASET_ROOT = Path(os.getenv("FUSIONNET_DATASET_ROOT", "/kaggle/input/your-dataset"))

# %% Dataset: pull from Kaggle API key or an uploaded ZIP
work_dir = Path("/kaggle/working/data")
work_dir.mkdir(parents=True, exist_ok=True)

# Option A: download a published dataset via Kaggle API (requires kaggle.json with API key/secret)
if DATASET_SLUG:
    subprocess.run([
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(work_dir),
        "--unzip",
    ], check=True)
    root = work_dir
# Option B: user-uploaded ZIP in /kaggle/input (or override via FUSIONNET_DATASET_ZIP)
elif DATASET_ZIP.exists():
    with zipfile.ZipFile(DATASET_ZIP) as zf:
        zf.extractall(work_dir)
    root = work_dir
# Fallback: mounted dataset directory in /kaggle/input
else:
    root = DATASET_ROOT

image_size = 256
train_ds = YoloDataset(root, image_size=image_size)
num_classes = len(train_ds.label_map) or 80
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_yolo)

# %% Model
model = DetectionTransformer(num_classes=num_classes, num_queries=50, d_model=192)
try:
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    use_xla = True
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_xla = False

model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# %% Training loop
for epoch in range(2):
    model.train()
    for images, boxes, classes in train_dl:
        images = images.to(device)
        boxes = boxes.to(device)
        classes = classes.to(device)
        out = model(images)
        loss = compute_loss(out, boxes, classes, num_classes=num_classes, criterion=criterion)
        optimizer.zero_grad()
        loss.backward()
        if use_xla:
            xm.optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()
    checkpoint_path = Path(f"fusionnet_epoch{epoch}.pt")
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    if BACKEND_API and API_TOKEN:
        files = {"checkpoint": checkpoint_path.open("rb")}
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        requests.post(f"{BACKEND_API}/upload/checkpoint", files=files, headers=headers, timeout=30)

# %% Optional: register dataset + send metrics to backend/front-end
if BACKEND_API and API_TOKEN:
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    requests.post(
        f"{BACKEND_API}/dataset/register",
        data={"dataset_root": str(root), "image_size": image_size, "max_detections": 50},
        headers=headers,
        timeout=15,
    )

# Push a synthetic metrics row to the dashboard API (visible on Codespaces Streamlit)
if FRONTEND_API and API_TOKEN:
    payload = {"count_overall": 0, "count_classwise": {}, "rms_velocity_overall": 0.0, "rms_velocity_classwise": {}}
    requests.post(
        f"{FRONTEND_API}/frontend/metrics/push",
        data={"payload": json.dumps(payload)},
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        timeout=10,
    )
