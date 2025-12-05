"""
Notebook-style script for Kaggle TPU/GPUs.
Copy/paste cells into a Kaggle notebook. Fill API_TOKEN with your backend URL.
"""

# %% Imports
import os
from pathlib import Path

import torch
from torchvision import transforms

from src.config import TrainingConfig
from src.dataset import YoloDataset, collate_yolo
from src.model import DetectionTransformer
from src.train import compute_loss

API_ENDPOINT = ""  # <- place backend endpoint here
API_TOKEN = ""  # <- add secret token for syncing artifacts

# %% Dataset
root = Path("/kaggle/input/your-dataset")
image_size = 256
train_ds = YoloDataset(root / "images", root / "labels", image_size=image_size)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_yolo)

# %% Model
model = DetectionTransformer(num_classes=len(train_ds.label_map), num_queries=50, d_model=192)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# %% Training loop
for epoch in range(2):
    model.train()
    for images, boxes, classes in train_dl:
        images = images.to(model.query_embed.weight.device)
        boxes = boxes.to(model.query_embed.weight.device)
        classes = classes.to(model.query_embed.weight.device)
        out = model(images)
        loss = compute_loss(out, boxes, classes, num_classes=len(train_ds.label_map), criterion=criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save({"model_state_dict": model.state_dict()}, f"fusionnet_epoch{epoch}.pt")

# %% Sync artifacts (pseudo-code)
if API_ENDPOINT and API_TOKEN:
    import requests

    for ckpt in Path(".").glob("fusionnet_epoch*.pt"):
        resp = requests.post(
            API_ENDPOINT,
            headers={"Authorization": f"Bearer {API_TOKEN}"},
            files={"file": ckpt.open("rb")},
        )
        print("Uploaded", ckpt, resp.status_code)
