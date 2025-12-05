"""FastAPI bridge for FusionNet backend and frontend clients.

Endpoints include dataset upload/registration, training, checkpoint sync,
and lightweight inference/metrics retrieval so the Streamlit dashboard and
Kaggle notebook can exchange artifacts. Protect the server with
``FUSIONNET_API_TOKEN``; clients should send ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import io as tv_io
from torchvision import transforms as tv_transforms

from .config import TrainingConfig
from .dataset import YoloDataset, collate_yolo
from .model import DetectionTransformer
from .pipeline import FusionNetPipeline
from .train import train_epoch


def _require_token(authorization: str = Header(default="")) -> None:
    expected = os.getenv("FUSIONNET_API_TOKEN", "")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid or missing token")


def _load_latest_jsonl(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


class _ServiceState:
    def __init__(self) -> None:
        self.pipeline: FusionNetPipeline | None = None
        self.label_map: list[str] = []
        self.checkpoint_dir = Path("checkpoints")
        self.data_root: Path | None = None
        self.image_size: int = 128


state = _ServiceState()
app = FastAPI(title="FusionNet Service", version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "pipeline_ready": str(state.pipeline is not None)}


@app.post("/dataset/upload")
async def upload_dataset(
    archive: UploadFile = File(...),
    destination: str = Form("data/uploaded"),
    _: None = Depends(_require_token),
) -> Dict[str, str]:
    dest_root = Path(destination)
    dest_root.parent.mkdir(parents=True, exist_ok=True)
    archive_path = dest_root.with_suffix(dest_root.suffix + ".zip")
    with archive_path.open("wb") as f:
        shutil.copyfileobj(archive.file, f)
    shutil.unpack_archive(str(archive_path), str(dest_root))
    return {"dataset_root": str(dest_root)}


@app.post("/dataset/register")
async def register_dataset(
    dataset_root: str = Form(...),
    image_size: int = Form(256),
    max_detections: int = Form(25),
    _: None = Depends(_require_token),
) -> Dict:
    data_dir = Path(dataset_root)
    ds = YoloDataset(data_dir=data_dir, image_size=image_size, max_detections=max_detections)
    state.label_map = ds.label_map
    state.data_root = data_dir
    state.image_size = image_size
    return {"num_classes": len(ds.label_map), "samples": len(ds)}


def _build_model(cfg: TrainingConfig) -> DetectionTransformer:
    return DetectionTransformer(
        num_classes=cfg.num_classes,
        num_queries=cfg.num_queries,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    )


@app.post("/train")
async def train_model(
    epochs: int = Form(1),
    batch_size: int = Form(2),
    num_queries: int = Form(25),
    lr: float = Form(1e-4),
    weight_decay: float = Form(1e-4),
    image_size: int = Form(128),
    device: str = Form("cpu"),
    _: None = Depends(_require_token),
) -> Dict[str, str]:
    if state.data_root is None:
        raise HTTPException(status_code=400, detail="Register a dataset first")
    cfg = TrainingConfig(
        data_dir=state.data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_classes=len(state.label_map) or 80,
        num_queries=num_queries,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=epochs,
        device=device,
    )
    dataset = YoloDataset(data_dir=cfg.data_dir, image_size=cfg.image_size, max_detections=cfg.num_queries)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_yolo,
    )
    device_t = torch.device(cfg.device)
    model = _build_model(cfg).to(device_t)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    for _ in range(cfg.max_epochs):
        train_epoch(
            model,
            data_loader,
            optimizer,
            criterion,
            device_t,
            num_classes=cfg.num_classes,
            log_every=cfg.log_every,
        )
    state.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = state.checkpoint_dir / "det_transformer_api.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg.__dict__}, ckpt_path)
    state.pipeline = FusionNetPipeline(model, label_map=dataset.label_map, memory_path=Path("artifacts/memory.pt"))
    return {"checkpoint": str(ckpt_path)}


@app.post("/upload/checkpoint")
async def upload_checkpoint(
    checkpoint: UploadFile = File(...),
    _: None = Depends(_require_token),
) -> Dict[str, str]:
    state.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dest = state.checkpoint_dir / checkpoint.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(checkpoint.file, f)
    return {"saved": str(dest)}


@app.get("/frontend/metrics")
async def frontend_metrics(_: None = Depends(_require_token)) -> Dict:
    if state.pipeline is None:
        return {"message": "pipeline not initialized"}
    latest = _load_latest_jsonl(state.pipeline.artifacts.metrics_path)
    return latest or {"message": "no metrics yet"}


@app.post("/frontend/infer")
async def frontend_infer(
    image_path: str = Form(...),
    _: None = Depends(_require_token),
) -> Dict:
    if state.pipeline is None:
        raise HTTPException(status_code=400, detail="Train or load a checkpoint first")
    image = Path(image_path)
    if not image.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image}")
    tensor = tv_io.read_image(str(image)).float() / 255.0
    transform = tv_transforms.Resize((state.image_size, state.image_size))
    tensor = transform(tensor)
    with torch.inference_mode():
        outputs = state.pipeline.forward(tensor.unsqueeze(0).to(next(state.pipeline.model.parameters()).device))
    return outputs


@app.post("/frontend/metrics/push")
async def push_metrics(
    payload: str = Form(...),
    _: None = Depends(_require_token),
) -> Dict[str, str]:
    path = state.pipeline.artifacts.metrics_path if state.pipeline else Path("artifacts/metrics.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = json.loads(payload)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_payload) + "\n")
    return {"written": str(path)}


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("src.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":  # pragma: no cover
    run()
