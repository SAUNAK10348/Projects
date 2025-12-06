from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Thread
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st
import uvicorn


def render_overview(metrics: List[Dict]) -> None:
    st.header("FusionNet Live Dashboard")
    if not metrics:
        st.info("Waiting for metrics. Trigger training/inference to populate live feed.")
        return
    latest = metrics[-1]
    st.subheader("Latest Frame Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Objects/frame", latest.get("count_overall", 0))
    col2.metric("RMS velocity", f"{latest.get('rms_velocity_overall', 0):.3f}")
    col3.metric("Classes", len(latest.get("count_classwise", {})))

    st.subheader("Classwise counts")
    df = pd.DataFrame(
        {
            "class": list(latest.get("count_classwise", {}).keys()),
            "count": list(latest.get("count_classwise", {}).values()),
            "rms_velocity": [
                latest.get("rms_velocity_classwise", {}).get(cls, 0.0)
                for cls in latest.get("count_classwise", {})
            ],
        }
    )
    st.bar_chart(df.set_index("class"))
    st.table(df)


def render_video(path: Path) -> None:
    st.subheader("Live feed")
    if path.exists():
        st.video(str(path))
    else:
        st.info("Live video buffer not found. Ensure realtime script writes frames to artifacts/live.mp4")


def render_explainability(entries: List[Dict]) -> None:
    st.subheader("Explainability & Attention")
    if not entries:
        st.info("No explainability artifacts yet. Run inference with --export-attn")
        return
    latest = entries[-1]
    st.json(latest.get("attention", {}))


def load_metrics_local(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def fetch_backend(endpoint: str, base_url: str, token: str | None, **kwargs):
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.request(kwargs.pop("method", "get"), f"{base_url}{endpoint}", headers=headers, timeout=60, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Backend request failed: {exc}")
        return None
    if resp.status_code >= 400:
        st.error(f"Backend returned {resp.status_code}: {resp.text}")
        return None
    if "application/json" in resp.headers.get("content-type", ""):
        return resp.json()
    return resp.text


def ensure_backend_running(base_url: str, token: str | None) -> bool:
    """Check backend health and allow inline startup for one-command stack.

    Returns True when the backend responds to /health. When unreachable, this
    function surfaces a start button that spins up a local uvicorn server in a
    background thread so users who launched only the dashboard can still run the
    full pipeline.
    """

    healthy = fetch_backend("/health", base_url, token) is not None
    status_placeholder = st.empty()
    if healthy:
        status_placeholder.success(f"Backend live at {base_url}")
        return True

    status_placeholder.warning("Backend not reachable. Start it inline to proceed.")

    # Avoid double-start by tracking state.
    if "_local_backend" not in st.session_state:
        st.session_state._local_backend = None

    host_port = base_url.split("//")[-1]
    host_port = host_port.split("/")[0]
    host, _, port = host_port.partition(":")
    try:
        port_int = int(port) if port else 8000
    except ValueError:
        port_int = 8000

    if st.button("Start local backend server"):
        if st.session_state._local_backend is None:
            config = uvicorn.Config("src.api:app", host=host or "0.0.0.0", port=port_int, log_level="info", reload=False)
            server = uvicorn.Server(config)
            thread = Thread(target=server.run, daemon=True)
            thread.start()
            st.session_state._local_backend = server
            st.success(f"Started backend on {host or '0.0.0.0'}:{port_int}. Re-run health to confirm.")
        else:
            st.info("Backend start requested; waiting for it to become healthy.")

    return False


def render_dataset_controls(base_url: str, token: str | None) -> None:
    st.subheader("Dataset Upload & Registration")
    dataset_zip = st.file_uploader("Upload YOLO dataset (.zip with images/ and labels/)", type=["zip"])
    destination = st.text_input("Destination on backend", "data/uploaded")
    col1, col2, col3 = st.columns(3)
    with col1:
        image_size = st.number_input("Image size", min_value=64, max_value=1024, value=256, step=16)
    with col2:
        max_detections = st.number_input("Max detections", min_value=1, max_value=200, value=25, step=1)
    with col3:
        register_now = st.checkbox("Register after upload", value=True)

    if st.button("Upload dataset to backend"):
        if dataset_zip is None:
            st.error("Please select a dataset archive first.")
        else:
            files = {"archive": (dataset_zip.name, dataset_zip.getvalue(), "application/zip")}
            data = {"destination": destination}
            result = fetch_backend("/dataset/upload", base_url, token, method="post", files=files, data=data)
            if result:
                st.success(f"Uploaded to {result.get('dataset_root', destination)}")
                if register_now:
                    reg_payload = {
                        "dataset_root": result.get("dataset_root", destination),
                        "image_size": image_size,
                        "max_detections": max_detections,
                    }
                    reg_result = fetch_backend("/dataset/register", base_url, token, method="post", data=reg_payload)
                    if reg_result:
                        st.success(f"Registered {reg_result.get('samples', 0)} samples with {reg_result.get('num_classes', 0)} classes")


def render_training_controls(base_url: str, token: str | None) -> None:
    st.subheader("Training")
    c1, c2, c3 = st.columns(3)
    with c1:
        epochs = st.number_input("Epochs", min_value=1, max_value=200, value=5, step=1)
        batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=4, step=1)
    with c2:
        num_queries = st.number_input("Num queries", min_value=4, max_value=256, value=25, step=1)
        lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=1e-4, step=1e-5, format="%.6f")
    with c3:
        weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1e-2, value=1e-4, step=1e-5, format="%.6f")
        device = st.selectbox("Device", ["cpu", "cuda"])

    if st.button("Start training on backend"):
        payload = {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "num_queries": int(num_queries),
            "lr": lr,
            "weight_decay": weight_decay,
            "device": device,
        }
        result = fetch_backend("/train", base_url, token, method="post", data=payload)
        if result:
            st.success(f"Training completed. Checkpoint: {result.get('checkpoint', '')}")


def render_inference_controls(base_url: str, token: str | None) -> None:
    st.subheader("Single-image inference")
    image = st.file_uploader("Upload image for inference", type=["jpg", "jpeg", "png"])
    if st.button("Run inference"):
        if image is None:
            st.error("Please upload an image first.")
        else:
            files = {"image_file": (image.name, image.getvalue(), image.type or "application/octet-stream")}
            result = fetch_backend("/frontend/infer", base_url, token, method="post", files=files)
            if result:
                st.success("Inference complete")
                detections = result.get("detections", [])
                st.write(detections)


def render_metrics(metrics: List[Dict], explain_entries: List[Dict], video_path: Path) -> None:
    render_overview(metrics)
    render_video(video_path)
    render_explainability(explain_entries)


def main() -> None:
    st.set_page_config(page_title="FusionNet Dashboard", layout="wide")
    backend_default = os.getenv("FUSIONNET_BACKEND_URL", "http://localhost:8000")
    token_default = os.getenv("FUSIONNET_API_TOKEN", "")
    st.sidebar.header("Backend")
    backend_url = st.sidebar.text_input("Backend URL", backend_default)
    api_token = st.sidebar.text_input("API token", token_default, type="password")
    metrics_path = Path(st.sidebar.text_input("Local metrics JSONL", "artifacts/metrics.jsonl"))
    video_path = Path(st.sidebar.text_input("Video buffer", "artifacts/live.mp4"))
    explain_path = Path(st.sidebar.text_input("Explainability JSONL", "artifacts/attention.jsonl"))
    use_backend_metrics = st.sidebar.checkbox("Fetch latest metrics from backend", value=True)

    st.title("FusionNet Control Center")
    st.caption("Upload data, kick off training, run inference, and visualize metrics from one place.")

    ready = ensure_backend_running(backend_url, api_token or None)
    if not ready:
        st.stop()

    render_dataset_controls(backend_url, api_token or None)
    render_training_controls(backend_url, api_token or None)
    render_inference_controls(backend_url, api_token or None)

    st.divider()
    st.header("Live Metrics & Explainability")
    metrics = []
    explains: List[Dict] = []
    if use_backend_metrics:
        latest = fetch_backend("/frontend/metrics", backend_url, api_token or None)
        if latest:
            metrics = [latest]
    if not metrics:
        metrics = load_metrics_local(metrics_path)
    explains = load_metrics_local(explain_path)
    render_metrics(metrics, explains, video_path)

    st.caption(
        "When launched via scripts/start_stack.*, the backend and dashboard run together so you can operate everything from this page."
    )


if __name__ == "__main__":
    main()
