from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from .gnn import GNNMetrics


def load_metrics(path: Path) -> List[Dict]:
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


def main() -> None:
    st.set_page_config(page_title="FusionNet Dashboard", layout="wide")
    metrics_path = Path(st.sidebar.text_input("Metrics JSONL", "artifacts/metrics.jsonl"))
    video_path = Path(st.sidebar.text_input("Video buffer", "artifacts/live.mp4"))
    explain_path = Path(st.sidebar.text_input("Explainability JSONL", "artifacts/attention.jsonl"))
    refresh_sec = st.sidebar.slider("Refresh hint (seconds)", min_value=2, max_value=15, value=5)

    metrics = load_metrics(metrics_path)
    explains = load_metrics(explain_path)

    render_overview(metrics)
    render_video(video_path)
    render_explainability(explains)
    st.caption(
        "The dashboard reads JSONL metrics from disk. Streamlit's built-in rerun will refresh when you hit 'r' or the rerun icon."
    )


if __name__ == "__main__":
    main()
