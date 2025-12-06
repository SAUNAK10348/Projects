from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json
import torch

from .gnn import GraphAnalytics
from .knowledge import KnowledgeGraphMemory
from .model import DetectionTransformer


@dataclass
class PipelineArtifacts:
    metrics_path: Path = Path("artifacts/metrics.jsonl")
    attention_path: Path = Path("artifacts/attention.jsonl")
    video_buffer: Path = Path("artifacts/live.mp4")


class FusionNetPipeline:
    def __init__(self, model: DetectionTransformer, label_map: List[str]):
        self.model = model
        self.label_map = label_map
        self.memory = KnowledgeGraphMemory(device=str(next(model.parameters()).device))
        self.analytics = GraphAnalytics(label_map)
        self.artifacts = PipelineArtifacts()
        self.artifacts.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_jsonl(self, path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")

    def forward(self, images: torch.Tensor) -> Dict:
        outputs = self.model(images, memory_bank=self.memory)
        detections = []
        for idx, (logits, boxes, tokens) in enumerate(
            zip(outputs["class_logits"], outputs["boxes"], outputs["tokens"])
        ):
            probs = logits.softmax(dim=-1)
            conf, cls = probs.max(dim=-1)
            for q in range(logits.size(0)):
                detections.append(
                    {
                        "id": f"sample{idx}_q{q}",
                        "score": float(conf[q].item()),
                        "class": self.label_map[int(cls[q].item())],
                        "box_xywh": boxes[q].tolist(),
                        "features": tokens[q].tolist(),
                    }
                )
        metrics = self.analytics.compute_metrics(detections)
        self._log_jsonl(
            self.artifacts.metrics_path,
            {
                "count_overall": metrics.count_overall,
                "count_classwise": metrics.count_classwise,
                "rms_velocity_overall": metrics.rms_velocity_overall,
                "rms_velocity_classwise": metrics.rms_velocity_classwise,
            },
        )
        return {"detections": detections, "metrics": metrics}
