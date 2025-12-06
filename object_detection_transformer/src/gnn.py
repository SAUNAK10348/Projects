from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import torch
from torch import Tensor, nn


@dataclass
class GNNMetrics:
    rms_velocity_classwise: Dict[str, float]
    rms_velocity_overall: float
    count_classwise: Dict[str, int]
    count_overall: int
    tracked_objects: Dict[str, Tuple[float, float, float, float]]


class GraphAggregator(nn.Module):
    """A lightweight GNN to aggregate detection tokens into graph metrics."""

    def __init__(self, d_model: int, hidden: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(d_model, hidden), nn.ReLU(inplace=True)])
            d_model = hidden
        self.projection = nn.Sequential(*layers)
        self.velocity_head = nn.Linear(hidden, 1)

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        # adjacency: (N, N) symmetric matrix
        agg = torch.matmul(adjacency, node_features)
        fused = node_features + agg
        return self.projection(fused)


class GraphAnalytics:
    def __init__(self, label_map: List[str], seed: int | None = 0) -> None:
        self.label_map = label_map
        self.prev_positions: Dict[str, Tensor] = {}
        self.smoothing = 0.9
        self.graph = nx.Graph()
        self.seed = seed
        self._gnn: GraphAggregator | None = None

    def _init_gnn(self, d_model: int) -> None:
        if self._gnn is None or self._gnn.projection[0].in_features != d_model:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            self._gnn = GraphAggregator(d_model=d_model)
            self._gnn.eval()

    def build_graph(self, detections: List[Dict]) -> Tuple[Tensor, Tensor]:
        if len(detections) == 0:
            return torch.zeros((1, 4)), torch.zeros((1, 1))
        features = []
        positions = []
        for det in detections:
            x, y, w, h = det.get("box_xywh", [0, 0, 0, 0])
            features.append(torch.tensor(det.get("features", [x, y, w, h]), dtype=torch.float32))
            positions.append(torch.tensor([x, y, w, h], dtype=torch.float32))
        feats = torch.stack(features)
        adjacency = torch.ones((len(detections), len(detections)), dtype=torch.float32)
        adjacency.fill_diagonal_(0)
        return feats, adjacency

    def compute_metrics(self, detections: List[Dict]) -> GNNMetrics:
        feats, adjacency = self.build_graph(detections)
        if feats.numel() == 0:
            return GNNMetrics({}, 0.0, {}, 0, {})
        self._init_gnn(d_model=feats.size(-1))
        assert self._gnn is not None  # for type checkers
        with torch.no_grad():
            enriched = self._gnn(feats, adjacency)
            velocities = torch.abs(enriched).mean(dim=-1)
        classwise_counts: Dict[str, int] = {}
        classwise_vel: Dict[str, List[float]] = {}
        tracked: Dict[str, Tuple[float, float, float, float]] = {}
        for det, vel in zip(detections, velocities):
            cls = det.get("class", "unknown")
            classwise_counts[cls] = classwise_counts.get(cls, 0) + 1
            classwise_vel.setdefault(cls, []).append(float(vel.item()))
            tracked[det.get("id", cls)] = tuple(det.get("box_xywh", [0, 0, 0, 0]))
        rms_classwise = {
            cls: float(torch.sqrt(torch.tensor(vals).pow(2).mean()).item()) for cls, vals in classwise_vel.items()
        }
        all_velocities = [vel for values in classwise_vel.values() for vel in values]
        overall_rms = (
            float(torch.sqrt(torch.tensor(all_velocities, dtype=torch.float32).pow(2).mean()).item())
            if all_velocities
            else 0.0
        )
        return GNNMetrics(
            rms_velocity_classwise=rms_classwise,
            rms_velocity_overall=overall_rms,
            count_classwise=classwise_counts,
            count_overall=sum(classwise_counts.values()),
            tracked_objects=tracked,
        )
