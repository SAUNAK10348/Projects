from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn


def _normalize(x: Tensor) -> Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


@dataclass
class MemoryEntry:
    embeddings: Tensor
    labels: Tensor


@dataclass
class KnowledgeGraphMemory:
    """Lightweight class-wise memory for encoder outputs.

    Stores patch-level embeddings grouped by class id and provides cosine-similarity
    retrieval for Vision RAG style conditioning.
    """

    max_per_class: int = 256
    device: str = "cpu"
    store: Dict[int, MemoryEntry] = field(default_factory=dict)

    def ingest(self, embeddings: Tensor, class_logits: Tensor) -> None:
        """Save encoder embeddings alongside predicted classes.

        Args:
            embeddings: Tensor of shape (B, Q, D) representing detection tokens.
            class_logits: Tensor of shape (B, Q, C).
        """
        probs = class_logits.softmax(dim=-1)
        conf, cls = probs.max(dim=-1)
        for sample_idx in range(embeddings.size(0)):
            for token_idx in range(embeddings.size(1)):
                cls_id = int(cls[sample_idx, token_idx].item())
                feature = embeddings[sample_idx, token_idx].detach().to(self.device)
                logit = class_logits[sample_idx, token_idx].detach().to(self.device)
                feature = _normalize(feature.unsqueeze(0))
                if cls_id not in self.store:
                    self.store[cls_id] = MemoryEntry(embeddings=feature, labels=logit.unsqueeze(0))
                else:
                    entry = self.store[cls_id]
                    entry.embeddings = torch.cat([entry.embeddings, feature], dim=0)[-self.max_per_class :]
                    entry.labels = torch.cat([entry.labels, logit.unsqueeze(0)], dim=0)[-self.max_per_class :]

    def retrieve(self, query_embeddings: Tensor, top_k: int = 4) -> Tuple[Tensor, Tensor]:
        """Return nearest neighbors for each query embedding.

        Args:
            query_embeddings: Tensor of shape (B, Q, D)
        Returns:
            Tuple of (retrieved_features, retrieved_logits) each of shape (B, Q, D/C).
            If a class has no memory, zeros are returned.
        """
        bsz, num_queries, dim = query_embeddings.shape
        retrieved_feat = torch.zeros((bsz, num_queries, dim), device=query_embeddings.device)
        retrieved_logits = torch.zeros((bsz, num_queries, 1), device=query_embeddings.device)
        for cls_id, entry in self.store.items():
            bank_feat = _normalize(entry.embeddings.to(query_embeddings.device))
            sims = torch.einsum("bqd,kd->bqk", _normalize(query_embeddings), bank_feat)
            top_sim, top_idx = sims.topk(min(top_k, bank_feat.size(0)), dim=-1)
            bank_expanded = bank_feat.unsqueeze(0).unsqueeze(0).expand(bsz, num_queries, -1, -1)
            gathered = bank_expanded.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, bank_feat.size(1))).mean(dim=2)
            retrieved_feat += gathered * 1.0  # additive conditioning
            retrieved_logits += top_sim.mean(dim=-1, keepdim=True)
        return retrieved_feat, retrieved_logits

    def save(self, path: str | Path) -> None:
        serialized = {cls: {"embeddings": entry.embeddings.cpu(), "labels": entry.labels.cpu()} for cls, entry in self.store.items()}
        torch.save({"max_per_class": self.max_per_class, "store": serialized}, path)

    def load(self, path: str | Path) -> None:
        blob = torch.load(path, map_location=self.device)
        self.max_per_class = blob.get("max_per_class", self.max_per_class)
        restored = {}
        for cls, payload in blob.get("store", {}).items():
            restored[int(cls)] = MemoryEntry(embeddings=payload["embeddings"].to(self.device), labels=payload["labels"].to(self.device))
        self.store = restored

    def summary(self) -> List[str]:
        return [f"class {cls}: {entry.embeddings.size(0)} embeddings" for cls, entry in self.store.items()]


class KnowledgeFusion(nn.Module):
    """Fuse encoder tokens with retrieved memory.

    This module keeps the transformer encoder pure while enabling Vision RAG style
    feature augmentation.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, tokens: Tensor, retrieved: Tensor) -> Tensor:
        combined = torch.cat([tokens, retrieved], dim=-1)
        gated = torch.sigmoid(self.gate(combined))
        return tokens + gated
