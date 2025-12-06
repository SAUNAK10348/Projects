from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .encoder import ImageEncoder
from .knowledge import KnowledgeFusion, KnowledgeGraphMemory


class DetectionTransformer(nn.Module):
    """Encoder-only detection transformer with memory fusion.

    Query embeddings are appended to the flattened image tokens so the stack remains
    encoder-only while still emitting detection tokens.
    """

    def __init__(
        self,
        num_classes: int,
        num_queries: int = 25,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        token_size: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.encoder = ImageEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            token_size=token_size,
        )
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.fusion = KnowledgeFusion(d_model)

        self.class_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, 4),
            nn.Sigmoid(),
        )

    def forward(
        self, images: Tensor, memory_bank: Optional[KnowledgeGraphMemory] = None
    ) -> Dict[str, Tensor]:
        queries = self.query_embed.weight.unsqueeze(0).expand(images.size(0), -1, -1)
        encoded, num_image_tokens = self.encoder(images, query_tokens=queries)
        token_split = encoded[:, num_image_tokens:, :]

        retrieved = None
        if memory_bank is not None and len(memory_bank.store) > 0:
            retrieved, _ = memory_bank.retrieve(token_split)
        if retrieved is None:
            retrieved = torch.zeros_like(token_split)
        fused = self.fusion(token_split, retrieved)

        class_logits = self.class_head(fused)
        box_preds = self.box_head(fused)

        if memory_bank is not None:
            memory_bank.ingest(fused.detach(), class_logits.detach())

        return {"class_logits": class_logits, "boxes": box_preds, "tokens": fused}

    def explain(self) -> Dict[str, Tuple[Tensor, ...]]:
        return {"encoder_attention": self.encoder.export_attention_maps()}
