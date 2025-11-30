from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from .encoder import ImageEncoder


class DetectionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_queries: int = 25,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        token_size: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = ImageEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            token_size=token_size,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.class_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, 4),
            nn.Sigmoid(),
        )

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        memory = self.encoder(images)
        queries = self.query_embed.weight.unsqueeze(0).expand(images.size(0), -1, -1)
        decoded = self.decoder(queries, memory)

        class_logits = self.class_head(decoded)
        box_preds = self.box_head(decoded)
        return {"class_logits": class_logits, "boxes": box_preds}

    def explain(self) -> Dict[str, Tuple[Tensor, ...]]:
        return {"encoder_attention": self.encoder.export_attention_maps()}
