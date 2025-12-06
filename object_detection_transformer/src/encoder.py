from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


def patchify(image: Tensor, token_size: int = 16) -> Tensor:
    """Convert an image tensor (B, C, H, W) into (B, N, token_size).

    The function flattens spatial dimensions into overlapping 1D vectors and pads
    the last token if necessary so that every sample is a clean 1D sequence.
    """

    bsz, channels, height, width = image.shape
    flattened = image.flatten(2)  # (B, C, H*W)
    flattened = flattened.transpose(1, 2)  # (B, H*W, C)
    flattened = flattened.reshape(bsz, height * width, channels)

    seq = flattened.reshape(bsz, -1)
    remainder = seq.shape[1] % token_size
    if remainder != 0:
        pad = token_size - remainder
        seq = torch.nn.functional.pad(seq, (0, pad))

    tokens = seq.view(bsz, -1, token_size)
    return tokens


class ImageEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        token_size: int = 16,
    ) -> None:
        super().__init__()
        self.token_size = token_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(token_size, d_model)
        self.positional = nn.Parameter(torch.zeros(1, 4096, d_model))

    def forward(self, images: Tensor) -> Tensor:
        tokens = patchify(images, token_size=self.token_size)
        projected = self.input_proj(tokens)
        max_len = self.positional.shape[1]
        if projected.size(1) > max_len:
            raise ValueError(
                f"Sequence length {projected.size(1)} exceeds positional capacity {max_len}."
            )
        positional = self.positional[:, : projected.size(1), :]
        encoded = self.encoder(projected + positional)
        return encoded

    def export_attention_maps(self) -> Tuple[Tensor, ...]:
        maps = []
        for layer in self.encoder.layers:
            attn = layer.self_attn
            maps.append(attn.in_proj_weight.detach().cpu())
        return tuple(maps)
