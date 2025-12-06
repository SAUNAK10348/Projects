from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from torch.nn.functional import scaled_dot_product_attention


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


class FusedEncoderLayer(nn.Module):
    """Encoder block that leans on fused SDPA kernels when available."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.nhead, -1).transpose(1, 2)
        k = k.view(bsz, seqlen, self.nhead, -1).transpose(1, 2)
        v = v.view(bsz, seqlen, self.nhead, -1).transpose(1, 2)
        attn_out = scaled_dot_product_attention(q, k, v)  # uses flash/sdp kernels when possible
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seqlen, self.d_model)
        attn_out = self.out_proj(attn_out)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_out.detach().cpu()


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
        self.layers = nn.ModuleList(
            [FusedEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.input_proj = nn.Linear(token_size, d_model)
        self.positional = nn.Parameter(torch.zeros(1, 4096, d_model))
        self._attn_cache: list[Tensor] = []

    def _ensure_positional_capacity(self, required_len: int) -> None:
        """Expand the learnable positional tensor if a longer sequence is needed."""

        current_len = self.positional.shape[1]
        if required_len <= current_len:
            return

        # Grow geometrically to reduce reallocations for larger inputs (e.g., 256x256).
        new_len = max(required_len, current_len * 2)
        expanded = torch.zeros(
            1,
            new_len,
            self.positional.shape[2],
            device=self.positional.device,
            dtype=self.positional.dtype,
        )
        expanded[:, :current_len] = self.positional.data
        self.positional = nn.Parameter(expanded)

    def forward(self, images: Tensor, query_tokens: Optional[Tensor] = None) -> Tuple[Tensor, int]:
        tokens = patchify(images, token_size=self.token_size)
        projected = self.input_proj(tokens)
        if query_tokens is not None:
            projected = torch.cat([projected, query_tokens], dim=1)
        self._ensure_positional_capacity(projected.size(1))
        positional = self.positional[:, : projected.size(1), :]
        x = projected + positional
        self._attn_cache.clear()
        for layer in self.layers:
            x, attn = layer(x)
            self._attn_cache.append(attn)
        return x, tokens.size(1)

    def export_attention_maps(self) -> Tuple[Tensor, ...]:
        return tuple(self._attn_cache)
