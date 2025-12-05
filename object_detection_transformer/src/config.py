from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TrainingConfig:
    data_dir: Path
    image_size: int = 128
    batch_size: int = 2
    num_workers: int = 2
    num_classes: int = 80
    num_queries: int = 25
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 2
    device: str = "cpu"
    log_every: int = 10


@dataclass
class InferenceConfig:
    checkpoint: Path
    image: Path
    label_map: List[str]
    image_size: int = 128
    num_queries: int = 25
    device: str = "cpu"
