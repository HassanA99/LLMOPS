# config/base_config.py
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class BaseConfig:
    """Base configuration class"""
    project_name: str = "fula_tts"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True