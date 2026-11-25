"""
Configuration loader for SPA-VAEGAN.
Loads YAML configuration into a structured Python dictionary or dataclass.
"""

import os
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load and return configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg: Dict[str, Any], save_path: str) -> None:
    """Save configuration dictionary to a YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
