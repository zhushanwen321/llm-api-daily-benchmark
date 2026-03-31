"""配置管理：从 YAML 文件加载配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CONFIG_DIR = Path(__file__).parent / "configs"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载默认配置，可被指定路径覆盖。

    Args:
        config_path: 配置文件路径。为 None 时使用 configs/default.yaml。

    Returns:
        配置字典。
    """
    path = Path(config_path) if config_path else _CONFIG_DIR / "default.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_models_config(models_path: str | Path | None = None) -> dict[str, Any]:
    """加载模型 API 配置。

    Args:
        models_path: 模型配置路径。为 None 时使用 configs/models.yaml。

    Returns:
        模型配置字典，包含 models 键。
    """
    path = Path(models_path) if models_path else _CONFIG_DIR / "models.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Models config not found: {path}. "
            "Copy configs/models.yaml.example to configs/models.yaml and fill in your API keys."
        )
    with open(path) as f:
        return yaml.safe_load(f)


def get_model_config(model_name: str) -> dict[str, Any]:
    """获取指定模型的配置。

    Args:
        model_name: 模型名称（如 glm-4.7）。

    Returns:
        该模型的配置字典。

    Raises:
        ValueError: 模型未在配置中定义。
    """
    models_cfg = load_models_config()
    models = models_cfg.get("models", {})
    if model_name not in models:
        available = ", ".join(models.keys()) if models else "none"
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")
    return models[model_name]
