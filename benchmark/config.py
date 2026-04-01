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
        模型配置字典（providers -> models 两层结构）。
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
    """获取指定模型的完整配置（合并 provider 和 model 层级的配置）。

    在 providers 下遍历所有 provider，找到包含该模型名的 provider，
    返回合并后的配置：api_key, api_base 来自 provider，max_tokens 来自 model（可选）。

    Args:
        model_name: 模型名称（如 glm-4.7）。

    Returns:
        合并后的配置字典，包含 api_key, api_base, max_tokens 等字段。

    Raises:
        ValueError: 模型未在任何 provider 下找到。
    """
    cfg = load_models_config()
    providers = cfg.get("providers", {})

    for provider_name, provider_cfg in providers.items():
        models = provider_cfg.get("models", {})
        if model_name in models:
            model_cfg = models[model_name] or {}
            return {
                "provider": provider_name,
                "api_key": provider_cfg["api_key"],
                "api_base": provider_cfg["api_base"],
                "max_tokens": model_cfg.get("max_tokens", 4096),
            }

    # 收集所有可用模型名用于错误提示
    all_models: list[str] = []
    for provider_cfg in providers.values():
        all_models.extend((provider_cfg.get("models") or {}).keys())
    available = ", ".join(all_models) if all_models else "none"
    raise ValueError(f"Model '{model_name}' not found. Available: {available}")
