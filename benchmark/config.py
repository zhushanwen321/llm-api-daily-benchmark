"""配置管理：从 YAML 文件加载配置。"""

from __future__ import annotations

import os
import re
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


_ENV_VAR_RE = re.compile(r"^\$\{(\w+)\}$")


def _resolve_env_var(value: str, field_name: str = "api_key") -> str:
    """若 value 匹配 ${ENV_VAR} 则从环境变量解析，否则原样返回。"""
    m = _ENV_VAR_RE.match(value)
    if m:
        var_name = m.group(1)
        resolved = os.environ.get(var_name)
        if resolved is None:
            raise ValueError(
                f"Environment variable '{var_name}' is not set "
                f"(referenced in {field_name})"
            )
        return resolved
    return value


def get_model_config(model_name: str, models_path: str | Path | None = None) -> dict[str, Any]:
    """获取指定模型的完整配置（合并 provider 和 model 层级的配置）。

    Args:
        model_name: 模型标识，格式为 provider/model（如 glm/glm-4.7）。
        models_path: 模型配置路径。为 None 时使用 configs/models.yaml。

    Returns:
        合并后的配置字典，包含 provider, api_key, api_base, max_tokens, rate_limit 字段。

    Raises:
        ValueError: 格式错误、provider 不存在或 model 不存在。
    """
    if "/" not in model_name:
        raise ValueError(
            f"Model identifier must be 'provider/model' format, got '{model_name}'. "
            "Example: glm/glm-4.7"
        )

    provider_name, model_id = model_name.split("/", 1)

    cfg = load_models_config(models_path)
    providers = cfg.get("providers", {})

    if provider_name not in providers:
        available = ", ".join(providers.keys()) or "none"
        raise ValueError(
            f"Provider '{provider_name}' not found. Available: {available}"
        )

    provider_cfg = providers[provider_name]
    models = provider_cfg.get("models", {})

    if model_id not in models:
        available = ", ".join(models.keys()) or "none"
        raise ValueError(
            f"Model '{model_id}' not found under provider '{provider_name}'. "
            f"Available: {available}"
        )

    model_cfg = models[model_id] or {}
    # 读取全局默认值，默认为 131072
    defaults = cfg.get("defaults", {})
    default_max_tokens = defaults.get("max_tokens", 131072)
    return {
        "provider": provider_name,
        "api_key": _resolve_env_var(provider_cfg["api_key"], "api_key"),
        "api_base": provider_cfg["api_base"],
        "max_tokens": model_cfg.get("max_tokens", default_max_tokens),
        "rate_limit": float(provider_cfg["rate_limit"]) if "rate_limit" in provider_cfg else None,
        "thinking": model_cfg.get("thinking", {}),
    }
