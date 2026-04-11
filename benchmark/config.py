"""配置管理：从 YAML 文件加载配置。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


_CONFIG_DIR = Path(__file__).parent / "configs"
_SETTINGS_PATH = _CONFIG_DIR / "settings.yml"


def load_settings(settings_path: str | Path | None = None) -> dict[str, Any]:
    """加载统一配置文件 settings.yml。

    Args:
        settings_path: 配置文件路径。为 None 时使用 configs/settings.yml。

    Returns:
        包含 defaults, model_defaults, providers 等区段的配置字典。
    """
    path = Path(settings_path) if settings_path else _SETTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载默认配置。

    当传入显式路径时直接加载该文件；否则优先从 settings.yml 的 defaults 区段读取，
    如 settings.yml 不存在则回退到 default.yaml（已废弃）。

    Args:
        config_path: 配置文件路径。为 None 时自动检测。

    Returns:
        配置字典。
    """
    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "model_defaults" in data:
            return data.get("defaults", {})
        return data

    if _SETTINGS_PATH.exists():
        settings = load_settings(_SETTINGS_PATH)
        return settings.get("defaults", {})

    path = _CONFIG_DIR / "default.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_models_config(models_path: str | Path | None = None) -> dict[str, Any]:
    """加载模型 API 配置。

    当传入显式路径时直接加载该文件；否则优先从 settings.yml 构造，
    如 settings.yml 不存在则回退到 models.yaml（已废弃）。

    Args:
        models_path: 模型配置路径。为 None 时自动检测。

    Returns:
        模型配置字典（providers -> models 两层结构）。
    """
    if models_path is not None:
        path = Path(models_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Models config not found: {path}. "
                "Copy configs/models.yaml.example to configs/models.yaml and fill in your API keys."
            )
        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "model_defaults" in data:
            return {
                "defaults": data.get("model_defaults", {}),
                "providers": data.get("providers", {}),
            }
        return data

    if _SETTINGS_PATH.exists():
        settings = load_settings(_SETTINGS_PATH)
        return {
            "defaults": settings.get("model_defaults", {}),
            "providers": settings.get("providers", {}),
        }

    path = _CONFIG_DIR / "models.yaml"
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


def get_model_config(
    model_name: str, models_path: str | Path | None = None
) -> dict[str, Any]:
    """获取指定模型的完整配置（合并 provider 和 model 层级的配置）。

    Args:
        model_name: 模型标识，格式为 provider/model（如 glm/glm-4.7）。
        models_path: 模型配置路径。为 None 时自动检测。

    Returns:
        合并后的配置字典，包含 provider, api_key, api_base, max_tokens, max_concurrency 字段。

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
    defaults = cfg.get("defaults", {})
    default_max_tokens = defaults.get("max_tokens", 131072)

    max_concurrency = None
    if "max_concurrency" in provider_cfg:
        max_concurrency = provider_cfg["max_concurrency"]
    elif "rate_limit" in provider_cfg:
        import warnings

        warnings.warn(
            f"Provider '{provider_name}': 'rate_limit' 配置已废弃，"
            f"请改用 'max_concurrency'。当前值 {provider_cfg['rate_limit']} "
            f"将自动映射为 max_concurrency。",
            DeprecationWarning,
            stacklevel=2,
        )
        max_concurrency = provider_cfg["rate_limit"]

    if max_concurrency is not None:
        max_concurrency = int(max_concurrency)
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")

    return {
        "provider": provider_name,
        "api_key": _resolve_env_var(provider_cfg["api_key"], "api_key"),
        "api_base": provider_cfg["api_base"],
        "max_tokens": model_cfg.get("max_tokens", default_max_tokens),
        "max_concurrency": max_concurrency,
        "thinking": model_cfg.get("thinking", {}),
    }
