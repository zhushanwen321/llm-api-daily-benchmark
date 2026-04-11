"""config.py 的 provider/model 解析测试。"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from benchmark.config import (
    get_model_config,
    load_settings,
    load_config,
    load_models_config,
)


def _write_test_config(tmp_path: Path, config: dict) -> str:
    """写入临时 models.yaml 并返回路径。"""
    p = tmp_path / "models.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return str(p)


def _write_test_settings(tmp_path: Path, settings: dict) -> str:
    """写入临时 settings.yml 并返回路径。"""
    p = tmp_path / "settings.yml"
    with open(p, "w") as f:
        yaml.dump(settings, f)
    return str(p)


_SAMPLE_SETTINGS = {
    "defaults": {
        "model": "glm-4.7",
        "temperature": 0.0,
        "max_tokens": 4096,
        "max_retries": 3,
        "timeout": 300,
        "dataset_root": "benchmark/datasets",
        "difficulty_weights": {"easy": 1.0, "medium": 1.5, "hard": 2.0},
    },
    "model_defaults": {
        "max_tokens": 131072,
    },
    "providers": {
        "zai": {
            "api_key": "test-zai-key",
            "api_base": "https://open.bigmodel.cn/api/coding/paas/v4",
            "max_concurrency": 2,
            "models": {
                "glm-4.7": {"thinking": {"enabled": True}},
                "glm-5": {
                    "thinking": {
                        "enabled": True,
                        "reasoning_field": "reasoning_content",
                    }
                },
            },
        },
        "minimax": {
            "api_key": "test-minimax-key",
            "api_base": "https://api.minimaxi.com/v1",
            "rate_limit": 2,
            "models": {
                "MiniMax-M2.7": {},
            },
        },
    },
}


class TestGetModelConfig:
    def test_valid_provider_model(self, tmp_path):
        """glm/glm-4.7 应正确解析。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "test-key",
                        "api_base": "https://api.test.com/v1/",
                        "models": {"glm-4.7": {"max_tokens": 4096}},
                    }
                }
            },
        )
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["provider"] == "glm"
        assert result["api_key"] == "test-key"
        assert result["max_tokens"] == 4096

    def test_max_concurrency_returned(self, tmp_path):
        """有 max_concurrency 时应包含在返回值中。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "k",
                        "api_base": "https://api.test.com/v1/",
                        "max_concurrency": 5,
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["max_concurrency"] == 5
        assert "rate_limit" not in result

    def test_no_max_concurrency_returns_none(self, tmp_path):
        """无 max_concurrency 时返回 None。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "k",
                        "api_base": "https://api.test.com/v1/",
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["max_concurrency"] is None

    def test_rate_limit_deprecated_mapped_to_max_concurrency(self, tmp_path):
        """旧配置 rate_limit 应映射到 max_concurrency 并打印 deprecation warning。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "k",
                        "api_base": "https://api.test.com/v1/",
                        "rate_limit": 3,
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        with pytest.warns(DeprecationWarning, match="rate_limit.*max_concurrency"):
            result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["max_concurrency"] == 3
        assert "rate_limit" not in result

    def test_max_concurrency_negative_raises(self, tmp_path):
        """max_concurrency <= 0 时应抛出 ValueError。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "k",
                        "api_base": "https://api.test.com/v1/",
                        "max_concurrency": 0,
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="max_concurrency must be positive"):
            get_model_config("glm/glm-4.7", models_path=cfg_path)

    def test_bare_model_name_raises(self, tmp_path):
        """裸名（如 glm-4.7）应报错提示格式。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "k",
                        "api_base": "https://api.test.com/v1/",
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="provider/model"):
            get_model_config("glm-4.7", models_path=cfg_path)

    def test_unknown_provider_raises(self, tmp_path):
        """provider 不存在时报错。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "k",
                        "api_base": "https://api.test.com/v1/",
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="Provider 'openai' not found"):
            get_model_config("openai/gpt-4", models_path=cfg_path)

    def test_unknown_model_raises(self, tmp_path):
        """provider 下无该 model 时报错。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "k",
                        "api_base": "https://api.test.com/v1/",
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        with pytest.raises(
            ValueError, match="Model 'glm-4-flash' not found under provider 'glm'"
        ):
            get_model_config("glm/glm-4-flash", models_path=cfg_path)


class TestEnvVarOverride:
    def test_env_var_resolved(self, tmp_path, monkeypatch):
        """api_key 为 ${ENV_VAR} 格式时应从环境变量解析。"""
        monkeypatch.setenv("MY_API_KEY", "resolved-key-123")
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "${MY_API_KEY}",
                        "api_base": "https://api.test.com/v1/",
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["api_key"] == "resolved-key-123"

    def test_plain_key_unchanged(self, tmp_path):
        """api_key 为明文字符串时应原样返回。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "plain-key",
                        "api_base": "https://api.test.com/v1/",
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["api_key"] == "plain-key"

    def test_env_var_not_set_raises(self, tmp_path):
        """${ENV_VAR} 对应的环境变量不存在时应抛出 ValueError。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "${MISSING_KEY}",
                        "api_base": "https://api.test.com/v1/",
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        with pytest.raises(ValueError, match="MISSING_KEY"):
            get_model_config("glm/glm-4.7", models_path=cfg_path)


class TestLoadSettings:
    def test_load_settings_returns_dict(self, tmp_path):
        """load_settings 应返回包含 defaults 和 providers 的字典。"""
        path = _write_test_settings(tmp_path, _SAMPLE_SETTINGS)
        result = load_settings(path)
        assert isinstance(result, dict)
        assert "defaults" in result
        assert "providers" in result

    def test_load_settings_file_not_found(self, tmp_path):
        """settings.yml 不存在时应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="Settings file not found"):
            load_settings(str(tmp_path / "nonexistent.yml"))

    def test_load_settings_contains_all_sections(self, tmp_path):
        """settings.yml 应包含 defaults, model_defaults, providers 区段。"""
        path = _write_test_settings(tmp_path, _SAMPLE_SETTINGS)
        result = load_settings(path)
        assert "defaults" in result
        assert "model_defaults" in result
        assert "providers" in result
        assert result["defaults"]["model"] == "glm-4.7"
        assert result["model_defaults"]["max_tokens"] == 131072

    def test_load_settings_env_var_preserved(self, tmp_path):
        """settings.yml 中 ${ENV_VAR} 语法应被原样保留，由后续解析处理。"""
        settings = {
            "defaults": {"model": "glm-4.7"},
            "model_defaults": {},
            "providers": {
                "zai": {
                    "api_key": "${MY_API_KEY}",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            },
        }
        path = _write_test_settings(tmp_path, settings)
        result = load_settings(path)
        assert result["providers"]["zai"]["api_key"] == "${MY_API_KEY}"


class TestLoadConfigFromSettings:
    def test_load_config_from_settings(self, tmp_path):
        """load_config 从 settings.yml 的 defaults 区段读取。"""
        settings_path = _write_test_settings(tmp_path, _SAMPLE_SETTINGS)
        result = load_config(settings_path)
        assert result["model"] == "glm-4.7"
        assert result["temperature"] == 0.0
        assert result["max_tokens"] == 4096
        assert result["max_retries"] == 3
        assert result["timeout"] == 300

    def test_load_config_from_yaml_fallback(self, tmp_path):
        """load_config 传入 .yaml 路径时仍可直接加载。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "model": "test-model",
                "temperature": 0.5,
            },
        )
        result = load_config(cfg_path)
        assert result["model"] == "test-model"


class TestLoadModelsConfigFromSettings:
    def test_load_models_config_from_settings(self, tmp_path):
        """load_models_config 从 settings.yml 构造 {defaults, providers} 结构。"""
        settings_path = _write_test_settings(tmp_path, _SAMPLE_SETTINGS)
        result = load_models_config(settings_path)
        assert "defaults" in result
        assert "providers" in result
        assert result["defaults"]["max_tokens"] == 131072
        assert "zai" in result["providers"]

    def test_load_models_config_from_yaml_fallback(self, tmp_path):
        """load_models_config 传入 .yaml 路径时仍可直接加载。"""
        cfg_path = _write_test_config(
            tmp_path,
            {
                "providers": {
                    "glm": {
                        "api_key": "k",
                        "api_base": "https://api.test.com/v1/",
                        "models": {"glm-4.7": {}},
                    }
                }
            },
        )
        result = load_models_config(cfg_path)
        assert "providers" in result


class TestGetModelConfigFromSettings:
    def test_get_model_config_via_settings(self, tmp_path):
        """get_model_config 通过 settings.yml 路径正确解析模型配置。"""
        settings_path = _write_test_settings(tmp_path, _SAMPLE_SETTINGS)
        result = get_model_config("zai/glm-4.7", models_path=settings_path)
        assert result["provider"] == "zai"
        assert result["api_key"] == "test-zai-key"
        assert result["api_base"] == "https://open.bigmodel.cn/api/coding/paas/v4"
        assert result["max_concurrency"] == 2
        assert result["max_tokens"] == 131072

    def test_get_model_config_thinking_via_settings(self, tmp_path):
        """通过 settings.yml 读取 thinking 配置。"""
        settings_path = _write_test_settings(tmp_path, _SAMPLE_SETTINGS)
        result = get_model_config("zai/glm-5", models_path=settings_path)
        assert result["thinking"]["enabled"] is True
        assert result["thinking"]["reasoning_field"] == "reasoning_content"

    def test_get_model_config_rate_limit_via_settings(self, tmp_path):
        """settings.yml 中 rate_limit 字段仍映射到 max_concurrency。"""
        settings_path = _write_test_settings(tmp_path, _SAMPLE_SETTINGS)
        with pytest.warns(DeprecationWarning):
            result = get_model_config("minimax/MiniMax-M2.7", models_path=settings_path)
        assert result["max_concurrency"] == 2

    def test_env_var_resolved_via_settings(self, tmp_path, monkeypatch):
        """settings.yml 中 ${ENV_VAR} 语法正确解析。"""
        monkeypatch.setenv("TEST_API_KEY", "resolved-key-456")
        settings = {
            "defaults": {},
            "model_defaults": {},
            "providers": {
                "zai": {
                    "api_key": "${TEST_API_KEY}",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            },
        }
        settings_path = _write_test_settings(tmp_path, settings)
        result = get_model_config("zai/glm-4.7", models_path=settings_path)
        assert result["api_key"] == "resolved-key-456"
