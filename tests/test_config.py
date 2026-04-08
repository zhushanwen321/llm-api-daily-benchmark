"""config.py 的 provider/model 解析测试。"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch

from benchmark.config import get_model_config


def _write_test_config(tmp_path: Path, config: dict) -> str:
    """写入临时 models.yaml 并返回路径。"""
    p = tmp_path / "models.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return str(p)


class TestGetModelConfig:
    def test_valid_provider_model(self, tmp_path):
        """glm/glm-4.7 应正确解析。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "test-key",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {"max_tokens": 4096}},
                }
            }
        })
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["provider"] == "glm"
        assert result["api_key"] == "test-key"
        assert result["max_tokens"] == 4096

    def test_max_concurrency_returned(self, tmp_path):
        """有 max_concurrency 时应包含在返回值中。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "max_concurrency": 5,
                    "models": {"glm-4.7": {}},
                }
            }
        })
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["max_concurrency"] == 5
        assert "rate_limit" not in result

    def test_no_max_concurrency_returns_none(self, tmp_path):
        """无 max_concurrency 时返回 None。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["max_concurrency"] is None

    def test_rate_limit_deprecated_mapped_to_max_concurrency(self, tmp_path):
        """旧配置 rate_limit 应映射到 max_concurrency 并打印 deprecation warning。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "rate_limit": 3,
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with pytest.warns(DeprecationWarning, match="rate_limit.*max_concurrency"):
            result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["max_concurrency"] == 3
        assert "rate_limit" not in result

    def test_max_concurrency_negative_raises(self, tmp_path):
        """max_concurrency <= 0 时应抛出 ValueError。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "max_concurrency": 0,
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with pytest.raises(ValueError, match="max_concurrency must be positive"):
            get_model_config("glm/glm-4.7", models_path=cfg_path)

    def test_bare_model_name_raises(self, tmp_path):
        """裸名（如 glm-4.7）应报错提示格式。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with pytest.raises(ValueError, match="provider/model"):
            get_model_config("glm-4.7", models_path=cfg_path)

    def test_unknown_provider_raises(self, tmp_path):
        """provider 不存在时报错。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with pytest.raises(ValueError, match="Provider 'openai' not found"):
            get_model_config("openai/gpt-4", models_path=cfg_path)

    def test_unknown_model_raises(self, tmp_path):
        """provider 下无该 model 时报错。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "k",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with pytest.raises(ValueError, match="Model 'glm-4-flash' not found under provider 'glm'"):
            get_model_config("glm/glm-4-flash", models_path=cfg_path)


class TestEnvVarOverride:
    def test_env_var_resolved(self, tmp_path, monkeypatch):
        """api_key 为 ${ENV_VAR} 格式时应从环境变量解析。"""
        monkeypatch.setenv("MY_API_KEY", "resolved-key-123")
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "${MY_API_KEY}",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["api_key"] == "resolved-key-123"

    def test_plain_key_unchanged(self, tmp_path):
        """api_key 为明文字符串时应原样返回。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "plain-key",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["api_key"] == "plain-key"

    def test_env_var_not_set_raises(self, tmp_path):
        """${ENV_VAR} 对应的环境变量不存在时应抛出 ValueError。"""
        cfg_path = _write_test_config(tmp_path, {
            "providers": {
                "glm": {
                    "api_key": "${MISSING_KEY}",
                    "api_base": "https://api.test.com/v1/",
                    "models": {"glm-4.7": {}},
                }
            }
        })
        with pytest.raises(ValueError, match="MISSING_KEY"):
            get_model_config("glm/glm-4.7", models_path=cfg_path)
