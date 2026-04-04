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

    def test_rate_limit_returned(self, tmp_path):
        """有 rate_limit 时应包含在返回值中。"""
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
        result = get_model_config("glm/glm-4.7", models_path=cfg_path)
        assert result["rate_limit"] == 3.0

    def test_no_rate_limit_returns_none(self, tmp_path):
        """无 rate_limit 时返回 None。"""
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
        assert result["rate_limit"] is None

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
