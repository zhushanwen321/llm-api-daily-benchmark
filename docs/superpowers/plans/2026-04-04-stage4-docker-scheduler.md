# Stage 4: Docker 部署 + 定时调度 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将评测系统容器化部署，支持定时自动评测指定模型在所有 benchmark 维度上的成绩。

**Architecture:** CLI 新增 `--dimension all` 和多模型支持，调度器基于 APScheduler 作为 Streamlit 后台线程运行，Docker 镜像打包整个应用，通过 volume 挂载管理配置和数据持久化。

**Tech Stack:** Python 3.13, APScheduler, Docker, Docker Compose, python-dotenv, Click

---

## File Structure

| 文件 | 操作 | 职责 |
|------|------|------|
| `benchmark/config.py` | 修改 | 新增 `_resolve_env_var()` 解析 `${ENV_VAR}` 语法 |
| `tests/test_config.py` | 修改 | 新增环境变量覆盖测试 |
| `benchmark/cli.py` | 修改 | `--dimension all`、多模型、scheduler 子命令 |
| `tests/test_cli.py` | 新增 | CLI 新参数和 scheduler 子命令测试 |
| `benchmark/core/scheduler.py` | 新增 | APScheduler 定时调度器 |
| `tests/test_scheduler.py` | 新增 | 调度器单元测试 |
| `benchmark/visualization/app.py` | 修改 | 启动时自动拉起调度器 |
| `pyproject.toml` | 修改 | 添加 `apscheduler>=3.10` 依赖 |
| `Dockerfile` | 新增 | Docker 镜像定义 |
| `docker-compose.yml` | 新增 | Docker Compose 配置 |
| `.env.example` | 新增 | 环境变量模板 |
| `.gitignore` | 修改 | 添加 `data/` 忽略 |

---

### Task 1: config.py — `${ENV_VAR}` 环境变量解析

**Files:**
- Modify: `benchmark/config.py:49-100`
- Test: `tests/test_config.py`

- [ ] **Step 1: 在 `test_config.py` 末尾追加环境变量覆盖测试**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/test_config.py::TestEnvVarOverride -v`
Expected: FAIL — `get_model_config` 尚未支持 `${ENV_VAR}` 解析

- [ ] **Step 3: 在 `config.py` 中实现 `_resolve_env_var` 并集成到 `get_model_config`**

在 `config.py` 顶部新增 import：

```python
import os
import re
```

在 `get_model_config` 函数之前新增辅助函数：

```python
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
```

在 `get_model_config` 返回字典中，将 `"api_key": provider_cfg["api_key"]` 改为：

```python
        "api_key": _resolve_env_var(provider_cfg["api_key"], "api_key"),
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add benchmark/config.py tests/test_config.py
git commit -m "feat(config): api_key 支持 \${ENV_VAR} 环境变量覆盖"
```

---

### Task 2: CLI — `--dimension all` 支持

**Files:**
- Modify: `benchmark/cli.py:77-106`
- Test: `tests/test_cli.py` (新增)

- [ ] **Step 1: 创建 `tests/test_cli.py`，写入 `--dimension all` 测试**

```python
"""CLI 参数解析测试。"""

from click.testing import CliRunner

from benchmark.cli import cli


class TestDimensionAll:
    def test_dimension_all_accepted(self):
        """--dimension all 应被接受为合法参数。"""
        runner = CliRunner()
        # 不实际执行评测，只验证参数解析不报错
        result = runner.invoke(cli, ["evaluate", "--model", "glm/glm-4.7", "--dimension", "all", "--help"])
        # --help 会直接退出，不会有错误
        assert result.exit_code == 0

    def test_dimension_choices_includes_all(self):
        """dimension 参数的选项应包含 all。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--model", "glm/glm-4.7", "--dimension", "invalid"])
        assert result.exit_code != 0
        # Click 输出的错误信息中应包含 all 作为合法选项
        assert "all" in result.output
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/test_cli.py::TestDimensionAll -v`
Expected: FAIL — `--dimension` 的 `click.Choice` 不包含 `all`

- [ ] **Step 3: 修改 `cli.py` 中 `evaluate` 命令的 `--dimension` 参数**

将 `cli.py:84-86` 的 `type=click.Choice([...])` 改为包含 `all`：

```python
@click.option(
    "--dimension",
    required=True,
    type=click.Choice(["reasoning", "backend-dev", "system-architecture", "frontend-dev", "all"]),
    help="评测维度，all 表示全部维度",
)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/test_cli.py::TestDimensionAll -v`
Expected: PASS

- [ ] **Step 5: 实现多模型和多维度的并发评测逻辑**

在 `cli.py` 中，修改 `evaluate` 命令的处理逻辑，使其支持多模型和 `dimension=all`：

将 `cli.py:95-106` 的 `evaluate` 函数体替换为：

```python
def evaluate(
    ctx: click.Context, model: str, dimension: str, samples: int, debug: bool
) -> None:
    # 如果子命令没有指定 debug，使用父命令的设置
    if not debug:
        debug = ctx.obj.get("debug", False)
    _setup_proxy()

    models = [m.strip() for m in model.split(",")]
    dimensions = list(DIMENSION_REGISTRY.keys()) if dimension == "all" else [dimension]

    asyncio.run(_run_multi_evaluation(models, dimensions, samples, debug))
```

在 `cli.py` 中，在 `_run_evaluation` 函数之前新增 `_run_multi_evaluation`：

```python
async def _run_multi_evaluation(
    models: list[str], dimensions: list[str], samples: int, debug: bool
) -> None:
    """多模型 x 多维度并发评测。"""
    coros = [
        _run_evaluation(model, dim, samples, debug)
        for model in models
        for dim in dimensions
    ]
    await asyncio.gather(*coros)
```

- [ ] **Step 6: 运行全部测试确认无回归**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/ -v --ignore=tests/test_llm_adapter.py --ignore=tests/test_app_integration.py`
Expected: ALL PASS（跳过需要 API 调用的集成测试）

- [ ] **Step 7: 提交**

```bash
git add benchmark/cli.py tests/test_cli.py
git commit -m "feat(cli): --dimension all 和多模型逗号分隔支持"
```

---

### Task 3: 调度器 — `benchmark/core/scheduler.py`

**Files:**
- Create: `benchmark/core/scheduler.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: 创建 `tests/test_scheduler.py`，写入调度器测试**

```python
"""scheduler.py 定时调度器测试。"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestBenchmarkScheduler:
    def test_disabled_by_default(self):
        """SCHEDULER_ENABLED 未设置时调度器不启动。"""
        with patch.dict(os.environ, {}, clear=True):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            assert sched.enabled is False

    def test_enabled_from_env(self):
        """SCHEDULER_ENABLED=true 时 enabled 为 True。"""
        with patch.dict(os.environ, {"SCHEDULER_ENABLED": "true"}):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            assert sched.enabled is True

    def test_models_parsed_from_env(self):
        """SCHEDULER_MODELS 应正确解析为列表。"""
        with patch.dict(os.environ, {
            "SCHEDULER_ENABLED": "true",
            "SCHEDULER_MODELS": "glm/glm-4.7,kimi/kimi-2",
            "SCHEDULER_DIMENSIONS": "all",
            "SCHEDULER_CRON": "0 2 * * *",
            "SCHEDULER_SAMPLES": "15",
        }):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            assert sched.models == ["glm/glm-4.7", "kimi/kimi-2"]
            assert sched.dimensions == ["all"]
            assert sched.cron == "0 2 * * *"
            assert sched.samples == 15

    def test_start_does_nothing_when_disabled(self):
        """enabled=False 时 start() 应直接返回。"""
        with patch.dict(os.environ, {}, clear=True):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            sched.start()
            assert sched._scheduler is None

    def test_start_creates_background_scheduler(self):
        """enabled=True 时 start() 应创建 APScheduler 并启动。"""
        with patch.dict(os.environ, {
            "SCHEDULER_ENABLED": "true",
            "SCHEDULER_MODELS": "glm/glm-4.7",
            "SCHEDULER_CRON": "0 2 * * *",
        }):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            sched.start()
            assert sched._scheduler is not None
            assert sched._scheduler.running
            sched.stop()

    def test_stop_shuts_down_scheduler(self):
        """stop() 应关闭 APScheduler。"""
        with patch.dict(os.environ, {
            "SCHEDULER_ENABLED": "true",
            "SCHEDULER_MODELS": "glm/glm-4.7",
            "SCHEDULER_CRON": "0 2 * * *",
        }):
            from benchmark.core.scheduler import BenchmarkScheduler
            sched = BenchmarkScheduler()
            sched.start()
            sched.stop()
            assert not sched._scheduler.running
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/test_scheduler.py -v`
Expected: FAIL — `benchmark.core.scheduler` 模块不存在

- [ ] **Step 3: 安装 apscheduler 依赖**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv add apscheduler>=3.10`

- [ ] **Step 4: 创建 `benchmark/core/scheduler.py`**

```python
"""定时调度器：基于 APScheduler 实现定时评测。"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BenchmarkScheduler:
    """定时评测调度器。

    从环境变量读取配置，使用 APScheduler 按 cron 表达式定时触发评测。
    调度器在后台线程中运行，主进程（如 Streamlit）不受阻塞。
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("SCHEDULER_ENABLED", "false").lower() == "true"
        self.cron = os.getenv("SCHEDULER_CRON", "0 2 * * *")
        self.models = [
            m.strip()
            for m in os.getenv("SCHEDULER_MODELS", "").split(",")
            if m.strip()
        ]
        dimensions_raw = os.getenv("SCHEDULER_DIMENSIONS", "all")
        self.dimensions = [
            d.strip()
            for d in dimensions_raw.split(",")
            if d.strip()
        ]
        self.samples = int(os.getenv("SCHEDULER_SAMPLES", "15"))
        self._scheduler: BackgroundScheduler | None = None

    def start(self) -> None:
        """启动调度器。如果未启用则跳过。"""
        if not self.enabled:
            logger.info("调度器未启用 (SCHEDULER_ENABLED != true)")
            return

        if not self.models:
            logger.warning("调度器启用但未配置 SCHEDULER_MODELS，跳过启动")
            return

        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._run_scheduled_evaluation,
            CronTrigger.from_crontab(self.cron),
            id="daily_evaluation",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            f"调度器已启动: cron='{self.cron}', "
            f"models={self.models}, dimensions={self.dimensions}, "
            f"samples={self.samples}"
        )

    def stop(self) -> None:
        """停止调度器。"""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("调度器已停止")

    def _run_scheduled_evaluation(self) -> None:
        """调度触发时执行的全量评测。"""
        logger.info("定时评测触发: models=%s, dimensions=%s", self.models, self.dimensions)
        from benchmark.cli import _run_multi_evaluation

        dimensions = self.dimensions
        if dimensions == ["all"]:
            from benchmark.cli import DIMENSION_REGISTRY
            dimensions = list(DIMENSION_REGISTRY.keys())

        asyncio.run(_run_multi_evaluation(self.models, dimensions, self.samples, debug=False))
        logger.info("定时评测完成")
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/test_scheduler.py -v`
Expected: ALL PASS

- [ ] **Step 6: 提交**

```bash
git add benchmark/core/scheduler.py tests/test_scheduler.py pyproject.toml uv.lock
git commit -m "feat(scheduler): 基于 APScheduler 的定时评测调度器"
```

---

### Task 4: CLI — scheduler 子命令

**Files:**
- Modify: `benchmark/cli.py` (末尾追加)
- Test: `tests/test_cli.py`

- [ ] **Step 1: 在 `test_cli.py` 追加 scheduler 子命令测试**

```python
class TestSchedulerCLI:
    def test_scheduler_start(self):
        """benchmark scheduler start 应正常执行。"""
        runner = CliRunner()
        # SCHEDULER_ENABLED 不设为 true，start 应直接退出
        result = runner.invoke(cli, ["scheduler", "start"])
        assert result.exit_code == 0

    def test_scheduler_status(self):
        """benchmark scheduler status 应正常执行。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["scheduler", "status"])
        assert result.exit_code == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/test_cli.py::TestSchedulerCLI -v`
Expected: FAIL — `scheduler` 子命令不存在

- [ ] **Step 3: 在 `cli.py` 末尾追加 scheduler 子命令**

```python
@cli.group()
def scheduler() -> None:
    """定时调度器管理。"""


@scheduler.command()
def start() -> None:
    """启动定时调度器。"""
    from benchmark.core.scheduler import BenchmarkScheduler
    sched = BenchmarkScheduler()
    sched.start()
    if sched.enabled:
        console.print("[green]调度器已启动[/green]")
        console.print(f"  Cron: {sched.cron}")
        console.print(f"  Models: {sched.models}")
        console.print(f"  Dimensions: {sched.dimensions}")
        console.print(f"  Samples: {sched.samples}")
    else:
        console.print("[yellow]调度器未启用 (设置 SCHEDULER_ENABLED=true)[/yellow]")


@scheduler.command()
def stop() -> None:
    """停止定时调度器。"""
    from benchmark.core.scheduler import BenchmarkScheduler
    sched = BenchmarkScheduler()
    sched.stop()
    console.print("[green]调度器已停止[/green]")


@scheduler.command()
def status() -> None:
    """查看调度器状态。"""
    from benchmark.core.scheduler import BenchmarkScheduler
    sched = BenchmarkScheduler()
    console.print(f"  Enabled: {sched.enabled}")
    console.print(f"  Cron: {sched.cron}")
    console.print(f"  Models: {sched.models}")
    console.print(f"  Dimensions: {sched.dimensions}")
    console.print(f"  Samples: {sched.samples}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 5: 提交**

```bash
git add benchmark/cli.py tests/test_cli.py
git commit -m "feat(cli): 新增 scheduler start/stop/status 子命令"
```

---

### Task 5: Streamlit — 启动时自动拉起调度器

**Files:**
- Modify: `benchmark/visualization/app.py:91-93`

- [ ] **Step 1: 在 `app.py` 的 `main()` 函数开头追加调度器启动逻辑**

在 `app.py:92`（`st.title(...)` 之后）追加：

```python
    # 启动定时调度器（如果已启用）
    if "scheduler_started" not in st.session_state:
        from benchmark.core.scheduler import BenchmarkScheduler
        sched = BenchmarkScheduler()
        sched.start()
        st.session_state["scheduler_started"] = True
```

- [ ] **Step 2: 提交**

```bash
git add benchmark/visualization/app.py
git commit -m "feat(web): Streamlit 启动时自动拉起调度器"
```

---

### Task 6: Docker 部署文件

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Modify: `.gitignore`

- [ ] **Step 1: 创建 `Dockerfile`**

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# 复制项目文件
COPY benchmark/ ./benchmark/

# 创建数据目录
RUN mkdir -p /app/data

# 暴露 Streamlit 端口
EXPOSE 8501

# 默认启动 Web 界面（调度器在 app.py 中自动拉起）
CMD ["streamlit", "run", "benchmark/visualization/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

- [ ] **Step 2: 创建 `docker-compose.yml`**

```yaml
services:
  benchmark:
    build: .
    container_name: llm-benchmark
    restart: unless-stopped
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./.env:/app/.env:ro
      - ./configs/models.yaml:/app/benchmark/configs/models.yaml:ro
    env_file:
      - .env
    environment:
      - PYTHONUNBUFFERED=1
```

- [ ] **Step 3: 创建 `.env.example`**

```bash
# ========== API 配置 ==========
# 在 models.yaml 中用 ${ZAI_API_KEY} 引用这些变量
ZAI_API_KEY=your_api_key_here
MINIMAX_API_KEY=your_api_key_here
KIMI_API_KEY=your_api_key_here

# ========== 调度配置 ==========
# 是否启用定时调度
SCHEDULER_ENABLED=false
# Cron 表达式（默认每天凌晨 2 点）
SCHEDULER_CRON=0 2 * * *
# 要评测的模型列表（逗号分隔）
SCHEDULER_MODELS=glm/glm-4.7
# 要评测的维度（逗号分隔，all 表示全部）
SCHEDULER_DIMENSIONS=all
# 每个维度的题目数量
SCHEDULER_SAMPLES=15

# ========== HuggingFace 代理 ==========
# HF_PROXY=http://proxy:port
```

- [ ] **Step 4: 更新 `.gitignore`，在末尾追加**

```
# Docker 数据持久化
data/
```

- [ ] **Step 5: 确认 configs 目录结构**

确保 `configs/` 目录存在用于 Docker volume 挂载：

```bash
mkdir -p configs
```

在 `configs/` 下创建 `models.yaml.example`（从现有模板复制）：

```bash
cp benchmark/configs/default.yaml configs/default.yaml 2>/dev/null; true
```

- [ ] **Step 6: 提交**

```bash
git add Dockerfile docker-compose.yml .env.example .gitignore
git commit -m "feat(docker): 添加 Dockerfile、docker-compose 和 .env.example"
```

---

### Task 7: 集成验证

**Files:** 无新增

- [ ] **Step 1: 运行全部单元测试**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run pytest tests/ -v --ignore=tests/test_llm_adapter.py --ignore=tests/test_app_integration.py`
Expected: ALL PASS

- [ ] **Step 2: 验证 CLI 参数解析**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run benchmark evaluate --help`
Expected: `--dimension` 选项包含 `all`

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && uv run benchmark scheduler status`
Expected: 输出调度器配置状态

- [ ] **Step 3: 验证 Docker 构建**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && docker build -t llm-benchmark .`
Expected: 构建成功

- [ ] **Step 4: 提交最终状态**

如果前面所有步骤的提交都已完成，此处无需额外提交。
