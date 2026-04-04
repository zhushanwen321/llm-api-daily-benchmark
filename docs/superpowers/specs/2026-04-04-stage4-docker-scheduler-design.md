# Stage 4 设计文档：Docker 部署 + 定时调度

**日期**: 2026-04-04
**分支**: stage4
**状态**: 已确认

---

## 目标

将评测系统部署到生产环境，实现 Docker 容器化 + 定时自动评测指定模型在所有 benchmark 维度上的成绩。

## 核心决策

1. **执行粒度**：全部并行，受 rate_limit 约束
2. **容器主进程**：Streamlit Web + 调度器后台线程
3. **API Key 管理**：models.yaml 中 `api_key` 支持 `${ENV_VAR}` 语法，运行时从环境变量解析
4. **调度方案**：APScheduler 内嵌，作为 Streamlit 后台线程

---

## 1. CLI 改造

### 1.1 `--dimension` 支持 `all`

**文件**：`benchmark/cli.py`

- `--dimension` 参数新增 `all` 选项
- 当 `dimension=all` 时，对 `DIMENSION_REGISTRY` 中的所有维度并发评测
- 底层复用 `_run_evaluation`，用 `asyncio.gather` 并发执行
- 每个维度创建独立的 `EvalRun` 记录

### 1.2 `--model` 支持多模型

- `--model` 参数接受逗号分隔的列表（如 `glm/glm-4.7,kimi/kimi-2`）
- 多模型 x 多维度全部并行，受各 provider 的 rate_limit 约束

---

## 2. API Key 环境变量覆盖

### 2.1 `config.py` 改造

**文件**：`benchmark/config.py`

- `get_model_config()` 读取到 `api_key` 字段后，检查是否匹配 `${...}` 模式
- 如果是 `${ENV_VAR_NAME}`，从 `os.environ` 中读取对应环境变量
- 向后兼容：如果 `api_key` 不是 `${...}` 格式，直接使用 YAML 中的明文值

示例 `models.yaml`：
```yaml
providers:
  glm:
    api_key: ${ZAI_API_KEY}
    api_base: https://open.bigmodel.cn/api/coding/paas/v4
```

### 2.2 `.env` 文件

- 通过 `python-dotenv` 的 `load_dotenv()` 在启动时加载
- `.env` 中存放所有 provider 的 API Key 和调度配置
- `.env.example` 作为模板提交到 git，包含所有配置项的说明

---

## 3. 定时调度器

### 3.1 核心类

**文件**：`benchmark/core/scheduler.py`

- `BenchmarkScheduler` 类，基于 APScheduler 的 `BackgroundScheduler`
- 启动时读取环境变量配置
- 每次触发时调用 `_run_full_evaluation()`：
  - 解析 `SCHEDULER_MODELS` 和 `SCHEDULER_DIMENSIONS`
  - 用 `asyncio.gather` 全部并行执行
  - 每个模型 x 维度组合创建独立的 `EvalRun`

### 3.2 配置项

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `SCHEDULER_ENABLED` | 是否启用调度 | `false` |
| `SCHEDULER_CRON` | cron 表达式 | `0 2 * * *` |
| `SCHEDULER_MODELS` | 逗号分隔的模型列表 | - |
| `SCHEDULER_DIMENSIONS` | 逗号分隔的维度列表，`all` 表示全部 | `all` |
| `SCHEDULER_SAMPLES` | 每个维度的题目数量 | `15` |

### 3.3 启动方式

- Streamlit `app.py` 启动时检查 `SCHEDULER_ENABLED`，为 `true` 时自动拉起调度器后台线程
- CLI 支持 `benchmark scheduler start/stop/status` 手动管理

---

## 4. Docker 部署

### 4.1 Dockerfile

- 基于 `python:3.13-slim`
- 安装系统依赖（gcc）
- `pip install -e .` 安装项目依赖
- 复制 `benchmark/`、`pyproject.toml`
- 数据集目录（`benchmark/datasets/`）在构建时打入镜像
- 创建 `/app/data` 目录

### 4.2 docker-compose.yml

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

### 4.3 启动流程

```bash
cp .env.example .env
vim .env                      # 填入 API Key 和调度配置
vim configs/models.yaml       # 配置 provider（api_key 用 ${ENV_VAR} 占位）
docker-compose up -d
```

---

## 5. 新增依赖

- `apscheduler>=3.10` — 定时调度

---

## 6. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `Dockerfile` | 新增 | Docker 镜像定义 |
| `docker-compose.yml` | 新增 | Docker Compose 配置 |
| `.env.example` | 新增 | 环境变量模板 |
| `.gitignore` | 更新 | 忽略 `.env`、`data/` |
| `benchmark/cli.py` | 更新 | `--dimension all`、多模型支持、scheduler 子命令 |
| `benchmark/config.py` | 更新 | `${ENV_VAR}` 语法解析 |
| `benchmark/core/scheduler.py` | 新增 | APScheduler 定时调度器 |
| `benchmark/visualization/app.py` | 更新 | 启动时自动拉起调度器 |
| `pyproject.toml` | 更新 | 添加 `apscheduler` 依赖 |
