# LLM Benchmark - Stage 4 规格概要

**阶段**: 生产部署  
**版本**: 1.0  
**创建日期**: 2026-04-02  
**状态**: 规划中  
**前置条件**: Stage 3 完成并验收  
**估算工作量**: 12-18 小时

---

## 概述

### 目标

将评测系统部署到生产环境，实现服务器上的自动化定时评测。

### 交付价值

- ✅ Docker容器化部署
- ✅ .env配置支持
- ✅ 定时自动化评测
- ✅ 配置文件外部化

---

## 新增功能

### 1. Docker容器化

**文件**：`Dockerfile`、`docker-compose.yml`

#### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# 复制项目文件
COPY benchmark/ ./benchmark/
COPY datasets/ ./datasets/

# 创建数据目录
RUN mkdir -p /app/data

# 暴露Streamlit端口
EXPOSE 8501

# 默认启动Web界面
CMD ["streamlit", "run", "benchmark/visualization/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  benchmark:
    build: .
    container_name: llm-benchmark
    restart: unless-stopped
    ports:
      - "8501:8501"
    volumes:
      # 数据持久化
      - ./data:/app/data
      # 配置文件外部映射
      - ./configs/models.yaml:/app/benchmark/configs/models.yaml:ro
      - ./configs/.env:/app/.env:ro
    env_file:
      - .env
    environment:
      - PYTHONUNBUFFERED=1
```

---

### 2. 环境变量配置支持

**文件**：`benchmark/config.py`、`.env.example`

#### 配置读取逻辑

更新配置读取优先级：
1. 环境变量（最高优先级）
2. .env文件
3. YAML配置文件
4. 默认值

#### .env.example

```bash
# ========== API配置 ==========
# ZAI API配置
ZAI_API_KEY=your_api_key_here
ZAI_API_BASE=https://open.bigmodel.cn/api/coding/paas/v4
ZAI_RATE_LIMIT=2

# MiniMax API配置
MINIMAX_API_KEY=your_api_key_here
MINIMAX_API_BASE=https://api.minimaxi.com/v1
MINIMAX_RATE_LIMIT=2

# Kimi API配置
KIMI_API_KEY=your_api_key_here
KIMI_API_BASE=https://api.kimi.com/coding/v1
KIMI_RATE_LIMIT=2

# ========== 调度配置 ==========
# 是否启用定时调度
SCHEDULER_ENABLED=true
# 调度时间表达式（cron格式）
SCHEDULER_CRON=0 2 * * *
# 要评测的模型列表（逗号分隔）
SCHEDULER_MODELS=glm-4.7,gpt-4
# 要评测的维度（逗号分隔，all表示全部）
SCHEDULER_DIMENSIONS=all

# ========== 数据库配置 ==========
# 数据库路径
DB_PATH=/app/data/results.db

# ========== Web服务配置 ==========
# Streamlit端口
WEB_PORT=8501
# Streamlit地址
WEB_ADDRESS=0.0.0.0
```

---

### 3. 定时调度器

**文件**：`benchmark/core/scheduler.py`

#### 功能

- 读取.env中的调度配置
- APScheduler实现cron调度
- PID文件管理（防止重复启动）
- 调度日志记录
- 支持动态配置（.env修改后自动重载）

#### CLI命令

```bash
# 启动调度器
python -m benchmark scheduler start

# 查看状态
python -m benchmark scheduler status

# 停止调度器
python -m benchmark scheduler stop
```

#### 配置说明

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| SCHEDULER_ENABLED | 是否启用调度 | false |
| SCHEDULER_CRON | Cron表达式 | 0 2 * * * |
| SCHEDULER_MODELS | 评测模型列表 | - |
| SCHEDULER_DIMENSIONS | 评测维度 | all |

#### 调度器逻辑

```python
class Scheduler:
    """定时调度器"""

    def __init__(self):
        self.enabled = os.getenv("SCHEDULER_ENABLED", "false").lower() == "true"
        self.cron = os.getenv("SCHEDULER_CRON", "0 2 * * *")
        self.models = os.getenv("SCHEDULER_MODELS", "").split(",")
        self.dimensions = os.getenv("SCHEDULER_DIMENSIONS", "all")

    def start(self):
        """启动调度器"""
        if not self.enabled:
            logger.info("调度器未启用")
            return

        scheduler = BackgroundScheduler()
        scheduler.add_job(
            self.run_evaluation,
            CronTrigger.from_crontab(self.cron),
            id="daily_evaluation"
        )
        scheduler.start()

        logger.info(f"调度器已启动: {self.cron}")

    def run_evaluation(self):
        """执行评测任务"""
        for model in self.models:
            logger.info(f"开始评测: {model}")
            # 调用评测逻辑
            evaluate(model, self.dimensions)
```

---

### 4. 配置文件外部化

**目的**：支持部署时通过volume挂载的方式管理配置，无需重新构建镜像

#### docker-compose.yml配置

```yaml
services:
  benchmark:
    volumes:
      # models.yaml外部映射
      - ./configs/models.yaml:/app/benchmark/configs/models.yaml:ro
      # .env外部映射
      - ./configs/.env:/app/.env:ro
```

#### 使用方式

```bash
# 1. 准备配置文件目录
mkdir -p configs

# 2. 复制示例配置
cp .env.example configs/.env
cp benchmark/configs/models.yaml configs/

# 3. 编辑配置（敏感信息）
vim configs/.env
vim configs/models.yaml

# 4. 启动容器
docker-compose up -d

# 5. 修改配置后重启
vim configs/models.yaml
docker-compose restart benchmark
```

---

## 验收标准

### 1. Docker部署

```bash
# 构建镜像
docker-compose build

# 启动容器
docker-compose up -d

# 预期：
# - 容器成功启动
# - 访问 http://localhost:8501 可以看到Web界面
# - 数据目录创建成功
```

### 2. 环境变量配置

```bash
# 设置环境变量
export ZAI_API_KEY=test_key

# 运行评测
python -m benchmark evaluate --model glm-4.7 --dimension reasoning

# 预期：成功读取环境变量中的API Key
```

### 3. 定时调度

```bash
# 启用调度器
export SCHEDULER_ENABLED=true
export SCHEDULER_CRON="*/5 * * * *"  # 每5分钟执行一次（测试用）

# 启动调度器
python -m benchmark scheduler start

# 预期：
# - 每5分钟自动执行一次评测
# - 日志显示调度记录
```

### 4. 配置文件外部化

```bash
# 修改外部models.yaml
vim configs/models.yaml

# 重启容器
docker-compose restart benchmark

# 预期：
# - 容器使用新的配置
# - 无需重新构建镜像
```

---

## 部署流程

### 开发环境

```bash
# 1. 复制环境变量模板
cp .env.example .env

# 2. 编辑配置
vim .env

# 3. 运行
uv run python -m benchmark evaluate --model glm-4.7 --dimension reasoning
```

### 生产环境

```bash
# 1. 准备配置目录
mkdir -p /opt/benchmark/configs
mkdir -p /opt/benchmark/data

# 2. 复制配置文件
cp .env.example /opt/benchmark/configs/.env
cp benchmark/configs/models.yaml /opt/benchmark/configs/

# 3. 编辑生产配置
vim /opt/benchmark/configs/.env
vim /opt/benchmark/configs/models.yaml

# 4. 启动容器
docker-compose -f docker-compose.prod.yml up -d

# 5. 查看日志
docker-compose logs -f benchmark
```

---

## 目录结构

```
llm-api-daily-benchmark/
├── Dockerfile                          # 🆕 Docker镜像定义
├── docker-compose.yml                  # 🆕 Docker Compose配置
├── docker-compose.prod.yml             # 🆕 生产环境配置
├── .env.example                        # 🆕 环境变量模板
├── benchmark/
│   ├── core/
│   │   └── scheduler.py                # 🆕 定时调度器
│   └── config.py                       # 更新：支持.env读取
└── configs/                            # 🆕 配置文件目录（用于部署）
    ├── models.yaml                     # 🆕 模型配置（可映射）
    └── .env                            # 🆕 环境变量（可映射）
```

---

## 依赖更新

**无新增Python依赖**

**系统依赖**：
- Docker
- Docker Compose

---

## 安全注意事项

1. **敏感信息管理**
   - .env文件包含API Key，必须加入.gitignore
   - 生产环境建议使用secrets管理工具（如Docker Secrets）

2. **配置文件权限**
   - models.yaml映射为只读（:ro）
   - .env文件权限设为600

3. **容器安全**
   - 使用非root用户运行（可选）
   - 限制容器资源（可选）

---

## 状态追踪

| 组件 | 状态 | 说明 |
|------|------|------|
| Dockerfile | ⏳ 待实施 | 文件：`Dockerfile` |
| docker-compose.yml | ⏳ 待实施 | 文件：`docker-compose.yml` |
| .env支持 | ⏳ 待实施 | 更新：`config.py` |
| Scheduler | ⏳ 待实施 | 文件：`core/scheduler.py` |

---

## 下一步

⏳ 等待Stage 3完成并验收后开始实施

---

## 实施顺序建议

1. **Day 1**: 实现Dockerfile和docker-compose.yml
2. **Day 2**: 实现.env配置支持（更新config.py）
3. **Day 3**: 实现定时调度器

**总工作量**：12-18小时
