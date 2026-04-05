# === 构建阶段：安装编译工具和 Python 依赖 ===
FROM python:3.13-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY benchmark/ ./benchmark/
RUN pip install --no-cache-dir .

# === 运行阶段：只复制运行时产物，不含编译工具 ===
FROM python:3.13-slim

ENV TZ=Asia/Shanghai

WORKDIR /app

# 安装运行时必需的系统库（不含 gcc 等编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制 Python 包和可执行文件
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目源码
COPY benchmark/ ./benchmark/

# 卸载 pip 减小体积
RUN pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.13/ensurepip

# 创建数据目录
RUN mkdir -p /app/data

# 暴露 Streamlit 端口
EXPOSE 8501

# 默认启动 Web 界面（调度器在 app.py 中自动拉起）
CMD ["streamlit", "run", "benchmark/visualization/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
