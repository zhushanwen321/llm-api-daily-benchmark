FROM python:3.13-slim

ENV TZ=Asia/Shanghai

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    tzdata \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存
COPY pyproject.toml ./
COPY benchmark/ ./benchmark/
RUN pip install --no-cache-dir .

# 创建数据目录
RUN mkdir -p /app/data

# 暴露 Streamlit 端口
EXPOSE 8501

# 默认启动 Web 界面（调度器在 app.py 中自动拉起）
CMD ["streamlit", "run", "benchmark/visualization/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
