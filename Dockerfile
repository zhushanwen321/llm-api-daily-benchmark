# === 构建阶段：安装 Python 依赖 ===
FROM python:3.13-slim AS builder

WORKDIR /app

COPY pyproject.toml ./
COPY benchmark/ ./benchmark/
RUN pip install --no-cache-dir .

# === 运行阶段：只复制运行时产物 ===
FROM python:3.13-slim

ENV TZ=Asia/Shanghai

WORKDIR /app

# 安装运行时必需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制 Python 包和可执行文件
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制项目源码（streamlit run 需要直接路径访问 app.py）
COPY benchmark/ ./benchmark/

# 让 site-packages 中的 configs 指向工作目录，这样容器挂载的配置文件
# (models.yaml 等) 能被 __file__ 解析到 site-packages 的代码正确找到
RUN rm -rf /usr/local/lib/python3.13/site-packages/benchmark/configs \
    && ln -s /app/benchmark/configs /usr/local/lib/python3.13/site-packages/benchmark/configs

# 清理不需要的运行时文件以减小镜像体积
RUN pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.13/ensurepip \
    && rm -rf /usr/local/lib/python3.13/site-packages/pydeck* \
    && find /usr/local/lib/python3.13 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.13 -type d -name tests -path '*/site-packages/*' -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.13 -type f -name '*.pyi' -delete 2>/dev/null; \
    true

# 创建数据目录
RUN mkdir -p /app/data

# 暴露 Streamlit 端口
EXPOSE 8501

# 默认启动 Web 界面（调度器在 app.py 中自动拉起）
CMD ["streamlit", "run", "benchmark/visualization/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
