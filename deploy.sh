#!/usr/bin/env bash
#
# LLM Benchmark 一键部署脚本
# 用法: ./deploy.sh [TAG]
#   TAG: 镜像标签，默认 latest
#
# 前置条件:
#   - 已安装 docker 和 docker compose
#   - 已登录 ghcr.io: docker login ghcr.io -u <username> -p <token>
#
# 代理处理:
#   访问 ghcr.io 需要代理，脚本会自动:
#   1. 注释 /etc/hosts 中 github 相关行
#   2. 启动 HTTPS 代理
#   3. 拉取镜像
#   4. 还原 /etc/hosts 和代理配置
#

set -euo pipefail

IMAGE="ghcr.io/zhushanwen321/llm-api-daily-benchmark"
TAG="${1:-latest}"
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY="http://192.168.1.102:7890"
HOSTS_FILE="/etc/hosts"
HOSTS_BACKUP="/tmp/hosts.benchmark.bak"

echo "=== LLM Benchmark 部署 ==="
echo "镜像: ${IMAGE}:${TAG}"
echo "目录: ${DEPLOY_DIR}"
echo ""

# 创建必要目录
mkdir -p "${DEPLOY_DIR}/data"
mkdir -p "${DEPLOY_DIR}/configs"

# 初始化 .env（如果不存在）
if [ ! -f "${DEPLOY_DIR}/.env" ]; then
    if [ -f "${DEPLOY_DIR}/.env.example" ]; then
        cp "${DEPLOY_DIR}/.env.example" "${DEPLOY_DIR}/.env"
        echo "[!] 已从 .env.example 创建 .env，请编辑后重新运行"
        echo "    vim ${DEPLOY_DIR}/.env"
        exit 1
    else
        echo "[!] 缺少 .env 文件，请手动创建"
        exit 1
    fi
fi

# 初始化 models.yaml（如果不存在）
if [ ! -f "${DEPLOY_DIR}/configs/models.yaml" ]; then
    echo "[!] 缺少 configs/models.yaml，请创建后重新运行"
    echo "    参考 benchmark/configs/ 下的模板，api_key 使用 \${ENV_VAR} 格式"
    exit 1
fi

# --- 代理和 hosts 管理 ---

disable_github_hosts() {
    # 备份原始 hosts
    if [ ! -f "${HOSTS_BACKUP}" ]; then
        sudo cp "${HOSTS_FILE}" "${HOSTS_BACKUP}"
        echo "[proxy] 已备份 ${HOSTS_FILE} -> ${HOSTS_BACKUP}"
    fi
    # 注释掉包含 github 相关域名的行
    sudo sed -i -E 's/^([^#].*github.*)$/#\1/' "${HOSTS_FILE}"
    echo "[proxy] 已注释 ${HOSTS_FILE} 中的 github 相关行"
}

restore_hosts() {
    if [ -f "${HOSTS_BACKUP}" ]; then
        sudo cp "${HOSTS_BACKUP}" "${HOSTS_FILE}"
        rm -f "${HOSTS_BACKUP}"
        echo "[proxy] 已还原 ${HOSTS_FILE}"
    fi
}

cleanup() {
    echo "[proxy] 清理代理环境..."
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
    docker config rm proxy-config 2>/dev/null || true
    restore_hosts
    echo "[proxy] 清理完成"
}

trap cleanup EXIT

# --- 拉取镜像 ---

echo "[1/4] 配置代理并拉取镜像..."
disable_github_hosts

export http_proxy="${PROXY}"
export https_proxy="${PROXY}"
export all_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}"
export HTTPS_PROXY="${PROXY}"
export ALL_PROXY="${PROXY}"
echo "[proxy] 已设置代理: ${PROXY}"

docker pull "${IMAGE}:${TAG}"

# 拉取完成后立即清除代理环境（后续容器不需要代理）
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
restore_hosts
echo "[1/4] 镜像拉取完成，代理已清除"

# 生成 docker-compose.prod.yml
echo "[2/4] 生成 docker-compose.prod.yml..."
cat > "${DEPLOY_DIR}/docker-compose.prod.yml" <<EOF
services:
  benchmark:
    image: ${IMAGE}:${TAG}
    container_name: llm-benchmark
    restart: unless-stopped
    ports:
      - "8501:8501"
    volumes:
      - ${DEPLOY_DIR}/data:/app/data
      - ${DEPLOY_DIR}/.env:/app/.env:ro
      - ${DEPLOY_DIR}/configs/models.yaml:/app/benchmark/configs/models.yaml:ro
    env_file:
      - ${DEPLOY_DIR}/.env
    environment:
      - PYTHONUNBUFFERED=1
EOF

# 停止旧容器
echo "[3/4] 停止旧容器..."
docker compose -f "${DEPLOY_DIR}/docker-compose.prod.yml" down 2>/dev/null || true

# 启动新容器
echo "[4/4] 启动新容器..."
docker compose -f "${DEPLOY_DIR}/docker-compose.prod.yml" up -d

echo ""
echo "=== 部署完成 ==="
echo "Web 界面: http://localhost:8501"
echo "查看日志: docker compose -f ${DEPLOY_DIR}/docker-compose.prod.yml logs -f"
echo ""
echo "常用命令:"
echo "  停止:  docker compose -f ${DEPLOY_DIR}/docker-compose.prod.yml down"
echo "  重启:  docker compose -f ${DEPLOY_DIR}/docker-compose.prod.yml restart"
echo "  日志:  docker compose -f ${DEPLOY_DIR}/docker-compose.prod.yml logs -f"
echo "  状态:  docker compose -f ${DEPLOY_DIR}/docker-compose.prod.yml ps"
