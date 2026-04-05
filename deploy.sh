#!/usr/bin/env bash
#
# LLM Benchmark 一键部署脚本
# 用法: ./deploy.sh [TAG]
#   TAG: 镜像标签，默认 latest
#
# 部署目录结构:
#   deploy.sh            ← 本脚本
#   .env                 ← API Key + 调度配置（首次运行自动生成模板）
#   models.yaml          ← 模型 provider 配置（首次运行自动生成模板）
#   data/                ← 评测数据持久化（自动创建）
#   dataset/             ← 数据集缓存（HF 下载后持久化，避免重复下载）
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

# 宎宿主机: dataset 缓存目录; 容器内: benchmark/datasets 数据集加载目录
DATASET_HOST_DIR="${DEPLOY_DIR}/dataset"
DATASET_FLAG="${DATASET_HOST_DIR}/.download-complete"

echo "=== LLM Benchmark 部署 ==="
echo "镜像: ${IMAGE}:${TAG}"
echo "目录: ${DEPLOY_DIR}"
echo ""

# --- 初始化配置文件 ---

init_env() {
    if [ ! -f "${DEPLOY_DIR}/.env" ]; then
        cat > "${DEPLOY_DIR}/.env" <<'ENVEOF'
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
# 要评测的模型列表（逗号分隔，对应 models.yaml 中的 provider/model）
SCHEDULER_MODELS=glm/glm-4.7
# 要评测的维度（逗号分隔，all 表示全部）
SCHEDULER_DIMENSIONS=all
# 每个维度的题目数量
SCHEDULER_SAMPLES=15

# ========== HuggingFace 代理 ==========
# HF_PROXY=http://proxy:port
# 数据集离线模式（首次运行自动下载数据集后，在 dataset/ 下创建 .download-complete 标志文件即可自动启用）
# HF_DATASETS_OFFLINE=0
ENVEOF
        echo "[!] 已生成 .env 模板，请编辑后重新运行:"
        echo "    vim ${DEPLOY_DIR}/.env"
        echo ""
        echo "    必填项:"
        echo "      - ZAI_API_KEY / MINIMAX_API_KEY / KIMI_API_KEY"
        echo "      - SCHEDULER_MODELS (要评测的模型列表)"
        echo "      - SCHEDULER_ENABLED=true (启用定时调度)"
        exit 1
    fi
}

init_models() {
    if [ ! -f "${DEPLOY_DIR}/models.yaml" ]; then
        cat > "${DEPLOY_DIR}/models.yaml" <<'YAMLEOF'
# 模型 Provider 配置
# api_key 使用 ${ENV_VAR} 格式引用 .env 中的变量

defaults:
  max_tokens: 131072

providers:
  glm:
    api_key: ${ZAI_API_KEY}
    api_base: https://open.bigmodel.cn/api/coding/paas/v4
    rate_limit: 2
    models:
      glm-4.7:
        max_tokens: 131072

  # minimax:
  #   api_key: ${MINIMAX_API_KEY}
  #   api_base: https://api.minimaxi.com/v1
  #   rate_limit: 2
  #   models:
  #     minimax-2:
  #       max_tokens: 131072
  #       thinking:
  #         enabled: true
  #         request_params:
  #           reasoning_split: true
  #         reasoning_field: reasoning_details

  # kimi:
  #   api_key: ${KIMI_API_KEY}
  #   api_base: https://api.kimi.com/coding/v1
  #   rate_limit: 2
  #   models:
  #     kimi-2:
  #       max_tokens: 131072
YAMLEOF
        echo "[!] 已生成 models.yaml 模板，请编辑后重新运行:"
        echo "    vim ${DEPLOY_DIR}/models.yaml"
        echo ""
        echo "    配置说明:"
        echo "      - 取消注释需要的 provider"
        echo "      - api_key 已使用 \${ENV_VAR} 引用 .env 中的变量"
        echo "      - 新增 provider 时同步在 .env 中添加对应 API Key"
        exit 1
    fi
}

mkdir -p "${DEPLOY_DIR}/data" "${DATASET_HOST_DIR}"
init_env
init_models

# 检测数据集缓存标志，决定是否需要网络下载
if [ -f "${DATASET_FLAG}" ]; then
    echo "[dataset] 检测到完整缓存标志 (.download-complete)，将自动启用离线模式"
    # 在 .env 中设置离线模式（如果用户没有手动设置）
    if grep -q "^HF_DATASETS_OFFLINE=1" "${DEPLOY_DIR}/.env" 2>/dev/null; then
        echo "[dataset] .env 已配置 HF_DATASETS_OFFLINE=1"
    else
        # 追加或修改离线设置
        if grep -q "^HF_DATASETS_OFFLINE=" "${DEPLOY_DIR}/.env" 2>/dev/null; then
            sed -i 's/^HF_DATASETS_OFFLINE=.*/HF_DATASETS_OFFLINE=1/' "${DEPLOY_DIR}/.env"
        else
            echo "" >> "${DEPLOY_DIR}/.env"
            echo "# 数据集已缓存，启用离线模式" >> "${DEPLOY_DIR}/.env"
            echo "HF_DATASETS_OFFLINE=1" >> "${DEPLOY_DIR}/.env"
        fi
        echo "[dataset] 已自动设置 HF_DATASETS_OFFLINE=1"
    fi
else
    echo "[dataset] 未检测到缓存标志，首次运行将从 HuggingFace 下载数据集"
    echo "[dataset] 下载完成后请在 dataset/ 目录下创建标志文件:"
    echo "    touch ${DATASET_HOST_DIR}/.download-complete"
fi

echo "[配置] .env 和 models.yaml 已就绪"
echo ""

# --- 代理和 hosts 管理 ---

disable_github_hosts() {
    if [ ! -f "${HOSTS_BACKUP}" ]; then
        sudo cp "${HOSTS_FILE}" "${HOSTS_BACKUP}"
        echo "[proxy] 已备份 ${HOSTS_FILE}"
    fi
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
000    restore_hosts
    echo "[proxy] 清理完成"
}

trap cleanup EXIT

# --- 拉取镜像 ---

echo "[1/4] 配置代理并拉取镜像..."
disable_github_hosts

export http_proxy="${PROXY}" https_proxy="${PROXY}" all_proxy="${PROXY}"
export HTTP_PROXY="${PROXY}" HTTPS_PROXY="${PROXY}" ALL_PROXY="${PROXY}"
echo "[proxy] 已设置代理: ${PROXY}"

000
docker pull "${IMAGE}:${TAG}"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
restore_hosts
echo "[1/4] 镜像拉取完成，代理已清除"

# --- 生成 compose 配置并启动 ---

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
      - ${DATASET_HOST_DIR}:/app/benchmark/datasets
      - ${DEPLOY_DIR}/.env:/app/.env:ro
      - ${DEPLOY_DIR}/models.yaml:/app/benchmark/configs/models.yaml:ro
    env_file:
      - ${DEPLOY_DIR}/.env
    environment:
      - PYTHONUNBUFFERED=1
EOF

echo "[3/4] 停止旧容器..."
docker compose -f "${DEPLOY_DIR}/docker-compose.prod.yml" down 2>/dev/null || true

 000
echo "[4/4] 启动新容器..."
docker compose -f "${DEPLOY_DIR}/docker-compose.prod.yml" up -d

echo ""
echo "=== 部署完成 ==="
echo "Web 界面: http://localhost:8501"
echo ""
echo "配置文件:"
echo "  .env:        ${DEPLOY_DIR}/.env"
echo "  models.yaml: ${DEPLOY_DIR}/models.yaml"
echo "  数据目录:    ${DEPLOY_DIR}/data/"
echo "  数据集缓存:  ${DATASET_HOST_DIR}/"
echo ""
echo "数据集缓存:"
echo "  首次运行会自动从 HuggingFace 下载数据集到 dataset/ 目录"
echo "  下载完成后创建标志: touch ${DATASET_HOST_DIR}/.download-complete"
echo "  创建标志后，后续运行将自动启用离线模式（HF_DATASETS_OFFLINE=1）"
echo ""
echo "修改配置后重启:"
echo "  docker compose -f ${DEPLOY_DIR}/docker-compose.prod.yml restart"
echo ""
echo "常用命令:"
echo "  日志:  docker compose -f ${DEPLOY_DIR}/docker-compose.prod.yml logs -f"
echo "  停止:  docker compose -f ${DEPLOY_DIR}/docker-compose.prod.yml down"
echo "  状态:  docker compose -f ${DEPLOY_DIR}/docker-compose.prod.yml ps"
