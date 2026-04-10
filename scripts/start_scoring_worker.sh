#!/usr/bin/env bash
# 启动异步评分 Worker
# 用法: ./scripts/start_scoring_worker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 激活虚拟环境（如果存在）
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Starting Scoring Worker..."
echo "  SCORING_BACKEND_TYPE: ${SCORING_BACKEND_TYPE:-qwen_cli}"
echo "  SCORING_WORKER_POLL_INTERVAL: ${SCORING_WORKER_POLL_INTERVAL:-5}s"
echo "  SCORING_WORKER_BATCH_SIZE: ${SCORING_WORKER_BATCH_SIZE:-10}"
echo ""

exec python -m benchmark.workers.scoring_worker
