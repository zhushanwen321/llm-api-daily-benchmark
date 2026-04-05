# 主计划：镜像精简 + 流量控制优化

日期：2026-04-05
规格文档：[2026-04-05-image-and-flow-optimization-design.md](../specs/2026-04-05-image-and-flow-optimization-design.md)

## 任务总览

| Task | 模块 | 描述 | 详细计划 |
|------|------|------|----------|
| 1-4 | 流量控制 | AsyncConcurrencyLimiter + config + llm_adapter + 删除 rate_limiter | [module1-flow-control.md](module1-flow-control.md) |
| 5-6 | 统计模块 | 纯 Python t.ppf + 纯 Python bootstrap/t-test | [module2-statistics.md](module2-statistics.md) |
| 7-8 | 数据集加载 | hf_loader.py + 5 个 adapter 改造 | [module3-hf-loader.md](module3-hf-loader.md) |
| 9-10 | 移除依赖 | 移除 sympy + 移除 pandas | [module4-sympy-pandas.md](module4-sympy-pandas.md) |
| 11 | 收尾 | pyproject.toml + Dockerfile + models.yaml.example | 见下方 |

## 执行顺序

```
Task 1 ──→ Task 2 ──→ Task 3 ──→ Task 4
Task 5 ──→ Task 6
Task 7 ──→ Task 8
Task 9
Task 10
                     ┌──────────────────────────────────────┐
                     │ Task 1-10 全部完成后 ──→ Task 11     │
                     └──────────────────────────────────────┘
```

- Task 1→2→3→4 串行（每步依赖上一步的模块）
- Task 5→6 串行（Task 6 依赖 Task 5 的 `_t_ppf`）
- Task 7→8 串行（Task 8 依赖 Task 7 的 `hf_loader`）
- Task 9、10 互不依赖，可并行
- Task 11 必须在所有其他 Task 完成后执行

---

## Task 11: 收尾 — pyproject.toml + Dockerfile + models.yaml.example

**前置条件:** Task 1-10 全部完成并通过测试。

**Files:**
- Modify: `pyproject.toml`
- Modify: `Dockerfile`
- Modify: `benchmark/configs/models.yaml.example`

---

- [ ] **Step 1: 更新 pyproject.toml 依赖列表**

从 `dependencies` 中删除 4 个已不再使用的包：

```toml
# 删除以下行：
"datasets>=2.14",
"pandas>=2.0",
"scipy>=1.11",
"sympy>=1.12",
```

最终 `dependencies` 应为：

```toml
dependencies = [
    "pydantic>=2.0",
    "click>=8.0",
    "rich>=13.0",
    "streamlit>=1.28",
    "pyyaml>=6.0",
    "requests>=2.31",
    "httpx>=0.27",
    "python-dotenv>=1.0",
    "matplotlib>=3.7",
    "jinja2>=3.1",
    "apscheduler>=3.10",
]
```

同时在 `dev` 依赖中添加 `requests-mock`（Task 7 的测试需要）：

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "requests-mock>=1.11"]
```

> 注意：如果 Task 7 的 Step 4 已添加 `requests-mock`，此处仅确认即可。

- [ ] **Step 2: 更新 Dockerfile — 添加 pydeck 清理**

在运行阶段的 `pip uninstall` 之后添加 pydeck 清理：

```dockerfile
# 卸载 pip 减小体积 + 清理未使用的 pydeck
RUN pip uninstall -y pip \
    && rm -rf /usr/local/lib/python3.13/ensurepip \
    && rm -rf /usr/local/lib/python3.13/site-packages/pydeck*
```

- [ ] **Step 3: 更新 models.yaml.example**

将 `rate_limit` 改为 `max_concurrency`，更新注释：

```yaml
# 模型配置模板
# 复制为 models.yaml 并填入真实 API key
# models.yaml 已加入 .gitignore，不会被提交
#
# 模型标识格式: provider/model（如 glm/glm-4.7）
# max_concurrency: 可选，同一 provider 最大并发流数，不配置则不限流

providers:
  glm:
    api_key: "YOUR_GLM_API_KEY_HERE"
    api_base: "https://open.bigmodel.cn/api/paas/v4/"
    max_concurrency: 2
    models:
      glm-4.7:
        max_tokens: 4096
      glm-4-flash: {}

  openai:
    api_key: "YOUR_OPENAI_API_KEY_HERE"
    api_base: "https://api.openai.com/v1/"
    models:
      gpt-4:
        max_tokens: 4096
      gpt-4o: {}
```

- [ ] **Step 4: 提交**

```
git add pyproject.toml Dockerfile benchmark/configs/models.yaml.example
git commit -m "chore: 移除已废弃依赖 + Dockerfile 清理 pydeck + 配置迁移 max_concurrency"
```

---

## 自检清单

Task 1-11 全部完成后，执行以下检查：

### 1. 依赖清理验证

```bash
# 确认源码中不再引用已移除的库
grep -r "from datasets import\|import datasets" --include="*.py" benchmark/ tests/
grep -r "from scipy\|import scipy" --include="*.py" benchmark/ tests/
grep -r "from sympy\|import sympy" --include="*.py" benchmark/ tests/
grep -r "from pandas import\|import pandas" --include="*.py" benchmark/ tests/
grep -r "import numpy\|from numpy" --include="*.py" benchmark/ tests/
grep -r "rate_limiter\|TokenBucketRateLimiter" --include="*.py" benchmark/ tests/
```

Expected: 全部无输出。

### 2. 全量测试

```bash
cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest -v
```

Expected: 全部通过，无 skip、无 warning。

### 3. 运行时 import 检查

```bash
python -c "
import sys
# 验证核心模块可正常导入
from benchmark.core.concurrency import AsyncConcurrencyLimiter
from benchmark.core.statistics import calculate_confidence_interval
from benchmark.core.advanced_statistics import bootstrap_confidence_interval
from benchmark.adapters.hf_loader import load_hf_dataset
# 验证已移除的库未被加载
for mod in ['scipy', 'numpy', 'sympy', 'pandas', 'datasets']:
    assert mod not in sys.modules, f'{mod} should not be imported'
print('OK')
"
```

Expected: 输出 `OK`。

### 4. Docker 镜像体积（可选，需要 Docker 环境）

```bash
docker build -t benchmark:check .
docker images benchmark:check --format "{{.Size}}"
```

Expected: ~370 MB（对比原 953 MB，降幅约 61%）。

### 5. 功能冒烟测试

```bash
# 验证 CLI 可用
python -m benchmark.cli --help

# 验证 adapter 能正常加载数据（需网络或本地缓存）
python -c "from benchmark.adapters.math_adapter import MathAdapter; print('MathAdapter OK')"
python -c "from benchmark.adapters.gsm8k_adapter import GSM8KAdapter; print('GSM8KAdapter OK')"
python -c "from benchmark.adapters.mmlu_adapter import MMLUAdapter; print('MMLUAdapter OK')"
python -c "from benchmark.adapters.mmlu_pro_adapter import MMLUProAdapter; print('MMLUProAdapter OK')"
python -c "from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter; print('BigCodeBenchAdapter OK')"
```

Expected: 全部输出 `OK`。

---

## Git 提交历史（预期）

```
feat(core): 新增 AsyncConcurrencyLimiter 并发控制器              # Task 1
refactor(config): rate_limit 迁移为 max_concurrency              # Task 2
refactor(llm_adapter): 用并发控制器替换令牌桶限流器               # Task 3
chore: 删除已废弃的令牌桶限流器模块及测试                         # Task 4
refactor(statistics): 用纯 Python Cornish-Fisher 展开替换 scipy   # Task 5
refactor(advanced_statistics): 用纯 Python 替换 numpy/scipy 依赖  # Task 6
feat(adapters): 新增 hf_loader，用 HF API 替代 datasets 库       # Task 7
refactor(adapters): 所有 adapter 改用 hf_loader 加载数据          # Task 8
refactor(math_scorer): 移除 sympy fallback                       # Task 9
refactor(app): 移除 pandas，改用 sqlite3 原生查询 + list[dict]    # Task 10
chore: 移除已废弃依赖 + Dockerfile 清理 + 配置迁移                # Task 11
```
