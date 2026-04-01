# Quick Start

## 环境准备

要求 Python >= 3.11。

```bash
# 克隆仓库
git clone <repo-url>
cd llm-api-daily-benchmark

# 创建虚拟环境并安装依赖（推荐 uv）
uv sync

# 或使用标准方式
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

## 配置 API Key

```bash
# 从模板创建模型配置
cp benchmark/configs/models.yaml.example benchmark/configs/models.yaml

# 编辑 models.yaml，填入真实 API key
vim benchmark/configs/models.yaml
```

`models.yaml` 格式（provider -> model 两层结构）：

```yaml
providers:
  glm:
    api_key: "你的API Key"
    api_base: "https://open.bigmodel.cn/api/paas/v4/"
    models:
      glm-4.7:
        max_tokens: 4096
      glm-4-flash: {}              # max_tokens 可省略，默认 4096

  openai:
    api_key: "你的OpenAI Key"
    api_base: "https://api.openai.com/v1/"
    models:
      gpt-4:
        max_tokens: 4096
```

- `api_key` 和 `api_base` 配置在 provider 层级，同一 provider 下的模型共享
- `max_tokens` 配置在 model 层级，可省略（默认 4096）
- `models.yaml` 已在 `.gitignore` 中，不会被提交

## 运行评测

如果使用 `uv`，请在命令前加上 `uv run`：

```bash
# 使用 uv（推荐）
uv run python -m benchmark <command>

# 或激活虚拟环境后使用
source .venv/bin/activate
python -m benchmark <command>
```

### 1. 查看可用数据集

```bash
uv run python -m benchmark list-datasets
```

### 2. 运行评测

```bash
# 数学推理（GSM8K 最难5题）
uv run python -m benchmark evaluate --model glm-4.7 --dimension reasoning

# 代码生成（BigCodeBench-Hard 随机5题）
uv run python -m benchmark evaluate --model glm-4.7 --dimension backend-dev

# 指定题目数量
uv run python -m benchmark evaluate --model glm-4.7 --dimension reasoning --samples 3
```

评测完成后结果自动保存到 `benchmark/data/results.db`（SQLite）。

### 3. 导出结果

```bash
# 导出为 JSON
uv run python -m benchmark export --format json --output results.json

# 导出为 CSV，按模型过滤
uv run python -m benchmark export --format csv --output results.csv --model glm-4.7
```

## 可视化

```bash
uv run streamlit run benchmark/visualization/app.py
```

浏览器打开后可按模型/维度过滤查看结果，点击查看单题详情。

## 评测维度

| 维度 | 数据集 | 题目选取 | 评分方式 |
|------|--------|----------|----------|
| `reasoning` | GSM8K | 解答步骤最多的 5 题 | 数值精确匹配 |
| `backend-dev` | BigCodeBench-Hard | 随机 5 题（种子固定可复现） | 代码执行 + 测试用例 |

## 自定义模型

在 `benchmark/configs/models.yaml` 中添加新模型即可，要求 API 兼容 OpenAI `/chat/completions` 接口：

```yaml
providers:
  my-provider:
    api_key: "sk-xxx"
    api_base: "https://your-api-endpoint/v1/"
    models:
      my-model:
        max_tokens: 4096
      my-model-lite: {}    # 省略 max_tokens，使用默认值
```

然后运行：

```bash
uv run python -m benchmark evaluate --model my-model --dimension reasoning
```

## 项目结构

```
benchmark/
├── cli.py                     # CLI 命令入口
├── config.py                  # 配置加载
├── adapters/                  # 数据集适配器
│   ├── gsm8k_adapter.py       # GSM8K（数学推理）
│   └── bigcodebench_adapter.py # BigCodeBench（代码生成）
├── scorers/                   # 评分器
│   ├── exact_match_scorer.py  # 数值精确匹配
│   └── execution_scorer.py    # 代码执行验证
├── core/
│   └── llm_adapter.py         # LLM API 调用（OpenAI 兼容）
├── models/
│   ├── schemas.py             # 数据模型（Pydantic）
│   └── database.py            # SQLite 存储
├── visualization/
│   └── app.py                 # Streamlit 界面
└── configs/
    ├── default.yaml           # 默认配置
    └── models.yaml.example    # 模型配置模板
```
