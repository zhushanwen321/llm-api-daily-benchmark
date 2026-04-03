"""结构化 Prompt 构建器.

在各维度的任务 prompt 后追加 JSON 格式要求，引导模型返回结构化输出。
"""

from __future__ import annotations

import json

_REASONING_SCHEMA = {
    "example": {"answer": "42"},
    "fields": {
        "answer": "（字符串）最终数值答案，纯数字，不含单位或其他文字",
    },
}

_BACKEND_DEV_SCHEMA = {
    "example": {
        "code": "import ...\n\ndef task_func(...):\n    ...",
    },
    "fields": {
        "code": "（字符串）完整的、可直接执行的 Python 代码，不包含 markdown 标记",
    },
}

_FRONTEND_DEV_SCHEMA = {
    "example": {
        "code": "<html>\n<!-- or CSS/JS/React code -->\n...",
    },
    "fields": {
        "code": "（字符串）完整的前端代码（HTML/CSS/JavaScript/React），不包含 markdown 标记",
    },
}

_MATH_SCHEMA = {
    "instruction": (
        "请先展示解题过程，然后将最终答案放在 \\boxed{} 中。\n"
        "例如：如果答案是 42，请写 \\boxed{42}；如果答案是 3/5，请写 \\boxed{\\frac{3}{5}}。\n"
        "不要使用 JSON 格式回答。"
    ),
}

_SCHEMAS = {
    "reasoning": _REASONING_SCHEMA,
    "backend-dev": _BACKEND_DEV_SCHEMA,
    "frontend-dev": _FRONTEND_DEV_SCHEMA,
}


def build_structured_prompt(task_prompt: str, dimension: str, dataset: str = "") -> str:
    """在原始任务 prompt 后追加 JSON 格式要求.

    Args:
        task_prompt: 原始任务描述
        dimension: 评测维度（"reasoning" 或 "backend-dev"）
        dataset: 数据集名称（"math" 时使用 \boxed{} 格式而非 JSON）

    Returns:
        带有 JSON 格式要求的完整 prompt
    """
    # MATH 数据集：追加 \boxed{} 指令
    if dataset == "math":
        return f"{task_prompt}\n\n---\n{_MATH_SCHEMA['instruction']}"

    schema = _SCHEMAS.get(dimension)
    if not schema:
        return task_prompt

    example = json.dumps(schema["example"], ensure_ascii=False, indent=2)

    parts = [
        task_prompt,
        "",
        "---",
        "请严格按照以下 JSON 格式返回结果，不要在 JSON 之外包含任何其他文本：",
        "```json",
        example,
        "```",
        "",
        "字段说明：",
    ]
    for field, desc in schema["fields"].items():
        parts.append(f"- {field}: {desc}")

    parts.append("")
    parts.append("注意：只返回这个 JSON 对象，不要包含任何其他内容。")
    return "\n".join(parts)
