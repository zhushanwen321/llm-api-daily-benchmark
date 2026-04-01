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

_SCHEMAS = {
    "reasoning": _REASONING_SCHEMA,
    "backend-dev": _BACKEND_DEV_SCHEMA,
}


def build_structured_prompt(task_prompt: str, dimension: str) -> str:
    """在原始任务 prompt 后追加 JSON 格式要求.

    Args:
        task_prompt: 原始任务描述
        dimension: 评测维度（"reasoning" 或 "backend-dev"）

    Returns:
        带有 JSON 格式要求的完整 prompt
    """
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
