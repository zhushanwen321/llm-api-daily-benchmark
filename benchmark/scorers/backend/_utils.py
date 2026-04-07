"""Backend scorer 公共工具函数。"""

from __future__ import annotations

import ast
import json


def extract_code(model_answer: str) -> str:
    """从 model_answer 中提取代码。

    支持 JSON {"code": "..."} 格式。
    """
    try:
        data = json.loads(model_answer)
        if isinstance(data, dict) and "code" in data:
            return str(data["code"])
    except (json.JSONDecodeError, TypeError):
        pass
    return model_answer


def safe_parse_ast(code: str) -> ast.AST | None:
    """安全解析 AST，失败返回 None。"""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None
