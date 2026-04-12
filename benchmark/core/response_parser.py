"""LLM 响应解析器.

将模型输出解析为 ParsedResponse（think 固定为空，answer 为最终答案）。
支持处理：JSON 格式、markdown code block、纯文本 fallback。
推理内容由 adapter 层分离（reasoning_content 字段）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# 匹配 ```json ... ``` 代码块
_JSON_BLOCK_RE = re.compile(r"```json\s*\n?(.*?)\n?\s*```", re.DOTALL)
# 匹配 ```python ... ``` 代码块（backend-dev fallback）
_PYTHON_BLOCK_RE = re.compile(r"```(?:python)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
# 匹配 \boxed{...}，支持嵌套花括号
_BOXED_RE = re.compile(r"\\boxed\s*\{")


@dataclass
class ParsedResponse:
    """解析后的模型响应.

    Attributes:
        think: 模型的思考/推理过程（可能为空）
        answer: 最终提交的答案内容（reasoning 为数字，backend-dev 为纯代码）
    """

    think: str
    answer: str


def extract_json_object(text: str) -> dict | None:
    """从文本中提取 JSON 对象.

    尝试三种策略：
    1. 整个文本直接 json.loads
    2. 从 ```json ... ``` code block 提取
    3. 从第一个 { 到最后一个 } 的子串
    """
    # 策略1：直接解析
    text_stripped = text.strip()
    if text_stripped.startswith("{"):
        try:
            return json.loads(text_stripped)
        except json.JSONDecodeError:
            pass

    # 策略2：从 ```json code block 提取
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 策略3：花括号匹配
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def extract_python_code(text: str) -> str | None:
    """从 markdown code block 中提取 Python 代码."""
    match = _PYTHON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_boxed(text: str) -> str:
    """从 LaTeX 文本中提取 \\boxed{...} 内的内容.

    处理嵌套花括号，如 \\boxed{\\frac{14}{3}}.
    处理双重嵌套，如 \\boxed{\\boxed{42}} -> 42.
    """
    match = _BOXED_RE.search(text)
    if not match:
        return ""

    # 从 \boxed{ 后面开始，逐字符平衡匹配花括号
    start = match.end()  # 指向 { 后面的第一个字符
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    if depth == 0:
        result = text[start : pos - 1].strip()
    else:
        result = text[start:].strip()

    # 处理双重嵌套: \boxed{\boxed{42}} -> 递归提取
    if result.startswith("\\boxed"):
        inner = extract_boxed(result)
        if inner:
            return inner

    return result


def _extract_answer_from_json(data: dict, dimension: str) -> str:
    """从 JSON 数据中按维度提取 answer 字段."""
    if dimension == "reasoning":
        # 优先取 answer 字段，fallback 到 reasoning 字段（某些模型可能用错字段名）
        return str(data.get("answer", data.get("result", "")))
    elif dimension in ("backend-dev", "frontend-dev"):
        # 优先取 code 字段
        return str(data.get("code", ""))
    elif dimension == "probe":
        if "answer" in data:
            return str(data["answer"])
        for key in ["result", "data", "status", "value", "items"]:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    return str(value)
                return str(value)
    return ""


def parse_response(raw: str, dimension: str) -> ParsedResponse:
    """解析模型 content（已由 adapter 分离 reasoning_content），提取最终答案.

    Args:
        raw: 模型的 content 部分（不含 reasoning_content）
        dimension: 评测维度（"reasoning" 或 "backend-dev"）

    Returns:
        ParsedResponse，think 固定为空（推理内容由 adapter 层分离）
    """
    if not raw:
        return ParsedResponse(think="", answer="")

    # Step 0: reasoning 维度先尝试 \boxed{} 提取（MATH 数据集）
    if dimension == "reasoning":
        boxed = extract_boxed(raw)
        if boxed:
            return ParsedResponse(think="", answer=boxed)

    # Step 1: 尝试 JSON 解析
    json_data = extract_json_object(raw)
    if json_data:
        answer = _extract_answer_from_json(json_data, dimension)
        return ParsedResponse(think="", answer=answer)

    # Step 2: JSON 解析失败的 fallback
    if dimension == "backend-dev":
        code = extract_python_code(raw)
        if code:
            return ParsedResponse(think="", answer=code)

    # 最终 fallback：原文整体作为 answer
    return ParsedResponse(think="", answer=raw)
