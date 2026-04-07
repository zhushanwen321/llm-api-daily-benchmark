# scripts/generate_frontcode_tasks.py
"""用 LLM 批量生成 FrontCode 题目，辅助人工审核。"""

from __future__ import annotations

import asyncio
import json
import re
import sys

from benchmark.core.llm_adapter import LLMEvalAdapter

_BATCHES = [
    {"type": "html", "count": 2, "difficulty": "easy", "description": "语义化 HTML 结构和表单"},
    {"type": "css", "count": 3, "difficulty": "medium", "description": "响应式布局、Grid、动画、Flexbox"},
    {"type": "javascript", "count": 3, "difficulty": "medium", "description": "debounce、throttle、策略模式、日期处理"},
    {"type": "react", "count": 3, "difficulty": "medium", "description": "组件、Hooks、状态管理、表单"},
    {"type": "typescript", "count": 2, "difficulty": "medium", "description": "类型定义、泛型、接口"},
    {"type": "accessibility", "count": 2, "difficulty": "hard", "description": "ARIA、键盘导航、屏幕阅读器"},
    {"type": "complex", "count": 2, "difficulty": "hard", "description": "完整页面，多技术栈组合"},
]

_SYSTEM_PROMPT = """你是一个前端评测题目出题专家。你需要为 LLM Benchmark 生成前端编码题目。
题目要求模型输出一段完整的前端代码。你需要:
1. 用中文描述题目要求
2. 列出代码中必须包含的关键词（用于关键词匹配评分）
3. 列出 3-5 个 DOM/逻辑断言（用于自动化功能测试）
4. 指定题目难度"""

_TASK_TEMPLATE = """请为前端评测生成 {count} 道「{type}」类型的题目。

要求:
- 难度: {difficulty}
- 覆盖范围: {description}

每道题返回以下 JSON 格式（不要包含其他文字）:
```json
[
  {{
    "id": "frontcode_{type}_{序号}",
    "type": "{type}",
    "prompt": "中文题目描述...",
    "keywords": ["keyword1", "keyword2", ...],
    "test_cases": ["断言1", "断言2", ...],
    "difficulty": "{difficulty}"
  }}
]
```

注意:
- prompt 必须用中文
- keywords 是代码中必须包含的技术关键词
- test_cases 是用自然语言描述的功能断言，后续会用于自动化评分
- 每道题的 id 序号从当前最大序号+1开始"""


async def generate_batch(llm: LLMEvalAdapter, batch: dict, existing_count: int) -> list[dict]:
    prompt = _TASK_TEMPLATE.format(**batch)
    resp = await llm.agenerate(
        prompt=prompt,
        model="zai/glm-5.1",
        temperature=0.7,
        system_message=_SYSTEM_PROMPT,
    )
    json_match = re.search(r'\[.*\]', resp.content, re.DOTALL)
    if not json_match:
        print(f"  [WARN] {batch['type']}: 无法从响应中提取 JSON")
        return []
    try:
        tasks = json.loads(json_match.group())
        for i, task in enumerate(tasks):
            task["id"] = f"frontcode_{batch['type']}_{existing_count + i + 1}"
        return tasks
    except json.JSONDecodeError as e:
        print(f"  [WARN] {batch['type']}: JSON 解析失败: {e}")
        return []


async def main() -> None:
    tasks_file = "benchmark/datasets/frontcode/tasks.json"
    with open(tasks_file) as f:
        data = json.load(f)
    existing_tasks = data.get("tasks", [])
    print(f"现有题目数: {len(existing_tasks)}")

    llm = LLMEvalAdapter(model="zai/glm-5.1")
    all_new_tasks = []
    offset = len(existing_tasks)

    for batch in _BATCHES:
        print(f"\n生成 {batch['type']} ({batch['count']} 题)...")
        new_tasks = await generate_batch(llm, batch, offset)
        all_new_tasks.extend(new_tasks)
        offset += len(new_tasks)
        print(f"  成功生成: {len(new_tasks)} 题")

    merged = existing_tasks + all_new_tasks
    print(f"\n总计: {len(merged)} 题")

    output_path = tasks_file.replace(".json", "_generated.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"tasks": merged}, f, ensure_ascii=False, indent=2)
    print(f"\n生成结果已写入: {output_path}")
    print("请人工审核后替换 tasks.json")


if __name__ == "__main__":
    asyncio.run(main())
