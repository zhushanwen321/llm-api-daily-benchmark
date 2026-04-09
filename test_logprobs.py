#!/usr/bin/env python3
"""测试 API 是否支持 logprobs 参数."""

import asyncio
import httpx
import json
from datetime import datetime

PROVIDER = "zai"
MODEL = "glm-4.7"
API_KEY = "f457c4c0cfc147178d15296acf28c502.VxsvosYr4aZ0Mg2u"
API_BASE = "https://open.bigmodel.cn/api/coding/paas/v4"

TEST_PROMPT = "What is 2 + 2? Answer with just the number."


async def test_logprobs():
    print("=" * 60)
    print("Logprobs API 支持测试")
    print("=" * 60)
    print(f"Provider: {PROVIDER}")
    print(f"Model: {MODEL}")
    print(f"API: {API_BASE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "temperature": 0.0,
        "max_tokens": 10,
        "logprobs": True,
        "top_logprobs": 3,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]

                    if "logprobs" in choice:
                        print("\n✅ API 支持 logprobs!")
                        print(f"\nLogprobs 数据:")
                        print(
                            json.dumps(
                                choice["logprobs"], indent=2, ensure_ascii=False
                            )[:500]
                        )
                        return True
                    else:
                        print("\n❌ API 不支持 logprobs")
                        print(f"响应字段: {list(choice.keys())}")
                        return False
            else:
                print(f"\n❌ 请求失败: {response.status_code}")
                print(f"响应: {response.text[:200]}")
                return False

        except Exception as e:
            print(f"\n❌ 异常: {e}")
            return False


if __name__ == "__main__":
    result = asyncio.run(test_logprobs())
    exit(0 if result else 1)
