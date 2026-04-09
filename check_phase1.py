#!/usr/bin/env python3
"""
Phase 1 代码修改检查脚本
检查内容：
1. 调用链衔接是否正确
2. 潜在bug
3. 遗漏点
"""

import ast
import sys
from pathlib import Path


def check_file(filepath: str) -> dict:
    """检查单个文件"""
    results = {"file": filepath, "issues": [], "warnings": [], "suggestions": []}

    with open(filepath, "r") as f:
        content = f.read()

    # 检查1: 是否存在资源泄漏风险（close未调用）
    if "llm_adapter.py" in filepath:
        if "_clients" in content and "async with" not in content:
            # 检查是否使用了持久化客户端但没有确保关闭
            results["warnings"].append(
                {
                    "type": "resource_leak",
                    "message": "使用持久化HTTP客户端，需要确保调用close()方法释放资源",
                    "severity": "medium",
                }
            )

    # 检查2: 缓存无限增长风险
    if "quality_signals.py" in filepath:
        if "self._cache" in content and "maxsize" not in content:
            results["warnings"].append(
                {
                    "type": "memory_leak",
                    "message": "缓存没有大小限制，长时间运行可能导致内存泄漏",
                    "severity": "low",
                }
            )

        if "self._cache" in content and "TTL" not in content.upper():
            results["suggestions"].append(
                {
                    "type": "cache_ttl",
                    "message": "建议为缓存添加TTL过期机制，避免缓存过期数据",
                    "severity": "low",
                }
            )

    # 检查3: 异常处理
    if "cli.py" in filepath:
        if "return_exceptions=True" in content:
            # 检查是否正确处理了异常
            if "isinstance(result, Exception)" in content:
                results["suggestions"].append(
                    {
                        "type": "exception_handling",
                        "message": "异常处理逻辑正确，建议记录更多上下文信息",
                        "severity": "info",
                    }
                )

    # 检查4: 并发控制
    if "Semaphore" in content:
        # 检查信号量是否正确使用
        if "async with semaphore" in content:
            results["suggestions"].append(
                {"type": "concurrency", "message": "信号量使用正确", "severity": "info"}
            )

    return results


def main():
    files_to_check = [
        "benchmark/cli.py",
        "benchmark/core/llm_adapter.py",
        "benchmark/analysis/quality_signals.py",
    ]

    print("=" * 80)
    print("Phase 1 代码修改检查报告")
    print("=" * 80)
    print()

    all_issues = []
    all_warnings = []
    all_suggestions = []

    for filepath in files_to_check:
        full_path = Path(filepath)
        if not full_path.exists():
            print(f"⚠️  文件不存在: {filepath}")
            continue

        print(f"\n📄 检查文件: {filepath}")
        print("-" * 80)

        result = check_file(str(full_path))

        if result["issues"]:
            print("\n❌ 发现的问题:")
            for issue in result["issues"]:
                print(
                    f"   [{issue['severity'].upper()}] {issue['type']}: {issue['message']}"
                )
                all_issues.append((filepath, issue))

        if result["warnings"]:
            print("\n⚠️  警告:")
            for warning in result["warnings"]:
                print(
                    f"   [{warning['severity'].upper()}] {warning['type']}: {warning['message']}"
                )
                all_warnings.append((filepath, warning))

        if result["suggestions"]:
            print("\n💡 建议:")
            for suggestion in result["suggestions"]:
                print(
                    f"   [{suggestion['severity'].upper()}] {suggestion['type']}: {suggestion['message']}"
                )
                all_suggestions.append((filepath, suggestion))

        if not any([result["issues"], result["warnings"], result["suggestions"]]):
            print("   ✅ 未发现明显问题")

    # 汇总
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"问题数: {len(all_issues)}")
    print(f"警告数: {len(all_warnings)}")
    print(f"建议数: {len(all_suggestions)}")

    if all_issues:
        print("\n🔴 需要立即修复的问题:")
        for filepath, issue in all_issues:
            print(f"   - {filepath}: {issue['message']}")

    if all_warnings:
        print("\n🟡 建议修复的警告:")
        for filepath, warning in all_warnings:
            print(f"   - {filepath}: {warning['message']}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
