#!/usr/bin/env python3
"""Phase 1 性能基准测试."""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

from benchmark.core.llm_adapter import LLMEvalAdapter
from benchmark.core.tz import now
from benchmark.analysis.quality_signals import QualitySignalCollector


class PerformanceBenchmark:
    def __init__(self, model: str, samples: int = 10):
        self.model = model
        self.samples = samples
        self.results: dict[str, Any] = {}

    async def run_all_benchmarks(self) -> dict[str, Any]:
        print(f"\n{'=' * 60}")
        print(f"Phase 1 性能基准测试")
        print(f"Model: {self.model}")
        print(f"Samples: {self.samples}")
        print(f"Time: {now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")

        await self.benchmark_concurrent_execution()
        await self.benchmark_cache_performance()
        await self.benchmark_connection_pool()
        await self.benchmark_end_to_end()

        return self.results

    async def benchmark_concurrent_execution(self) -> None:
        print("[1/4] 测试并发执行性能...")

        async def mock_task(task_id: int) -> float:
            await asyncio.sleep(0.1)
            return task_id

        start = time.time()
        for i in range(5):
            await mock_task(i)
        serial_time = time.time() - start

        start = time.time()
        await asyncio.gather(*[mock_task(i) for i in range(5)])
        concurrent_time = time.time() - start

        speedup = serial_time / concurrent_time if concurrent_time > 0 else 0

        self.results["concurrent_execution"] = {
            "serial_time": round(serial_time, 3),
            "concurrent_time": round(concurrent_time, 3),
            "speedup": round(speedup, 2),
            "tasks": 5,
        }

        print(f"  串行执行: {serial_time:.3f}s")
        print(f"  并发执行: {concurrent_time:.3f}s")
        print(f"  加速比: {speedup:.2f}x")
        print()

    async def benchmark_cache_performance(self) -> None:
        print("[2/4] 测试缓存性能...")

        mock_repo = MockRepo()
        collector = QualitySignalCollector(mock_repo, self.model)  # type: ignore[arg-type]

        start = time.time()
        await collector._get_history_stats(
            query_key="test",
            filters={"dimension": "probe", "task_id": "task_1"},
            value_expr="LENGTH(model_output)",
        )
        first_query_time = time.time() - start

        start = time.time()
        await collector._get_history_stats(
            query_key="test",
            filters={"dimension": "probe", "task_id": "task_1"},
            value_expr="LENGTH(model_output)",
        )
        second_query_time = time.time() - start

        speedup = first_query_time / second_query_time if second_query_time > 0 else 0

        self.results["cache_performance"] = {
            "first_query_time_ms": round(first_query_time * 1000, 3),
            "cached_query_time_ms": round(second_query_time * 1000, 3),
            "speedup": round(speedup, 2),
            "cache_size": len(collector._cache),
        }

        print(f"  首次查询: {first_query_time * 1000:.3f}ms")
        print(f"  缓存查询: {second_query_time * 1000:.3f}ms")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  缓存条目数: {len(collector._cache)}")
        print()

    async def benchmark_connection_pool(self) -> None:
        print("[3/4] 测试连接池性能...")

        adapter = LLMEvalAdapter()

        start = time.time()
        client1 = adapter._get_client("zai")
        first_client_time = time.time() - start

        start = time.time()
        client2 = adapter._get_client("zai")
        reuse_time = time.time() - start

        is_same = client1 is client2

        await adapter.close()

        self.results["connection_pool"] = {
            "first_client_time_ms": round(first_client_time * 1000, 3),
            "reuse_time_ms": round(reuse_time * 1000, 3),
            "is_same_instance": is_same,
        }

        print(f"  首次创建: {first_client_time * 1000:.3f}ms")
        print(f"  复用获取: {reuse_time * 1000:.3f}ms")
        print(f"  同一实例: {is_same}")
        print()

    async def benchmark_end_to_end(self) -> None:
        print("[4/4] 测试端到端性能...")

        from benchmark.adapters.probe_adapter import ProbeAdapter

        adapter = ProbeAdapter()
        tasks = adapter.load()

        if not tasks:
            print("  警告: 没有加载到任务，跳过端到端测试")
            return

        tasks = tasks[: self.samples]

        print(f"  加载了 {len(tasks)} 个任务")

        llm = LLMEvalAdapter(model=self.model)

        start = time.time()
        success_count = 0
        fail_count = 0

        for task in tasks:
            try:
                await llm.agenerate(
                    prompt=task.prompt,
                    model=self.model,
                    temperature=0.0,
                    max_tokens=50,
                )
                success_count += 1
            except Exception as e:
                fail_count += 1
                print(f"  任务失败: {task.task_id}, 错误: {e}")

        total_time = time.time() - start

        await llm.close()

        self.results["end_to_end"] = {
            "total_time_sec": round(total_time, 2),
            "tasks_completed": success_count,
            "tasks_failed": fail_count,
            "avg_time_per_task_sec": round(total_time / len(tasks), 3) if tasks else 0,
            "throughput_tps": round(len(tasks) / total_time, 2)
            if total_time > 0
            else 0,
        }

        print(f"  总耗时: {total_time:.2f}s")
        print(f"  成功: {success_count}, 失败: {fail_count}")
        print(f"  平均每个任务: {total_time / len(tasks):.3f}s")
        print(f"  吞吐量: {len(tasks) / total_time:.2f} tasks/s")
        print()

    def generate_report(self) -> str:
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("性能基准测试报告")
        lines.append("=" * 60)
        lines.append(f"测试时间: {now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"测试模型: {self.model}")
        lines.append("")

        if "concurrent_execution" in self.results:
            r = self.results["concurrent_execution"]
            lines.append("[并发执行性能]")
            lines.append(f"  串行执行: {r['serial_time']}s")
            lines.append(f"  并发执行: {r['concurrent_time']}s")
            lines.append(f"  加速比: {r['speedup']}x")
            lines.append("")

        if "cache_performance" in self.results:
            r = self.results["cache_performance"]
            lines.append("[缓存性能]")
            lines.append(f"  首次查询: {r['first_query_time_ms']}ms")
            lines.append(f"  缓存查询: {r['cached_query_time_ms']}ms")
            lines.append(f"  加速比: {r['speedup']}x")
            lines.append(f"  缓存大小: {r['cache_size']}")
            lines.append("")

        if "connection_pool" in self.results:
            r = self.results["connection_pool"]
            lines.append("[连接池性能]")
            lines.append(f"  首次创建: {r['first_client_time_ms']}ms")
            lines.append(f"  复用获取: {r['reuse_time_ms']}ms")
            lines.append(f"  同一实例: {r['is_same_instance']}")
            lines.append("")

        if "end_to_end" in self.results:
            r = self.results["end_to_end"]
            lines.append("[端到端性能]")
            lines.append(f"  总耗时: {r['total_time_sec']}s")
            lines.append(f"  完成任务: {r['tasks_completed']}")
            lines.append(f"  失败任务: {r['tasks_failed']}")
            lines.append(f"  平均耗时: {r['avg_time_per_task_sec']}s/task")
            lines.append(f"  吞吐量: {r['throughput_tps']} tasks/s")
            lines.append("")

        lines.append("=" * 60)
        lines.append("测试完成")
        lines.append("=" * 60)

        return "\n".join(lines)


class MockRepo:
    """Minimal mock for QualitySignalCollector benchmark."""

    def get_runs(self, **kwargs):
        return [{"created_at": "2026-01-01T00:00:00", "run_id": "run-1"}]

    async def aget_results(self, **kwargs):
        return [{"task_id": "task_1", "model_output": "x" * 100}]


async def main():
    parser = argparse.ArgumentParser(description="Phase 1 性能基准测试")
    parser.add_argument("--model", default="zai/glm-4.7", help="测试模型")
    parser.add_argument("--samples", type=int, default=10, help="测试样本数")
    parser.add_argument("--output", default=None, help="输出文件路径")
    args = parser.parse_args()

    benchmark = PerformanceBenchmark(args.model, args.samples)
    await benchmark.run_all_benchmarks()

    report = benchmark.generate_report()
    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n报告已保存: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
