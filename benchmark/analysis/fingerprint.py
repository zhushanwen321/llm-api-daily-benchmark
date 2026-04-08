"""模型指纹管理器。

生成、存储、对比模型行为指纹（33 维向量），
用于检测模型是否发生隐式变更（如降级、替换等）。

指纹结构：
  - 20 维：每题归一化分数 (0-1)
  - 13 维：聚合质量信号（取 20 题均值）
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# 13 个需要聚合的质量信号键名，顺序与任务规格一致
_AGGREGATE_SIGNAL_KEYS: tuple[str, ...] = (
    "format_compliance",
    "repetition_ratio",
    "garbled_text_ratio",
    "refusal_detected",
    "language_consistency",
    "output_length_zscore",
    "thinking_ratio",
    "empty_reasoning",
    "truncated",
    "token_efficiency_zscore",
    "tps_zscore",
    "ttft_zscore",
    "answer_entropy",
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _sanitize_model(model: str) -> str:
    """将模型名中的 / 替换为 __，用于安全目录名。"""
    return model.replace("/", "__")


class FingerprintManager:
    """模型指纹管理器。生成、存储、对比模型行为指纹。"""

    def __init__(self, fingerprint_dir: str = "fingerprint_db") -> None:
        self._dir = Path(fingerprint_dir)

    # ── 公共 API ──

    async def generate_fingerprint(
        self,
        model: str,
        scores: list[float],
        quality_signals: list[dict],
    ) -> dict:
        """生成指纹向量（33 维）并保存。"""
        return self.generate_fingerprint_sync(model, scores, quality_signals)

    def generate_fingerprint_sync(
        self,
        model: str,
        scores: list[float],
        quality_signals: list[dict],
    ) -> dict:
        """同步版本的指纹生成。"""
        vector = self._build_vector(scores, quality_signals)
        fingerprint = {
            "model": model,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "run_id": "",
            "vector": vector,
        }

        model_dir = self._dir / _sanitize_model(model)
        model_dir.mkdir(parents=True, exist_ok=True)

        # 写入时间戳文件
        ts_file = model_dir / f"{fingerprint['timestamp']}.json"
        ts_file.write_text(json.dumps(fingerprint, ensure_ascii=False), encoding="utf-8")

        # 如果基线不存在则创建
        baseline_file = model_dir / "baseline.json"
        if not baseline_file.exists():
            baseline_file.write_text(
                json.dumps(fingerprint, ensure_ascii=False), encoding="utf-8"
            )

        return fingerprint

    def compare_with_baseline(
        self,
        model: str,
        fingerprint: dict | None = None,
        threshold: float = 0.85,
    ) -> dict:
        """对比指纹与基线的相似度。"""
        model_dir = self._dir / _sanitize_model(model)
        baseline_file = model_dir / "baseline.json"

        if not baseline_file.exists():
            return {
                "model": model,
                "similarity": 0.0,
                "status": "no_baseline",
                "baseline_timestamp": None,
                "current_timestamp": None,
            }

        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))

        # 如果没有传入 fingerprint，读取最新的非基线文件
        if fingerprint is None:
            fingerprint = self._load_latest_fingerprint(model_dir)
            if fingerprint is None:
                # 只有基线，没有其他指纹
                return {
                    "model": model,
                    "similarity": 1.0,
                    "status": "match",
                    "baseline_timestamp": baseline.get("timestamp"),
                    "current_timestamp": baseline.get("timestamp"),
                }

        similarity = _cosine_similarity(baseline["vector"], fingerprint["vector"])
        status: Literal["match", "suspected_model_change"] = (
            "suspected_model_change" if similarity < threshold else "match"
        )

        return {
            "model": model,
            "similarity": round(similarity, 6),
            "status": status,
            "baseline_timestamp": baseline.get("timestamp"),
            "current_timestamp": fingerprint.get("timestamp"),
        }

    def get_fingerprint_history(
        self,
        model: str,
    ) -> list[dict]:
        """获取模型的所有历史指纹，按时间升序排列。"""
        model_dir = self._dir / _sanitize_model(model)
        if not model_dir.exists():
            return []

        results: list[dict] = []
        for fp_file in sorted(model_dir.glob("*.json")):
            try:
                data = json.loads(fp_file.read_text(encoding="utf-8"))
                data["_source_file"] = fp_file.name
                results.append(data)
            except (json.JSONDecodeError, OSError):
                continue

        return results

    # ── 内部方法 ──

    def _build_vector(
        self, scores: list[float], quality_signals: list[dict]
    ) -> list[float]:
        """构建 33 维指纹向量。

        前 20 维 = 归一化分数（除以 100），
        后 13 维 = 聚合质量信号（取 20 题均值）。
        """
        # 归一化分数
        norm_scores = [s / 100.0 for s in scores]

        # 聚合质量信号
        agg = []
        for key in _AGGREGATE_SIGNAL_KEYS:
            values = [qs.get(key, 0.0) for qs in quality_signals]
            agg.append(sum(values) / len(values) if values else 0.0)

        return norm_scores + agg

    def _load_latest_fingerprint(self, model_dir: Path) -> dict | None:
        """加载指定模型目录下最新的非基线指纹文件。"""
        candidates = [
            f
            for f in model_dir.glob("*.json")
            if f.name != "baseline.json"
        ]
        if not candidates:
            return None

        # 按文件名（即时间戳）降序排列，取最新
        latest = max(candidates, key=lambda f: f.name)
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
