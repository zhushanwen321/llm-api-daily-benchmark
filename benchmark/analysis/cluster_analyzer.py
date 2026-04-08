"""指纹聚类分析与模型分类器。

对同一模型的历史指纹做 DBSCAN 聚类，检测模型是否被替换。
跨模型训练 KNN 分类器，识别新指纹属于哪个已知模型。
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

from benchmark.analysis.models import ClusterInfo, ClusterReport

# DBSCAN 默认参数
_DEFAULT_EPS = 0.15
_DEFAULT_MIN_SAMPLES = 3

# KNN 默认参数
_DEFAULT_K = 3


class FingerprintClusterAnalyzer:
    """对模型指纹做 DBSCAN 聚类，检测身份变化。"""

    def __init__(self, fingerprint_dir: str = "fingerprint_db") -> None:
        self._dir = Path(fingerprint_dir)

    def analyze(
        self,
        model: str,
        eps: float = _DEFAULT_EPS,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
    ) -> ClusterReport:
        """对模型的所有指纹做 DBSCAN 聚类。

        Args:
            model: 模型名
            eps: DBSCAN eps 参数（余弦距离阈值）
            min_samples: DBSCAN 最小簇大小
        """
        history = self._load_fingerprints(model)
        if len(history) < min_samples:
            return ClusterReport(
                model=model,
                n_clusters=0,
                n_noise=0,
                summary=f"数据不足（{len(history)} 条，需要 >= {min_samples}）",
            )

        vectors = np.array([fp["vector"] for fp in history])
        timestamps = [fp["timestamp"] for fp in history]

        clustering = DBSCAN(
            metric="cosine", eps=eps, min_samples=min_samples,
        ).fit(vectors)
        labels = clustering.labels_

        unique_labels = sorted(set(labels) - {-1})
        noise_mask = labels == -1
        n_noise = int(noise_mask.sum())

        # 构建各簇信息
        clusters: list[ClusterInfo] = []
        for label_id in unique_labels:
            mask = labels == label_id
            cluster_vectors = vectors[mask]
            cluster_ts = [timestamps[i] for i in range(len(timestamps)) if mask[i]]

            centroid = cluster_vectors.mean(axis=0).tolist()

            # 归一化分数均值：前 20 维 * 100
            avg_score = float(np.mean(cluster_vectors[:, :20]) * 100)

            clusters.append(
                ClusterInfo(
                    cluster_id=int(label_id),
                    size=int(mask.sum()),
                    time_range=(min(cluster_ts), max(cluster_ts)),
                    centroid=centroid,
                    avg_score=round(avg_score, 2),
                )
            )

        # 检测模型变化：按时间排序，相邻指纹属于不同簇
        suspected_changes = self._detect_changes(labels, timestamps, vectors)

        summary = self._build_summary(len(unique_labels), n_noise, suspected_changes)

        return ClusterReport(
            model=model,
            n_clusters=len(unique_labels),
            n_noise=n_noise,
            clusters=clusters,
            suspected_changes=suspected_changes,
            summary=summary,
        )

    def _detect_changes(
        self,
        labels: np.ndarray,
        timestamps: list[str],
        vectors: np.ndarray,
    ) -> list[dict]:
        """检测时间序列中的聚类变化。"""
        changes: list[dict] = []
        for i in range(1, len(labels)):
            if labels[i] == labels[i - 1]:
                continue
            if labels[i] == -1 or labels[i - 1] == -1:
                continue
            cos_sim = float(
                np.dot(vectors[i], vectors[i - 1])
                / (
                    np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i - 1])
                    + 1e-10
                )
            )
            changes.append(
                {
                    "from_cluster": int(labels[i - 1]),
                    "to_cluster": int(labels[i]),
                    "at": timestamps[i],
                    "cosine_similarity": round(cos_sim, 6),
                }
            )
        return changes

    @staticmethod
    def _build_summary(
        n_clusters: int, n_noise: int, changes: list[dict]
    ) -> str:
        parts: list[str] = []
        if n_clusters <= 1 and n_noise == 0:
            return "Model identity consistent."
        if n_clusters > 1:
            parts.append(f"{n_clusters} distinct clusters detected")
        if n_noise > 0:
            parts.append(f"{n_noise} noise points")
        if changes:
            parts.append(f"{len(changes)} cluster transitions")
        return "; ".join(parts) + "."

    def _load_fingerprints(self, model: str) -> list[dict]:
        """从指纹库加载模型的所有指纹，按时间排序。"""
        from benchmark.analysis.fingerprint import FingerprintManager

        # 复用 FingerprintManager 的历史加载逻辑
        fm = FingerprintManager(str(self._dir))
        history = fm.get_fingerprint_history(model)
        # 排除 baseline（与第一次运行重复）
        return [
            fp
            for fp in history
            if fp.get("_source_file") != "baseline.json"
        ]


class ModelClassifier:
    """跨模型 KNN 分类器，识别指纹属于哪个已知模型。"""

    def __init__(self, fingerprint_dir: str = "fingerprint_db") -> None:
        self._dir = Path(fingerprint_dir)
        self._clf: KNeighborsClassifier | None = None
        self._model_names: list[str] = []

    def train(
        self,
        models: list[str] | None = None,
        k: int = _DEFAULT_K,
    ) -> dict:
        """从指纹库训练 KNN 分类器。

        Returns:
            训练报告：各模型样本数、总样本数
        """
        X, y, model_names = self._load_training_data(models)
        if len(X) == 0:
            return {"status": "no_data", "message": "No fingerprint data available"}

        self._clf = KNeighborsClassifier(
            n_neighbors=min(k, len(X) - 1), metric="cosine",
        )
        self._clf.fit(X, y)
        self._model_names = model_names

        per_model = {}
        for name in model_names:
            count = int((np.array(y) == name).sum())
            if count > 0:
                per_model[name] = count

        return {
            "status": "trained",
            "total_samples": len(X),
            "models": per_model,
            "k": min(k, len(X) - 1),
        }

    def predict(self, vector: list[float]) -> dict:
        """预测指纹属于哪个模型。

        Returns:
            {"predicted_model": str, "probabilities": dict, "confidence": float}
        """
        if self._clf is None:
            return {"status": "not_trained"}

        X = np.array([vector])
        predicted = self._clf.predict(X)[0]
        distances, indices = self._clf.kneighbors(X)

        # 基于距离计算伪概率
        raw_dists = distances[0]
        inv_dists = 1.0 / (raw_dists + 1e-10)
        total = inv_dists.sum()
        # classes_ 将内部数字标签映射回原始字符串标签
        classes = self._clf.classes_
        neighbor_labels = [str(classes[j]) for j in self._clf._y[indices[0]]]

        probs: dict[str, float] = {}
        for label, w in zip(neighbor_labels, inv_dists):
            label_str = str(label)
            probs[label_str] = probs.get(label_str, 0.0) + w
        for key in probs:
            probs[key] = round(probs[key] / total, 4)

        return {
            "predicted_model": str(predicted),
            "probabilities": probs,
            "confidence": round(probs.get(str(predicted), 0.0), 4),
        }

    def cross_validate(
        self, models: list[str] | None = None, k: int = _DEFAULT_K,
    ) -> dict:
        """留一法交叉验证。

        Returns:
            {"accuracy": float, "per_model": dict, "total_samples": int}
        """
        X, y, model_names = self._load_training_data(models)
        if len(X) < 2:
            return {"accuracy": 0.0, "total_samples": len(X), "per_model": {}}

        y_arr = np.array(y)
        n_neighbors = min(k, len(X) - 1)
        loo = LeaveOneOut()
        correct = 0
        per_model_correct: dict[str, int] = {}
        per_model_total: dict[str, int] = {}

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)[0]

            label = str(y_test[0])
            per_model_total[label] = per_model_total.get(label, 0) + 1
            if pred == y_test[0]:
                correct += 1
                per_model_correct[label] = per_model_correct.get(label, 0) + 1

        total_tests = sum(per_model_total.values())
        accuracy = correct / total_tests if total_tests > 0 else 0.0

        per_model_acc = {}
        for name in per_model_total:
            t = per_model_total[name]
            c = per_model_correct.get(name, 0)
            per_model_acc[name] = round(c / t, 4) if t > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "total_samples": len(X),
            "per_model": per_model_acc,
        }

    def _load_training_data(
        self, models: list[str] | None = None,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """从指纹库加载所有模型的指纹作为训练数据。"""
        from benchmark.analysis.fingerprint import FingerprintManager

        fm = FingerprintManager(str(self._dir))

        if models is None:
            # 扫描所有模型目录
            if not self._dir.exists():
                return np.array([]), [], []
            model_dirs = [
                d for d in self._dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        else:
            model_dirs = []
            for m in models:
                from benchmark.analysis.fingerprint import _sanitize_model
                p = self._dir / _sanitize_model(m)
                if p.exists():
                    model_dirs.append(p)

        all_vectors: list[list[float]] = []
        all_labels: list[str] = []
        model_names: list[str] = []

        for model_dir in sorted(model_dirs):
            history = fm.get_fingerprint_history(model_dir.name.replace("__", "/"))
            if len(history) < 2:
                continue

            # 排除 baseline
            fps = [
                fp for fp in history
                if fp.get("_source_file") != "baseline.json"
            ]
            if not fps:
                continue

            model_name = fps[0]["model"]
            model_names.append(model_name)
            for fp in fps:
                all_vectors.append(fp["vector"])
                all_labels.append(model_name)

        if not all_vectors:
            return np.array([]), [], []

        return np.array(all_vectors), all_labels, model_names
