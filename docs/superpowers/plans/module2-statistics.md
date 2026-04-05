# 模块 2：纯 Python 统计（Task 5-6）

### Task 5: 纯 Python t 分布 ppf 替换 scipy

**Files:**
- Modify: `benchmark/core/statistics.py`
- Modify: `tests/test_statistics.py`

---

- [ ] **Step 1: 为 `_t_ppf` 添加单元测试**

在 `tests/test_statistics.py` 末尾追加：

```python
import math
from benchmark.core.statistics import _t_ppf

def test_t_ppf_known_values():
    """验证 _t_ppf 在已知小 df 下的基本正确性."""
    val = _t_ppf(0.975, df=1)
    assert abs(val - 12.706) < 1.0, f"df=1 p=0.975: got {val}, expected ~12.706"

    val = _t_ppf(0.975, df=5)
    assert abs(val - 2.571) < 0.05, f"df=5 p=0.975: got {val}, expected ~2.571"

    val = _t_ppf(0.95, df=10)
    assert abs(val - 1.812) < 0.05, f"df=10 p=0.95: got {val}, expected ~1.812"

def test_t_ppf_symmetry():
    """t 分布关于 0 对称: ppf(p) = -ppf(1-p)."""
    for df in [2, 5, 15, 30]:
        for p in [0.8, 0.9, 0.95, 0.975, 0.99]:
            pos = _t_ppf(p, df=df)
            neg = _t_ppf(1 - p, df=df)
            assert abs(pos + neg) < 1e-10, f"symmetry broken: df={df}, p={p}"

def test_t_ppf_monotonic():
    """ppf 应随 p 单调递增."""
    df = 20
    prev = _t_ppf(0.5, df=df)
    for p in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
        cur = _t_ppf(p, df=df)
        assert cur > prev, f"not monotonic: p={p}, prev={prev}, cur={cur}"
        prev = cur

def test_t_ppf_accuracy_df_ge_15():
    """df >= 15 时误差 < 0.001。"""
    scipy_reference = {
        (0.95, 15): 1.753050, (0.975, 15): 2.131450, (0.99, 15): 2.602480,
        (0.95, 20): 1.724718, (0.975, 20): 2.085963, (0.99, 20): 2.527977,
        (0.95, 30): 1.697261, (0.975, 30): 2.042272, (0.99, 30): 2.457262,
        (0.95, 50): 1.675905, (0.975, 50): 2.008559, (0.99, 50): 2.403272,
        (0.95, 100): 1.660234, (0.975, 100): 1.983972, (0.99, 100): 2.364217,
    }
    for (p, df), expected in scipy_reference.items():
        got = _t_ppf(p, df=df)
        assert abs(got - expected) < 0.001, f"df={df}, p={p}: got {got}, expected {expected}"

def test_t_ppf_zero():
    """ppf(0.5) 应接近 0."""
    for df in [1, 2, 5, 10, 30, 100]:
        val = _t_ppf(0.5, df=df)
        assert abs(val) < 1e-10, f"ppf(0.5, df={df}) = {val}"
```

- [ ] **Step 2: 验证测试失败（`_t_ppf` 尚不存在）**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_statistics.py -k "t_ppf" -v`
Expected: `ImportError: cannot import name '_t_ppf'`

- [ ] **Step 3: 实现纯 Python `_t_ppf` 并更新 `calculate_confidence_interval`**

将 `benchmark/core/statistics.py` 整体替换为：

```python
"""统计计算模块：均值、标准差、置信区间（纯 Python，无 scipy 依赖）."""
from __future__ import annotations

import math
from statistics import mean, stdev


def _inv_erf(x: float) -> float:
    """erf 的反函数（Abramowitz & Stegun 近似，最大误差约 1.5e-7）。"""
    a = max(-0.99999, min(0.99999, x))
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    sign = 1.0 if a >= 0 else -1.0
    t = 1.0 / (1.0 + p * abs(a))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-a * a)
    return sign * y


def _norm_ppf(p: float) -> float:
    """标准正态分布的分位数函数。Phi^{-1}(p) = sqrt(2) * erf^{-1}(2p-1)"""
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if abs(p - 0.5) < 1e-16:
        return 0.0
    return math.sqrt(2.0) * _inv_erf(2.0 * p - 1.0)


def _t_ppf(p: float, df: int) -> float:
    """t 分布的分位数函数（Cornish-Fisher 展开，df >= 15 时误差 < 0.001）。"""
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    z = _norm_ppf(p)
    z2, z3, z5 = z * z, z * z * z, z * z * z * z * z
    df_f = float(df)

    # Cornish-Fisher 二阶展开
    return z + (z3 + z) / (4.0 * df_f) + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df_f * df_f)


def calculate_mean(scores: list[float]) -> float:
    if not scores:
        raise ValueError("Cannot calculate mean of empty list")
    return mean(scores)


def calculate_std(scores: list[float]) -> float:
    if len(scores) < 2:
        raise ValueError("Cannot calculate std with less than 2 samples")
    return stdev(scores)


def calculate_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    if len(scores) < 2:
        raise ValueError("Cannot calculate CI with less than 2 samples")
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1 (exclusive)")

    n = len(scores)
    sample_mean = mean(scores)
    sample_std = stdev(scores)
    standard_error = sample_std / (n ** 0.5)
    t_critical = _t_ppf((1 + confidence) / 2, df=n - 1)
    margin_of_error = t_critical * standard_error
    return (sample_mean - margin_of_error, sample_mean + margin_of_error)
```

- [ ] **Step 4: 运行全部 statistics 测试**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_statistics.py -v`
Expected: 全部通过（原有 4 个 + 新增 5 个 `_t_ppf` 测试）

- [ ] **Step 5: 验证 scipy 不再被 import**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -c "import benchmark.core.statistics; import sys; assert 'scipy' not in sys.modules; print('OK')"`
Expected: 输出 `OK`

- [ ] **Step 6: 提交**

```
git add benchmark/core/statistics.py tests/test_statistics.py
git commit -m "refactor(statistics): 用纯 Python Cornish-Fisher 展开替换 scipy t.ppf"
```

---

### Task 6: 纯 Python bootstrap + t-test 替换 numpy 和 scipy

**Files:**
- Modify: `benchmark/core/advanced_statistics.py`
- Test: `tests/test_advanced_statistics.py`

**前置条件:** Task 5 已完成，`_t_ppf` 可从 `benchmark.core.statistics` 导入。

---

- [ ] **Step 1: 运行现有测试确认当前状态**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_advanced_statistics.py -v`
Expected: 全部通过

- [ ] **Step 2: 用纯 Python 重写 `advanced_statistics.py`**

将 `benchmark/core/advanced_statistics.py` 整体替换为：

```python
"""高级统计分析模块：Bootstrap 置信区间 + t-test 显著性检验（纯 Python）."""
from __future__ import annotations

import itertools
import math
import random
import statistics
from typing import Any

from benchmark.core.statistics import _t_ppf


def _percentile(sorted_data: list[float], q: float) -> float:
    """排序数据的第 q 百分位数（线性插值法，与 numpy 默认一致）。"""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    rank = q / 100.0 * (n - 1)
    lower = int(math.floor(rank))
    upper = lower + 1
    if upper >= n:
        return sorted_data[-1]
    frac = rank - lower
    return sorted_data[lower] + frac * (sorted_data[upper] - sorted_data[lower])


def _t_cdf_bisect(t_stat: float, df: float) -> float:
    """通过二分法 + _t_ppf 求 t 分布的 CDF。"""
    if abs(t_stat) < 1e-15:
        return 0.5
    lo, hi = (0.5, 1.0) if t_stat > 0 else (0.0, 0.5)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        try:
            val = _t_ppf(mid, df=int(round(df)))
        except ValueError:
            hi = mid
            continue
        if val < t_stat:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _ttest_ind_p_value(mean_a, mean_b, var_a, var_b, n_a, n_b):
    """Welch's t-test 双侧 p 值。"""
    var_sum = var_a / n_a + var_b / n_b
    if var_sum == 0:
        return 1.0
    se = math.sqrt(var_sum)
    t_stat = (mean_a - mean_b) / se

    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    if denom == 0:
        return 1.0
    df = num / denom

    p_one_side = _t_cdf_bisect(t_stat, df)
    return 2.0 * min(p_one_side, 1.0 - p_one_side)


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    rng = random.Random(42)
    n = len(scores)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choices(scores, k=n)
        bootstrap_means.append(statistics.mean(sample))
    bootstrap_means.sort()
    lower = _percentile(bootstrap_means, (1 - confidence) / 2 * 100)
    upper = _percentile(bootstrap_means, (1 + confidence) / 2 * 100)
    return (lower, upper)


def ttest_significance(
    scores_a: list[float],
    scores_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    if len(scores_a) < 2 or len(scores_b) < 2:
        raise ValueError("Each group needs at least 2 samples for t-test")

    n_a, n_b = len(scores_a), len(scores_b)
    mean_a, mean_b = statistics.mean(scores_a), statistics.mean(scores_b)
    var_a, var_b = statistics.variance(scores_a), statistics.variance(scores_b)

    p_value = _ttest_ind_p_value(mean_a, mean_b, var_a, var_b, n_a, n_b)

    pooled_std = math.sqrt((var_a + var_b) / 2.0)
    effect_size = float((mean_a - mean_b) / pooled_std) if pooled_std > 0 else 0.0
    is_significant = bool(p_value < alpha)

    if is_significant:
        if mean_a > mean_b:
            conclusion = f"model_a 显著优于 model_b (p={p_value:.4f}, d={effect_size:.2f})"
        else:
            conclusion = f"model_b 显著优于 model_a (p={p_value:.4f}, d={effect_size:.2f})"
    else:
        conclusion = f"无显著差异 (p={p_value:.4f})"

    return {
        "p_value": float(p_value),
        "is_significant": is_significant,
        "effect_size": effect_size,
        "conclusion": conclusion,
    }


def pairwise_comparison(
    model_scores: dict[str, list[float]],
    alpha: float = 0.05,
) -> list[dict]:
    models = list(model_scores.keys())
    results = []
    for model_a, model_b in itertools.combinations(models, 2):
        test_result = ttest_significance(
            model_scores[model_a], model_scores[model_b], alpha
        )
        results.append({"model_a": model_a, "model_b": model_b, **test_result})
    return results
```

- [ ] **Step 3: 运行全部 advanced_statistics 测试**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -m pytest tests/test_advanced_statistics.py -v`
Expected: 全部 6 个测试通过

- [ ] **Step 4: 验证 numpy 和 scipy 不再被 import**

Run: `cd /home/zhushanwen/github/llm-api-daily-benchmark && python -c "import benchmark.core.advanced_statistics; import sys; assert 'numpy' not in sys.modules; assert 'scipy' not in sys.modules; print('OK')"`
Expected: 输出 `OK`

- [ ] **Step 5: 提交**

```
git add benchmark/core/advanced_statistics.py
git commit -m "refactor(advanced_statistics): 用纯 Python 替换 numpy/scipy 依赖"
```
