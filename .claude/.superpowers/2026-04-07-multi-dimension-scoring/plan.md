# 多维度评分系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 4 个评测维度的 binary 评分（0/100）升级为多维度细粒度评分，通过 CompositeScorer 组合 24 个子 Scorer 计算加权总分。

**Architecture:** CompositeScorer 模式 — 每个维度由多个独立 BaseScorer 子类按权重组合。子 Scorer 异常时默认 100 分（不惩罚），确保工具缺失不阻断评测。

**Tech Stack:** Python 3.12+, asyncio, pylint, flake8, bandit, semgrep, radon, beautifulsoup4, Node.js, Playwright, axe-core, stylelint, eslint

**设计文档:** `docs/superpowers/specs/2026-04-07-multi-dimension-scoring-design.md`

---

## 总览

| Phase | 范围 | 子 Scorer 数 | 核心依赖 | 详细计划 |
|---|---|---|---|---|
| **1** | CompositeScorer + Backend + System-Arch | 13 | pylint, flake8, bandit, semgrep, radon | [phase1-infra-backend-sysarch.md](phase1-infra-backend-sysarch.md) |
| **2** | Frontend + MATH (含 LLM Judge) | 12 | Playwright, axe-core, stylelint, eslint, LLM API | [phase2-frontend-math.md](phase2-frontend-math.md) |
| **3** | FrontCode 扩展 + 报告展示 | 0 (题目+报告) | glm-5.1 (题目生成) | [phase3-frontcode-reports.md](phase3-frontcode-reports.md) |

## Phase 1 任务清单

> 详细计划: [phase1-infra-backend-sysarch.md](phase1-infra-backend-sysarch.md)

| Task | 组件 | 文件 | 权重 |
|---|---|---|---|
| 1 | CompositeScorer 基础设施 | `benchmark/scorers/composite.py` | - |
| 2 | BigCodeBenchAdapter 添加 canonical_solution | `benchmark/adapters/bigcodebench_adapter.py` | - |
| 3 | Backend TestCoverageScorer | `benchmark/scorers/backend/test_coverage.py` | 40% |
| 4 | Backend PerformanceScorer | `benchmark/scorers/backend/performance.py` | 25% |
| 5 | Backend 静态分析 Scorers (×5) | `benchmark/scorers/backend/` | 35% |
| 6 | System-Architecture Scorers (×5) | `benchmark/scorers/system_architecture/` | 100% |
| 7 | DIMENSION_REGISTRY 更新 + 集成 | `benchmark/cli.py` | - |

## Phase 2 任务清单

> 详细计划: [phase2-frontend-math.md](phase2-frontend-math.md)

| Task | 组件 | 文件 | 权重 |
|---|---|---|---|
| 1 | MATH AnswerCorrectnessScorer | `benchmark/scorers/reasoning/answer_correctness.py` | 40% |
| 2 | MATH ReasoningCompletenessScorer | `benchmark/scorers/reasoning/reasoning_completeness.py` | 25% |
| 3 | MATH ReasoningValidityScorer (LLM Judge) | `benchmark/scorers/reasoning/reasoning_validity.py` | 20% |
| 4 | MATH MethodEleganceScorer | `benchmark/scorers/reasoning/method_elegance.py` | 10% |
| 5 | MATH DifficultyAdaptationScorer | `benchmark/scorers/reasoning/difficulty_adaptation.py` | 5% |
| 6 | Frontend FunctionalityScorer | `benchmark/scorers/frontend/functionality.py` | 30% |
| 7 | Frontend HTMLSemanticScorer | `benchmark/scorers/frontend/html_semantic.py` | 20% |
| 8 | Frontend AccessibilityScorer | `benchmark/scorers/frontend/accessibility.py` | 15% |
| 9 | Frontend CSSQualityScorer | `benchmark/scorers/frontend/css_quality.py` | 15% |
| 10 | Frontend CodeOrganizationScorer | `benchmark/scorers/frontend/code_organization.py` | 10% |
| 11 | Frontend PerformanceScorer | `benchmark/scorers/frontend/performance.py` | 5% |
| 12 | Frontend BrowserCompatScorer | `benchmark/scorers/frontend/browser_compat.py` | 5% |
| 13 | DIMENSION_REGISTRY 更新 + 集成 | `benchmark/cli.py` | - |

## Phase 3 任务清单

> 详细计划: [phase3-frontcode-reports.md](phase3-frontcode-reports.md)

| Task | 组件 | 文件 |
|---|---|---|
| 1 | FrontCode 题目扩展 (5→17题) | `benchmark/datasets/frontcode/tasks.json` |
| 2 | FrontCodeAdapter 支持 test_cases | `benchmark/adapters/frontcode_adapter.py` |
| 3 | 报告雷达图 (纯 SVG) | `benchmark/core/reporter.py` |
| 4 | 报告维度分数表格 | `benchmark/core/reporter.py` |
| 5 | 集成验证 | - |

## 文件结构（最终）

```
benchmark/scorers/
├── base.py                      # 不变
├── composite.py                 # 新增: CompositeScorer
├── execution_scorer.py          # 保留
├── math_scorer.py              # 保留
├── choice_match_scorer.py       # 保留
├── keyword_match_scorer.py      # 保留
├── backend/
│   ├── __init__.py
│   ├── test_coverage.py         # 40%
│   ├── performance.py           # 25%
│   ├── code_style.py            # 15%
│   ├── robustness.py            # 10%
│   ├── architecture.py          # 5%
│   ├── security.py              # 3%
│   └── extensibility.py         # 2%
├── frontend/
│   ├── __init__.py
│   ├── functionality.py         # 30%
│   ├── html_semantic.py         # 20%
│   ├── accessibility.py         # 15%
│   ├── css_quality.py           # 15%
│   ├── code_organization.py     # 10%
│   ├── performance.py           # 5%
│   └── browser_compat.py        # 5%
├── reasoning/
│   ├── __init__.py
│   ├── answer_correctness.py    # 40%
│   ├── reasoning_completeness.py # 25%
│   ├── reasoning_validity.py    # 20%
│   ├── method_elegance.py       # 10%
│   └── difficulty_adaptation.py # 5%
└── system_architecture/
    ├── __init__.py
    ├── answer_correctness.py    # 30%
    ├── reasoning_completeness.py # 25%
    ├── option_analysis.py       # 20%
    ├── reasoning_confidence.py  # 15%
    └── subject_adaptation.py    # 10%
```

## 关键约束

1. 子 Scorer 异常 → 默认 100 分，不拉低总分
2. `reasoning_content` 为空 → 依赖推理的 Scorer 返回 100 分
3. 权重之和必须等于 1.0
4. `passed` = `score >= 60`
5. `functional_score` 改为加权总分
6. `details` JSON 存储 `composite.weights` + `composite.scores` + 各子 scorer 原始输出
