# LLM Benchmark - Stage 3 规格概要

**阶段**: 高级 + 完善  
**版本**: 1.0  
**创建日期**: 2026-03-31  
**状态**: 规划中  
**前置条件**: Stage 2 完成并验收  
**估算工作量**: 17-26 小时

---

## 概述

### 目标

实现完整的多维度评测能力，提供深度分析和专业报告。

### 交付价值

- ✅ 完整的5维度评测
- ✅ 显著性检验（模型对比）
- ✅ 专业PDF报告

### 数据集

| 维度 | 数据集 | 题目数 | 状态 |
|------|--------|--------|------|
| reasoning | GSM8K最难的5题 | 5题 | ✅ Stage 1 |
| backend-dev | BigCodeBench-Hard 5题 | 5题 | ✅ Stage 1 |
| system-architecture | MMLU（法律/道德） | 5题 | ✅ Stage 2 |
| frontend-dev | FrontCode自建 | 5题 | ✅ Stage 2 |
| **tool-use-agentic** | **AgentBench（WebShop/Mind2Web）** | **5题** | **🆕 Stage 3** |
| **总计** | | **25题** | |

---

## 新增功能

### 1. Agent Loop 实现

**文件**：`benchmark/core/agent_loop.py`

**概念**：
Agent Loop是多轮工具调用循环，用于测试模型的能力使用工具解决问题的能力。

**实现逻辑**：

```python
async def run_agent_loop(
    task: TaskDefinition,
    model: str,
    tools: List[dict],
    max_iterations: int = 10
) -> ExecutionResult:
    """运行Agent Loop
    
    Args:
        task: 任务定义
        model: 模型名称
        tools: 可用工具列表
        max_iterations: 最大迭代次数
        
    Returns:
        执行结果（包含工具调用记录）
    """
    messages = [{"role": "user", "content": task.prompt}]
    all_tool_calls = []
    
    for i in range(max_iterations):
        # 调用LLM
        response = await llm.generate_with_tools(messages, tools, model)
        
        # 如果模型不发起工具调用，视为最终答案
        if not response.tool_calls:
            break
        
        # 执行工具调用
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            all_tool_calls.append({
                "iteration": i,
                "tool": tool_call.name,
                "args": tool_call.args,
                "result": result
            })
            
            # 将结果添加到消息
            messages.append({
                "role": "tool",
                "content": str(result)
            })
    
    return ExecutionResult(
        final_answer=response.content,
        tool_calls=all_tool_calls,
        iterations=i
    )
```

**工具模拟器**：

```python
# 定义可用工具
tools = [
    {
        "name": "read_file",
        "description": "读取文件内容",
        "parameters": {
            "path": {"type": "string", "description": "文件路径"}
        }
    },
    {
        "name": "write_file",
        "description": "写入文件内容",
        "parameters": {
            "path": {"type": "string"},
            "content": {"type": "string"}
        }
    },
    {
        "name": "run_command",
        "description": "执行命令",
        "parameters": {
            "command": {"type": "string"}
        }
    }
]

def execute_tool(tool_call: ToolCall) -> Any:
    """执行工具调用（在subprocess沙箱中）"""
    if tool_call.name == "read_file":
        return read_file_sandbox(tool_call.args["path"])
    elif tool_call.name == "write_file":
        return write_file_sandbox(tool_call.args["path"], tool_call.args["content"])
    elif tool_call.name == "run_command":
        return run_command_sandbox(tool_call.args["command"])
```

**终止条件**：
- 模型返回纯文本（不发起工具调用）
- 达到最大迭代次数

---

### 2. AgentBench Adapter

**文件**：`benchmark/adapters/agentbench_adapter.py`

**数据集**：AgentBench（选择WebShop或Mind2Web环境）

**选择策略**：
- WebShop环境：最复杂的环境
- 或Mind2Web环境：最长的交互链路
- 选择评估准确率最低的5题

**评分器**：AgentLoopScorer

**评分逻辑**：
```python
class AgentLoopScorer(BaseScorer):
    def score(self, result: ExecutionResult, task: TaskDefinition) -> ScoreResult:
        """Agent Loop评分
        
        考虑因素：
        1. 任务是否完成
        2. 迭代次数是否合理
        3. 工具调用成功率
        """
        # 检查任务完成
        completed = check_task_completion(result, task)
        
        # 检查工具调用成功
        tool_success_rate = sum(1 for t in result.tool_calls if t["result"]["success"]) / len(result.tool_calls)
        
        # 检查迭代次数
        efficiency = max(0, 1 - result.iterations / task.max_iterations)
        
        score = (completed * 60 + tool_success_rate * 30 + efficiency * 10)
        
        return ScoreResult(
            score=score,
            passed=completed,
            details={
                "iterations": result.iterations,
                "tool_calls": len(result.tool_calls),
                "tool_success_rate": tool_success_rate
            }
        )
```

---

### 3. 高级统计（Bootstrap + t-test）

**文件**：`benchmark/core/advanced_statistics.py`

**功能**：

#### Bootstrap置信区间

```python
def bootstrap_confidence_interval(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """Bootstrap置信区间
    
    Args:
        scores: 分数列表
        confidence: 置信水平（默认95%）
        n_bootstrap: Bootstrap采样次数
        
    Returns:
        (lower_bound, upper_bound)
    """
    import numpy as np
    
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    
    return (lower, upper)
```

#### t-test显著性检验

```python
def ttest_significance(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """t-test显著性检验
    
    Args:
        scores_a: 模型A的分数列表
        scores_b: 模型B的分数列表
        alpha: 显著性水平（默认0.05）
        
    Returns:
        {
            "p_value": float,
            "is_significant": bool,
            "effect_size": float,
            "conclusion": str
        }
    """
    from scipy import stats
    
    # t-test
    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
    
    # 效应量（Cohen's d）
    cohens_d = (np.mean(scores_a) - np.mean(scores_b)) / np.sqrt(
        (np.std(scores_a) ** 2 + np.std(scores_b) ** 2) / 2
    )
    
    # 判断显著性
    is_significant = p_value < alpha
    
    # 结论
    if is_significant:
        if np.mean(scores_a) > np.mean(scores_b):
            conclusion = f"模型A显著优于模型B（p={p_value:.4f}）"
        else:
            conclusion = f"模型B显著优于模型A（p={p_value:.4f}）"
    else:
        conclusion = f"两个模型无显著差异（p={p_value:.4f}）"
    
    return {
        "p_value": p_value,
        "is_significant": is_significant,
        "effect_size": cohens_d,
        "conclusion": conclusion
    }
```

---

### 4. 报告生成

**文件**：`benchmark/core/reporter.py`

**功能**：

#### HTML报告

```python
def generate_html_report(
    run_ids: List[str],
    output_path: str
) -> str:
    """生成HTML报告
    
    Args:
        run_ids: 运行记录ID列表
        output_path: 输出文件路径
        
    Returns:
        HTML文件路径
    """
    from jinja2 import Environment, FileSystemLoader
    
    # 加载模板
    env = Environment(loader=FileSystemLoader('benchmark/templates'))
    template = env.get_template('report.html')
    
    # 查询数据
    data = query_report_data(run_ids)
    
    # 渲染HTML
    html = template.render(data)
    
    # 写入文件
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path
```

#### PDF报告（可选）

```python
def generate_pdf_report(
    html_path: str,
    output_path: str
) -> str:
    """从HTML生成PDF
    
    Args:
        html_path: HTML文件路径
        output_path: 输出PDF路径
        
    Returns:
        PDF文件路径
    """
    from weasyprint import HTML
    
    HTML(html_path).write_pdf(output_path)
    
    return output_path
```

**报告内容**：

1. **概览**：评测时间范围、模型列表、数据集信息
2. **趋势分析**：各维度分数随时间变化的折线图
3. **统计检验**：模型之间是否有显著差异
4. **详细结果**：每个题目的分数明细

---

## CLI 命令更新

### 新增命令

```bash
# 生成报告
python -m benchmark report \
  --models glm-4.7,gpt-4 \
  --dimensions reasoning,backend-dev \
  --date-range 2024-01-01,2024-01-31 \
  --output report.pdf

# 预期：生成PDF报告
```

```bash
# 运行Agent Loop评测
python -m benchmark evaluate --model glm-4.7 --dimension tool-use-agentic --samples 5

# 预期：可以看到多轮工具调用记录
# [1/5] Task ID: agentbench_xxx
#   Iteration 1: read_file(path="xxx")
#   Iteration 2: write_file(path="yyy", content="...")
#   Iteration 3: run_command(command="ls")
#   ...
#   Score: 75, Iterations: 5
```

---

## 评分策略完整版

### 所有维度评分

| 维度 | 评分器 | 权重 |
|------|--------|------|
| reasoning | ExactMatchScorer | auto=0.8, judge=0.2 |
| backend-dev | ExecutionScorer | auto=0.8, judge=0.2 |
| system-architecture | ExactMatchScorer | auto=0.8, judge=0.2 |
| frontend-dev | LLMJudgeScorer | auto=0.2, judge=0.8 |
| tool-use-agentic | AgentLoopScorer | auto=0.5, judge=0.5 |

---

## 验收标准

### 1. Agent Loop

```bash
# 运行评测
python -m benchmark evaluate --model glm-4.7 --dimension tool-use-agentic --samples 5

# 预期：
# - 可以看到多轮工具调用记录
# - 显示迭代次数
# - 显示工具调用成功率
```

### 2. 高级统计

```bash
# 启动Streamlit
streamlit run benchmark/visualization/app.py

# 预期：
# - 可以看到Bootstrap置信区间
# - 可以看到显著性检验结果
# - 可以看到"模型A是否显著优于模型B"
```

### 3. 报告生成

```bash
# 生成PDF报告
python -m benchmark report --models glm-4.7,gpt-4 --output report.pdf

# 预期：
# - 生成PDF文件
# - 包含趋势图
# - 包含统计表格
# - 包含显著性检验结果
```

---

## 依赖更新

**新增依赖**：
- `weasyprint>=60` - PDF生成（可选）

---

## 不在范围内

Stage 3 **不包含**：

- ❌ Docker容器化（不在任何Stage）
- ❌ 浏览器渲染验证（不在任何Stage）
- ❌ 分布式评测（不在任何Stage）
- ❌ 实时评测（不在任何Stage）

---

## 状态追踪

| 组件 | 状态 | 说明 |
|------|------|------|
| AgentLoop | ⏳ 待实施 | 文件：`core/agent_loop.py` |
| AgentBenchAdapter | ⏳ 待实施 | 文件：`adapters/agentbench_adapter.py` |
| AgentLoopScorer | ⏳ 待实施 | 文件：`scorers/agent_loop_scorer.py` |
| AdvancedStatistics | ⏳ 待实施 | 文件：`core/advanced_statistics.py` |
| Reporter | ⏳ 待实施 | 文件：`core/reporter.py` |
| Report Templates | ⏳ 待实施 | 文件：`templates/report.html` |

---

## 下一步

⏳ 等待Stage 2完成并验收后开始实施

---

## 实施顺序建议

1. **Week 1**: 实现Agent Loop + AgentBench适配器
2. **Week 2**: 实现高级统计（Bootstrap + t-test）
3. **Week 3**: 实现报告生成模块

**总工作量**：17-26小时