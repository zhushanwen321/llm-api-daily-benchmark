# LLM Benchmark - Stage 1 详细规格

**阶段**: MVP 核心评测能力  
**版本**: 1.0  
**创建日期**: 2026-03-31  
**状态**: 待实施  
**前置条件**: 无  
**估算工作量**: 15-21 小时

---

## 概述

### 目标

构建可运行的最小可用版本（MVP），能够：
- 手动评测单个模型
- 计算 2 个维度的分数
- 查看评测结果

### 交付价值

- ✅ 验证核心架构是否正确
- ✅ 确认数据集加载和评分逻辑
- ✅ 用户可以看到评测结果

### 数据集

| 维度 | 数据集 | 题目数 |
|------|--------|--------|
| **reasoning** | GSM8K（最难的5题） | 5题 |
| **backend-dev** | BigCodeBench-Hard（5题） | 5题 |
| **总计** | | **10题** |

---

## 架构设计

### 组件职责

```
┌─────────────────────────────────────┐
│         CLI (cli.py)                │
│  evaluate | list-datasets | export │
└────────────┬────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐      ┌────▼─────┐
│Adapters│      │ Scorers  │
│(数据集) │      │ (评分)   │
└───┬────┘      └────┬─────┘
    │                │
    └────────┬───────┘
             │
      ┌──────▼──────┐
      │   Storage   │
      │  (SQLite)   │
      └─────────────┘
```

### 文件结构

```
benchmark/
├── __init__.py
├── __main__.py                 # 入口：python -m benchmark
├── cli.py                      # CLI命令
├── config.py                   # 配置加载
│
├── adapters/
│   ├── __init__.py
│   ├── base.py                 # DatasetAdapter基类
│   ├── gsm8k_adapter.py         # GSM8K适配器
│   └── bigcodebench_adapter.py  # BigCodeBench适配器
│
├── scorers/
│   ├── __init__.py
│   ├── base.py                 # BaseScorer基类
│   ├── exact_match_scorer.py   # 精确匹配评分器
│   └── execution_scorer.py      # 执行验证评分器
│
├── models/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic数据模型
│   └── database.py             # SQLite操作
│
├── core/
│   ├── __init__.py
│   └── llm_adapter.py           # LLM API调用适配
│
├── visualization/
│   ├── __init__.py
│   └── app.py                  # Streamlit应用
│
└── configs/
    ├── default.yaml            # 默认配置
    └── models.yaml             # 模型API配置
```

---

## 核心接口设计

### 1. DatasetAdapter（数据集适配器）

**职责**：加载和验证数据集

**基类**：`benchmark/adapters/base.py`

```python
from abc import ABC, abstractmethod
from typing import List
from benchmark.models.schemas import TaskDefinition

class DatasetAdapter(ABC):
    """数据集适配器基类"""
    
    @abstractmethod
    def load(self, path: str) -> List[TaskDefinition]:
        """加载数据集
        
        Args:
            path: 数据集路径（本地或HuggingFace）
            
        Returns:
            任务定义列表
        """
        pass
    
    @abstractmethod
    def validate(self, task: TaskDefinition) -> bool:
        """验证任务格式
        
        Args:
            task: 任务定义
            
        Returns:
            是否有效
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> str:
        """返回所属维度
        
        Returns:
            维度名称（reasoning, backend-dev等）
        """
        pass
```

---

### 2. GSM8KAdapter（GSM8K适配器）

**文件**：`benchmark/adapters/gsm8k_adapter.py`

**职责**：加载GSM8K最难的5题

**实现逻辑**：

```python
class GSM8KAdapter(DatasetAdapter):
    """GSM8K适配器，加载最难的5题"""
    
    def load(self, path: str) -> List[TaskDefinition]:
        """从HuggingFace加载GSM8K，选择最难的5题"""
        from datasets import load_dataset
        
        # 加载数据集
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        
        # 筛选最难的题目（解答步骤最多）
        # 解答长度 = 思维链长度
        difficulties = []
        for item in dataset:
            solution = item['solution']
            steps = len(solution.split('\n'))
            difficulties.append((item, steps))
        
        # 按步骤数降序排序，选前5题
        difficulties.sort(key=lambda x: x[1], reverse=True)
        hardest_5 = [item for item, _ in difficulties[:5]]
        
        # 转换为TaskDefinition
        tasks = []
        for item in hardest_5:
            # 提取最终答案（格式：#### 42）
            answer = item['answer'].split('####')[1].strip()
            
            task = TaskDefinition(
                task_id=f"gsm8k_{item['question'][:20]}",
                dimension="reasoning",
                dataset="gsm8k",
                prompt=item['question'],
                expected_output=answer,
                metadata={
                    "difficulty": "hard",
                    "steps": steps
                }
            )
            tasks.append(task)
        
        return tasks
```

---

### 3. BigCodeBenchAdapter（BigCodeBench适配器）

**文件**：`benchmark/adapters/bigcodebench_adapter.py`

**职责**：加载BigCodeBench-Hard子集的5题

**实现逻辑**：

```python
class BigCodeBenchAdapter(DatasetAdapter):
    """BigCodeBench适配器，加载Hard子集"""
    
    def load(self, path: str) -> List[TaskDefinition]:
        """从HuggingFace加载BigCodeBench-Hard，随机选5题"""
        from datasets import load_dataset
        import random
        
        # 加载Hard子集
        dataset = load_dataset(
            "bigcode/bigcodebench-hard",
            split="test"
        )
        
        # 随机选择5题
        indices = random.sample(range(len(dataset)), 5)
        selected = [dataset[i] for i in indices]
        
        # 转换为TaskDefinition
        tasks = []
        for item in selected:
            task = TaskDefinition(
                task_id=f"bigcodebench_{item['task_id']}",
                dimension="backend-dev",
                dataset="bigcodebench",
                prompt=item['instruct'],
                expected_output="",  # 代码不需要精确匹配
                test_cases=item['test'],  # 测试用例
                metadata={
                    "difficulty": "hard",
                    "entry_point": item['entry_point']
                }
            )
            tasks.append(task)
        
        return tasks
```

---

### 4. BaseScorer（评分器基类）

**文件**：`benchmark/scorers/base.py`

```python
from abc import ABC, abstractmethod
from benchmark.models.schemas import TaskDefinition, ScoreResult

class BaseScorer(ABC):
    """评分器基类"""
    
    @abstractmethod
    def score(
        self, 
        model_output: str, 
        expected: str, 
        task: TaskDefinition
    ) -> ScoreResult:
        """评分
        
        Args:
            model_output: 模型输出
            expected: 期望输出
            task: 任务定义
            
        Returns:
            评分结果
        """
        pass
    
    @abstractmethod
    def get_metric_name(self) -> str:
        """返回指标名称"""
        pass
```

---

### 5. ExactMatchScorer（精确匹配评分器）

**文件**：`benchmark/scorers/exact_match_scorer.py`

**职责**：用于reasoning维度（GSM8K）

```python
import re
from benchmark.scorers.base import BaseScorer, ScoreResult
from benchmark.models.schemas import TaskDefinition

class ExactMatchScorer(BaseScorer):
    """精确匹配评分器，用于reasoning维度"""
    
    def score(
        self, 
        model_output: str, 
        expected: str, 
        task: TaskDefinition
    ) -> ScoreResult:
        """精确匹配评分
        
        Args:
            model_output: 模型输出的最终答案
            expected: 期望答案（数字）
            task: 任务定义
            
        Returns:
            ScoreResult(score=100 或 0, passed=bool)
        """
        # 提取数字
        numbers = re.findall(r'-?\d+\.?\d*', model_output)
        
        if not numbers:
            return ScoreResult(
                score=0,
                passed=False,
                details={"error": "No number found in output"},
                reasoning="Model output contains no numeric answer"
            )
        
        # 取最后一个数字作为答案
        predicted = numbers[-1]
        
        # 精确匹配
        passed = (predicted == expected)
        score = 100 if passed else 0
        
        return ScoreResult(
            score=score,
            passed=passed,
            details={
                "predicted": predicted,
                "expected": expected
            },
            reasoning=f"{'Correct' if passed else 'Incorrect'}: predicted {predicted}, expected {expected}"
        )
    
    def get_metric_name(self) -> str:
        return "exact_match"
```

---

### 6. ExecutionScorer（执行验证评分器）

**文件**：`benchmark/scorers/execution_scorer.py`

**职责**：用于backend-dev维度（BigCodeBench）

```python
import subprocess
import tempfile
from benchmark.scorers.base import BaseScorer, ScoreResult
from benchmark.models.schemas import TaskDefinition

class ExecutionScorer(BaseScorer):
    """执行验证评分器，用于backend-dev维度"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def score(
        self, 
        model_output: str, 
        expected: str, 
        task: TaskDefinition
    ) -> ScoreResult:
        """执行代码并验证测试用例
        
        Args:
            model_output: 模型生成的代码
            expected: 不使用
            task: 任务定义（包含test_cases）
            
        Returns:
            ScoreResult(score=100 或 0, passed=bool)
        """
        test_cases = task.metadata.get('test_cases', [])
        entry_point = task.metadata.get('entry_point', 'solution')
        
        # 创建临时文件执行代码
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False
        ) as f:
            # 写入模型生成的代码
            f.write(model_output)
            # 写入测试用例
            f.write('\n\n# Test cases\n')
            for test in test_cases:
                f.write(f'{test}\n')
            temp_file = f.name
        
        try:
            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # 检查退出码
            if result.returncode == 0:
                return ScoreResult(
                    score=100,
                    passed=True,
                    details={
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    },
                    reasoning="All test cases passed"
                )
            else:
                return ScoreResult(
                    score=0,
                    passed=False,
                    details={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode
                    },
                    reasoning=f"Test execution failed with return code {result.returncode}"
                )
        
        except subprocess.TimeoutExpired:
            return ScoreResult(
                score=0,
                passed=False,
                details={"error": "Timeout"},
                reasoning=f"Execution timed out after {self.timeout} seconds"
            )
        
        except Exception as e:
            return ScoreResult(
                score=0,
                passed=False,
                details={"error": str(e)},
                reasoning=f"Execution error: {str(e)}"
            )
        
        finally:
            # 清理临时文件
            import os
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def get_metric_name(self) -> str:
        return "execution"
```

---

### 7. Data Models（数据模型）

**文件**：`benchmark/models/schemas.py`

```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class TaskDefinition(BaseModel):
    """任务定义"""
    task_id: str
    dimension: str  # reasoning, backend-dev, etc.
    dataset: str    # gsm8k, bigcodebench, etc.
    prompt: str
    expected_output: str
    test_cases: Optional[List[str]] = []
    metadata: Dict[str, Any] = {}

class ScoreResult(BaseModel):
    """评分结果"""
    score: float          # 0-100
    passed: bool          # 是否通过
    details: Dict[str, Any] = {}  # 详细信息
    reasoning: str        # 评分理由

class EvalRun(BaseModel):
    """评测运行记录"""
    run_id: str
    model: str
    dimension: str
    dataset: str
    started_at: datetime
    finished_at: Optional[datetime]
    status: str  # running, completed, failed
    config_hash: str

class EvalResult(BaseModel):
    """单题结果"""
    result_id: str
    run_id: str
    task_id: str
    task_content: str
    model_output: str
    functional_score: float
    quality_score: float = 0.0
    final_score: float
    passed: bool
    details: Dict[str, Any] = {}
    execution_time: float
    created_at: datetime
```

---

### 8. Database（数据库操作）

**文件**：`benchmark/models/database.py`

```python
import sqlite3
from typing import List, Optional
from benchmark.models.schemas import EvalRun, EvalResult

class Database:
    """SQLite数据库操作"""
    
    def __init__(self, db_path: str = "benchmark/data/results.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建eval_runs表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                model TEXT NOT NULL,
                dimension TEXT NOT NULL,
                dataset TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                finished_at TIMESTAMP,
                status TEXT NOT NULL,
                config_hash TEXT
            )
        """)
        
        # 创建eval_results表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id TEXT UNIQUE NOT NULL,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                task_content TEXT,
                model_output TEXT,
                functional_score REAL,
                quality_score REAL,
                final_score REAL,
                passed INTEGER,
                details TEXT,
                execution_time REAL,
                created_at TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_run(self, run: EvalRun) -> str:
        """创建评测运行记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO eval_runs 
            (run_id, model, dimension, dataset, started_at, status, config_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run.run_id, run.model, run.dimension, run.dataset,
            run.started_at, run.status, run.config_hash
        ))
        
        conn.commit()
        conn.close()
        
        return run.run_id
    
    def save_result(self, result: EvalResult):
        """保存单题结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO eval_results
            (result_id, run_id, task_id, task_content, model_output,
             functional_score, quality_score, final_score, passed,
             details, execution_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.result_id, result.run_id, result.task_id,
            result.task_content, result.model_output,
            result.functional_score, result.quality_score,
            result.final_score, result.passed,
            str(result.details), result.execution_time,
            result.created_at
        ))
        
        conn.commit()
        conn.close()
```

---

### 9. LLM Adapter（LLM调用适配器）

**文件**：`benchmark/core/llm_adapter.py`

```python
import requests
from typing import Dict, Any
from benchmark.config import load_config

class LLMEvalAdapter:
    """LLM调用适配器"""
    
    def __init__(self):
        self.config = load_config()
        self.timeout = self.config.get('timeout', 300)
        self.max_retries = self.config.get('max_retries', 3)
    
    def generate(
        self, 
        prompt: str, 
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> str:
        """生成文本
        
        Args:
            prompt: 输入提示
            model: 模型名称
            temperature: 温度（评测固定为0）
            max_tokens: 最大token数
            
        Returns:
            模型生成的文本
        """
        # 获取模型配置
        model_config = self.config['models'].get(model)
        if not model_config:
            raise ValueError(f"Model {model} not found in config")
        
        # 构造请求
        url = f"{model_config['api_base']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {model_config['api_key']}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # 发送请求（带重试）
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                return data['choices'][0]['message']['content']
            
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                # 指数退避重试
                import time
                time.sleep(2 ** attempt)
```

---

### 10. CLI Commands（CLI命令）

**文件**：`benchmark/cli.py`

```python
import click
from datetime import datetime
import uuid
from benchmark.adapters.gsm8k_adapter import GSM8KAdapter
from benchmark.adapters.bigcodebench_adapter import BigCodeBenchAdapter
from benchmark.scorers.exact_match_scorer import ExactMatchScorer
from benchmark.scorers.execution_scorer import ExecutionScorer
from benchmark.models.database import Database
from benchmark.models.schemas import EvalRun, EvalResult
from benchmark.core.llm_adapter import LLMEvalAdapter

@click.group()
def cli():
    """LLM Benchmark评测工具"""
    pass

@cli.command()
@click.option('--model', required=True, help='模型名称')
@click.option('--dimension', required=True, 
              type=click.Choice(['reasoning', 'backend-dev']),
              help='评测维度')
@click.option('--samples', default=5, help='样本数量')
def evaluate(model: str, dimension: str, samples: int):
    """运行评测"""
    # 初始化组件
    db = Database()
    llm = LLMEvalAdapter()
    
    # 选择适配器和评分器
    if dimension == 'reasoning':
        adapter = GSM8KAdapter()
        scorer = ExactMatchScorer()
    elif dimension == 'backend-dev':
        adapter = BigCodeBenchAdapter()
        scorer = ExecutionScorer()
    
    # 加载数据集
    tasks = adapter.load('')[:samples]
    
    # 创建运行记录
    run_id = str(uuid.uuid4())
    run = EvalRun(
        run_id=run_id,
        model=model,
        dimension=dimension,
        dataset=adapter.get_dimension(),
        started_at=datetime.now(),
        status='running',
        config_hash=''
    )
    db.create_run(run)
    
    click.echo(f"Starting evaluation: {dimension} with {len(tasks)} tasks")
    
    # 评测每道题
    for i, task in enumerate(tasks, 1):
        click.echo(f"[{i}/{len(tasks)}] Task ID: {task.task_id}")
        
        # 调用LLM
        start_time = datetime.now()
        model_output = llm.generate(task.prompt, model)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # 评分
        score_result = scorer.score(model_output, task.expected_output, task)
        
        # 保存结果
        result = EvalResult(
            result_id=str(uuid.uuid4()),
            run_id=run_id,
            task_id=task.task_id,
            task_content=task.prompt,
            model_output=model_output,
            functional_score=score_result.score,
            final_score=score_result.score,
            passed=score_result.passed,
            details=score_result.details,
            execution_time=execution_time,
            created_at=datetime.now()
        )
        db.save_result(result)
        
        click.echo(f"  Score: {score_result.score}, Passed: {score_result.passed}")
    
    click.echo(f"Complete evaluation: {run_id}")

@cli.command()
def list_datasets():
    """列出可用数据集"""
    click.echo("Available datasets:")
    click.echo("  - reasoning: GSM8K (hardest 5 tasks)")
    click.echo("  - backend-dev: BigCodeBench-Hard (5 tasks)")

@cli.command()
@click.option('--format', type=click.Choice(['json', 'csv']), default='json')
@click.option('--output', default='results.json', help='输出文件')
def export(format: str, output: str):
    """导出结果"""
    # TODO: 实现导出
    click.echo(f"Exporting results to {output} in {format} format")

if __name__ == '__main__':
    cli()
```

---

### 11. Streamlit界面

**文件**：`benchmark/visualization/app.py`

```python
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="LLM Benchmark",
    page_icon="📊",
    layout="wide"
)

# 数据库连接
@st.cache_resource
def get_db():
    return sqlite3.connect('benchmark/data/results.db')

# 主页面
def main():
    st.title("LLM Benchmark Results")
    
    # 侧边栏过滤器
    st.sidebar.header("Filters")
    
    model_filter = st.sidebar.selectbox(
        "Model",
        ["All"] + get_models()
    )
    
    dimension_filter = st.sidebar.selectbox(
        "Dimension",
        ["All", "reasoning", "backend-dev"]
    )
    
    # 查询结果
    results = get_results(model_filter, dimension_filter)
    
    # 显示表格
    st.subheader("Evaluation Results")
    st.dataframe(results)
    
    # 显示详情
    if st.checkbox("Show Details"):
        selected = st.selectbox("Select Result", results['result_id'])
        show_result_details(selected)

def get_models():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT DISTINCT model FROM eval_runs")
    return [row[0] for row in cursor.fetchall()]

def get_results(model, dimension):
    db = get_db()
    query = """
        SELECT 
            r.result_id,
            e.model,
            e.dimension,
            r.task_id,
            r.final_score,
            r.passed,
            r.execution_time,
            e.created_at
        FROM eval_results r
        JOIN eval_runs e ON r.run_id = e.run_id
        WHERE 1=1
    """
    params = []
    
    if model != "All":
        query += " AND e.model = ?"
        params.append(model)
    
    if dimension != "All":
        query += " AND e.dimension = ?"
        params.append(dimension)
    
    df = pd.read_sql_query(query, db, params=params)
    return df

def show_result_details(result_id):
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT * FROM eval_results WHERE result_id = ?",
        (result_id,)
    )
    result = cursor.fetchone()
    
    if result:
        st.subheader(f"Result: {result_id}")
        st.write(f"**Task**: {result[3]}")
        st.write(f"**Score**: {result[6]}")
        st.write(f"**Passed**: {result[8]}")
        st.write(f"**Execution Time**: {result[7]:.2f}s")
        st.write("**Model Output**:")
        st.code(result[5])

if __name__ == "__main__":
    main()
```

---

## 配置文件

### 1. default.yaml

**文件**：`benchmark/configs/default.yaml`

```yaml
# 默认配置
model: "glm-4.7"
temperature: 0.0
max_tokens: 4096
max_retries: 3
timeout: 300

# 数据集根目录
dataset_root: "benchmark/datasets"

# 维度权重配置
dimensions:
  reasoning:
    adapter: "gsm8k"
    auto_weight: 0.8
    judge_weight: 0.2
  
  backend-dev:
    adapter: "bigcodebench"
    auto_weight: 0.8
    judge_weight: 0.2
```

### 2. models.yaml

**文件**：`benchmark/configs/models.yaml`

```yaml
# 模型配置（明文存储，个人项目）
# 注意：此文件应加入 .gitignore

models:
  glm-4.7:
    provider: "glm"
    api_key: "your_glm_api_key_here"
    api_base: "https://open.bigmodel.cn/api/paas/v4/"
    max_tokens: 4096
  
  gpt-4:
    provider: "openai"
    api_key: "your_openai_api_key_here"
    api_base: "https://api.openai.com/v1/"
    max_tokens: 4096
```

---

## 依赖列表

**文件**：`pyproject.toml`

```toml
[project]
name = "llm-benchmark"
version = "0.1.0"
dependencies = [
    "pydantic>=2.0",
    "rich>=13.0",
    "streamlit>=1.28",
    "datasets>=2.14",
    "pyyaml>=6.0",
    "requests>=2.31",
]

[project.scripts]
benchmark = "benchmark.cli:cli"
```

---

## 验收标准

### 1. CLI命令

```bash
# 命令可用
python -m benchmark --help
# 预期输出：显示所有可用命令

# 数据集列表
python -m benchmark list-datasets
# 预期输出：
# Available datasets:
#   - reasoning: GSM8K (hardest 5 tasks)
#   - backend-dev: BigCodeBench-Hard (5 tasks)

# 运行评测
python -m benchmark evaluate --model glm-4.7 --dimension reasoning --samples 5
# 预期输出：
# Loading GSM8K dataset...
# [1/5] Task ID: gsm8k_xxx
#   Score: 100, Passed: True
# [2/5] Task ID: gsm8k_yyy
#   Score: 0, Passed: False
# ...
# Average Score: 75.6
# Complete evaluation: <run_id>
```

### 2. SQLite存储

```bash
sqlite3 benchmark/data/results.db
# 预期：
# - eval_runs 表存在
# - eval_results 表存在
# - 评测结果正确写入
```

### 3. Streamlit界面

```bash
streamlit run benchmark/visualization/app.py
# 预期：
# - 页面加载成功
# - 显示结果列表
# - 可以按 model/dimension 过滤
# - 可以查看单题详情
```

---

## 不在范围内

Stage 1 **不包含**：

- ❌ 定时调度器（Stage 2）
- ❌ Agent Loop（Stage 3）
- ❌ LLM Judge评分器（Stage 2）
- ❌ 趋势图（Stage 2）
- ❌ 统计检验（Stage 3）
- ❌ 报告生成（Stage 3）
- ❌ frontend-dev维度（Stage 2）
- ❌ system-architecture维度（Stage 2）
- ❌ tool-use-agentic维度（Stage 3）

---

## 下一步

1. ✅ Stage 1 规格已文档化
2. 🔄 创建 Stage 1 实施计划（使用writing-plans技能）
3. ⏳ 开始实施 Stage 1

---

## 状态追踪

| 组件 | 状态 | 说明 |
|------|------|------|
| DatasetAdapter基类 | ⏳ 待实施 | 文件：`adapters/base.py` |
| GSM8KAdapter | ⏳ 待实施 | 文件：`adapters/gsm8k_adapter.py` |
| BigCodeBenchAdapter | ⏳ 待实施 | 文件：`adapters/bigcodebench_adapter.py` |
| BaseScorer基类 | ⏳ 待实施 | 文件：`scorers/base.py` |
| ExactMatchScorer | ⏳ 待实施 | 文件：`scorers/exact_match_scorer.py` |
| ExecutionScorer | ⏳ 待实施 | 文件：`scorers/execution_scorer.py` |
| Data Models | ⏳ 待实施 | 文件：`models/schemas.py` |
| Database | ⏳ 待实施 | 文件：`models/database.py` |
| LLM Adapter | ⏳ 待实施 | 文件：`core/llm_adapter.py` |
| CLI Commands | ⏳ 待实施 | 文件：`cli.py` |
| Streamlit界面 | ⏳ 待实施 | 文件：`visualization/app.py` |
| 配置文件 | ⏳ 待实施 | 文件：`configs/*.yaml` |