# Phase 1 迁移指南

## 变更摘要

### 1. 新增功能

#### 并发执行 (Task 1.1)
- 同 provider 的多个模型现在可以并发执行
- 使用 `asyncio.Semaphore` 控制并发度（默认 2）
- 从配置读取 `max_concurrency` 或 `rate_limit`

#### 历史统计缓存 (Task 1.2)
- `QualitySignalCollector` 现在缓存历史统计数据
- 缓存键: `{model}:{query_key}:{dimension}:{task_id}`
- 缓存大小限制: 1000 条
- 缓存 TTL: 3600 秒 (1小时)

#### HTTP连接池 (Task 1.3)
- `LLMEvalAdapter` 现在使用连接池复用 HTTP 连接
- 每个 provider 有独立的 `httpx.AsyncClient`
- 连接池配置: max_connections=10, max_keepalive_connections=5
- 支持异步上下文管理器 (`async with`)

#### 探针层框架 (Task 2.1)
- 新增 `benchmark/probes/` 目录结构
- 添加 `BaseProbe` 抽象基类
- 实现 `CapabilityProbe` 高频探针

### 2. 破坏性变更

#### 移除 system-architecture 维度 (Task 2.2)
**移除内容**:
- `system-architecture` 不再是一个有效的评测维度
- 相关的 `MMLUProAdapter` 和 `create_sysarch_composite` 导入已移除
- `download` 命令不再下载 MMLU-Pro 数据集

**影响**:
- 以前使用 `--dimension system-architecture` 的命令将失败
- 现有数据库中的历史数据仍然保留，可以查询

**迁移建议**:
- 使用 `probe` 维度替代，进行高频轻量级监控
- 或使用 `reasoning` 维度，包含数学推理题目

### 3. API 变更

#### LLMEvalAdapter
```python
# 新方法：异步上下文管理器
async with LLMEvalAdapter(model="zai/glm-4.7") as llm:
    response = await llm.agenerate(prompt="Hello")
# 自动调用 close() 释放连接

# 或者手动调用 close()
llm = LLMEvalAdapter(model="zai/glm-4.7")
try:
    response = await llm.agenerate(prompt="Hello")
finally:
    await llm.close()
```

#### QualitySignalCollector
```python
# 新增可选参数
collector = QualitySignalCollector(
    db=db,
    model="zai/glm-4.7",
    max_cache_size=1000,  # 缓存大小限制
    cache_ttl=3600,       # 缓存过期时间（秒）
)
```

## 性能提升

| 优化项 | 预期提升 |
|--------|----------|
| 同 Provider 并发 | 50-70% 时间减少 |
| 历史统计缓存 | 90%+ 查询减少 |
| HTTP 连接池 | 20-30% 吞吐量提升 |

## 测试

所有测试通过:
- `test_concurrent_execution.py` - 并发执行测试
- `test_quality_signals_cache.py` - 缓存功能测试
- `test_connection_pool.py` - 连接池测试
- `test_probe_framework.py` - 探针框架测试

## 下一步

Phase 2 将添加更多探针类型:
- 安全探针 (SafetyProbe)
- 特异性指纹探针 (FingerprintProbe)
- 语义一致性探针 (ConsistencyProbe)