# 项目规范

## 时区

**禁止直接使用 `datetime.now()`。** 所有获取当前时间的代码必须通过统一工具函数。

### 规则

1. 使用 `from benchmark.core.tz import now`，调用 `now()` 获取当前时间
2. 时区通过 `.env` 的 `APP_TIMEZONE` 配置，默认 `UTC`
3. ruff DTZ 规则会拦截裸 `datetime.now()`，CI 会失败

### 正确

```python
from benchmark.core.tz import now

created_at=now()
cutoff = now() - timedelta(days=7)
timestamp = now().timestamp()
```

### 错误

```python
from datetime import datetime
datetime.now()                    # ruff DTZ003 会报错
datetime.now(timezone.utc)       # 绕过了统一配置，也不允许
```

### 原因

数据库写入（repository 层）和分析比较（analysis 层）必须在同一时区，
否则时间比较会产生偏移（如 UTC+8 服务器上会差 8 小时）。
通过 `.env` 统一配置确保所有模块行为一致。
