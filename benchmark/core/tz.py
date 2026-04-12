"""项目统一时区工具。

所有需要当前时间的地方，必须使用本模块的 now() 函数，
而非标准库的 datetime.now()。时区通过环境变量 APP_TIMEZONE 配置，
默认 UTC。

用法::

    from benchmark.core.tz import now

    ts = now()                    # 带时区信息的当前时间
    ts.isoformat()                # "2025-01-01T00:00:00+00:00"
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_DEFAULT_TZ = "UTC"


def get_timezone() -> timezone | ZoneInfo:
    """获取项目配置的时区。

    从环境变量 APP_TIMEZONE 读取，默认 "UTC"。
    支持所有 IANA 时区名（如 "Asia/Shanghai"、"America/New_York"）。
    """
    tz_name = os.environ.get("APP_TIMEZONE", _DEFAULT_TZ)
    if tz_name.upper() == "UTC":
        return timezone.utc
    return ZoneInfo(tz_name)


def now() -> datetime:
    """返回带时区信息的当前时间。"""
    return datetime.now(get_timezone())
