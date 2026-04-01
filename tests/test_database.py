import tempfile
from datetime import datetime
from pathlib import Path

from benchmark.models.database import Database
from benchmark.models.schemas import ApiCallMetrics


def _test_db() -> Database:
    """创建临时数据库。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db = Database(db_path=Path(tmp.name))
    return db


def test_save_and_query_metrics():
    db = _test_db()
    now = datetime.now()
    metrics = ApiCallMetrics(
        result_id="r001",
        prompt_tokens=100,
        completion_tokens=50,
        duration=2.0,
        tokens_per_second=25.0,
        created_at=now,
    )
    db.save_metrics(metrics)

    conn = db._get_conn()
    row = conn.execute(
        "SELECT * FROM api_call_metrics WHERE result_id = ?", ("r001",)
    ).fetchone()
    cols = [d[0] for d in conn.execute("SELECT * FROM api_call_metrics WHERE result_id = ?", ("r001",)).description]
    result = dict(zip(cols, row))

    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 50
    assert result["duration"] == 2.0
    assert result["tokens_per_second"] == 25.0
    db.close()
