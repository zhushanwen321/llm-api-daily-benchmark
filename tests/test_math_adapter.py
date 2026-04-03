from benchmark.adapters.math_adapter import MATHAdapter


def test_math_adapter_loads_tasks():
    adapter = MATHAdapter()
    tasks = adapter.load()

    assert len(tasks) == 15
    assert tasks[0].dimension == "reasoning"
    assert tasks[0].dataset == "math"
    assert tasks[0].expected_output  # answer 不为空
    assert tasks[0].metadata["level"] >= 3
    assert tasks[0].metadata["subject"]


def test_math_adapter_covers_multiple_subjects():
    adapter = MATHAdapter()
    tasks = adapter.load()
    subjects = {t.metadata["subject"] for t in tasks}
    assert len(subjects) >= 4  # 至少覆盖 4 个学科


def test_math_adapter_includes_hard_levels():
    adapter = MATHAdapter()
    tasks = adapter.load()
    levels = [t.metadata["level"] for t in tasks]
    assert max(levels) >= 4  # 包含 Level 4 或 5


def test_math_adapter_validate():
    adapter = MATHAdapter()
    tasks = adapter.load()
    for task in tasks:
        assert adapter.validate(task)


def test_math_adapter_get_dimension():
    adapter = MATHAdapter()
    assert adapter.get_dimension() == "reasoning"
