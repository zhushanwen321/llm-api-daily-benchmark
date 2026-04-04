from benchmark.adapters.mmlu_pro_adapter import MMLUProAdapter


def test_mmlu_pro_loads_tasks():
    adapter = MMLUProAdapter()
    tasks = adapter.load()

    assert len(tasks) == 15
    assert tasks[0].dimension == "system-architecture"
    assert tasks[0].dataset == "mmlu-pro"
    assert tasks[0].expected_output  # 期望字母不为空
    assert len(tasks[0].expected_output) == 1  # 单个字母


def test_mmlu_pro_prompt_has_options():
    adapter = MMLUProAdapter()
    tasks = adapter.load()
    for task in tasks:
        assert "A." in task.prompt or "B." in task.prompt
        assert "Answer with the letter" in task.prompt


def test_mmlu_pro_covers_multiple_categories():
    adapter = MMLUProAdapter()
    tasks = adapter.load()
    categories = {t.metadata["category"] for t in tasks}
    assert len(categories) >= 2


def test_mmlu_pro_validate():
    adapter = MMLUProAdapter()
    tasks = adapter.load()
    for task in tasks:
        assert adapter.validate(task)


def test_mmlu_pro_get_dimension():
    adapter = MMLUProAdapter()
    assert adapter.get_dimension() == "system-architecture"
