"""测试动态探针生成器."""

import unittest
from datetime import datetime

from benchmark.probes.dynamic.probe_generator import (
    DynamicProbeGenerator,
    GeneratedProbe,
    ProbeTemplate,
    ProbeTemplateLibrary,
    VariationStrategy,
)


class TestProbeTemplate(unittest.TestCase):
    """测试探针模板."""

    def test_template_creation(self):
        template = ProbeTemplate(
            template_id="test_template",
            category="safety",
            base_prompt="测试{topic}的{aspect}",
            variation_rules=[
                {"type": "topic", "values": ["AI", "ML"]},
                {"type": "aspect", "values": ["安全性", "性能"]},
            ],
            expected_patterns=["测试"],
            tags=["test"],
        )

        self.assertEqual(template.template_id, "test_template")
        self.assertEqual(template.category, "safety")
        self.assertEqual(len(template.variation_rules), 2)


class TestProbeTemplateLibrary(unittest.TestCase):
    """测试探针模板库."""

    def setUp(self):
        self.library = ProbeTemplateLibrary()

    def test_default_templates_loaded(self):
        templates = self.library.list_templates()
        self.assertGreater(len(templates), 0)

    def test_get_template(self):
        template = self.library.get_template("safety_boundary")
        self.assertIsNotNone(template)
        self.assertEqual(template.template_id, "safety_boundary")

    def test_list_templates_by_category(self):
        safety_templates = self.library.list_templates("safety")
        self.assertEqual(len(safety_templates), 1)
        self.assertEqual(safety_templates[0].category, "safety")

    def test_add_template(self):
        new_template = ProbeTemplate(
            template_id="new_test",
            category="fingerprint",
            base_prompt="测试",
        )
        self.library.add_template(new_template)

        retrieved = self.library.get_template("new_test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.template_id, "new_test")


class TestVariationStrategy(unittest.TestCase):
    """测试变异策略."""

    def setUp(self):
        self.template = ProbeTemplate(
            template_id="test",
            category="safety",
            base_prompt="测试{topic}的{aspect}",
            variation_rules=[
                {"type": "topic", "values": ["AI", "ML"]},
                {"type": "aspect", "values": ["安全性"]},
            ],
        )
        self.strategy = VariationStrategy()

    def test_apply_template_variation(self):
        prompt = self.strategy.apply_template_variation(self.template, seed=0)
        self.assertNotIn("{", prompt)
        self.assertTrue("AI" in prompt or "ML" in prompt)

    def test_generate_variations(self):
        variations = self.strategy.generate_variations(self.template, count=3)
        self.assertEqual(len(variations), 3)
        self.assertGreaterEqual(len(set(variations)), 2)

    def test_semantic_variation_rephrase(self):
        result = self.strategy.semantic_variation("请说明问题", "rephrase")
        self.assertNotEqual(result, "请说明问题")
        self.assertIn("能否", result)  # "请" 应该被替换为 "能否"

    def test_semantic_variation_expand(self):
        result = self.strategy.semantic_variation("测试", "expand")
        self.assertIn("详细", result)
        self.assertIn("具体例子", result)

    def test_semantic_variation_simplify(self):
        result = self.strategy.semantic_variation("详细测试具体说明", "simplify")
        self.assertNotIn("详细", result)
        self.assertNotIn("具体", result)


class TestDynamicProbeGenerator(unittest.TestCase):
    """测试动态探针生成器."""

    def setUp(self):
        self.generator = DynamicProbeGenerator()

    def test_generate_from_template(self):
        probes = self.generator.generate_from_template("safety_boundary", count=3)

        self.assertEqual(len(probes), 3)
        for probe in probes:
            self.assertIsInstance(probe, GeneratedProbe)
            self.assertTrue(probe.probe_id.startswith("safety_boundary"))
            self.assertEqual(probe.category, "safety")
            self.assertGreater(len(probe.prompt), 0)

    def test_generate_from_nonexistent_template(self):
        probes = self.generator.generate_from_template("nonexistent", count=3)
        self.assertEqual(len(probes), 0)

    def test_generate_by_category(self):
        probes = self.generator.generate_by_category("safety", probes_per_template=2)

        self.assertGreater(len(probes), 0)
        for probe in probes:
            self.assertEqual(probe.category, "safety")

    def test_evaluate_probe_effectiveness(self):
        # 先创建一些探针
        probes = self.generator.generate_from_template("safety_boundary", count=1)
        probe_id = probes[0].probe_id

        # 模拟测试结果
        test_results = [
            {"passed": True, "response": "回答1"},
            {"passed": True, "response": "回答2"},
            {"passed": False, "response": "回答3"},
        ]

        score = self.generator.evaluate_probe_effectiveness(probe_id, test_results)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_evaluate_probe_effectiveness_empty_results(self):
        probes = self.generator.generate_from_template("safety_boundary", count=1)
        probe_id = probes[0].probe_id

        score = self.generator.evaluate_probe_effectiveness(probe_id, [])
        self.assertEqual(score, 0.0)

    def test_select_effective_probes(self):
        # 创建探针并评估
        probes = self.generator.generate_from_template("safety_boundary", count=3)

        # 为每个探针设置不同的效果分数
        for i, probe in enumerate(probes):
            test_results = [{"passed": i % 2 == 0, "response": f"response{i}"}]
            self.generator.evaluate_probe_effectiveness(probe.probe_id, test_results)

        # 选择有效的探针
        effective = self.generator.select_effective_probes(threshold=0.0, max_count=2)
        self.assertLessEqual(len(effective), 2)

    def test_to_task_definition(self):
        probes = self.generator.generate_from_template("safety_boundary", count=1)
        probe = probes[0]

        task_def = self.generator.to_task_definition(probe)

        self.assertEqual(task_def.task_id, probe.probe_id)
        self.assertEqual(task_def.dimension, "dynamic_probe")
        self.assertEqual(task_def.dataset, probe.category)
        self.assertEqual(task_def.prompt, probe.prompt)

    def test_get_generation_stats(self):
        # 生成一些探针
        self.generator.generate_from_template("safety_boundary", count=2)
        self.generator.generate_from_template("instruction_following", count=2)

        stats = self.generator.get_generation_stats()

        self.assertEqual(stats["total_generated"], 4)
        self.assertEqual(stats["by_category"]["safety"], 2)
        self.assertEqual(stats["by_category"]["fingerprint"], 2)
        self.assertGreater(stats["templates_available"], 0)


if __name__ == "__main__":
    unittest.main()
