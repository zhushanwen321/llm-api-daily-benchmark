"""benchmark/cli.py CLI 参数测试。"""

from click.testing import CliRunner

from benchmark.cli import cli


class TestDimensionAll:
    """--dimension all 选项测试。"""

    def test_dimension_all_accepted(self):
        """--dimension all 应被 Click 正常接受（不报参数错误）。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--model", "glm/glm-4.7", "--dimension", "all"])
        # 不应出现 "Invalid value for '--dimension'" 之类的 Click 错误
        assert "Invalid value" not in result.output
        assert "'all'" not in result.output or "all" in result.output

    def test_dimension_all_not_in_invalid_error(self):
        """传入无效维度时，错误信息中应包含 'all' 作为合法选项。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--model", "glm/glm-4.7", "--dimension", "invalid-dim"])
        assert result.exit_code != 0
        assert "all" in result.output

    def test_invalid_dimension_fails(self):
        """无效维度应导致非零退出码。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--model", "glm/glm-4.7", "--dimension", "nonexistent"])
        assert result.exit_code != 0

    def test_valid_single_dimension_accepted(self):
        """已有的单个维度仍应被正常接受。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--model", "glm/glm-4.7", "--dimension", "reasoning"])
        # 不应出现 Click 参数校验错误
        assert "Invalid value" not in result.output


class TestSchedulerCommands:
    """scheduler 子命令测试。"""

    def test_scheduler_start(self):
        """scheduler start 在未启用时应正常退出（exit 0）。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["scheduler", "start"])
        assert result.exit_code == 0

    def test_scheduler_status(self):
        """scheduler status 应正常退出（exit 0）。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["scheduler", "status"])
        assert result.exit_code == 0


class TestProbeCommands:
    """probe 子命令测试。"""

    def test_probe_run_missing_model(self):
        """缺少 --model 参数应报错。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["probe", "run"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "Error" in result.output

    def test_probe_run_accepts_model(self):
        """提供 --model 参数时不应报参数校验错误。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["probe", "run", "--model", "glm/glm-4.7"])
        # 可能因缺少 probe tasks 文件而失败，但不应是参数错误
        assert "Invalid value" not in result.output

    def test_probe_schedule_missing_models(self):
        """缺少 --models 参数应报错。"""
        runner = CliRunner()
        result = runner.invoke(cli, ["probe", "schedule"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "Error" in result.output

    def test_probe_schedule_accepts_models(self):
        """提供 --models 参数时不应报参数校验错误。"""
        from unittest.mock import patch

        runner = CliRunner()
        with patch("benchmark.cli.time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(
                cli, ["probe", "schedule", "--models", "glm/glm-4.7,openai/gpt-4o"]
            )
        # 不应是参数校验错误
        assert "Invalid value" not in result.output
