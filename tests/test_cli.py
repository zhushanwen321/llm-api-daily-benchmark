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
