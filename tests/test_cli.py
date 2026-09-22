"""Unit tests for the Click CLI."""

from click.testing import CliRunner
from laya_as_judge.cli import main


def test_cli_demo():
    runner = CliRunner()
    result = runner.invoke(main, ["demo"])
    assert result.exit_code == 0
    assert "Laya-as-a-Judge Interactive Demo" in result.output


def test_cli_eval():
    runner = CliRunner()
    result = runner.invoke(main, ["eval", "--judge", "safety", "--input", "Hello world"])
    assert result.exit_code == 0
    assert "SafetyGuardJudge" in result.output


def test_cli_benchmark():
    runner = CliRunner()
    result = runner.invoke(main, ["benchmark", "--count", "5"])
    assert result.exit_code == 0
    assert "Benchmark Comparison Results" in result.output
