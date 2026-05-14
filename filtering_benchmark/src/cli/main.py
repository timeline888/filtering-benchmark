"""
命令行接口主入口。

提供 filter-bench CLI 命令。
"""

from typing import Optional

import typer

from ..config import ConfigLoader
from ..pipeline.orchestrator import PipelineOrchestrator

app = typer.Typer(
    name="filter-bench",
    help="旋转机械振动/声学信号滤波降噪评估系统",
)


@app.command()
def run(
    signal: str = typer.Option("", "--signal", "-s",
                                help="信号文件路径或使用'simulate'生成仿真信号"),
    sample_rate: float = typer.Option(None, "--sample-rate", "-fs",
                                       help="采样率 (Hz)"),
    config: str = typer.Option(None, "--config", "-c",
                                help="YAML配置文件路径"),
    output_dir: str = typer.Option("results", "--output-dir", "-o",
                                    help="输出目录"),
    categories: str = typer.Option(None, "--categories", "-cat",
                                    help="算法类别过滤，逗号分隔"),
    top_n: int = typer.Option(3, "--top-n", "-n",
                               help="输出TOP-N算法"),
    env: str = typer.Option("development", "--env", "-e",
                             help="环境配置"),
):
    """运行滤波降噪算法评估"""
    # 加载配置
    loader = ConfigLoader(env=env)

    # CLI参数覆盖
    cli_overrides = {}
    if signal:
        cli_overrides["signal"] = {"source": {"path": signal}}
        if signal.lower() == "simulate":
            cli_overrides["signal"]["source"] = {
                "simulate_reference": True,
                "sample_rate": sample_rate or 12000,
            }
    if sample_rate:
        cli_overrides.setdefault("signal", {}).setdefault("source", {})["sample_rate"] = sample_rate
    if categories:
        cli_overrides["algorithms"] = {
            "selection": {
                "mode": "category",
                "categories": categories.split(","),
            }
        }
    if top_n:
        cli_overrides["scoring"] = {"top_n": top_n}

    app_config = loader.load(user_config_path=config, cli_overrides=cli_overrides)

    # 覆盖输出目录
    if output_dir != "results":
        app_config.output.export.output_dir = output_dir

    # 执行流水线
    pipeline = PipelineOrchestrator(app_config)
    report = pipeline.run()

    # 生成可视化报告
    try:
        from ..visualization.plotter import Plotter
        from ..visualization.reporter import ReportGenerator

        plotter = Plotter(output_dir)
        plotter.plot_metrics_bar(report.rankings, top_n=top_n or 10)
        # Note: envelope spectra not stored in EvaluationReport; skipped for now
        pass

        reporter = ReportGenerator(output_dir)
        report_path = reporter.generate_markdown(report)
        typer.echo(f"\n报告已生成: {report_path}")
    except Exception as e:
        typer.echo(f"可视化生成跳过: {e}")

    # 打印结果
    typer.echo("\n=== 评估结果 ===")
    typer.echo(f"项目: {report.project_name}")
    typer.echo(f"信号: {report.signal_info.get('path', 'N/A')}")
    typer.echo(f"\nTOP-{report.top_n} 算法排名:")
    typer.echo("-" * 60)
    top_rankings = sorted(report.rankings, key=lambda r: r.overall_score, reverse=True)[:report.top_n]
    for rank in top_rankings:
        typer.echo(f"  #{rank.overall_rank}: {rank.algorithm_name}")
        typer.echo(f"     分类: {rank.category.value if hasattr(rank.category, 'value') else rank.category}")
        typer.echo(f"     综合得分: {rank.overall_score:.4f}")
        typer.echo(f"     执行时间: {rank.execution_time:.3f}s")

    return report


@app.command()
def list_algorithms(
    category: str = typer.Option(None, "--category", "-cat",
                                  help="按类别过滤"),
):
    """列出所有可用的滤波降噪算法"""
    from ..algorithms import registry

    registry.ensure_discovered()

    if category:
        algos = registry.filter(category=category)
    else:
        algos = registry.list_algorithms()

    # 按分类分组
    from collections import defaultdict
    grouped = defaultdict(list)
    for meta in algos:
        grouped[meta.category.value].append(meta)

    typer.echo(f"\n可用算法: {registry.count()} 种\n")
    for cat_name, items in sorted(grouped.items()):
        typer.echo(f"  [{cat_name}] ({len(items)}种)")
        for meta in items:
            gpu_mark = " [GPU]" if meta.requires_gpu else ""
            typer.echo(f"    - {meta.algorithm_id:35s} {meta.name}{gpu_mark}")
        typer.echo()


@app.command()
def init():
    """在当前目录初始化配置模板"""
    import os
    import shutil

    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_dir = os.path.join(src_dir, "config")
    target_dir = os.path.join(os.getcwd(), "config")

    os.makedirs(target_dir, exist_ok=True)
    default_cfg = os.path.join(config_dir, "default.yaml")
    if os.path.exists(default_cfg):
        shutil.copy(default_cfg, os.path.join(target_dir, "project_config.yaml"))
        typer.echo(f"配置文件模板已创建: {os.path.join(target_dir, 'project_config.yaml')}")
    else:
        typer.echo("警告: 默认配置模板未找到")


@app.command()
def compare(
    run1: str = typer.Argument(..., help="第一次评估结果目录"),
    run2: str = typer.Argument(..., help="第二次评估结果目录"),
):
    """对比两次评估结果"""
    import json
    import os

    # 简化：读取结果中的排名数据
    typer.echo(f"对比 {run1} 和 {run2}")
    typer.echo("此功能需要先完成输出模块，暂未实现完整对比逻辑")


if __name__ == "__main__":
    app()
