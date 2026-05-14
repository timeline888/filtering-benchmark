import os
import datetime
from loguru import logger


class ReportGenerator:
    """Generate evaluation reports in markdown format."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_markdown(self, report) -> str:
        """Generate a full markdown evaluation report."""
        lines = []
        self._render_header(report, lines)
        self._render_config(report, lines)
        self._render_top3(report, lines)
        self._render_all_results(report, lines)

        safe_name = str(report.signal_info.get("path", "signal")).replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}_evaluation_report.md"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("//n".join(lines))
        logger.info(f"Report saved: {path}")
        return path

    def _render_header(self, report, lines):
        lines.append("# Signal Filtering & Denoising Evaluation Report")
        lines.append("")
        signal_path = report.signal_info.get("path", "N/A")
        sample_rate = report.signal_info.get("sample_rate", "N/A")
        n_channels = report.signal_info.get("channels", "N/A")
        duration = report.signal_info.get("duration", "N/A")
        lines.append(f"- **Signal**: {signal_path}")
        lines.append(f"- **Sample Rate**: {sample_rate} Hz")
        if duration != "N/A":
            lines.append(f"- **Duration**: {duration:.3f} s")
        lines.append(f"- **Channels**: {n_channels}")
        lines.append(f"- **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("---")
        lines.append("")

    def _render_config(self, report, lines):
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"- **Algorithms Tested**: {len(report.rankings)}")
        lines.append(f"- **Top-N Output**: {report.top_n}")
        if report.rankings:
            metric_names = list(report.rankings[0].metric_scores.keys())
            lines.append(f"- **Metrics Used**: {', '.join(metric_names)}")
        lines.append("")
        lines.append("---")
        lines.append("")

    def _render_top3(self, report, lines):
        lines.append("## Top-3 Recommended Algorithms")
        lines.append("")
        lines.append("| Rank | Algorithm | Category | Score | Key Strengths |")
        lines.append("|------|-----------|----------|-------|---------------|")
        rankings = sorted(report.rankings,
                         key=lambda r: r.overall_score, reverse=True)
        for i, ranking in enumerate(rankings[:3], 1):
            strengths = self._get_strengths(ranking)
            cat = ranking.category.value if ranking.category else "N/A"
            lines.append(f"| {i} | {ranking.algorithm_name} | {cat} | {ranking.overall_score:.4f} | {strengths} |")
        lines.append("")
        lines.append("> The top-3 algorithms are selected based on combined weighted scoring across all evaluation metrics.")
        lines.append("")

    def _render_all_results(self, report, lines):
        lines.append("## Full Algorithm Ranking")
        lines.append("")
        lines.append("| Rank | Algorithm | Category | Score | FFA | RFFI | FBR | Kurtosis | GI |")
        lines.append("|------|-----------|----------|-------|-----|------|-----|----------|-----|")
        rankings = sorted(report.rankings,
                         key=lambda r: r.overall_score, reverse=True)
        for i, ranking in enumerate(rankings, 1):
            cat = ranking.category.value if ranking.category else "N/A"
            ms = ranking.metric_scores
            ffa = f"{ms.get('ffa', 0):.3f}" if isinstance(ms, dict) else "N/A"
            rffi = f"{ms.get('rffi', 0):.3f}" if isinstance(ms, dict) else "N/A"
            fbr = f"{ms.get('fbr', 0):.3f}" if isinstance(ms, dict) else "N/A"
            kurt = f"{ms.get('kurtosis', 0):.3f}" if isinstance(ms, dict) else "N/A"
            gi_val = ms.get('gini_index', ms.get('gi', 0)) if isinstance(ms, dict) else 0
            gi = f"{gi_val:.3f}"
            lines.append(f"| {i} | {ranking.algorithm_name} | {cat} | {ranking.overall_score:.4f} | {ffa} | {rffi} | {fbr} | {kurt} | {gi} |")
        lines.append("")

    def _get_strengths(self, ranking):
        if not hasattr(ranking, 'metric_scores') or not isinstance(ranking.metric_scores, dict):
            return "Balanced performance"
        ms = ranking.metric_scores
        strengths = []
        if ms.get('ffa', 0) > 0.8:
            strengths.append("High FFA")
        if ms.get('kurtosis', 0) > 5:
            strengths.append("Strong impulsiveness")
        gi_val = ms.get('gini_index', ms.get('gi', 0))
        if gi_val > 0.7:
            strengths.append("High sparsity")
        if ms.get('rffi', 0) > 0.7:
            strengths.append("Good feature enhancement")
        return ", ".join(strengths) if strengths else "Balanced"
