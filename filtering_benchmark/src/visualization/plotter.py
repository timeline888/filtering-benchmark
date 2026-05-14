"""Visualization module for evaluation results."""

import os
import numpy as np
from loguru import logger


class Plotter:
    """Generate plots for signal denoising evaluation."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._matplotlib_available = False
        self._import_attempted = False

    def _ensure_imports(self):
        if self._import_attempted:
            return self._matplotlib_available
        self._import_attempted = True
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self.plt = plt
            self._matplotlib_available = True
        except ImportError:
            logger.warning("matplotlib not available, plots will be skipped")
            self._matplotlib_available = False
        return self._matplotlib_available

    def plot_time_domain(self, signals_dict, sample_rate, title_prefix=""):
        """Plot time-domain waveforms for multiple signals."""
        if not self._ensure_imports():
            return None
        n_signals = len(signals_dict)
        fig, axes = self.plt.subplots(n_signals, 1, figsize=(12, 3 * n_signals),
                                       squeeze=False)
        t = np.arange(len(list(signals_dict.values())[0])) / sample_rate
        for i, (name, signal) in enumerate(signals_dict.items()):
            ax = axes[i][0]
            ax.plot(t, signal, linewidth=0.5)
            ax.set_title(f"{title_prefix}{name}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
        self.plt.tight_layout()
        path = os.path.join(self.output_dir, "time_domain_comparison.png")
        fig.savefig(path, dpi=150)
        self.plt.close(fig)
        logger.info(f"Time domain plot saved: {path}")
        return path

    def plot_envelope_spectra(self, spectra_dict, sample_rate, title_prefix="",
                              max_freq=None):
        """Plot envelope spectra comparison for multiple algorithms."""
        if not self._ensure_imports():
            return None
        n_spectra = len(spectra_dict)
        if n_spectra == 0:
            return None
        n_cols = min(3, n_spectra)
        n_rows = (n_spectra + n_cols - 1) // n_cols
        fig, axes = self.plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows),
                                       squeeze=False)
        freq_ax = np.fft.rfftfreq(len(list(spectra_dict.values())[0]), 1 / sample_rate)
        if max_freq:
            freq_limit = max_freq
        else:
            freq_limit = freq_ax[-1] / 2
        for i, (name, spectrum) in enumerate(spectra_dict.items()):
            row, col = i // n_cols, i % n_cols
            ax = axes[row][col]
            mask = freq_ax <= freq_limit
            ax.plot(freq_ax[mask], spectrum[mask], linewidth=0.6)
            ax.set_title(f"{title_prefix}{name}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
        for i in range(n_spectra, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row][col].set_visible(False)
        self.plt.tight_layout()
        path = os.path.join(self.output_dir, "envelope_spectra_comparison.png")
        fig.savefig(path, dpi=150)
        self.plt.close(fig)
        logger.info(f"Envelope spectra plot saved: {path}")
        return path

    def plot_metrics_bar(self, rankings, metric_names=None, top_n=10):
        """Plot bar chart of composite scores for top-N algorithms."""
        if not self._ensure_imports():
            return None
        rankings = sorted(rankings, key=lambda r: r.overall_score, reverse=True)[:top_n]
        names = [r.algorithm_name[:20] for r in rankings]
        scores = [r.overall_score for r in rankings]

        fig, ax = self.plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
        colors = self.plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = ax.barh(range(len(names)), scores, color=colors[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Composite Score")
        ax.set_title("Algorithm Ranking by Composite Score")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va="center", fontsize=8)

        self.plt.tight_layout()
        path = os.path.join(self.output_dir, "algorithm_ranking.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        self.plt.close(fig)
        logger.info(f"Ranking bar chart saved: {path}")
        return path

    def plot_radar(self, rankings, metric_names, top_n=5):
        """Plot radar chart comparing top-N algorithms across key metrics."""
        if not self._ensure_imports():
            return None
        rankings = sorted(rankings, key=lambda r: r.overall_score, reverse=True)[:top_n]
        n_metrics = len(metric_names)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = self.plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = self.plt.cm.tab10(np.linspace(0, 1, len(rankings)))

        for i, ranking in enumerate(rankings):
            values = [ranking.metric_scores.get(m, 0) for m in metric_names]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=1.5, label=ranking.algorithm_name[:15],
                    color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=9)
        ax.set_title("Algorithm Comparison Radar", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        self.plt.tight_layout()
        path = os.path.join(self.output_dir, "algorithm_radar.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        self.plt.close(fig)
        logger.info(f"Radar chart saved: {path}")
        return path
