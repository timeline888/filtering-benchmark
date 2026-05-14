"""
matplotlib 画布组件 - 嵌入 PyQt6。

提供多种图表绘制方法供结果展示组件调用。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ChartCanvas(FigureCanvasQTAgg):
    """通用图表画布"""

    def __init__(self, parent=None, width: int = 6, height: int = 4, dpi: int = 100):
        # 确保中文字体可渲染
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False

        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)

    def clear(self):
        self.figure.clear()

    def draw_ranking_bar(self, names: List[str], scores: List[float], title: str = "算法排名"):
        """绘制排名柱状图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        colors = plt_cm_viridis([max(0.3, min(1.0, s / max(scores))) for s in scores]) if scores else []
        bars = ax.barh(range(len(names)), scores, color=colors if len(colors) > 0 else '#2ecc71')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("综合得分")
        ax.set_title(title)
        ax.invert_yaxis()
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.01 * max(scores), bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va='center')
        self.figure.tight_layout()
        self.draw()

    def draw_radar(self, categories: List[str], values_dict: Dict[str, List[float]],
                   title: str = "算法多指标对比"):
        """绘制雷达图"""
        self.figure.clear()
        n_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        ax = self.figure.add_subplot(111, polar=True)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        for idx, (name, values) in enumerate(values_dict.items()):
            if len(values) != n_cats:
                continue
            values_closed = values + values[:1]
            ax.plot(angles, values_closed, 'o-', linewidth=1.5,
                    label=name, color=colors[idx % len(colors)])
            ax.fill(angles, values_closed, alpha=0.1, color=colors[idx % len(colors)])

        ax.set_title(title, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        self.figure.tight_layout()
        self.draw()

    def draw_metric_comparison(self, metric_names: List[str],
                                algorithms: Dict[str, List[float]],
                                title: str = "指标对比"):
        """绘制指标对比柱状图（分组）"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        n_groups = len(metric_names)
        n_algo = len(algorithms)
        if n_groups == 0 or n_algo == 0:
            self.draw()
            return

        index = np.arange(n_groups)
        bar_width = 0.8 / n_algo
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']

        for idx, (name, values) in enumerate(algorithms.items()):
            offset = (idx - n_algo / 2 + 0.5) * bar_width
            bars = ax.bar(index + offset, values, bar_width,
                         label=name, color=colors[idx % len(colors)])

        ax.set_xticks(index)
        ax.set_xticklabels(metric_names, fontsize=8, rotation=45, ha='right')
        ax.set_title(title)
        ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.draw()

    def draw_waveform(self, time: np.ndarray, data_dict: Dict[str, np.ndarray],
                      title: str = "波形对比", xlabel: str = "时间 (s)", ylabel: str = "幅值"):
        """绘制波形对比图"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        colors = ['#2c3e50', '#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        for idx, (name, signal) in enumerate(data_dict.items()):
            ax.plot(time, signal, linewidth=0.5, label=name,
                    color=colors[idx % len(colors)])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.draw()


def plt_cm_viridis(values):
    """viridis colormap approximation"""
    try:
        from matplotlib import colormaps
        return colormaps['viridis'](values)
    except (ImportError, KeyError):
        import matplotlib.cm as cm
        return cm.viridis(values)
