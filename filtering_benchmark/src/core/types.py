"""
核心数据类型定义。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AlgorithmCategory(str, Enum):
    """算法分类枚举"""
    CLASSICAL_FILTER = "经典滤波方法"
    TIME_DOMAIN = "时域降噪方法"
    FREQ_DOMAIN = "频域降噪方法"
    WAVELET = "小波变换类"
    EMD = "经验模态分解类"
    ADAPTIVE = "自适应滤波"
    SVD = "奇异值分解类"
    SPARSE = "稀疏表示类"
    BSS = "盲源分离类"
    DEEP_LEARNING = "深度学习类"
    ROTATING_SPECIFIC = "旋转机械专用"
    ACOUSTIC = "语音/声学降噪"
    ADVANCED = "其他前沿方法"


class AlgorithmComplexity(str, Enum):
    """算法复杂度"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"


class ResourceConsumption(str, Enum):
    """资源消耗等级"""
    VERY_LOW = "极低"
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"
    VERY_HIGH = "极高"


@dataclass
class ComplexityAnalysis:
    """算法复杂度详细分析结果"""
    algorithm_id: str
    algorithm_name: str
    category: AlgorithmCategory
    complexity: AlgorithmComplexity
    resource_level: ResourceConsumption = ResourceConsumption.MEDIUM
    time_complexity: str = ""  # 时间复杂度描述，如 "O(N)", "O(N²)"
    space_complexity: str = ""  # 空间复杂度描述
    estimated_time_per_10k: str = ""  # 每10000点信号预估耗时
    cpu_usage: str = ""  # CPU使用评估
    memory_usage: str = ""  # 内存使用评估
    optimization_tips: str = ""  # 优化建议
    bottlenecks: str = ""  # 主要性能瓶颈


class SignalDomain(str, Enum):
    """信号域"""
    TIME_DOMAIN = "time"
    FREQ_DOMAIN = "freq"
    TIME_FREQ = "time_freq"


@dataclass
class SignalData:
    """统一信号数据结构"""
    data: np.ndarray  # shape: (n_channels, n_samples) 或 (n_samples,)
    sample_rate: float  # 采样率 (Hz)
    channel_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None
    unit: str = ""

    def __post_init__(self):
        if self.data.ndim == 1:
            self.data = self.data.reshape(1, -1)
        if not self.channel_names:
            n_chan = self.data.shape[0]
            self.channel_names = [f"ch_{i}" for i in range(n_chan)]

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate

    def get_channel(self, idx: int = 0) -> np.ndarray:
        return self.data[idx]

    def select_channels(self, indices: List[int]) -> "SignalData":
        return SignalData(
            data=self.data[indices],
            sample_rate=self.sample_rate,
            channel_names=[self.channel_names[i] for i in indices],
            metadata=self.metadata,
            source_path=self.source_path,
            unit=self.unit,
        )


@dataclass
class DenoisedResult:
    """降噪结果"""
    algorithm_id: str
    algorithm_name: str
    category: AlgorithmCategory
    denoised_signal: np.ndarray  # shape: (n_channels, n_samples)
    execution_time: float  # 耗时 (秒)
    params: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EnvelopeResult:
    """包络谱分析结果"""
    algorithm_id: str
    envelope_signal: np.ndarray  # 包络时域信号
    envelope_spectrum: np.ndarray  # 包络谱幅值
    freq_axis: np.ndarray  # 频率轴
    dominant_freqs: List[Tuple[float, float]] = field(default_factory=list)  # (freq, amplitude)
    fault_freq_matches: Dict[str, float] = field(default_factory=dict)  # {freq_name: amplitude}


@dataclass
class MetricResult:
    """单指标计算结果"""
    metric_name: str
    value: float
    higher_is_better: bool
    category: str = ""
    normalized_value: Optional[float] = None


@dataclass
class AlgorithmRanking:
    """算法排名"""
    algorithm_id: str
    algorithm_name: str
    category: AlgorithmCategory
    overall_score: float
    overall_rank: int
    metric_scores: Dict[str, float] = field(default_factory=dict)
    metric_ranks: Dict[str, int] = field(default_factory=dict)
    execution_time: float = 0.0


@dataclass
class EvaluationReport:
    """完整评估报告"""
    project_name: str
    signal_info: Dict[str, Any]
    rankings: List[AlgorithmRanking]
    top_n: int = 3
    # 保存 TOP-N 算法的降噪后信号（用于 GUI 信号预览对比）
    top_denoised_results: Dict[str, DenoisedResult] = field(default_factory=dict)

    @property
    def top_algorithms(self) -> List[AlgorithmRanking]:
        return self.rankings[:self.top_n]
