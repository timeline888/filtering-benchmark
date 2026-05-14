"""
系统常量与枚举定义。
"""

from enum import Enum, auto


class FaultFrequencyType(str, Enum):
    """故障特征频率类型"""
    BPFO = "bpfo"  # 外圈故障频率
    BPFI = "bpfi"  # 内圈故障频率
    BSF = "bsf"    # 滚动体故障频率
    FTF = "ftf"    # 保持架故障频率
    GMF = "gmf"    # 齿轮啮合频率
    FR = "fr"      # 转频


class MetricCategory(str, Enum):
    """指标分类"""
    TIME_DOMAIN = "时域指标"
    FREQ_DOMAIN = "频域指标"
    ENVELOPE = "包络域指标"
    PERCEPTUAL = "感知指标"
    COMPUTATIONAL = "计算效率指标"


class ExecutionBackend(str, Enum):
    """并行执行后端"""
    SEQUENTIAL = "sequential"
    MULTIPROCESSING = "multiprocessing"
    RAY = "ray"


class SignalFormat(str, Enum):
    """信号文件格式"""
    CSV = "csv"
    MAT = "mat"
    TDMS = "tdms"
    UFF = "uff"
    WAV = "wav"
    HDF5 = "hdf5"
    TXT = "txt"


class InputMode(str, Enum):
    """故障频率输入模式"""
    MANUAL = "manual"
    AUTO = "auto"


class ScoringMethod(str, Enum):
    """评分方法"""
    WEIGHTED_SUM = "weighted_sum"
    TOPSIS = "topsis"
    BORDA_COUNT = "borda_count"
    PARETO = "pareto"


class NormalizationMethod(str, Enum):
    """归一化方法"""
    MINMAX = "minmax"
    ZSCORE = "zscore"
    RANK = "rank"


# 默认综合评分权重
DEFAULT_METRIC_WEIGHTS = {
    "ffr": 0.20,      # 故障特征比
    "lsnr": 0.20,     # 局部信噪比
    "her": 0.15,      # 谐波能量占比
    "esk": 0.15,      # 包络谱峰度
    "gi": 0.15,       # 基尼指数
    "execution_time": 0.10,  # 执行时间
    "other": 0.05,    # 其他指标
}

# 默认的轴承型号参数库
BEARING_PARAMETERS = {
    "6205": {  # SKF 6205 深沟球轴承
        "pitch_diameter": 39.04,   # 节圆直径 (mm)
        "roller_diameter": 7.94,   # 滚动体直径 (mm)
        "n_rollers": 9,            # 滚动体数量
        "contact_angle": 0,        # 接触角 (度)
    },
    "6203": {
        "pitch_diameter": 28.50,
        "roller_diameter": 6.75,
        "n_rollers": 8,
        "contact_angle": 0,
    },
    "NU205": {
        "pitch_diameter": 38.50,
        "roller_diameter": 7.50,
        "n_rollers": 12,
        "contact_angle": 0,
    },
}

# 时间单位常量
SECONDS_PER_MINUTE = 60.0

# 默认算法超时配置（秒）
DEFAULT_TIMEOUT = {
    "低": 30,
    "中": 120,
    "高": 600,
}
