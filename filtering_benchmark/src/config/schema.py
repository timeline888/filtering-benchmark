"""
Pydantic 配置模型定义。

所有配置校验逻辑集中在 schema.py，确保类型安全。
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# 信号配置
# ============================================================

class PreprocessingConfig(BaseModel):
    """预处理配置"""
    remove_dc: bool = True
    resample: Optional[float] = None  # 重采样到指定采样率，None表示不重采样
    normalize: bool = False
    segment_duration_sec: Optional[float] = None
    segment_overlap: float = 0.0


class SignalSourceConfig(BaseModel):
    """信号源配置"""
    path: str = ""
    sample_rate: Optional[float] = None  # 覆盖自动检测
    channel: int = 0  # 使用第几个通道
    reference_signal_path: Optional[str] = None  # 参考信号路径
    simulate_reference: bool = False  # 是否仿真生成参考信号


class SignalConfig(BaseModel):
    """信号整体配置"""
    source: SignalSourceConfig = Field(default_factory=SignalSourceConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


# ============================================================
# 算法配置
# ============================================================

class AlgorithmSelectionConfig(BaseModel):
    """算法选择配置"""
    mode: str = "all"  # all | selected | category
    categories: List[str] = Field(default_factory=list)
    selected_ids: List[str] = Field(default_factory=list)
    param_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("mode")
    def validate_mode(cls, v: str) -> str:
        allowed = {"all", "selected", "category"}
        if v not in allowed:
            raise ValueError(f"mode 必须是 {allowed} 之一，得到 '{v}'")
        return v


class AlgorithmConfig(BaseModel):
    """算法配置"""
    selection: AlgorithmSelectionConfig = Field(default_factory=AlgorithmSelectionConfig)


# ============================================================
# 包络谱配置
# ============================================================

class BandpassConfig(BaseModel):
    """带通滤波配置（包络分析前的可选前置滤波）"""
    enabled: bool = False
    low_cutoff: float = 2000.0
    high_cutoff: float = 5000.0
    order: int = 4


class FaultFrequencyManualConfig(BaseModel):
    """手动输入特征频率"""
    bpfo: Optional[float] = None
    bpfi: Optional[float] = None
    bsf: Optional[float] = None
    ftf: Optional[float] = None
    gmf: Optional[float] = None
    fr: Optional[float] = None  # 转频


class BearingParamsConfig(BaseModel):
    """轴承参数"""
    bearing_type: Optional[str] = None  # 标准型号，如 "6205"
    pitch_diameter: Optional[float] = None  # 节圆直径 (mm)
    roller_diameter: Optional[float] = None  # 滚动体直径 (mm)
    n_rollers: Optional[int] = None  # 滚动体个数
    contact_angle: float = 0  # 接触角 (度)
    shaft_rpm: Optional[float] = None  # 转速 (RPM)
    gear_teeth: Optional[int] = None  # 齿轮齿数


class FaultFrequencyAutoConfig(BaseModel):
    """自动计算参数"""
    bearing_params: BearingParamsConfig = Field(default_factory=BearingParamsConfig)


class FaultFrequencyConfig(BaseModel):
    """特征频率配置"""
    mode: str = "manual"  # manual | auto
    manual: FaultFrequencyManualConfig = Field(default_factory=FaultFrequencyManualConfig)
    auto: FaultFrequencyAutoConfig = Field(default_factory=FaultFrequencyAutoConfig)

    @field_validator("mode")
    def validate_mode(cls, v: str) -> str:
        allowed = {"manual", "auto"}
        if v not in allowed:
            raise ValueError(f"特征频率 mode 必须是 {allowed} 之一，得到 '{v}'")
        return v


class EnvelopeConfig(BaseModel):
    """包络谱配置"""
    bandpass: BandpassConfig = Field(default_factory=BandpassConfig)
    n_fft: Optional[int] = None  # FFT点数，None自动设置
    fault_frequencies: FaultFrequencyConfig = Field(default_factory=FaultFrequencyConfig)


# ============================================================
# 评估指标配置
# ============================================================

class MetricsConfig(BaseModel):
    """指标配置"""
    enabled: List[str] = Field(default_factory=lambda: [
        "ffr", "lsnr", "her", "esk", "gi", "ese",
        "ffa", "rffi", "fbr", "er",
        "kurtosis", "sk", "skewness",
        "hsi", "sln", "hln",
        "si", "cf", "if", "ser",
    ])
    weights: Dict[str, float] = Field(default_factory=dict)


# ============================================================
# 评分与排序配置
# ============================================================

class ScoringConfig(BaseModel):
    """评分配置"""
    method: str = "weighted_sum"  # weighted_sum | topsis | borda_count | pareto
    top_n: int = 3
    normalization: str = "minmax"  # minmax | zscore | rank

    @field_validator("method")
    def validate_method(cls, v: str) -> str:
        allowed = {"weighted_sum", "topsis", "borda_count", "pareto"}
        if v not in allowed:
            raise ValueError(f"评分 method 必须是 {allowed} 之一，得到 '{v}'")
        return v

    @field_validator("normalization")
    def validate_norm(cls, v: str) -> str:
        allowed = {"minmax", "zscore", "rank"}
        if v not in allowed:
            raise ValueError(f"归一化方法必须是 {allowed} 之一，得到 '{v}'")
        return v


# ============================================================
# 执行配置
# ============================================================

class ExecutionConfig(BaseModel):
    """执行配置"""
    parallel_backend: str = "sequential"  # sequential | multiprocessing
    max_workers: int = 2
    timeout_per_algorithm_sec: int = 300
    gpu_indices: List[int] = Field(default_factory=list)

    # 资源控制（节流）
    process_priority: str = "normal"  # normal | below_normal | idle
    max_cpu_cores: int = 0  # 0=所有核心, >0则限制使用N个核心

    @field_validator("parallel_backend")
    def validate_backend(cls, v: str) -> str:
        allowed = {"sequential", "multiprocessing"}
        if v not in allowed:
            raise ValueError(f"并行后端必须是 {allowed} 之一，得到 '{v}'")
        return v

    @field_validator("max_workers")
    def validate_workers(cls, v: int) -> int:
        import os
        max_cpu = os.cpu_count() or 1
        if v < 1:
            v = 1
        if v > max_cpu:
            v = max_cpu
        return v

    @field_validator("process_priority")
    def validate_priority(cls, v: str) -> str:
        allowed = {"normal", "below_normal", "idle"}
        if v not in allowed:
            raise ValueError(f"进程优先级必须是 {allowed} 之一，得到 '{v}'")
        return v

    @field_validator("max_cpu_cores")
    def validate_cores(cls, v: int) -> int:
        if v < 0:
            return 0
        return v


# ============================================================
# 输出配置
# ============================================================

class ReportConfig(BaseModel):
    """报告配置"""
    format: str = "html"  # html | pdf | both
    include_charts: bool = True


class ExportConfig(BaseModel):
    """导出配置"""
    formats: List[str] = Field(default_factory=lambda: ["csv", "hdf5"])
    output_dir: str = "results"


class OutputConfig(BaseModel):
    """输出配置"""
    report: ReportConfig = Field(default_factory=ReportConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


# ============================================================
# 项目配置（顶层）
# ============================================================

class ProjectConfig(BaseModel):
    """项目元信息"""
    name: str = "filter_benchmark"
    description: str = ""


class AppConfig(BaseModel):
    """应用顶层配置"""
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    algorithms: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    envelope: EnvelopeConfig = Field(default_factory=EnvelopeConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def check_algorithms(self):
        """确保算法选择与模式一致"""
        if self.algorithms.selection.mode == "selected" and not self.algorithms.selection.selected_ids:
            raise ValueError("mode='selected' 但未指定 selected_ids")
        if self.algorithms.selection.mode == "category" and not self.algorithms.selection.categories:
            raise ValueError("mode='category' 但未指定 categories")
        return self
