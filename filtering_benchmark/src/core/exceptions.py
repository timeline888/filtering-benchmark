"""
自定义异常体系。
"""


class FilteringBenchmarkError(Exception):
    """基类异常"""
    pass


class SignalReadError(FilteringBenchmarkError):
    """信号读取错误"""
    pass


class AlgorithmNotFoundError(FilteringBenchmarkError):
    """算法未找到"""
    def __init__(self, algorithm_id: str):
        super().__init__(f"未找到算法: {algorithm_id}")
        self.algorithm_id = algorithm_id


class AlgorithmExecutionError(FilteringBenchmarkError):
    """算法执行错误"""
    pass


class AlgorithmTimeoutError(AlgorithmExecutionError):
    """算法执行超时"""
    def __init__(self, algorithm_id: str, timeout: float):
        super().__init__(f"算法 '{algorithm_id}' 执行超时 ({timeout}s)")
        self.algorithm_id = algorithm_id
        self.timeout = timeout


class ConfigValidationError(FilteringBenchmarkError):
    """配置校验错误"""
    pass


class MetricComputationError(FilteringBenchmarkError):
    """指标计算错误"""
    pass


class EnvelopeAnalysisError(FilteringBenchmarkError):
    """包络谱分析错误"""
    pass


class PipelineStageError(FilteringBenchmarkError):
    """流水线阶段错误"""
    pass


class UnsupportedFormatError(SignalReadError):
    """不支持的信号格式"""
    def __init__(self, fmt: str):
        super().__init__(f"不支持的信号格式: {fmt}")
        self.format = fmt


class SignalTooLargeError(AlgorithmExecutionError):
    """信号长度超过算法的安全处理上限。

    用于保护高复杂度算法在大规模信号下卡死系统或耗尽内存。
    由 orchestrator 捕获后转为 success=False 结果。
    """
    def __init__(self, algorithm_id: str, n_samples: int, max_samples: int):
        super().__init__(
            f"算法 '{algorithm_id}' 的信号长度 {n_samples} 超过安全上限 {max_samples}"
        )
        self.algorithm_id = algorithm_id
        self.n_samples = n_samples
        self.max_samples = max_samples
