"""
Filtering Benchmark - 旋转机械振动/声学信号滤波降噪评估系统。

使用方式:
    from filtering_benchmark import Pipeline, Config
    config = Config.from_yaml("config.yaml")
    report = Pipeline(config).run()
"""

from .src.config import ConfigLoader, AppConfig
from .src.pipeline.orchestrator import PipelineOrchestrator
from .src.algorithms import registry
from .src.metrics import metric_registry

__version__ = "0.1.0"


class Config:
    """配置便捷类"""
    @staticmethod
    def from_yaml(path: str) -> AppConfig:
        loader = ConfigLoader()
        return loader.load(user_config_path=path)


class Pipeline:
    """评估流水线便捷入口"""
    def __init__(self, config: AppConfig):
        self._pipeline = PipelineOrchestrator(config)

    def run(self):
        # 确保所有算法和指标已被发现注册
        registry.ensure_discovered()
        return self._pipeline.run()


__all__ = ["Config", "Pipeline", "AppConfig", "registry", "metric_registry"]
