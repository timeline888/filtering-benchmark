"""
配置桥接 - UI 控件 ↔ AppConfig 双向转换。

支持从 UI 控件值构建配置字典，
以及从 AppConfig 填充 UI 控件。
"""

from typing import Any, Dict, Optional

import os


class ConfigBridge:
    """UI ↔ 配置双向桥接"""

    @staticmethod
    def ui_to_config(
        signal_data,
        sample_rate: float,
        source_path: str,
        algorithm_selected_ids,
        algorithm_mode: str,
        scoring_method: str,
        top_n: int,
        backend: str,
        channel: int = 0,
        # 资源控制参数
        enable_throttle: bool = True,
        process_priority: str = "below_normal",
        max_cpu_cores: int = 0,
        max_workers: int = 2,
    ) -> Dict[str, Any]:
        """从 UI 控件值构建配置字典"""
        is_simulated = source_path == "simulated"

        config: Dict[str, Any] = {
            "project": {
                "name": "滤波基石系统 - GUI评估",
                "description": "通过图形界面运行的滤波降噪评估",
            },
            "signal": {
                "source": {
                    "simulate_reference": is_simulated,
                    "path": "" if is_simulated else source_path,
                    "sample_rate": sample_rate,
                    "channel": channel,
                },
                "preprocessing": {
                    "remove_dc": True,
                    "resample": False,
                    "normalize": False,
                },
            },
            "algorithms": {
                "selection": {
                    "mode": algorithm_mode,
                    "selected_ids": list(algorithm_selected_ids) if algorithm_selected_ids else [],
                    "categories": [],
                    "param_overrides": {},
                },
            },
            "envelope": {
                "bandpass": {
                    "enabled": True,
                    "low_cutoff": 500.0,
                    "high_cutoff": 5000.0,
                    "order": 4,
                },
                "bp_low": 500,
                "bp_high": 5000,
                "fault_frequencies": {
                    "mode": "manual",
                    "manual": {"bpfo": 78.5, "bpfi": 60.0, "bsf": 40.0, "ftf": 30.0},
                },
            },
            "metrics": {
                "enabled": [
                    "ffr", "lsnr", "her", "esk", "gi", "ese",
                    "ffa", "rffi", "fbr", "er", "kurtosis", "sk",
                    "skewness", "hsi", "sln", "hln", "si", "cf", "if", "ser",
                ],
                "weights": {},
            },
            "scoring": {
                "method": scoring_method,
                "top_n": top_n,
                "normalization": "minmax",
            },
            "execution": {
                "backend": backend,
                "max_workers": max_workers if enable_throttle else (os.cpu_count() or 4),
                "timeout": 300,
                "gpu_indices": [],
                "process_priority": process_priority if enable_throttle else "normal",
                "max_cpu_cores": max_cpu_cores if enable_throttle else 0,
            },
            "output": {
                "report_format": "html",
                "export_formats": ["csv"],
                "output_dir": "output",
            },
        }
        return config

    @staticmethod
    def ui_to_appconfig(
        signal_data,
        sample_rate: float,
        source_path: str,
        algorithm_selected_ids,
        algorithm_mode: str,
        scoring_method: str,
        top_n: int,
        backend: str,
        channel: int = 0,
        # 资源控制参数
        enable_throttle: bool = True,
        process_priority: str = "below_normal",
        max_cpu_cores: int = 0,
        max_workers: int = 2,
    ):
        """从 UI 值构建 AppConfig 对象"""
        config_dict = ConfigBridge.ui_to_config(
            signal_data, sample_rate, source_path,
            algorithm_selected_ids, algorithm_mode,
            scoring_method, top_n, backend, channel,
            enable_throttle, process_priority, max_cpu_cores, max_workers,
        )
        from ...config.schema import AppConfig
        return AppConfig(**config_dict)

    @staticmethod
    def config_to_dict(config) -> Dict[str, Any]:
        """将 AppConfig 转为字典"""
        if hasattr(config, 'model_dump'):
            return config.model_dump()
        return dict(config)

    @staticmethod
    def load_yaml_to_dict(yaml_path: str) -> Dict[str, Any]:
        """从 YAML 文件加载配置"""
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def save_dict_to_yaml(config_dict: Dict[str, Any], yaml_path: str):
        """保存配置字典到 YAML 文件"""
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
