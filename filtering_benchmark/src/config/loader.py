"""
配置加载与合并管理。

支持多层配置叠加：
1. 默认配置 (config/default.yaml)
2. 环境配置 (config/env/{env}.yaml)
3. 用户项目配置文件 (--config)
4. 命令行参数 (最高优先级)
"""

import os
from typing import Optional

import yaml
from loguru import logger
from pydantic import ValidationError

from ..core.exceptions import ConfigValidationError
from .schema import AppConfig


class ConfigLoader:
    """配置加载器"""

    def __init__(self, env: str = "development"):
        self.env = env
        self._base_dir: Optional[str] = None
        self._config: Optional[AppConfig] = None

    def find_base_dir(self) -> str:
        """查找项目根目录"""
        if self._base_dir:
            return self._base_dir

        # 从当前文件位置向上查找
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            if os.path.exists(os.path.join(current, "config", "default.yaml")):
                self._base_dir = current
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

        # 回退到当前工作目录
        self._base_dir = os.getcwd()
        return self._base_dir

    def load_default(self) -> dict:
        """加载默认配置"""
        base = self.find_base_dir()
        path = os.path.join(base, "config", "default.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def load_env(self) -> dict:
        """加载环境配置"""
        base = self.find_base_dir()
        path = os.path.join(base, "config", "env", f"{self.env}.yaml")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def load_user(self, user_config_path: Optional[str] = None) -> dict:
        """加载用户配置文件"""
        if not user_config_path:
            return {}

        if not os.path.exists(user_config_path):
            raise FileNotFoundError(f"用户配置文件不存在: {user_config_path}")

        with open(user_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def merge(self, *configs: dict) -> dict:
        """深度合并多个配置字典（后面的覆盖前面的）"""
        merged = {}
        for config in configs:
            self._deep_merge(merged, config)
        return merged

    def _deep_merge(self, base: dict, override: dict) -> None:
        """递归合并字典"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def load(
        self,
        user_config_path: Optional[str] = None,
        cli_overrides: Optional[dict] = None,
    ) -> AppConfig:
        """
        加载并合并所有配置层。

        Args:
            user_config_path: 用户 YAML 配置文件路径
            cli_overrides: 命令行参数覆盖

        Returns:
            验证通过的 AppConfig 对象
        """
        # 四层配置叠加
        default_cfg = self.load_default()
        env_cfg = self.load_env()
        user_cfg = self.load_user(user_config_path)
        cli_cfg = cli_overrides or {}

        merged = self.merge(default_cfg, env_cfg, user_cfg, cli_cfg)

        try:
            self._config = AppConfig.model_validate(merged)
            logger.info(f"配置加载完成: project={self._config.project.name}, "
                        f"signal={self._config.signal.source.path}")
            return self._config
        except ValidationError as e:
            logger.error(f"配置校验失败: {e}")
            raise ConfigValidationError(f"配置校验失败: {e}")

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            raise RuntimeError("配置未加载，请先调用 load()")
        return self._config
