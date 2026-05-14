"""
SVD (奇异值分解) 降噪算法集合。

包含：
1. SVD降噪：对Hankel矩阵做SVD，保留主奇异值重构
2. Hankel-SVD降噪：显式构造Hankel矩阵后SVD分解

所有算法继承自 BaseAlgorithm，通过 @register_algorithm 装饰器注册。
"""

from typing import Any, ClassVar, Dict, Optional

import numpy as np

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


def _build_hankel(signal: np.ndarray, window_length: int) -> np.ndarray:
    """将1D信号构建为Hankel矩阵（轨迹矩阵）。

    Args:
        signal: 1D 输入信号，长度 N
        window_length: 窗口长度 L (1 < L < N)

    Returns:
        Hankel 矩阵，shape (L, K)，其中 K = N - L + 1
    """
    n = len(signal)
    k = n - window_length + 1  # 列数
    # 向量化实现：使用 sliding_window_view 避免 Python 循环
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        # windows shape: (k, window_length)
        windows = sliding_window_view(signal, window_length)
        return windows.T.copy()  # (L, K)
    except ImportError:
        H = np.zeros((window_length, k))
        for i in range(window_length):
            H[i, :] = signal[i : i + k]
        return H


def _hankel_reconstruct(H: np.ndarray, n: int) -> np.ndarray:
    """从Hankel矩阵通过反对角线平均重构1D信号。

    Args:
        H: Hankel 矩阵，shape (L, K)
        n: 原始信号长度

    Returns:
        重构的 1D 信号
    """
    L, K = H.shape
    # 向量化反对角平均：H[i, j] 贡献给 signal[i + j]
    ii, jj = np.indices((L, K))
    p_idx = ii + jj
    signal = np.zeros(n)
    count = np.zeros(n)
    np.add.at(signal, p_idx.ravel(), H.ravel())
    np.add.at(count, p_idx.ravel(), 1.0)
    return signal / np.maximum(count, 1.0)


def _select_n_components_from_energy(
    S: np.ndarray, retained_energy: float
) -> int:
    """根据保留能量比例选择奇异值个数。

    Args:
        S: 奇异值数组
        retained_energy: 保留能量比例 (0, 1]

    Returns:
        保留的奇异值个数
    """
    total_energy = np.sum(S**2)
    energy_cumsum = np.cumsum(S**2)
    n_components = int(np.searchsorted(energy_cumsum, retained_energy * total_energy) + 1)
    return min(n_components, len(S))


@register_algorithm(
    name="SVD降噪",
    category=AlgorithmCategory.SVD,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["svd", "singular-value", "hankel"],
)
class SVDDenoise(BaseAlgorithm):
    """SVD (奇异值分解) 降噪。

    将信号构造成Hankel矩阵后进行SVD分解，
    通过保留主奇异值（较大的奇异值）并置零小奇异值来实现降噪，
    最后通过反对角线平均重构时域信号。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_components": None,
        "retained_energy": 0.9,
    }

    # SVD 的 Hankel 矩阵内存为 O(N²/3)，n=30000 约耗 2.4GB
    max_signal_length: ClassVar[int] = 30000

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        self._check_signal_size(signal)
        params = {**self.params, **kwargs}
        n_components = params.get("n_components")
        retained_energy = float(params.get("retained_energy", 0.9))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]

            # 自适应选择窗口长度（建议为信号长度的 1/3 ~ 1/2）
            window_length = n_samples // 3
            window_length = max(3, min(window_length, n_samples - 1))

            # 构建 Hankel 矩阵
            H = _build_hankel(sig, window_length)

            # SVD 分解
            U, S, Vt = np.linalg.svd(H, full_matrices=False)

            # 确定保留的奇异值个数
            if n_components is not None:
                k = int(n_components)
            else:
                k = _select_n_components_from_energy(S, retained_energy)
            k = max(1, min(k, len(S)))

            # 截断SVD重构
            H_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

            # 反对角线平均重构信号
            output[ch, :] = _hankel_reconstruct(H_denoised, n_samples)

        return output[0] if is_1d else output


@register_algorithm(
    name="Hankel-SVD降噪",
    category=AlgorithmCategory.SVD,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["svd", "hankel", "trajectory-matrix"],
)
class HankelSVD(BaseAlgorithm):
    """Hankel-SVD 降噪。

    显式构造Hankel矩阵后进行SVD分解，对窗口长度更灵活可控。
    可以指定窗口长度或保留能量比例来降噪，
    适用于非平稳信号和冲击特征的提取。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "window_length": None,
        "n_components": None,
        "retained_energy": 0.9,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        window_length = params.get("window_length")
        n_components = params.get("n_components")
        retained_energy = float(params.get("retained_energy", 0.9))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]

            # 确定窗口长度
            if window_length is None:
                # 默认取信号长度的1/2，确保 Hankel 矩阵接近方阵
                L = n_samples // 2
            else:
                L = int(window_length)
            L = max(2, min(L, n_samples - 1))

            # 构建 Hankel 矩阵
            H = _build_hankel(sig, L)

            # SVD 分解
            U, S, Vt = np.linalg.svd(H, full_matrices=False)

            # 确定保留的奇异值个数
            if n_components is not None:
                k = int(n_components)
            else:
                k = _select_n_components_from_energy(S, retained_energy)
            k = max(1, min(k, len(S)))

            # 截断SVD重构
            H_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

            # 反对角线平均重构信号
            output[ch, :] = _hankel_reconstruct(H_denoised, n_samples)

        return output[0] if is_1d else output
