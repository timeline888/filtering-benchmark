"""
自适应滤波算法集合：LMS, NLMS, RLS, APA, 卡尔曼滤波, 维纳滤波。

所有算法继承自 BaseAlgorithm，通过 @register_algorithm 装饰器注册。
"""

from typing import Any, ClassVar, Dict

import numpy as np
from scipy import linalg

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


@register_algorithm(
    name="LMS自适应滤波",
    category=AlgorithmCategory.ADAPTIVE,
    complexity=AlgorithmComplexity.LOW,
    tags=["adaptive", "lms", "real-time"],
)
class LMSAdaptiveFilter(BaseAlgorithm):
    """LMS (Least Mean Squares) 自适应滤波。

    使用最速下降法，通过瞬时梯度估计更新滤波器系数。
    适用于实时在线降噪。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 32,
        "mu": 0.01,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        filter_order = int(params["filter_order"])
        mu = float(params["mu"])

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            w = np.zeros(filter_order)
            for i in range(filter_order, n_samples):
                x = sig[i - filter_order : i][::-1]
                y = np.dot(w, x)
                e = sig[i] - y
                w = w + 2.0 * mu * e * x
                output[ch, i] = y
            output[ch, :filter_order] = sig[:filter_order]

        return output[0] if is_1d else output


@register_algorithm(
    name="NLMS自适应滤波",
    category=AlgorithmCategory.ADAPTIVE,
    complexity=AlgorithmComplexity.LOW,
    tags=["adaptive", "nlms", "normalized"],
)
class NLMSAdaptiveFilter(BaseAlgorithm):
    """NLMS (Normalized Least Mean Squares) 自适应滤波。

    对LMS进行归一化处理，使用输入信号能量调整步长，
    收敛更稳定且不依赖信号幅度。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 32,
        "mu": 0.1,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        filter_order = int(params["filter_order"])
        mu = float(params["mu"])
        eps = 1e-10

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            w = np.zeros(filter_order)
            for i in range(filter_order, n_samples):
                x = sig[i - filter_order : i][::-1]
                y = np.dot(w, x)
                e = sig[i] - y
                norm = np.dot(x, x) + eps
                w = w + mu * e * x / norm
                output[ch, i] = y
            output[ch, :filter_order] = sig[:filter_order]

        return output[0] if is_1d else output


@register_algorithm(
    name="RLS自适应滤波",
    category=AlgorithmCategory.ADAPTIVE,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["adaptive", "rls", "recursive"],
)
class RLSAdaptiveFilter(BaseAlgorithm):
    """RLS (Recursive Least Squares) 自适应滤波。

    通过递归最小二乘准则更新滤波器系数，收敛速度快于LMS，
    但计算复杂度较高。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 32,
        "lambda_": 0.99,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        filter_order = int(params["filter_order"])
        lambda_ = float(params["lambda_"])
        delta = 0.1  # 正则化初始化参数

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            w = np.zeros(filter_order)
            # 逆相关矩阵初始化
            P = np.eye(filter_order) / delta
            for i in range(filter_order, n_samples):
                x = sig[i - filter_order : i][::-1]
                # 增益向量
                Pi = P @ x
                denom = lambda_ + np.dot(x, Pi)
                k = Pi / denom
                # 先验误差
                y = np.dot(w, x)
                e = sig[i] - y
                # 更新权值
                w = w + k * e
                # 更新逆相关矩阵
                P = (P - np.outer(k, Pi)) / lambda_
                output[ch, i] = y
            output[ch, :filter_order] = sig[:filter_order]

        return output[0] if is_1d else output


@register_algorithm(
    name="APA自适应滤波",
    category=AlgorithmCategory.ADAPTIVE,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["adaptive", "apa", "projection"],
)
class APAAdaptiveFilter(BaseAlgorithm):
    """APA (Affine Projection Algorithm) 自适应滤波。

    使用多个输入向量进行仿射投影更新，在白噪声和有色噪声
    环境下均有较好表现。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 16,
        "projection_order": 4,
        "mu": 0.1,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        filter_order = int(params["filter_order"])
        P_order = int(params["projection_order"])
        mu = float(params["mu"])
        eps = 1e-10

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            w = np.zeros(filter_order)
            for i in range(filter_order + P_order - 1, n_samples):
                # 构建输入矩阵 (P_order x filter_order)
                X = np.zeros((P_order, filter_order))
                d = np.zeros(P_order)
                for p in range(P_order):
                    X[p, :] = sig[i - p - filter_order + 1 : i - p + 1][::-1]
                    d[p] = sig[i - p]
                # 滤波输出
                y_vec = X @ w
                e_vec = d - y_vec
                # APA 更新
                XTX = X @ X.T
                w = w + mu * X.T @ np.linalg.solve(
                    XTX + eps * np.eye(P_order), e_vec
                )
                output[ch, i] = y_vec[0]
            output[ch, : filter_order + P_order - 1] = sig[
                : filter_order + P_order - 1
            ]

        return output[0] if is_1d else output


@register_algorithm(
    name="卡尔曼滤波",
    category=AlgorithmCategory.ADAPTIVE,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["adaptive", "kalman", "state-space"],
)
class KalmanFilter(BaseAlgorithm):
    """卡尔曼滤波降噪。

    基于状态空间模型，通过预测-更新递归框架对信号进行最优估计。
    适用于高斯噪声环境下的信号降噪。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "process_noise": 1e-5,
        "measurement_noise": 1e-3,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        Q = float(params["process_noise"])
        R = float(params["measurement_noise"])

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        # 状态转移矩阵: [x, dx]^T -> [x+dx, dx]^T
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        # 观测矩阵: 观测位置
        H = np.array([[1.0, 0.0]])
        I = np.eye(2)

        for ch in range(n_channels):
            sig = signal[ch]
            # 状态初始化: [位置, 速度]
            x_est = np.array([sig[0], 0.0])
            P_est = np.eye(2)

            for i in range(n_samples):
                # 预测步骤
                x_pred = F @ x_est
                P_pred = F @ P_est @ F.T + Q * np.eye(2)

                # 更新步骤
                y = sig[i]
                S = H @ P_pred @ H.T + R
                K = (P_pred @ H.T / S[0, 0]).flatten()  # (2,)
                innovation = y - (H @ x_pred)[0]
                x_est = x_pred + K * innovation
                P_est = (I - np.outer(K, H[0])) @ P_pred

                output[ch, i] = x_est[0]

        return output[0] if is_1d else output


@register_algorithm(
    name="维纳滤波",
    category=AlgorithmCategory.ADAPTIVE,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["adaptive", "wiener", "optimal"],
)
class WienerFilter(BaseAlgorithm):
    """维纳滤波降噪。

    基于最小均方误差准则的最优线性滤波器。
    通过求解 Wiener-Hopf 方程得到最优滤波器系数。
    适用于平稳随机信号的降噪处理。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 32,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        filter_order = int(params["filter_order"])

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]

            # 计算自相关函数
            r = np.correlate(sig, sig, mode="full")
            r = r[len(r) // 2 :]  # 正延迟部分: r[0], r[1], ...

            # 构建 Toeplitz 自相关矩阵
            R = linalg.toeplitz(r[:filter_order])

            # 构建互相关向量: r_dx(k) = E[s(n) * s(n-k-1)] = r(k+1)
            p = r[1 : filter_order + 1]

            # 解 Wiener-Hopf 方程: R * w = p
            try:
                w = np.linalg.solve(R, p)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(R, p, rcond=None)[0]

            # 应用滤波器
            # 卷积: y[n] = sum(w[k] * sig[n-k])
            # 维纳滤波器输出为对当前样本的估计
            y = np.convolve(sig, w, mode="full")

            # 延迟补偿: 首样本保持原始值
            output[ch, 0] = sig[0]
            output[ch, 1:] = y[: n_samples - 1]

        return output[0] if is_1d else output


@register_algorithm(
    name="变步长LMS自适应滤波",
    category=AlgorithmCategory.ADAPTIVE,
    complexity=AlgorithmComplexity.LOW,
    tags=["adaptive", "vss-lms", "variable-step"]
)
class VsslmsAdaptiveFilter(BaseAlgorithm):
    """变步长LMS自适应滤波 (VSSLMS)。

    步长根据误差信号动态调整：误差大时大步长加速收敛，
    误差小时小步长降低稳态失调。
    使用 Kwong 方案: mu(n+1) = alpha*mu(n) + gamma*e^2(n)
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 32,
        "mu_max": 0.1,
        "mu_min": 0.001,
        "alpha": 0.97,
        "gamma": 0.0005,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        filter_order = int(params["filter_order"])
        mu_max = float(params["mu_max"])
        mu_min = float(params["mu_min"])
        alpha = float(params["alpha"])
        gamma = float(params["gamma"])

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            w = np.zeros(filter_order)
            mu = mu_max

            for i in range(filter_order, n_samples):
                x = sig[i - filter_order : i][::-1]
                y = np.dot(w, x)
                e = sig[i] - y
                # 步长更新
                mu = alpha * mu + gamma * e**2
                mu = np.clip(mu, mu_min, mu_max)
                w += mu * e * x
                output[ch, i] = y
            output[ch, :filter_order] = sig[:filter_order]

        return output[0] if is_1d else output


@register_algorithm(
    name="泄露LMS自适应滤波",
    category=AlgorithmCategory.ADAPTIVE,
    complexity=AlgorithmComplexity.LOW,
    tags=["adaptive", "leaky-lms", "regularized"]
)
class LeakyLmsAdaptiveFilter(BaseAlgorithm):
    """泄露LMS自适应滤波 (Leaky LMS)。

    在权重更新中加入泄露项 (1-mu*gamma)，等价于 L2 正则化，
    抑制权重发散，适用于输入信号高度相关的场景。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 32,
        "mu": 0.01,
        "gamma": 0.001,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        filter_order = int(params["filter_order"])
        mu = float(params["mu"])
        gamma = float(params["gamma"])
        leak = 1.0 - mu * gamma

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            w = np.zeros(filter_order)

            for i in range(filter_order, n_samples):
                x = sig[i - filter_order : i][::-1]
                y = np.dot(w, x)
                e = sig[i] - y
                w = leak * w + mu * e * x
                output[ch, i] = y
            output[ch, :filter_order] = sig[:filter_order]

        return output[0] if is_1d else output
