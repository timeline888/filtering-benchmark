"""
联合优化包装算法。

包含将优化算法应用于降噪参数自动调优的 BaseAlgorithm 子类：
1. 差分进化优化 IIR 陷波 (15.10.1)
2. PSO 优化 LMS 步长 (15.10.2)
3. 贝叶斯优化小波参数 (15.10.3)
4. 贝叶斯优化卡尔曼滤波 (16.2.1)
5. PSO 优化 RLS 遗忘因子 (16.2.2)
6. GA 优化 OMP 稀疏度 (16.4.1)
7. 统一参数搜索器 (16.8)
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

from ...algorithms.base import BaseAlgorithm, _to_stereo, _from_stereo
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


# ========================================================================
# 1. 差分进化优化 IIR 陷波滤波器 (15.10.1)
# ========================================================================

@register_algorithm(
    name="DE优化IIR陷波",
    category=AlgorithmCategory.OPTIMIZATION,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["optimization", "de", "iir-notch", "auto-tuning"],
)
class DeOptimizedIIRNotch(BaseAlgorithm):
    """差分进化优化 IIR 陷波滤波器参数。

    使用差分进化自动搜索最优陷波频率 f0 和品质因数 Q，
    在目标噪声频率处最大化衰减，同时最小化信号通带失真。

    参考: 15.10.1 遗传算法优化 IIR 陷波滤波器参数
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "noise_freq": 50.0,        # 目标噪声频率 (Hz)
        "f0_bounds": (45, 55),     # 陷波频率搜索范围 (Hz)
        "q_bounds": (10, 50),      # 品质因数搜索范围
        "sample_rate": 1000.0,     # 采样率 (Hz)
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        from ..optimization.evolutionary_optimizers import differential_evolution

        params = {**self.params, **kwargs}
        noise_freq = float(params["noise_freq"])
        f0_bounds = params["f0_bounds"]
        q_bounds = params["q_bounds"]
        fs = float(params.get("sample_rate", sample_rate))

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            # 目标函数: 在噪声频率处衰减最大, 同时最小化信号失真
            def notch_cost(param_vec):
                f0, Q = param_vec
                w0 = f0 / (fs / 2.0)
                b, a = scipy_signal.iirnotch(w0, Q)
                w, h = scipy_signal.freqz(b, a, fs=fs)
                idx_noise = int(np.argmin(np.abs(w - noise_freq)))
                attenuation = 20 * np.log10(abs(h[idx_noise]) + 1e-10)
                # 通带平坦度惩罚（排除噪声频率附近 ±20Hz）
                signal_band = (w < noise_freq - 20) | (w > noise_freq + 20)
                flatness_penalty = np.std(
                    20 * np.log10(abs(h[signal_band]) + 1e-10)
                )
                return -attenuation + 0.1 * flatness_penalty

            bounds = [f0_bounds, q_bounds]
            best_params, _ = differential_evolution(
                notch_cost, bounds, pop_size=30, max_iter=50, random_state=42
            )
            # 用最优参数设计陷波滤波器并应用
            w0 = best_params[0] / (fs / 2.0)
            Q = best_params[1]
            b, a = scipy_signal.iirnotch(w0, Q)
            output[ch] = scipy_signal.filtfilt(b, a, sig)

        return _from_stereo(output, orig_signal)


# ========================================================================
# 2. PSO 优化 LMS 自适应步长 (15.10.2)
# ========================================================================

@register_algorithm(
    name="PSO优化LMS步长",
    category=AlgorithmCategory.OPTIMIZATION,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["optimization", "pso", "lms", "adaptive-step"],
)
class PsoOptimizedLMS(BaseAlgorithm):
    """PSO 优化 LMS 自适应滤波器步长。

    使用粒子群优化搜索最优步长 mu，使 LMS 的稳态 MSE 最小。
    适用于对收敛速度和稳态误差有严格要求的场景。

    参考: 15.10.2 PSO 优化自适应滤波器步长
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 32,
        "mu_min": 0.001,
        "mu_max": 0.5,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        from ..optimization.evolutionary_optimizers import particle_swarm_optimization

        params = {**self.params, **kwargs}
        filter_order = int(params["filter_order"])
        mu_min = float(params["mu_min"])
        mu_max = float(params["mu_max"])

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]

            # 使用 PSO 搜索最优步长
            def lms_mse(mu_vec):
                mu = float(mu_vec[0])
                w = np.zeros(filter_order)
                mse = 0.0
                count = 0
                for i in range(filter_order, n_samples, 5):  # 降采样加速
                    x = sig[i - filter_order:i][::-1]
                    e = sig[i] - np.dot(w, x)
                    w += 2.0 * mu * e * x
                    mse += e ** 2
                    count += 1
                return mse / max(count, 1)

            best_mu, _ = particle_swarm_optimization(
                lms_mse, 1, [(mu_min, mu_max)],
                n_particles=15, max_iter=20, random_state=42,
            )
            mu = float(best_mu[0])

            # 用最优步长执行完整 LMS
            w = np.zeros(filter_order)
            for i in range(filter_order, n_samples):
                x = sig[i - filter_order:i][::-1]
                y = np.dot(w, x)
                e = sig[i] - y
                w += 2.0 * mu * e * x
                output[ch, i] = y
            output[ch, :filter_order] = sig[:filter_order]

        return _from_stereo(output, orig_signal)


# ========================================================================
# 3. 贝叶斯优化小波阈值降噪参数 (15.10.3)
# ========================================================================

@register_algorithm(
    name="贝叶斯优化小波降噪",
    category=AlgorithmCategory.OPTIMIZATION,
    complexity=AlgorithmComplexity.HIGH,
    tags=["optimization", "bayesian", "wavelet", "auto-tuning"],
)
class BayesianOptimizedWavelet(BaseAlgorithm):
    """贝叶斯优化小波阈值降噪参数。

    使用 GP-UCB 自动搜索最优阈值缩放因子和分解层数，
    最大化输出信噪比 (SNR)。

    参考: 15.10.3 贝叶斯优化小波阈值去噪参数
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "wavelet": "db4",
        "threshold_mode": "soft",
        "n_init": 5,
        "n_iter": 30,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        from ..optimization.bayesian_optimizer import gp_ucb_optimize

        params = {**self.params, **kwargs}
        wavelet = str(params.get("wavelet", "db4"))
        thr_mode = str(params.get("threshold_mode", "soft"))
        n_init = int(params.get("n_init", 5))
        n_iter = int(params.get("n_iter", 30))

        import pywt

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]

            # 使用信号本身的标准差估计噪声 (无干净参考时使用自监督)
            def wavelet_score(threshold_scale: float, level: float) -> float:
                level = int(level)
                coeffs = pywt.wavedec(sig, wavelet, level=level)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(sig)))
                coeffs_th = [coeffs[0]] + [
                    pywt.threshold(c, threshold, mode=thr_mode)
                    for c in coeffs[1:]
                ]
                denoised = pywt.waverec(coeffs_th, wavelet)[:len(sig)]
                # 自监督目标: 最大化 SNR = var(signal) / var(noise)
                noise = sig - denoised
                snr = 10 * np.log10(np.var(sig) / (np.var(noise) + 1e-10))
                return snr

            def objective(x: list) -> float:
                return wavelet_score(x[0], x[1])

            best_x, _ = gp_ucb_optimize(
                objective,
                [(0.5, 5.0), (1.0, 8.0)],
                n_init=n_init,
                n_iter=n_iter,
                random_state=42,
            )

            # 用最优参数执行降噪
            level = max(1, int(round(best_x[1])))
            coeffs = pywt.wavedec(sig, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = best_x[0] * sigma * np.sqrt(2 * np.log(len(sig)))
            coeffs_th = [coeffs[0]] + [
                pywt.threshold(c, threshold, mode=thr_mode) for c in coeffs[1:]
            ]
            output[ch] = pywt.waverec(coeffs_th, wavelet)[:len(sig)]

        return _from_stereo(output, orig_signal)


# ========================================================================
# 4. 贝叶斯优化卡尔曼滤波 (16.2.1)
# ========================================================================

@register_algorithm(
    name="贝叶斯优化卡尔曼滤波",
    category=AlgorithmCategory.OPTIMIZATION,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["optimization", "bayesian", "kalman", "auto-tuning"],
)
class BayesianOptimizedKalman(BaseAlgorithm):
    """贝叶斯优化卡尔曼滤波的噪声协方差。

    自动搜索最优过程噪声 Q 和测量噪声 R，
    最大化卡尔曼滤波的 SNR 提升。

    参考: 16.2.1 贝叶斯优化 Kalman 滤波的噪声协方差
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_init": 8,
        "n_iter": 30,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        from ..optimization.bayesian_optimizer import gp_ucb_optimize

        params = {**self.params, **kwargs}
        n_init = int(params.get("n_init", 8))
        n_iter = int(params.get("n_iter", 30))

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]

            def kalman_score(q_scale: float, r_scale: float) -> float:
                """一维卡尔曼滤波并返回 SNR (最大化)"""
                n = len(sig)
                x_est = np.zeros(n)
                p_est = np.zeros(n)
                x_est[0] = sig[0]
                p_est[0] = 1.0
                for k in range(1, n):
                    x_pred = x_est[k - 1]
                    p_pred = p_est[k - 1] + q_scale
                    K = p_pred / (p_pred + r_scale)
                    x_est[k] = x_pred + K * (sig[k] - x_pred)
                    p_est[k] = (1.0 - K) * p_pred
                noise = sig - x_est
                snr = 10 * np.log10(np.var(sig) / (np.var(noise) + 1e-10))
                return snr

            def objective(x: list) -> float:
                return kalman_score(x[0], x[1])

            best_x, _ = gp_ucb_optimize(
                objective,
                [(1e-5, 0.5), (1e-5, 0.5)],
                n_init=n_init,
                n_iter=n_iter,
                random_state=42,
            )

            # 用最优参数执行卡尔曼滤波
            q_opt, r_opt = best_x
            n = len(sig)
            x_est = np.zeros(n)
            p_est = np.zeros(n)
            x_est[0] = sig[0]
            p_est[0] = 1.0
            for k in range(1, n):
                x_pred = x_est[k - 1]
                p_pred = p_est[k - 1] + q_opt
                K = p_pred / (p_pred + r_opt)
                x_est[k] = x_pred + K * (sig[k] - x_pred)
                p_est[k] = (1.0 - K) * p_pred
            output[ch] = x_est

        return _from_stereo(output, orig_signal)


# ========================================================================
# 5. PSO 优化 RLS 遗忘因子 (16.2.2)
# ========================================================================

@register_algorithm(
    name="PSO优化RLS遗忘因子",
    category=AlgorithmCategory.OPTIMIZATION,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["optimization", "pso", "rls", "forgetting-factor"],
)
class PsoOptimizedRLS(BaseAlgorithm):
    """PSO 优化 RLS 自适应滤波器的遗忘因子。

    RLS 的遗忘因子 lambda 控制对过去数据的记忆长度：
    lambda → 1 收敛于 Wiener 解（跟踪慢）。
    lambda 越小跟踪越快但稳态误差越大。

    参考: 16.2.2 PSO 优化 RLS 自适应滤波器的遗忘因子
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 16,
        "delta": 0.1,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        from ..optimization.evolutionary_optimizers import particle_swarm_optimization

        params = {**self.params, **kwargs}
        filter_order = int(params["filter_order"])
        delta = float(params.get("delta", 0.1))

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]

            def rls_mse(lam_vec):
                lam = float(lam_vec[0])
                w = np.zeros(filter_order)
                P = np.eye(filter_order) / delta
                mse = 0.0
                count = 0
                for i in range(filter_order, n_samples, 3):
                    u = sig[i - filter_order:i][::-1]
                    Pi = P @ u
                    k = Pi / (lam + u @ Pi)
                    e = sig[i] - w @ u
                    w += k * e
                    P = (P - np.outer(k, Pi)) / lam
                    mse += e ** 2
                    count += 1
                return mse / max(count, 1)

            best_lam, _ = particle_swarm_optimization(
                rls_mse, 1, [(0.8, 1.0)],
                n_particles=15, max_iter=20, random_state=42,
            )
            lam = float(best_lam[0])

            # 用最优遗忘因子执行完整 RLS
            w = np.zeros(filter_order)
            P = np.eye(filter_order) / delta
            for i in range(filter_order, n_samples):
                u = sig[i - filter_order:i][::-1]
                Pi = P @ u
                k = Pi / (lam + u @ Pi)
                y = w @ u
                e = sig[i] - y
                w += k * e
                P = (P - np.outer(k, Pi)) / lam
                output[ch, i] = y
            output[ch, :filter_order] = sig[:filter_order]

        return _from_stereo(output, orig_signal)


# ========================================================================
# 6. GA 优化 OMP 稀疏度 (16.4.1)
# ========================================================================

@register_algorithm(
    name="GA优化OMP稀疏度",
    category=AlgorithmCategory.OPTIMIZATION,
    complexity=AlgorithmComplexity.HIGH,
    tags=["optimization", "ga", "omp", "sparsity"],
)
class GaOptimizedOMP(BaseAlgorithm):
    """GA 优化 OMP 稀疏表示降噪的稀疏度。

    使用遗传算法搜索 OMP 最优稀疏度 s，
    平衡表示精度与计算开销。

    参考: 16.4.1 GA 优化 OMP 的稀疏度阈值
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "max_sparsity": 30,
        "n_atoms": 128,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        from ..optimization.evolutionary_optimizers import genetic_algorithm
        from ..sparse.sparse_denoise import _gabor_dictionary, _orthogonal_matching_pursuit

        params = {**self.params, **kwargs}
        max_sparsity = int(params.get("max_sparsity", 30))
        n_atoms = int(params.get("n_atoms", 128))

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        n_atoms = min(n_atoms, n_samples)
        dictionary = _gabor_dictionary(n_samples, n_atoms, sample_rate)

        for ch in range(n_channels):
            sig = signal[ch]

            def omp_score(s_vec):
                s = max(1, int(round(s_vec[0])))
                denoised = _orthogonal_matching_pursuit(sig, dictionary, sparsity=s)
                noise = sig - denoised
                snr = 10 * np.log10(np.var(sig) / (np.var(noise) + 1e-10))
                # 惩罚过大稀疏度（计算成本）
                return snr - 0.05 * s

            best_s = genetic_algorithm(
                omp_score, 1, [(1, max_sparsity)],
                pop_size=20, max_generations=20, random_state=42,
            )
            s = max(1, int(round(float(best_s[0]))))
            output[ch] = _orthogonal_matching_pursuit(sig, dictionary, sparsity=s)

        return _from_stereo(output, orig_signal)


# ========================================================================
# 7. 统一降噪参数自动搜索器 (16.8)
# ========================================================================

@register_algorithm(
    name="降噪参数自动搜索器",
    category=AlgorithmCategory.OPTIMIZATION,
    complexity=AlgorithmComplexity.HIGH,
    tags=["optimization", "auto-tuning", "meta-algorithm"],
)
class DenoiseParamSearcher(BaseAlgorithm):
    """统一的降噪参数自动搜索器。

    可配置使用贝叶斯优化或差分进化来搜索任意降噪算法的最优参数。
    使用信号方差比作为自监督目标（无需干净参考信号）。

    参考: 16.8 综合案例：智能降噪参数自动调优系统
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "algorithm_id": "dwt_threshold_denoise",
        "method": "bayes",
        "n_iter": 30,
        "param_bounds": "{}",
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        from ..registry import registry
        from ..optimization.bayesian_optimizer import gp_ucb_optimize
        from ..optimization.evolutionary_optimizers import differential_evolution

        params = {**self.params, **kwargs}
        algorithm_id = str(params.get("algorithm_id", "dwt_threshold_denoise"))
        method = str(params.get("method", "bayes"))
        n_iter = int(params.get("n_iter", 30))

        import json
        param_bounds = json.loads(str(params.get("param_bounds", "{}")))

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        try:
            algo_meta = registry.get(algorithm_id)
        except KeyError:
            return _from_stereo(signal, orig_signal)

        for ch in range(n_channels):
            sig = signal[ch]

            # 自监督目标：最大化 SNR = var(signal) / var(noise)
            def evaluate(**kwargs_dict):
                instance = algo_meta.algorithm_class(**kwargs_dict)
                denoised = instance.denoise(sig.copy(), sample_rate)
                noise = sig - denoised[:len(sig)]
                snr = 10 * np.log10(np.var(sig) / (np.var(noise) + 1e-10))
                return snr

            if method == "bayes" and param_bounds:
                bounds_list = [(v[0], v[1]) for v in param_bounds.values()]
                def obj_func(x):
                    return evaluate(**dict(zip(param_bounds.keys(), x)))

                best_x, _ = gp_ucb_optimize(
                    obj_func, bounds_list,
                    n_init=min(10, n_iter // 3),
                    n_iter=n_iter,
                    random_state=42,
                )
                best_params = dict(zip(param_bounds.keys(), best_x))

            elif method == "de" and param_bounds:
                bounds_list = [(v[0], v[1]) for v in param_bounds.values()]
                def de_obj(p):
                    return -evaluate(**dict(zip(param_bounds.keys(), p)))

                best_x, _ = differential_evolution(
                    de_obj, bounds_list,
                    pop_size=20, max_iter=n_iter, random_state=42,
                )
                best_params = dict(zip(param_bounds.keys(), best_x))

            else:
                best_params = {}

            # 用最优参数执行降噪
            instance = algo_meta.algorithm_class(**best_params)
            result = instance.denoise(sig.copy(), sample_rate)
            output[ch] = result[:n_samples] if len(result) >= n_samples else result

        return _from_stereo(output, orig_signal)
