"""
旋转机械故障诊断专用滤波降噪算法集合。

包含6种旋转机械故障诊断专用算法：
1. 最小熵解卷积 (MED)               — 最大化峭度恢复冲击序列
2. 最大相关峭度解卷积 (MCKD)        — 提取周期性冲击
3. 多点最优最小熵解卷积调整 (MOMEDA) — 无需迭代的最优解卷积
4. 谱峭度/Kurtogram滤波             — 自动选择最优解调频带
5. 共振解调（包络解调）             — 带通+希尔伯特+低通完整解调
6. 循环平稳分析                     — 基于谱相关的二阶循环平稳分析
"""

import warnings
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity

# ============================================================================
# 依赖检查与回退
# ============================================================================
try:
    from scipy import signal as scipy_signal
    from scipy.linalg import solve, inv, toeplitz
    from scipy.ndimage import uniform_filter1d as _uniform_filter1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn(
        "scipy 未安装，旋转机械专用算法将使用回退模式（返回原始信号）。"
        "请执行 'pip install scipy' 以启用完整功能。"
    )

# ============================================================================
# 内部辅助函数
# ============================================================================

def _ensure_1d(signal: np.ndarray) -> np.ndarray:
    """将输入规整为1维numpy数组"""
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim == 1:
        return s
    if s.ndim == 2 and s.shape[0] == 1:
        return s[0].copy()
    if s.ndim == 2:
        return s[0].copy()
    return s.ravel()


def _kurtosis(x: np.ndarray) -> float:
    """计算峭度（归一化四阶矩）"""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4:
        return 0.0
    n = x.size
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std == 0.0:
        return 0.0
    return float(np.sum((x - mean) ** 4) / (n * std ** 4))


def _correlated_kurtosis(x: np.ndarray, period: int, n_shift: int = 1) -> float:
    """计算相关峭度 (Correlated Kurtosis)

    CK_M(T) = sum_n (prod_{m=0}^{M} y[n-mT])^2 / (sum_n y[n]^2)^{M+1}
    对于 n_shift=1: CK_1(T) = sum_n (y[n] * y[n-T])^2 / (sum_n y[n]^2)^2
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    numerator = 0.0
    for m in range(1, n_shift + 1):
        shift = m * period
        if shift >= n or shift < 1:
            continue
        valid_len = n - shift
        y0 = x[:valid_len]
        y_shifted = x[shift:]
        numerator += np.sum((y0 * y_shifted) ** 2)
    denom = np.sum(x ** 2)
    if denom == 0.0:
        return 0.0
    return float(numerator / (denom ** (n_shift + 1)) * n)


def _detect_period(signal: np.ndarray, sample_rate: float,
                   min_freq: float = 5.0, max_freq: float = 500.0) -> int:
    """通过包络谱峰值检测冲击周期（采样点）"""
    s = _ensure_1d(signal)
    n = s.size
    if _HAS_SCIPY:
        try:
            analytic = scipy_signal.hilbert(s)
            envelope = np.abs(analytic)
        except Exception:
            envelope = np.abs(s)
    else:
        envelope = np.abs(s)

    spectrum = np.fft.rfft(envelope)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    mag = np.abs(spectrum)

    valid = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    if len(valid) == 0:
        return max(n // 20, 2)
    peak_idx = valid[np.argmax(mag[valid])]
    peak_freq = freqs[peak_idx]
    if peak_freq <= 0.0:
        return max(n // 20, 2)
    period = int(round(sample_rate / peak_freq))
    return max(period, 2)


def _build_convolution_matrix(x: np.ndarray, L: int) -> np.ndarray:
    """构建卷积矩阵 X (N-L+1, L), 使 y = X @ f 实现有效卷积

    X[i, j] = x[i + L - 1 - j]   for j=0..L-1
    """
    n = x.size
    rows = n - L + 1
    if rows <= 0:
        raise ValueError(f"滤波器阶数 L={L} 不能大于信号长度 n={n}")
    X = np.zeros((rows, L), dtype=np.float64)
    for i in range(rows):
        X[i] = x[i:i + L][::-1]
    return X


def _next_pow2(n: int) -> int:
    """下一个2的幂"""
    return 1 << (n - 1).bit_length()


# ============================================================================
# 1. 最小熵解卷积 (MED)
# ============================================================================

@register_algorithm(
    name="最小熵解卷积(MED)",
    category=AlgorithmCategory.ROTATING_SPECIFIC,
    complexity=AlgorithmComplexity.HIGH,
    tags=["minimum-entropy", "deconvolution", "impulsive"],
)
class MEDDenoise(BaseAlgorithm):
    """最小熵解卷积 (Minimum Entropy Deconvolution)

    通过迭代优化FIR滤波器系数最大化输出信号的峭度，恢复被传递路径
    调制的冲击序列。适用于齿轮、轴承等旋转机械的早期故障特征提取。

    References:
        - Endo & Randall (2007). Enhancement of autoregressive model based
          gear tooth fault detection technique by the use of minimum entropy
          deconvolution filter. Mechanical Systems and Signal Processing.
        - Wiggins (1978). Minimum entropy deconvolution. Geoexploration.
    """
    # MED 迭代优化 O(N*L^2*iter)，卷积矩阵规模随 N 增大而膨胀，设安全上限。
    max_signal_length: ClassVar[int] = 50000

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 64,
        "max_iter": 30,
        "termination_tol": 1e-3,
    }

    def _validate_runtime_params(self, params: Dict[str, Any]) -> None:
        if params.get("filter_order", 64) < 2:
            raise ValueError("filter_order 必须 >= 2")
        if params.get("max_iter", 30) < 1:
            raise ValueError("max_iter 必须 >= 1")
        if params.get("termination_tol", 1e-3) <= 0:
            raise ValueError("termination_tol 必须 > 0")

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float,
                **kwargs) -> np.ndarray:
        self._check_signal_size(signal)
        if not _HAS_SCIPY:
            warnings.warn("scipy 不可用，MED 返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        params = {**self.params, **kwargs}
        L = int(params["filter_order"])
        max_iter = int(params["max_iter"])
        tol = float(params["termination_tol"])

        x = _ensure_1d(signal)
        n = x.size
        L = min(L, n // 3)
        if L < 2:
            return x.copy()

        # 构建卷积矩阵并预计算逆
        X = _build_convolution_matrix(x, L)
        XtX = X.T @ X + 1e-10 * np.eye(L)
        XtX_inv = inv(XtX)

        # 初始化滤波器（中心冲击）
        f = np.zeros(L)
        f[L // 2] = 1.0
        f = f / np.linalg.norm(f)

        # 迭代优化
        prev_kurt = -1.0
        for _ in range(max_iter):
            y = X @ f
            kurt = _kurtosis(y)

            if prev_kurt >= 0 and abs(kurt - prev_kurt) < tol:
                break
            prev_kurt = kurt

            # 梯度目标: v[n] = y[n]^3
            v = y ** 3
            f_new = XtX_inv @ (X.T @ v)
            norm = np.linalg.norm(f_new)
            if norm < 1e-15:
                break
            f = f_new / norm

        result = X @ f

        orig = np.asarray(signal, dtype=np.float64)
        if orig.ndim == 2 and orig.shape[0] == 1:
            return result.reshape(1, -1)
        return result


# ============================================================================
# 2. 最大相关峭度解卷积 (MCKD)
# ============================================================================

@register_algorithm(
    name="最大相关峭度解卷积(MCKD)",
    category=AlgorithmCategory.ROTATING_SPECIFIC,
    complexity=AlgorithmComplexity.HIGH,
    tags=["correlated-kurtosis", "deconvolution", "bearing"],
)
class MCKDDDenoise(BaseAlgorithm):
    """最大相关峭度解卷积 (Maximum Correlated Kurtosis Deconvolution)

    以相关峭度为目标函数进行解卷积，能有效提取被强噪声掩盖的周期性冲击
    序列。特别适用于滚动轴承故障诊断。

    References:
        - McDonald et al. (2012). Maximum Correlated Kurtosis Deconvolution
          and Application on Gear Tooth Chip Fault Detection.
          Journal of Sound and Vibration.
    """
    # MCKD 涉及相关峭度迭代计算，卷积矩阵构造消耗 O(N*L)，设安全上限。
    max_signal_length: ClassVar[int] = 50000

    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 64,
        "period": None,
        "max_iter": 30,
        "n_shift": 1,
    }

    def _validate_runtime_params(self, params: Dict[str, Any]) -> None:
        if params.get("filter_order", 64) < 2:
            raise ValueError("filter_order 必须 >= 2")
        period = params.get("period")
        if period is not None and period < 2:
            raise ValueError("period 必须 >= 2 或 None")
        if params.get("max_iter", 30) < 1:
            raise ValueError("max_iter 必须 >= 1")
        if params.get("n_shift", 1) < 1:
            raise ValueError("n_shift 必须 >= 1")

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float,
                **kwargs) -> np.ndarray:
        self._check_signal_size(signal)
        if not _HAS_SCIPY:
            warnings.warn("scipy 不可用，MCKD 返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        params = {**self.params, **kwargs}
        L = int(params["filter_order"])
        period = params.get("period")
        max_iter = int(params["max_iter"])
        n_shift = int(params["n_shift"])

        x = _ensure_1d(signal)
        n = x.size
        L = min(L, n // 3)

        if period is None:
            period = _detect_period(x, sample_rate)
        period = max(int(period), 2)

        X = _build_convolution_matrix(x, L)
        rows = X.shape[0]

        XtX = X.T @ X + 1e-10 * np.eye(L)
        XtX_inv = inv(XtX)

        f = np.ones(L) / np.sqrt(L)
        shifts = [m * period for m in range(1, n_shift + 1)]

        prev_ck = -1.0
        for _ in range(max_iter):
            y = X @ f
            ck = _correlated_kurtosis(y, period, n_shift)

            if prev_ck >= 0 and abs(ck - prev_ck) < 1e-3:
                break
            prev_ck = ck

            y_power = np.sum(y ** 2)
            if y_power == 0.0:
                break

            target_sum = np.zeros(rows)
            for s in shifts:
                if s < rows:
                    y_shifted = np.roll(y, -s)
                    y_shifted[rows - s:] = 0.0
                    target_sum += y_shifted

            t = y * target_sum / y_power
            f_new = XtX_inv @ (X.T @ t)
            norm = np.linalg.norm(f_new)
            if norm < 1e-15:
                break
            f = f_new / norm

        result = X @ f
        orig = np.asarray(signal, dtype=np.float64)
        if orig.ndim == 2 and orig.shape[0] == 1:
            return result.reshape(1, -1)
        return result


# ============================================================================
# 3. 多点最优最小熵解卷积调整 (MOMEDA)
# ============================================================================

@register_algorithm(
    name="多点最优最小熵解卷积调整(MOMEDA)",
    category=AlgorithmCategory.ROTATING_SPECIFIC,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["multi-point", "deconvolution", "optimal"],
)
class MOMEDADenoise(BaseAlgorithm):
    """多点最优最小熵解卷积调整 (Multi-point Optimal Minimum Entropy
    Deconvolution Adjusted)

    通过构造多点目标向量（周期位置为1，其余为0），直接求解最优FIR滤波
    器，无需迭代。相比MED/MCKD计算效率更高。

    References:
        - McDonald & Zhao (2017). Multipoint Optimal Minimum Entropy
          Deconvolution and Convolution Fix: Application to vibration
          fault detection. Mechanical Systems and Signal Processing.
    """
    default_params: ClassVar[Dict[str, Any]] = {
        "filter_order": 64,
        "period": None,
    }

    def _validate_runtime_params(self, params: Dict[str, Any]) -> None:
        if params.get("filter_order", 64) < 2:
            raise ValueError("filter_order 必须 >= 2")
        period = params.get("period")
        if period is not None and period < 2:
            raise ValueError("period 必须 >= 2 或 None")

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float,
                **kwargs) -> np.ndarray:
        if not _HAS_SCIPY:
            warnings.warn("scipy 不可用，MOMEDA 返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        params = {**self.params, **kwargs}
        L = int(params["filter_order"])
        period = params.get("period")

        x = _ensure_1d(signal)
        n = x.size
        L = min(L, n // 3)

        if period is None:
            period = _detect_period(x, sample_rate)
        period = max(int(period), 2)

        X = _build_convolution_matrix(x, L)
        rows = X.shape[0]

        # 多点目标向量
        t = np.zeros(rows)
        t[period - 1::period] = 1.0

        # 直接求解
        XtX = X.T @ X + 1e-10 * np.eye(L)
        XtX_inv = inv(XtX)
        f = XtX_inv @ (X.T @ t)

        norm = np.linalg.norm(f)
        if norm > 1e-15:
            f = f / norm

        result = X @ f

        # 能量缩放
        orig_power = np.sqrt(np.mean(x ** 2))
        result_power = np.sqrt(np.mean(result ** 2))
        if result_power > 1e-15:
            result = result * (orig_power / result_power)

        orig = np.asarray(signal, dtype=np.float64)
        if orig.ndim == 2 and orig.shape[0] == 1:
            return result.reshape(1, -1)
        return result


# ============================================================================
# 4. 谱峭度 / Kurtogram 滤波
# ============================================================================

@register_algorithm(
    name="谱峭度/Kurtogram滤波",
    category=AlgorithmCategory.ROTATING_SPECIFIC,
    complexity=AlgorithmComplexity.HIGH,
    tags=["spectral-kurtosis", "kurtogram", "band-selection"],
)
class SpectralKurtosisFilter(BaseAlgorithm):
    """谱峭度/Kurtogram滤波

    计算多层STFT谱峭度（不同窗长），自动定位最"非高斯"频带。
    在该频带设计带通滤波器提取故障冲击成分。

    References:
        - Antoni (2006). The spectral kurtosis: a useful tool for
          characterising non-stationary signals. Mechanical Systems
          and Signal Processing.
        - Antoni (2007). Fast computation of the kurtogram for the
          detection of transient faults. Mechanical Systems and
          Signal Processing.
    """
    default_params: ClassVar[Dict[str, Any]] = {
        "n_levels": 6,
        "threshold": 0.7,
    }

    def _validate_runtime_params(self, params: Dict[str, Any]) -> None:
        if params.get("n_levels", 6) < 1:
            raise ValueError("n_levels 必须 >= 1")
        th = params.get("threshold", 0.7)
        if not (0.0 < th <= 1.0):
            raise ValueError("threshold 必须在 (0, 1] 范围内")

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float,
                **kwargs) -> np.ndarray:
        if not _HAS_SCIPY:
            warnings.warn("scipy 不可用，SpectralKurtosis 返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        params = {**self.params, **kwargs}
        n_levels = int(params["n_levels"])
        threshold = float(params["threshold"])

        x = _ensure_1d(signal)
        n = x.size

        nfft_min = 32
        kurtograms = []

        for level in range(n_levels):
            nperseg = int(2 ** (np.log2(nfft_min) + level))
            nperseg = min(nperseg, n // 2, 8192)
            if nperseg < 8:
                continue

            noverlap = nperseg // 2

            try:
                freqs, times, Zxx = scipy_signal.stft(
                    x, fs=sample_rate,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window="hann",
                    boundary=None,
                    padded=True,
                )
            except Exception:
                continue

            mag_sq = np.abs(Zxx) ** 2
            mean_mag = np.mean(mag_sq, axis=1, keepdims=True)
            mean_mag = np.maximum(mean_mag, 1e-30)
            normalized = mag_sq / mean_mag
            sk = np.mean(normalized ** 2, axis=1) - 2.0
            sk = np.maximum(sk, 0.0)

            bandwidth = sample_rate / 2.0 / len(freqs)
            kurtograms.append((freqs, sk, level, nperseg, bandwidth))

        if not kurtograms:
            warnings.warn("Kurtogram 计算失败，返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        # 寻找最优频带
        best_sk = -np.inf
        best_level_idx = 0
        best_freq_idx = 0

        for idx, (freqs, sk, level, nperseg, bw) in enumerate(kurtograms):
            search_start = 2
            if len(sk) <= search_start:
                continue
            local_max = np.max(sk[search_start:])
            if local_max > best_sk:
                best_sk = local_max
                best_level_idx = idx
                best_freq_idx = search_start + np.argmax(sk[search_start:])

        freqs_best, sk_best, _, nperseg_best, bw_best = kurtograms[best_level_idx]
        center_freq = float(freqs_best[best_freq_idx])
        half_bw = bw_best * 1.5

        low_freq = max(0.0, center_freq - half_bw)
        high_freq = min(sample_rate / 2, center_freq + half_bw)

        if high_freq - low_freq < sample_rate * 0.01:
            low_freq = max(0.0, center_freq - sample_rate * 0.02)
            high_freq = min(sample_rate / 2, center_freq + sample_rate * 0.02)

        nyquist = sample_rate / 2.0
        norm_low = low_freq / nyquist
        norm_high = high_freq / nyquist

        if norm_low >= 1.0 or norm_high >= 1.0 or norm_low >= norm_high:
            warnings.warn("Kurtogram 选择频带无效，返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        try:
            sos = scipy_signal.butter(4, [norm_low, norm_high],
                                      btype="band", output="sos")
            result = scipy_signal.sosfiltfilt(sos, x, axis=-1)
        except Exception:
            warnings.warn("Kurtogram 带通滤波失败，返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        orig = np.asarray(signal, dtype=np.float64)
        if orig.ndim == 2 and orig.shape[0] == 1:
            return result.reshape(1, -1)
        return result


# ============================================================================
# 5. 共振解调（包络解调）
# ============================================================================

@register_algorithm(
    name="共振解调(包络解调)",
    category=AlgorithmCategory.ROTATING_SPECIFIC,
    complexity=AlgorithmComplexity.LOW,
    tags=["envelope", "demodulation", "resonance"],
)
class EnvelopeDemodulation(BaseAlgorithm):
    """共振解调（包络解调）

    完整解调流程：
    1. 带通滤波（可选）— 选择共振频带
    2. Hilbert变换 — 构建解析信号
    3. 取包络 — |Hilbert(x)|
    4. 低通滤波 — 平滑包络，去除残余载波

    包络中的频率成分对应调制频率（如故障特征频率），广泛用于
    滚动轴承和齿轮的故障诊断。
    """
    default_params: ClassVar[Dict[str, Any]] = {
        "bp_low": None,
        "bp_high": None,
        "order": 4,
    }

    def _validate_runtime_params(self, params: Dict[str, Any]) -> None:
        bp_low = params.get("bp_low")
        bp_high = params.get("bp_high")
        if bp_low is not None and bp_low < 0:
            raise ValueError("bp_low 必须 >= 0")
        if bp_high is not None and bp_high <= 0:
            raise ValueError("bp_high 必须 > 0")
        if bp_low is not None and bp_high is not None and bp_low >= bp_high:
            raise ValueError("bp_low 必须小于 bp_high")
        if params.get("order", 4) < 1:
            raise ValueError("order 必须 >= 1")

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float,
                **kwargs) -> np.ndarray:
        if not _HAS_SCIPY:
            warnings.warn("scipy 不可用，EnvelopeDemodulation 返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        params = {**self.params, **kwargs}
        bp_low = params.get("bp_low")
        bp_high = params.get("bp_high")
        order = int(params["order"])

        x = _ensure_1d(signal)

        # 步骤1: 带通滤波（可选）
        if bp_low is not None and bp_high is not None:
            nyquist = sample_rate / 2.0
            norm_low = bp_low / nyquist
            norm_high = bp_high / nyquist

            if norm_low < 0.0 or norm_high > 1.0 or norm_low >= norm_high:
                warnings.warn("带通参数无效，跳过带通滤波")
                filtered = x.copy()
            else:
                try:
                    sos = scipy_signal.butter(order, [norm_low, norm_high],
                                              btype="band", output="sos")
                    filtered = scipy_signal.sosfiltfilt(sos, x, axis=-1)
                except Exception:
                    filtered = x.copy()
        else:
            filtered = x.copy()

        # 步骤2 & 3: Hilbert + 包络
        try:
            analytic = scipy_signal.hilbert(filtered)
            envelope = np.abs(analytic)
        except Exception:
            envelope = np.abs(filtered)

        # 步骤4: 低通滤波平滑包络
        lp_cutoff = min(sample_rate * 0.1, sample_rate / 2 * 0.95)
        nyquist = sample_rate / 2.0
        norm_lp = lp_cutoff / nyquist
        norm_lp = min(norm_lp, 0.95)

        try:
            sos_lp = scipy_signal.butter(order, norm_lp, btype="low", output="sos")
            envelope_smooth = scipy_signal.sosfiltfilt(sos_lp, envelope, axis=-1)
        except Exception:
            envelope_smooth = envelope

        envelope_smooth = np.abs(envelope_smooth)

        orig = np.asarray(signal, dtype=np.float64)
        if orig.ndim == 2 and orig.shape[0] == 1:
            return envelope_smooth.reshape(1, -1)
        return envelope_smooth


# ============================================================================
# 6. 循环平稳分析
# ============================================================================

@register_algorithm(
    name="循环平稳分析",
    category=AlgorithmCategory.ROTATING_SPECIFIC,
    complexity=AlgorithmComplexity.HIGH,
    tags=["cyclostationary", "modulation", "CS2"],
)
class CyclostationaryAnalysis(BaseAlgorithm):
    """循环平稳分析 (Cyclostationary Analysis)

    基于谱相关的二阶循环平稳分析（CS2），检测信号中的循环调制现象。
    旋转机械故障信号通常表现出循环平稳特性（如齿轮啮合频率的幅值调制），
    通过提取循环频率成分可以识别故障特征。

    算法流程：
    1. 计算信号STFT
    2. 计算谱相关密度
    3. 提取显著循环频率
    4. 重构增强信号

    References:
        - Antoni (2007). Cyclic spectral analysis in practice.
          Mechanical Systems and Signal Processing.
        - Randall & Antoni (2011). Rolling element bearing diagnostics
          — A tutorial. Mechanical Systems and Signal Processing.
    """
    # 循环平稳分析需计算多帧 STFT 和谱相关，涉及 FFT 和循环频率提取，设安全上限。
    max_signal_length: ClassVar[int] = 200000

    default_params: ClassVar[Dict[str, Any]] = {
        "n_cyclic_freqs": 10,
        "max_cyclic_freq_ratio": 0.5,
    }

    def _validate_runtime_params(self, params: Dict[str, Any]) -> None:
        if params.get("n_cyclic_freqs", 10) < 1:
            raise ValueError("n_cyclic_freqs 必须 >= 1")
        ratio = params.get("max_cyclic_freq_ratio", 0.5)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("max_cyclic_freq_ratio 必须在 (0, 1] 范围内")

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float,
                **kwargs) -> np.ndarray:
        self._check_signal_size(signal)
        if not _HAS_SCIPY:
            warnings.warn("scipy 不可用，CyclostationaryAnalysis 返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        params = {**self.params, **kwargs}
        n_cyclic = int(params["n_cyclic_freqs"])
        max_cyclic_ratio = float(params["max_cyclic_freq_ratio"])

        x = _ensure_1d(signal)
        n = x.size

        nperseg = min(256, n // 4)
        if nperseg < 32:
            nperseg = min(32, n // 2)
        noverlap = nperseg // 2
        nfft = _next_pow2(nperseg)

        try:
            freqs, times, Zxx = scipy_signal.stft(
                x, fs=sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                window="hann",
                boundary=None,
                padded=True,
            )
        except Exception:
            warnings.warn("STFT 计算失败，返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        n_freq = Zxx.shape[0]
        n_time = Zxx.shape[1]

        half_alpha_bins = n_time // 2
        max_alpha_bins = int(half_alpha_bins * max_cyclic_ratio)
        max_alpha_bins = min(max_alpha_bins, half_alpha_bins)

        cyclic_freqs = np.fft.fftfreq(n_time, 1.0 / sample_rate)[:max_alpha_bins]
        gamma = np.zeros(max_alpha_bins)

        for fi in range(n_freq):
            Xf = Zxx[fi, :]
            spec_cyc = np.fft.fft(np.abs(Xf) ** 2)[:max_alpha_bins]
            gamma += np.abs(spec_cyc)

        gamma[0] = 0.0
        if np.max(gamma) > 0:
            gamma = gamma / np.max(gamma)

        peak_indices = np.argsort(gamma)[::-1][:n_cyclic]
        selected_alphas = cyclic_freqs[peak_indices]
        selected_gammas = gamma[peak_indices]

        valid = selected_gammas > 0.05
        selected_alphas = selected_alphas[valid]
        if len(selected_alphas) == 0:
            warnings.warn("未检测到显著循环频率，返回原始信号")
            return np.asarray(signal, dtype=np.float64)

        result = np.zeros_like(x)
        for i, alpha in enumerate(selected_alphas):
            if alpha <= 0:
                continue
            half_bw = alpha * 0.3
            low = max(0.0, alpha - half_bw)
            high = min(sample_rate / 2, alpha + half_bw)

            if low >= high:
                continue

            nyquist = sample_rate / 2.0
            if low / nyquist >= 1.0 or high / nyquist >= 1.0:
                continue

            try:
                sos = scipy_signal.butter(2, [low / nyquist, high / nyquist],
                                          btype="band", output="sos")
                filtered = scipy_signal.sosfiltfilt(sos, x, axis=-1)
                weight = selected_gammas[i]
                result += filtered * weight
            except Exception:
                continue

        if np.max(np.abs(result)) < 1e-15:
            return np.asarray(signal, dtype=np.float64)

        orig_rms = np.sqrt(np.mean(x ** 2))
        result_rms = np.sqrt(np.mean(result ** 2))
        if result_rms > 1e-15:
            result = result * (orig_rms / result_rms)

        orig = np.asarray(signal, dtype=np.float64)
        if orig.ndim == 2 and orig.shape[0] == 1:
            return result.reshape(1, -1)
        return result
