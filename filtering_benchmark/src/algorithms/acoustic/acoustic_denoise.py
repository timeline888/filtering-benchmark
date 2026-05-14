"""
语音/声学降噪算法集合。

包含：
1. MMSE-STSA: 最小均方误差短时谱幅度估计
2. Log-MMSE: 对数域最小均方误差估计
3. 子空间降噪: 基于信号子空间分解
4. 噪声门: 基于能量阈值的噪声抑制

所有算法继承自 BaseAlgorithm，通过 @register_algorithm 装饰器注册。
"""

from typing import Any, ClassVar, Dict, Optional

import numpy as np

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


# ==================== 辅助函数 ====================


def _stft(signal: np.ndarray, n_fft: int = 512, hop_length: Optional[int] = None,
          win_length: Optional[int] = None, window: str = "hann") -> np.ndarray:
    """短时傅里叶变换 (STFT)。

    Args:
        signal: 1D 输入信号
        n_fft: FFT 点数
        hop_length: 帧移，默认 n_fft // 4
        win_length: 窗长，默认 n_fft
        window: 窗函数类型

    Returns:
        复频谱矩阵，shape (n_freqs, n_frames)
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    # 窗函数
    if window == "hann":
        win = np.hanning(win_length)
    elif window == "hamming":
        win = np.hamming(win_length)
    elif window == "blackman":
        win = np.blackman(win_length)
    else:
        win = np.ones(win_length)

    # 信号填充
    pad_len = n_fft - (len(signal) % hop_length)
    if pad_len > 0 and pad_len < n_fft:
        signal_pad = np.pad(signal, (0, pad_len), mode="reflect")
    else:
        signal_pad = signal.copy()

    # 分帧
    n_frames = (len(signal_pad) - win_length) // hop_length + 1
    if n_frames < 1:
        n_frames = 1
        signal_pad = np.pad(signal_pad, (0, n_fft - len(signal_pad)), mode="constant")

    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)

    for t in range(n_frames):
        start = t * hop_length
        end = start + win_length
        if end > len(signal_pad):
            break
        frame = signal_pad[start:end] * win
        spectrum = np.fft.rfft(frame, n=n_fft)
        stft_matrix[:, t] = spectrum

    return stft_matrix


def _istft(stft_matrix: np.ndarray, n_fft: int = 512, hop_length: Optional[int] = None,
           win_length: Optional[int] = None, window: str = "hann",
           original_length: Optional[int] = None) -> np.ndarray:
    """短时傅里叶逆变换 (ISTFT)。

    Args:
        stft_matrix: 复频谱矩阵 (n_freqs, n_frames)
        n_fft: FFT 点数
        hop_length: 帧移
        win_length: 窗长
        window: 窗函数类型
        original_length: 原始信号长度（用于裁剪）

    Returns:
        重构的时域信号
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if window == "hann":
        win = np.hanning(win_length)
    elif window == "hamming":
        win = np.hamming(win_length)
    elif window == "blackman":
        win = np.blackman(win_length)
    else:
        win = np.ones(win_length)

    n_freqs, n_frames = stft_matrix.shape
    expected_length = hop_length * (n_frames - 1) + win_length
    signal = np.zeros(expected_length, dtype=float)
    window_sum = np.zeros(expected_length, dtype=float)

    for t in range(n_frames):
        start = t * hop_length
        end = start + win_length
        if end > expected_length:
            break
        spectrum = stft_matrix[:, t]
        frame = np.fft.irfft(spectrum, n=n_fft)[:win_length] * win
        signal[start:end] += frame
        window_sum[start:end] += win ** 2

    # 归一化（OLA 重叠相加）
    window_sum = np.where(window_sum > 1e-12, window_sum, 1.0)
    signal = signal / window_sum

    if original_length is not None:
        signal = signal[:original_length]

    return signal


def _estimate_noise_psd(magnitude: np.ndarray, n_frames: int = 10) -> np.ndarray:
    """使用前 n_frames 帧估计噪声功率谱密度 (PSD)。

    Args:
        magnitude: 幅度谱矩阵 (n_freqs, n_frames)
        n_frames: 用于估计噪声的初始帧数

    Returns:
        噪声PSD (n_freqs,)
    """
    n_frames = min(n_frames, magnitude.shape[1])
    noise_psd = np.mean(magnitude[:, :n_frames] ** 2, axis=1)
    return np.maximum(noise_psd, 1e-12)


def _compute_snr_prior(
    magnitude: np.ndarray,
    noise_psd: np.ndarray,
    gain_prev: np.ndarray,
    alpha: float = 0.98,
) -> np.ndarray:
    """计算先验信噪比 (decision-directed 方法)。

    Args:
        magnitude: 当前帧幅度谱
        noise_psd: 噪声功率谱
        gain_prev: 前一帧的增益
        alpha: 平滑因子

    Returns:
        先验SNR
    """
    # 后验SNR
    posterior_snr = magnitude ** 2 / np.maximum(noise_psd, 1e-12)
    # 先验SNR 的 MMSE 估计
    prior_snr = alpha * (gain_prev ** 2) * posterior_snr + (1 - alpha) * np.maximum(posterior_snr - 1, 0)
    return np.maximum(prior_snr, 1e-12)


# ==================== 1. MMSE-STSA ====================


@register_algorithm(
    name="MMSE-STSA降噪",
    category=AlgorithmCategory.ACOUSTIC,
    complexity=AlgorithmComplexity.HIGH,
    tags=["acoustic", "mmse", "stsa", "speech"],
)
class MmseStsaDenoise(BaseAlgorithm):
    """MMSE-STSA (最小均方误差短时谱幅度) 降噪。

    在频域对语音信号的短时谱幅度进行最小均方误差估计，
    假设语音和噪声的DFT系数服从独立复高斯分布。

    算法步骤:
    1. 对带噪信号做 STFT
    2. 估计噪声功率谱（使用前几帧）
    3. 计算先验和后验信噪比
    4. 计算 MMSE-STSA 增益函数
    5. 应用增益并做 ISTFT 重构

    特点:
    - 语音质量好，残留噪声自然（音乐噪声少）
    - 计算复杂度较高
    - 适用于平稳和非平稳噪声环境
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "noise_estimation_frames": 10,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        noise_frames = int(params.get("noise_estimation_frames", 10))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            try:
                output[ch] = self._mmse_stsa_channel(signal[ch], sample_rate, noise_frames)
            except Exception:
                output[ch] = signal[ch]

        return output[0] if is_1d else output

    def _mmse_stsa_channel(self, sig: np.ndarray, fs: float, noise_frames: int) -> np.ndarray:
        sig = sig.flatten()
        n_fft = 512
        hop = n_fft // 4

        # STFT
        spec = _stft(sig, n_fft=n_fft, hop_length=hop)
        magnitude = np.abs(spec)
        phase = np.angle(spec)

        # 噪声估计
        noise_psd = _estimate_noise_psd(magnitude, noise_frames)

        # MMSE-STSA 处理逐帧
        n_freqs, n_frames = magnitude.shape
        gain_prev = np.ones(n_freqs)

        for t in range(n_frames):
            mag_frame = magnitude[:, t]

            # 先验SNR
            prior_snr = _compute_snr_prior(mag_frame, noise_psd, gain_prev)

            # 后验SNR
            post_snr = mag_frame ** 2 / np.maximum(noise_psd, 1e-12)

            # MMSE-STSA 增益函数
            # gain = Gamma(1.5) * sqrt(nu) / gamma * exp(-nu/2) * [(1+nu)*I0(nu/2) + nu*I1(nu/2)]
            # 其中 nu = prior_snr / (1 + prior_snr) * post_snr
            nu = prior_snr / (1 + prior_snr) * post_snr

            # 近似计算增益 (Ephraim & Malah, 1984)
            # 使用近似公式: gain ≈ sqrt( (xi / (1+xi)) * ((1+nu)/nu) )
            # 更精确的版本涉及修正贝塞尔函数
            xi = prior_snr
            v = nu

            gain = np.zeros_like(xi)
            # 对每个频点计算
            for k in range(n_freqs):
                if v[k] < 1e-12:
                    gain[k] = 1.0
                    continue
                # 使用近似: G = sqrt( xi/(1+xi) * (1+v)/v )
                # 这是 MMSE-STSA 增益的简化形式
                gain[k] = np.sqrt(xi[k] / (1 + xi[k]) * (1 + v[k]) / v[k])
                gain[k] = min(gain[k], 1.0)

            gain_prev = gain

            # 应用增益
            magnitude[:, t] = mag_frame * gain

        # ISTFT 重构
        spec_denoised = magnitude * np.exp(1j * phase)
        return _istft(spec_denoised, n_fft=n_fft, hop_length=hop, original_length=len(sig))


# ==================== 2. Log-MMSE ====================


@register_algorithm(
    name="Log-MMSE降噪",
    category=AlgorithmCategory.ACOUSTIC,
    complexity=AlgorithmComplexity.HIGH,
    tags=["acoustic", "log-mmse", "speech", "log-spectral"],
)
class LogMmseDenoise(BaseAlgorithm):
    """Log-MMSE (对数域最小均方误差) 降噪。

    在 LOG 谱域进行最小均方误差估计，相比 MMSE-STSA，
    Log-MMSE 对谱幅度的对数进行估计，更符合人耳听觉特性，
    通常能进一步减少音乐噪声。

    算法步骤:
    1. 对带噪信号做 STFT
    2. 估计噪声功率谱
    3. 计算先验/后验信噪比
    4. 计算 Log-MMSE 增益函数
    5. 应用增益并做 ISTFT 重构

    特点:
    - 相比 MMSE-STSA，音乐噪声更少
    - 主观听感更好（符合人耳对数感知特性）
    - 与 MMSE-STSA 计算复杂度相近
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "noise_estimation_frames": 10,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        noise_frames = int(params.get("noise_estimation_frames", 10))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            try:
                output[ch] = self._log_mmse_channel(signal[ch], sample_rate, noise_frames)
            except Exception:
                output[ch] = signal[ch]

        return output[0] if is_1d else output

    def _log_mmse_channel(self, sig: np.ndarray, fs: float, noise_frames: int) -> np.ndarray:
        sig = sig.flatten()
        n_fft = 512
        hop = n_fft // 4

        # STFT
        spec = _stft(sig, n_fft=n_fft, hop_length=hop)
        magnitude = np.abs(spec)
        phase = np.angle(spec)

        # 噪声估计
        noise_psd = _estimate_noise_psd(magnitude, noise_frames)

        # Log-MMSE 逐帧处理
        n_freqs, n_frames = magnitude.shape
        gain_prev = np.ones(n_freqs)

        for t in range(n_frames):
            mag_frame = magnitude[:, t]

            # 先验/后验 SNR
            prior_snr = _compute_snr_prior(mag_frame, noise_psd, gain_prev)
            post_snr = mag_frame ** 2 / np.maximum(noise_psd, 1e-12)

            xi = prior_snr
            v = xi / (1 + xi) * post_snr

            # Log-MMSE 增益函数 (Ephraim & Malah, 1985)
            # G = xi/(1+xi) * exp( 0.5 * integral_{v}^{inf} e^{-t}/t dt )
            # 使用近似公式
            gain = np.zeros_like(xi)
            for k in range(n_freqs):
                if v[k] < 1e-12:
                    gain[k] = 1.0
                    continue
                # 指数积分 E1(v) 的近似
                e1 = self._exp_int(v[k])
                # Log-MMSE 增益: G = xi/(1+xi) * exp(0.5 * e1(v))
                gain[k] = xi[k] / (1 + xi[k]) * np.exp(0.5 * e1)
                gain[k] = min(gain[k], 1.0)

            gain_prev = gain
            magnitude[:, t] = mag_frame * gain

        # ISTFT
        spec_denoised = magnitude * np.exp(1j * phase)
        return _istft(spec_denoised, n_fft=n_fft, hop_length=hop, original_length=len(sig))

    @staticmethod
    def _exp_int(x: float) -> float:
        """计算指数积分 E1(x) = integral_{x}^{inf} e^{-t}/t dt 的近似值。

        使用有理函数近似，适用于 x > 0。
        """
        if x <= 0:
            return 0.0
        if x > 1.0:
            # 对大 x 的渐近展开
            a1, a2, a3, a4, a5 = 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857
            return np.exp(-x) / x * (a1 + a2 / x + a3 / x ** 2 + a4 / x ** 3 + a5 / x ** 4)
        else:
            # 对小 x 的级数展开
            return -np.log(x) - 0.57721566 + x - x ** 2 / 4 + x ** 3 / 18 - x ** 4 / 96 + x ** 5 / 600


# ==================== 3. 子空间降噪 ====================


@register_algorithm(
    name="子空间降噪(Subspace)",
    category=AlgorithmCategory.ACOUSTIC,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["acoustic", "subspace", "eigenvalue", "decomposition"],
)
class SubspaceDenoise(BaseAlgorithm):
    """子空间降噪 (信号子空间方法)。

    将带噪信号的向量空间分解为信号子空间和噪声子空间。
    通过在信号子空间中保留主要成分并舍弃噪声子空间来实现降噪。
    基于信号向量的协方差矩阵的特征值分解。

    算法步骤:
    1. 构造信号的延迟嵌入矩阵（轨迹矩阵）
    2. 计算协方差矩阵并进行特征值分解
    3. 根据特征值分布区分信号和噪声子空间
    4. 在信号子空间中重构信号

    特点:
    - 无需显式的噪声谱估计
    - 对有色噪声效果较好
    - 适用于短时平稳信号
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_components": None,
        "subspace_factor": 0.1,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        n_components = params.get("n_components")
        subspace_factor = float(params.get("subspace_factor", 0.1))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            try:
                output[ch] = self._subspace_channel(
                    signal[ch], n_components, subspace_factor,
                )
            except Exception:
                output[ch] = signal[ch]

        return output[0] if is_1d else output

    def _subspace_channel(
        self,
        sig: np.ndarray,
        n_components: Optional[int],
        subspace_factor: float,
    ) -> np.ndarray:
        sig = sig.flatten()
        n = len(sig)

        # 构造轨迹矩阵（Hankel 矩阵）
        # 窗口长度建议为 n/3 ~ n/2
        L = n // 3
        L = max(3, min(L, n - 1))
        K = n - L + 1

        H = np.zeros((L, K))
        for i in range(L):
            H[i, :] = sig[i:i + K]

        # 协方差矩阵
        C = H @ H.T / K

        # 特征值分解
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            return sig

        # 特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 确定信号子空间维度
        if n_components is not None:
            r = int(n_components)
        else:
            # 基于能量阈值自动选择
            total_energy = np.sum(eigenvalues)
            cumsum = np.cumsum(eigenvalues)
            # 找到累积能量超过 (1 - subspace_factor) 的最小维度
            threshold = (1.0 - subspace_factor) * total_energy
            r = int(np.searchsorted(cumsum, threshold) + 1)

        r = max(1, min(r, L))

        # 信号子空间投影
        U_signal = eigenvectors[:, :r]
        # 子空间权重（较大的特征值赋较高权重）
        weights = np.sqrt(np.maximum(eigenvalues[:r], 0))
        # 对Hankel矩阵的每一列投影到信号子空间
        H_denoised = U_signal @ (U_signal.T @ H)

        # 反对角线平均重构信号
        recon = np.zeros(n)
        count = np.zeros(n)
        for i in range(L):
            for j in range(K):
                recon[i + j] += H_denoised[i, j]
                count[i + j] += 1

        count = np.where(count > 0, count, 1.0)
        return recon / count


# ==================== 4. 噪声门 (Noise Gate) ====================


@register_algorithm(
    name="噪声门(Noise Gate)",
    category=AlgorithmCategory.ACOUSTIC,
    complexity=AlgorithmComplexity.LOW,
    tags=["acoustic", "noise-gate", "threshold", "real-time"],
)
class NoiseGateDenoise(BaseAlgorithm):
    """噪声门 (Noise Gate) 降噪。

    根据信号能量是否超过阈值来决定是否打开或关闭音频通道。
    当信号能量低于阈值时，认为是纯噪声段并进行衰减；
    当信号能量高于阈值时，认为是有效信号段并保持增益。

    算法步骤:
    1. 将信号分帧
    2. 计算每帧的能量（dB）
    3. 能量超过门限则开启/持续，低于门限则关闭/衰减
    4. 应用攻击/释放/保持时间实现平滑过渡

    特点:
    - 计算量极低，适合实时处理
    - 参数直观易调节
    - 对间歇性噪声（如录音中的背景噪声）效果明显
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "threshold_db": -40,
        "attack_ms": 10,
        "release_ms": 100,
        "hold_ms": 50,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        threshold_db = float(params.get("threshold_db", -40))
        attack_ms = float(params.get("attack_ms", 10))
        release_ms = float(params.get("release_ms", 100))
        hold_ms = float(params.get("hold_ms", 50))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            try:
                output[ch] = self._noise_gate_channel(
                    signal[ch], sample_rate,
                    threshold_db, attack_ms, release_ms, hold_ms,
                )
            except Exception:
                output[ch] = signal[ch]

        return output[0] if is_1d else output

    def _noise_gate_channel(
        self,
        sig: np.ndarray,
        fs: float,
        threshold_db: float,
        attack_ms: float,
        release_ms: float,
        hold_ms: float,
    ) -> np.ndarray:
        sig = sig.flatten()
        n = len(sig)

        # 帧长和帧移（选择小帧长以快速响应）
        frame_len = int(fs * 0.01)  # 10ms 帧
        if frame_len < 1:
            frame_len = 1
        hop = frame_len // 2  # 50% 重叠

        # 帧数
        n_frames = max(1, (n - frame_len) // hop + 1)

        # 计算每帧的 RMS 和增益
        frame_rms = np.zeros(n_frames)
        for t in range(n_frames):
            start = t * hop
            end = min(start + frame_len, n)
            frame = sig[start:end]
            frame_rms[t] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0

        # RMS 转 dB（避免 log(0)）
        frame_db = np.where(frame_rms > 1e-12, 20 * np.log10(frame_rms), -120)

        # 阈值线性化：threshold_db 转线性门限
        threshold_linear = 10 ** (threshold_db / 20) if threshold_db > -120 else 0

        # 攻击/释放/保持 时间转帧数
        attack_frames = max(1, int(attack_ms / 1000 * fs / hop))
        release_frames = max(1, int(release_ms / 1000 * fs / hop))
        hold_frames = max(0, int(hold_ms / 1000 * fs / hop))

        # 生成门控信号
        gate_open = frame_rms >= threshold_linear

        # 平滑门控（考虑保持时间）
        smoothed_gate = np.zeros(n_frames, dtype=float)
        gate_state = False
        hold_counter = 0

        for t in range(n_frames):
            if gate_open[t]:
                gate_state = True
                hold_counter = 0
            else:
                if hold_counter < hold_frames:
                    hold_counter += 1
                else:
                    gate_state = False

            smoothed_gate[t] = 1.0 if gate_state else 0.0

        # 应用攻击/释放包络
        envelope = np.zeros(n_frames)
        current_gain = 0.0
        for t in range(n_frames):
            target = smoothed_gate[t]
            if target > current_gain:
                # 攻击
                current_gain += (target - current_gain) / attack_frames
            else:
                # 释放
                current_gain += (target - current_gain) / release_frames
            envelope[t] = max(0.0, min(1.0, current_gain))

        # 应用增益到信号（逐帧重叠相加）
        output = np.zeros(n)
        weight = np.zeros(n)
        for t in range(n_frames):
            start = t * hop
            end = min(start + frame_len, n)
            gain = envelope[t]
            output[start:end] += sig[start:end] * gain
            weight[start:end] += 1.0

        weight = np.where(weight > 0, weight, 1.0)
        return output / weight
