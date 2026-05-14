"""
信号仿真生成模块。

用于生成标准测试信号的"干净参考信号"，以支持有监督评估。
包含：
1. 轴承故障仿真（Randall模型）
2. 齿轮故障仿真（AM-FM模型）
3. 通用合成信号（chirp/多谐波/脉冲）
"""

from typing import Optional, Tuple

import numpy as np


class SignalSimulator:
    """信号仿真器"""

    @staticmethod
    def bearing_fault(sample_rate: float, duration: float,
                      fault_freq: float, shaft_freq: float = 25.0,
                      impact_decay: float = 0.01, noise_level: float = 0.0,
                      n_harmonics: int = 3) -> np.ndarray:
        """
        生成轴承故障仿真信号（Randall模型）。

        数学模型：x(t) = Σ A_k · exp(-α·(t - kT)) · sin(2π·f_r·(t - kT))

        Args:
            sample_rate: 采样率 (Hz)
            duration: 信号时长 (秒)
            fault_freq: 故障特征频率 (Hz)
            shaft_freq: 转频 (Hz)
            impact_decay: 冲击衰减常数
            noise_level: 添加噪声的标准差
            n_harmonics: 谐波个数

        Returns:
            仿真的干净信号 (不含噪声)
        """
        n = int(sample_rate * duration)
        t = np.arange(n) / sample_rate
        signal = np.zeros(n)

        T_fault = 1.0 / max(fault_freq, 0.1)

        for h in range(1, n_harmonics + 1):
            T_h = T_fault / h
            for start in np.arange(0, duration, T_h):
                idx = int(start * sample_rate)
                if idx >= n:
                    break
                n_samples = min(int(impact_decay * sample_rate * 3), n - idx)
                if n_samples <= 0:
                    continue
                local_t = t[:n_samples]
                # 指数衰减正弦冲击
                impact = np.exp(-local_t / impact_decay) * np.sin(2 * np.pi * shaft_freq * h * local_t)
                # 随机幅值抖动
                amp = 1.0 + 0.2 * np.random.randn()
                signal[idx:idx + n_samples] += amp * impact * 0.5

        # 添加调制
        modulation = 1 + 0.3 * np.sin(2 * np.pi * shaft_freq * t)
        signal *= modulation

        # 添加噪声
        if noise_level > 0:
            signal += noise_level * np.random.randn(n)

        return signal

    @staticmethod
    def gear_fault(sample_rate: float, duration: float,
                   gmf: float = 500.0, shaft_freq: float = 25.0,
                   mod_depth: float = 0.3, noise_level: float = 0.0) -> np.ndarray:
        """
        生成齿轮故障仿真信号（AM-FM模型）。

        数学模型：x(t) = (1 + m_a·sin(2π·f_r·t)) · cos(2π·GMF·t + m_f·sin(2π·f_r·t))

        Args:
            sample_rate: 采样率 (Hz)
            duration: 信号时长 (秒)
            gmf: 齿轮啮合频率 (Hz)
            shaft_freq: 转频 (Hz)
            mod_depth: 调制深度
            noise_level: 噪声水平

        Returns:
            仿真的干净齿轮信号
        """
        n = int(sample_rate * duration)
        t = np.arange(n) / sample_rate

        # 载波 + AM + FM
        carrier = np.cos(2 * np.pi * gmf * t + mod_depth * np.sin(2 * np.pi * shaft_freq * t))
        am = 1 + mod_depth * np.cos(2 * np.pi * shaft_freq * t)
        signal = am * carrier

        if noise_level > 0:
            signal += noise_level * np.random.randn(n)

        return signal

    @staticmethod
    def chirp(sample_rate: float, duration: float,
              f0: float = 10.0, f1: float = 1000.0,
              noise_level: float = 0.0) -> np.ndarray:
        """生成扫频信号 (chirp)"""
        from scipy import signal as scipy_signal
        n = int(sample_rate * duration)
        t = np.arange(n) / sample_rate
        signal = scipy_signal.chirp(t, f0=f0, t1=duration, f1=f1, method='linear')

        if noise_level > 0:
            signal += noise_level * np.random.randn(n)

        return signal

    @staticmethod
    def multi_harmonic(sample_rate: float, duration: float,
                        harmonics: list, noise_level: float = 0.0) -> np.ndarray:
        """
        生成多谐波合成信号。

        Args:
            sample_rate: 采样率 (Hz)
            duration: 信号时长 (秒)
            harmonics: [(freq, amplitude, phase), ...]
            noise_level: 噪声水平
        """
        n = int(sample_rate * duration)
        t = np.arange(n) / sample_rate
        signal = np.zeros(n)

        for freq, amp, phase in harmonics:
            signal += amp * np.sin(2 * np.pi * freq * t + phase)

        if noise_level > 0:
            signal += noise_level * np.random.randn(n)

        return signal

    @staticmethod
    def mixed_signal(sample_rate: float, duration: float,
                     signals: list, noise_level: float = 0.0) -> np.ndarray:
        """
        生成混合信号。

        Args:
            sample_rate: 采样率 (Hz)
            duration: 信号时长 (秒)
            signals: [(generator_fn, weight), ...]
            noise_level: 噪声水平
        """
        n = int(sample_rate * duration)
        combined = np.zeros(n)

        for fn, weight in signals:
            sig = fn()
            if len(sig) > n:
                sig = sig[:n]
            elif len(sig) < n:
                sig = np.pad(sig, (0, n - len(sig)))
            combined += weight * sig

        if noise_level > 0:
            combined += noise_level * np.random.randn(n)

        return combined
