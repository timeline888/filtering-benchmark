"""
评估指标计算模块（20种指标）。

分为五大类：
1. 特征频率直接度量（FFA, RFFI, FBR, ER, LSNR, HER）
2. 统计分布特征（Kurtosis, SK, ESK, Skewness）
3. 综合特征比与范数（FFR, HSI, SLN, HLN）
4. 稀疏性与复杂度（GI, ESE, SI）
5. 时域与扩展（CF, IF, SER）
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from loguru import logger

from ..core.types import MetricResult, EnvelopeResult
from .base import BaseMetric
from .base import metric_registry


# ============================================================
# 第一类：特征频率直接度量
# ============================================================

@metric_registry.register
class FFAMetric(BaseMetric):
    """特征频率幅值 (FFA): 包络谱在故障特征频率处的幅值"""
    name = "ffa"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope and envelope.fault_freq_matches:
            return max(envelope.fault_freq_matches.values())
        return 0.0


@metric_registry.register
class RFFIMetric(BaseMetric):
    """特征频率相对强度 (RFFI): 特征频率幅值 / 噪声基底均值"""
    name = "rffi"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None or not envelope.fault_freq_matches:
            return 0.0
        noise_floor = np.mean(envelope.envelope_spectrum[envelope.freq_axis > 0])
        if noise_floor <= 0:
            return 0.0
        return max(envelope.fault_freq_matches.values()) / noise_floor


@metric_registry.register
class FBRMetric(BaseMetric):
    """频带占比 (FBR): 故障频带宽度占分析总带宽的比例"""
    name = "fbr"
    category = "包络域指标"
    higher_is_better = False  # 越小越好（信号集中）

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None:
            return 1.0
        total_bw = envelope.freq_axis[-1]
        if total_bw <= 0:
            return 1.0
        # 以最高峰为中心，半高宽作为特征带宽
        spec = envelope.envelope_spectrum
        max_idx = np.argmax(spec)
        half_max = spec[max_idx] / 2
        above_half = spec >= half_max
        if not np.any(above_half):
            return 0.1
        indices = np.where(above_half)[0]
        feature_bw = envelope.freq_axis[indices[-1]] - envelope.freq_axis[indices[0]]
        return min(feature_bw / total_bw, 1.0)


@metric_registry.register
class ERMetric(BaseMetric):
    """能量占比 (ER): 故障频带能量 / 总能量"""
    name = "er"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None:
            return 0.0
        spec = envelope.envelope_spectrum
        total_energy = np.sum(spec ** 2)
        if total_energy <= 0:
            return 0.0
        # 围绕主导频率的频带能量
        fault_freqs = list(envelope.fault_freq_matches.keys())
        if not fault_freqs:
            return 0.0
        # 取特征频率附近 ± 3% 带宽
        fault_config = kwargs.get("fault_freq_config", {})
        fault_energy = 0
        for fname in fault_freqs:
            if fname in fault_config:
                target = fault_config[fname]
                bw = target * 0.03
                mask = np.abs(envelope.freq_axis - target) <= bw
                fault_energy += np.sum(spec[mask] ** 2)
        return fault_energy / total_energy


@metric_registry.register
class LSNRMetric(BaseMetric):
    """局部信噪比 (LSNR): 20·log10(特征频率幅值 / 旁瓣噪声RMS)"""
    name = "lsnr"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None or not envelope.fault_freq_matches:
            return 0.0
        spec = envelope.envelope_spectrum
        max_fault_amp = max(envelope.fault_freq_matches.values())
        # 旁瓣噪声：排除前5%最大峰值的所有点
        sorted_spec = np.sort(spec)
        noise = sorted_spec[:max(int(len(sorted_spec) * 0.8), 1)]
        noise_rms = np.sqrt(np.mean(noise ** 2))
        if noise_rms <= 0:
            return 40.0
        return 20 * np.log10(max_fault_amp / noise_rms)


@metric_registry.register
class HERMetric(BaseMetric):
    """谐波能量占比 (HER): 特征频率及倍频能量 / 总能量"""
    name = "her"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None or not envelope.fault_freq_matches:
            return 0.0
        spec = envelope.envelope_spectrum
        total_energy = np.sum(spec ** 2)
        if total_energy <= 0:
            return 0.0
        # 对每个特征频率计算基频 + 3阶谐波能量
        harmonic_energy = 0.0
        for fname, famp in envelope.fault_freq_matches.items():
            if fname == "fr":
                continue
            for h in range(1, 4):
                hf = famp * h if False else self._get_freq_for_name(fname)
                if hf is None:
                    continue
                bw = hf * 0.02
                mask = np.abs(envelope.freq_axis - hf * h) <= bw
                harmonic_energy += np.sum(spec[mask] ** 2)
        return harmonic_energy / total_energy

    def _get_freq_for_name(self, name: str) -> Optional[float]:
        # 从envelope的fault_freq_matches中获取基频
        return None  # 简化处理


# ============================================================
# 第二类：统计分布特征
# ============================================================

@metric_registry.register
class KurtosisMetric(BaseMetric):
    """时域峭度 (Kurtosis): 信号分布的尖峰程度"""
    name = "kurtosis"
    category = "时域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        sig = denoised.ravel()
        return float(kurtosis(sig, fisher=False))


@metric_registry.register
class SKMetric(BaseMetric):
    """谱峭度 (SK): 频率分量的非平稳性度量"""
    name = "sk"
    category = "频域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        sig = denoised.ravel()
        # STFT -> 谱峭度
        f, t, Zxx = scipy_signal.stft(sig, fs=sample_rate, nperseg=min(256, len(sig) // 4))
        sk_val = np.mean(kurtosis(np.abs(Zxx), fisher=False, axis=1))
        return float(sk_val)


@metric_registry.register
class ESKMetric(BaseMetric):
    """包络谱峰度 (ESK): 包络信号频谱的峰度"""
    name = "esk"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None:
            return 0.0
        spec = envelope.envelope_spectrum[envelope.freq_axis > 0]
        if len(spec) == 0:
            return 0.0
        return float(kurtosis(spec, fisher=False))


@metric_registry.register
class SkewnessMetric(BaseMetric):
    """偏度 (Skewness): 信号分布的不对称性"""
    name = "skewness"
    category = "时域指标"
    higher_is_better = True  # 绝对值越大越好

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        return float(abs(skew(denoised.ravel())))


# ============================================================
# 第三类：综合特征比与范数
# ============================================================

@metric_registry.register
class FFRMetric(BaseMetric):
    """故障特征比 (FFR): 特征频率及谐波幅值和 / 全频带总幅值"""
    name = "ffr"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None or not envelope.fault_freq_matches:
            return 0.0
        total = np.sum(envelope.envelope_spectrum)
        if total <= 0:
            return 0.0
        fault_sum = sum(envelope.fault_freq_matches.values())
        return fault_sum / total


@metric_registry.register
class HSIMetric(BaseMetric):
    """谐波显著性指标 (HSI): 谐波峰值相对局部噪声的超出度"""
    name = "hsi"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None or not envelope.fault_freq_matches:
            return 0.0
        spec = envelope.envelope_spectrum
        scores = []
        for fname, famp in envelope.fault_freq_matches.items():
            f = 0
            # 查找特征频率
            for freq, amp in envelope.dominant_freqs:
                if abs(amp - famp) / max(famp, 1) < 0.1:
                    f = freq
                    break
            if f > 0:
                bw = max(f * 0.05, 1)
                mask = np.abs(envelope.freq_axis - f) <= bw
                local_mean = np.mean(spec[mask])
                if local_mean > 0:
                    scores.append((famp - local_mean) / local_mean)
        return np.mean(scores) if scores else 0.0


@metric_registry.register
class SLNMetric(BaseMetric):
    """谱 L2/L1 范数比 (SLN): 衡量包络谱的能量集中度"""
    name = "sln"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None:
            return 0.0
        spec = envelope.envelope_spectrum[envelope.freq_axis > 0]
        l2 = np.sqrt(np.sum(spec ** 2))
        l1 = np.sum(np.abs(spec))
        if l1 <= 0:
            return 0.0
        return float(l2 / l1)


@metric_registry.register
class HLNMetric(BaseMetric):
    """谐波 L2/L1 范数 (HLN): 特征频率处的L2/L1范数比"""
    name = "hln"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None or not envelope.fault_freq_matches:
            return 0.0
        amps = list(envelope.fault_freq_matches.values())
        l2 = np.sqrt(sum(a ** 2 for a in amps))
        l1 = sum(abs(a) for a in amps)
        if l1 <= 0:
            return 0.0
        return l2 / l1


# ============================================================
# 第四类：稀疏性与复杂度
# ============================================================

@metric_registry.register
class GIMetric(BaseMetric):
    """基尼指数 (GI): 包络谱能量的不均匀程度 [0, 1]"""
    name = "gi"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None:
            return 0.0
        spec = np.sort(envelope.envelope_spectrum[envelope.freq_axis > 0])
        if len(spec) == 0 or np.sum(spec) == 0:
            return 0.0
        n = len(spec)
        cumsum = np.cumsum(spec)
        return float(1 - 2 * np.sum((n - np.arange(1, n + 1) + 0.5) * spec) / (n * np.sum(spec)))


@metric_registry.register
class ESEMetric(BaseMetric):
    """包络谱熵 (ESE): 包络谱的信息熵"""
    name = "ese"
    category = "包络域指标"
    higher_is_better = False  # 越小越好

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None:
            return float('inf')
        p = envelope.envelope_spectrum[envelope.freq_axis > 0]
        p = p / (np.sum(p) + 1e-12)
        entropy = -np.sum(p * np.log2(p + 1e-12))
        # 归一化
        max_entropy = np.log2(len(p))
        return float(entropy / max_entropy) if max_entropy > 0 else 1.0


@metric_registry.register
class SIMetric(BaseMetric):
    """平滑度指标 (SI): 相邻频率幅值差的变异系数"""
    name = "si"
    category = "包络域指标"
    higher_is_better = False  # 越小越好

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None:
            return 1.0
        spec = envelope.envelope_spectrum[envelope.freq_axis > 0]
        if len(spec) <= 1:
            return 0.0
        diffs = np.abs(np.diff(spec))
        return float(np.std(diffs) / (np.mean(diffs) + 1e-12))


# ============================================================
# 第五类：时域与扩展指标
# ============================================================

@metric_registry.register
class CFMetric(BaseMetric):
    """峰值因子 (CF): max(|x|) / RMS(x)"""
    name = "cf"
    category = "时域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        sig = denoised.ravel()
        peak = np.max(np.abs(sig))
        rms = np.sqrt(np.mean(sig ** 2))
        return float(peak / rms) if rms > 0 else 1.0


@metric_registry.register
class IFMetric(BaseMetric):
    """脉冲因子 (IF): max(|x|) / mean(|x|)"""
    name = "if"
    category = "时域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        sig = denoised.ravel()
        peak = np.max(np.abs(sig))
        mean_abs = np.mean(np.abs(sig))
        return float(peak / mean_abs) if mean_abs > 0 else 1.0


@metric_registry.register
class SERMetric(BaseMetric):
    """边带能量比 (SER): 调制边带能量 / 总能量"""
    name = "ser"
    category = "包络域指标"
    higher_is_better = True

    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        if envelope is None:
            return 0.0
        spec = envelope.envelope_spectrum
        total_energy = np.sum(spec ** 2)
        if total_energy <= 0:
            return 0.0
        # 获取前3个主导频率，计算其 ± 转频 的边带能量
        sideband_energy = 0.0
        shaft_freq = 0
        # 从fault_freq_matches获取转频
        if "fr" in envelope.fault_freq_matches:
            shaft_freq = 0  # 需要实际频率值，简化处理
        for freq, amp in envelope.dominant_freqs[:3]:
            if freq > 0:
                bw = freq * 0.1
                mask = np.abs(envelope.freq_axis - freq) <= bw
                sideband_energy += np.sum(spec[mask] ** 2)
        # 总主导能量的比例
        total_dominant = sum(a ** 2 for _, a in envelope.dominant_freqs[:5])
        if total_dominant <= 0:
            return 0.0
        return sideband_energy / total_dominant
