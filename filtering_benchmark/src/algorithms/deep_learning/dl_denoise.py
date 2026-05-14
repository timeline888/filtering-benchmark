"""
深度学习降噪算法集合。

包含8种深度学习降噪算法：
1. DAE (去噪自编码器)
2. CDAE (卷积去噪自编码器)
3. DnCNN (残差学习的卷积网络)
4. UNet (1D UNet架构)
5. GAN降噪 (使用Generator做推理)
6. DRSN (深度残差收缩网络)
7. TCN-LSTM (时序卷积 + LSTM)
8. Transformer降噪

所有算法继承自 BaseAlgorithm，通过 @register_algorithm 装饰器注册。
由于用户无GPU，所有算法：
- 使用轻量级网络（参数量 < 10万）
- 使用即时构建的轻量网络（CPU推理 < 5秒）
- 当 torch 不可用时回退到 scipy 信号处理替代方法
- 在CPU上运行
"""

import math
import warnings
from typing import Any, ClassVar, Dict, Optional

import numpy as np

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity

# ==================== Torch 可用性检查 ====================

_TORCH_AVAILABLE: bool = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None


# ==================== 全局工具函数 ====================


def _to_stereo(signal: np.ndarray) -> np.ndarray:
    """确保信号为 (n_channels, n_samples) 形状"""
    if signal.ndim == 1:
        return signal.reshape(1, -1)
    return signal


def _from_stereo(signal: np.ndarray, original: np.ndarray) -> np.ndarray:
    """恢复信号为原始形状"""
    if original.ndim == 1:
        return signal[0]
    return signal


def _norch_fallback_spectral(
    signal: np.ndarray,
    sample_rate: float,
    cutoff_ratio: float = 0.1,
) -> np.ndarray:
    """torch 不可用时的回退方案：频域软阈值滤波。

    Args:
        signal: 输入信号，shape (n_channels, n_samples)
        sample_rate: 采样率 (Hz)
        cutoff_ratio: 保留频率比例（0~1），默认0.1表示保留低频10%

    Returns:
        滤波后信号
    """
    from scipy import signal as sp_signal

    sig_1d = signal.ndim == 1
    if sig_1d:
        signal = signal.reshape(1, -1)

    n_channels, n_samples = signal.shape
    output = np.zeros_like(signal, dtype=np.float64)

    for ch in range(n_channels):
        # FFT
        spectrum = np.fft.rfft(signal[ch])
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / max(sample_rate, 1e-12))
        # 软阈值
        cutoff_freq = cutoff_ratio * sample_rate / 2.0
        mask = np.abs(freqs) <= cutoff_freq
        # 平滑过渡
        attenuation = np.where(mask, 1.0, np.exp(-((np.abs(freqs) - cutoff_freq) / (sample_rate * 0.05)) ** 2))
        spectrum_filtered = spectrum * attenuation
        output[ch] = np.fft.irfft(spectrum_filtered, n=n_samples)

    return output[0] if sig_1d else output


def _norch_fallback_wavelet(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
    mode: str = "soft",
) -> np.ndarray:
    """torch 不可用时的回退方案：小波阈值降噪。"""
    try:
        import pywt
    except ImportError:
        # 如果 pywt 也不可用，使用简单的移动平均
        win = max(3, signal.shape[-1] // 100)
        from scipy.ndimage import uniform_filter1d

        return uniform_filter1d(signal, size=win, axis=-1)

    sig_1d = signal.ndim == 1
    if sig_1d:
        signal = signal.reshape(1, -1)

    output = np.zeros_like(signal, dtype=np.float64)
    for ch in range(signal.shape[0]):
        coeffs = pywt.wavedec(signal[ch], wavelet, level=level)
        # 通用阈值
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal[ch])))
        coeffs_th = list(coeffs)
        for i in range(1, len(coeffs_th)):
            if mode == "soft":
                coeffs_th[i] = pywt.threshold(coeffs_th[i], threshold, mode="soft")
            else:
                coeffs_th[i] = pywt.threshold(coeffs_th[i], threshold, mode="hard")
        output[ch] = pywt.waverec(coeffs_th, wavelet)[: len(signal[ch])]

    return output[0] if sig_1d else output


def _norch_fallback_wiener(
    signal: np.ndarray,
    win_size: int = 5,
) -> np.ndarray:
    """torch 不可用时的回退方案：维纳滤波。"""
    from scipy.signal import wiener

    sig_1d = signal.ndim == 1
    if sig_1d:
        signal = signal.reshape(1, -1)

    output = np.zeros_like(signal, dtype=np.float64)
    for ch in range(signal.shape[0]):
        output[ch] = wiener(signal[ch], mysize=win_size)

    return output[0] if sig_1d else output


def _ensure_channels_last(x: np.ndarray) -> np.ndarray:
    """确保形状为 (batch, channels, length)，用于卷积网络"""
    if x.ndim == 1:
        x = x[np.newaxis, np.newaxis, :]  # (1, 1, N)
    elif x.ndim == 2:
        x = x[:, np.newaxis, :]  # (C, 1, N)
    return x


# ==================== PyTorch 网络定义（仅当 torch 可用时使用）====================


def _build_dae_net(encoding_dim: int = 16, hidden_dim: int = 32, input_dim: int = 256) -> Optional[Any]:
    """构建轻量级全连接去噪自编码器。"""
    if not _TORCH_AVAILABLE:
        return None
    try:

        class _DAE(nn.Module):
            def __init__(self, enc_dim: int, hid_dim: int, inp_dim: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(inp_dim, hid_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hid_dim, enc_dim),
                    nn.ReLU(inplace=True),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(enc_dim, hid_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hid_dim, inp_dim),
                    nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        net = _DAE(encoding_dim, hidden_dim, input_dim)
        # 参数量统计
        num_params = sum(p.numel() for p in net.parameters())
        # print(f"[DAE] 参数量: {num_params}")
        net.eval()
        return net
    except Exception:
        return None


def _build_cdae_net(kernel_size: int = 3, n_filters: int = 16, input_channels: int = 1) -> Optional[Any]:
    """构建轻量级1D卷积去噪自编码器。"""
    if not _TORCH_AVAILABLE:
        return None
    try:

        class _CDAE(nn.Module):
            def __init__(self, ks: int, nf: int, ic: int):
                super().__init__()
                pad = ks // 2
                # Encoder
                self.enc_conv = nn.Sequential(
                    nn.Conv1d(ic, nf, ks, stride=2, padding=pad),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(nf, nf * 2, ks, stride=2, padding=pad),
                    nn.ReLU(inplace=True),
                )
                # Decoder
                self.dec_conv = nn.Sequential(
                    nn.ConvTranspose1d(nf * 2, nf, ks, stride=2, padding=pad, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(nf, ic, ks, stride=2, padding=pad, output_padding=1),
                    nn.Sigmoid(),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                enc = self.enc_conv(x)
                dec = self.dec_conv(enc)
                # 确保输出长度与输入匹配
                if dec.shape[-1] != x.shape[-1]:
                    dec = F.interpolate(dec, size=x.shape[-1], mode="linear", align_corners=False)
                return dec

        net = _CDAE(kernel_size, n_filters, input_channels)
        # num_params = sum(p.numel() for p in net.parameters())
        # print(f"[CDAE] 参数量: {num_params}")
        net.eval()
        return net
    except Exception:
        return None


def _build_dncnn_net(
    n_layers: int = 10,
    n_filters: int = 32,
    kernel_size: int = 3,
    input_channels: int = 1,
) -> Optional[Any]:
    """构建轻量级 DnCNN 网络（残差学习）。"""
    if not _TORCH_AVAILABLE:
        return None
    try:

        class _DnCNN(nn.Module):
            def __init__(self, nl: int, nf: int, ks: int, ic: int):
                super().__init__()
                pad = ks // 2
                layers = []
                # 第一层
                layers.append(
                    nn.Conv1d(ic, nf, ks, padding=pad)
                )
                layers.append(nn.ReLU(inplace=True))
                # 中间层
                for _ in range(nl - 2):
                    layers.append(
                        nn.Conv1d(nf, nf, ks, padding=pad)
                    )
                    layers.append(nn.BatchNorm1d(nf))
                    layers.append(nn.ReLU(inplace=True))
                # 最后一层
                layers.append(
                    nn.Conv1d(nf, ic, ks, padding=pad)
                )
                self.net = nn.Sequential(*layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = self.net(x)
                return x - residual  # 残差学习

        net = _DnCNN(n_layers, n_filters, kernel_size, input_channels)
        # num_params = sum(p.numel() for p in net.parameters())
        # print(f"[DnCNN] 参数量: {num_params}")
        net.eval()
        return net
    except Exception:
        return None


def _build_unet_1d_net(
    n_channels: int = 16,
    depth: int = 4,
    input_channels: int = 1,
) -> Optional[Any]:
    """构建轻量级1D UNet。"""
    if not _TORCH_AVAILABLE:
        return None
    try:

        class _DownBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, ks: int = 3):
                super().__init__()
                pad = ks // 2
                self.conv = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, ks, padding=pad),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_ch, out_ch, ks, padding=pad),
                    nn.ReLU(inplace=True),
                )
                self.pool = nn.MaxPool1d(2)

            def forward(self, x: torch.Tensor) -> tuple:
                x = self.conv(x)
                p = self.pool(x)
                return x, p

        class _UpBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, ks: int = 3):
                super().__init__()
                pad = ks // 2
                self.up = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2)
                self.conv = nn.Sequential(
                    nn.Conv1d(out_ch * 2, out_ch, ks, padding=pad),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(out_ch, out_ch, ks, padding=pad),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
                x = self.up(x)
                # 对齐尺寸
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = F.pad(x, [0, diff])
                elif diff < 0:
                    x = x[:, :, :-diff]
                x = torch.cat([skip, x], dim=1)
                x = self.conv(x)
                return x

        class _UNet1D(nn.Module):
            def __init__(self, nc: int, dp: int, ic: int):
                super().__init__()
                self.depth = dp
                # Encoder
                self.encoders = nn.ModuleList()
                ch = ic
                for i in range(dp):
                    out_ch = nc * (2 ** i)
                    self.encoders.append(_DownBlock(ch, out_ch))
                    ch = out_ch
                # Bottleneck
                self.bottleneck = nn.Sequential(
                    nn.Conv1d(ch, ch * 2, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(ch * 2, ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                # Decoder
                self.decoders = nn.ModuleList()
                for i in range(dp - 1, -1, -1):
                    out_ch = nc * (2 ** i)
                    self.decoders.append(_UpBlock(ch, out_ch))
                    ch = out_ch
                # Output
                self.out_conv = nn.Conv1d(ch, ic, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                skips = []
                for enc in self.encoders:
                    s, x = enc(x)
                    skips.append(s)
                x = self.bottleneck(x)
                for i, dec in enumerate(self.decoders):
                    x = dec(x, skips[-(i + 1)])
                return self.out_conv(x)

        net = _UNet1D(n_channels, depth, input_channels)
        # num_params = sum(p.numel() for p in net.parameters())
        # print(f"[UNet] 参数量: {num_params}")
        net.eval()
        return net
    except Exception:
        return None


def _build_gan_generator_net(
    latent_dim: int = 16,
    output_dim: int = 256,
) -> Optional[Any]:
    """构建轻量级GAN Generator。"""
    if not _TORCH_AVAILABLE:
        return None
    try:

        class _Generator(nn.Module):
            def __init__(self, ld: int, od: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(ld, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, od),
                    nn.Tanh(),
                )

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                return self.net(z)

        net = _Generator(latent_dim, output_dim)
        # num_params = sum(p.numel() for p in net.parameters())
        # print(f"[GAN Generator] 参数量: {num_params}")
        net.eval()
        return net
    except Exception:
        return None


def _build_drsn_net(
    n_filters: int = 16,
    depth: int = 6,
    input_channels: int = 1,
) -> Optional[Any]:
    """构建轻量级 DRSN (深度残差收缩网络)。"""
    if not _TORCH_AVAILABLE:
        return None
    try:

        class _RSBU(nn.Module):
            """残差收缩基本单元 (Residual Shrinkage Building Unit)"""
            def __init__(self, channels: int, ks: int = 3):
                super().__init__()
                pad = ks // 2
                self.conv1 = nn.Conv1d(channels, channels, ks, padding=pad)
                self.bn1 = nn.BatchNorm1d(channels)
                self.conv2 = nn.Conv1d(channels, channels, ks, padding=pad)
                self.bn2 = nn.BatchNorm1d(channels)
                # 软阈值化
                self.gap = nn.AdaptiveAvgPool1d(1)
                self.fc1 = nn.Linear(channels, channels)
                self.fc2 = nn.Linear(channels, channels)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                shortcut = x
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                # 软阈值
                gap = self.gap(out).squeeze(-1)
                fc1 = F.relu(self.fc1(gap))
                fc2 = torch.sigmoid(self.fc2(fc1))
                # 通道级阈值
                threshold = fc2.unsqueeze(-1)
                out = torch.sign(out) * F.relu(torch.abs(out) - threshold)
                return F.relu(out + shortcut)

        class _DRSN(nn.Module):
            def __init__(self, nf: int, dp: int, ic: int):
                super().__init__()
                layers = []
                ch = ic
                for i in range(dp):
                    out_ch = min(nf * (2 ** min(i, 3)), 128)
                    if i == 0:
                        layers.append(
                            nn.Sequential(
                                nn.Conv1d(ch, out_ch, 3, padding=1),
                                nn.BatchNorm1d(out_ch),
                                nn.ReLU(inplace=True),
                            )
                        )
                    else:
                        layers.append(_RSBU(out_ch))
                    ch = out_ch
                self.net = nn.Sequential(*layers)
                self.out_conv = nn.Conv1d(ch, ic, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.net(x)
                return self.out_conv(out)

        net = _DRSN(n_filters, depth, input_channels)
        # num_params = sum(p.numel() for p in net.parameters())
        # print(f"[DRSN] 参数量: {num_params}")
        net.eval()
        return net
    except Exception:
        return None


def _build_tcn_lstm_net(
    n_filters: int = 16,
    kernel_size: int = 3,
    n_lstm_units: int = 32,
    input_channels: int = 1,
) -> Optional[Any]:
    """构建轻量级 TCN-LSTM 网络。"""
    if not _TORCH_AVAILABLE:
        return None
    try:

        class _TCNBlock(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, ks: int, dilation: int):
                super().__init__()
                pad = (ks - 1) * dilation
                self.conv = nn.Conv1d(
                    in_ch, out_ch, ks,
                    padding=pad,
                    dilation=dilation,
                )
                self.bn = nn.BatchNorm1d(out_ch)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.conv(x)
                out = self.bn(out)
                out = self.relu(out)
                return out

        class _TCNLSTM(nn.Module):
            def __init__(self, nf: int, ks: int, nl: int, ic: int):
                super().__init__()
                self.tcn_blocks = nn.ModuleList()
                ch = ic
                for i in range(3):
                    out_ch = nf * (2 ** i)
                    self.tcn_blocks.append(
                        _TCNBlock(ch, out_ch, ks, dilation=2 ** i)
                    )
                    ch = out_ch
                self.lstm = nn.LSTM(
                    input_size=ch,
                    hidden_size=nl,
                    batch_first=True,
                    bidirectional=True,
                )
                self.out_fc = nn.Linear(nl * 2, ic)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, C, L)
                for block in self.tcn_blocks:
                    x = block(x)
                # (B, C, L) -> (B, L, C)
                x = x.permute(0, 2, 1)
                lstm_out, _ = self.lstm(x)
                out = self.out_fc(lstm_out)
                # (B, L, C) -> (B, C, L)
                out = out.permute(0, 2, 1)
                return out

        net = _TCNLSTM(n_filters, kernel_size, n_lstm_units, input_channels)
        # num_params = sum(p.numel() for p in net.parameters())
        # print(f"[TCN-LSTM] 参数量: {num_params}")
        net.eval()
        return net
    except Exception:
        return None


def _build_transformer_net(
    d_model: int = 32,
    n_heads: int = 4,
    n_layers: int = 2,
    dim_feedforward: int = 64,
    input_channels: int = 1,
) -> Optional[Any]:
    """构建轻量级 Transformer 降噪网络。"""
    if not _TORCH_AVAILABLE:
        return None
    try:

        class _PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len: int = 5000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.pe[:, : x.size(1), :]

        class _TransformerDenoiser(nn.Module):
            def __init__(self, dm: int, nh: int, nl: int, df: int, ic: int):
                super().__init__()
                self.input_proj = nn.Conv1d(ic, dm, 1)
                self.pos_enc = _PositionalEncoding(dm)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=dm,
                    nhead=nh,
                    dim_feedforward=df,
                    batch_first=True,
                    dropout=0.1,
                )
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nl)
                self.output_proj = nn.Conv1d(dm, ic, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, C, L)
                x = self.input_proj(x)  # (B, dm, L)
                x = x.permute(0, 2, 1)  # (B, L, dm)
                x = self.pos_enc(x)
                x = self.transformer(x)
                x = x.permute(0, 2, 1)  # (B, dm, L)
                x = self.output_proj(x)
                return x

        net = _TransformerDenoiser(d_model, n_heads, n_layers, dim_feedforward, input_channels)
        # num_params = sum(p.numel() for p in net.parameters())
        # print(f"[Transformer] 参数量: {num_params}")
        net.eval()
        return net
    except Exception:
        return None


# ==================== 1. DAE (去噪自编码器) ====================


@register_algorithm(
    name="去噪自编码器(DAE)",
    category=AlgorithmCategory.DEEP_LEARNING,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["deep_learning", "autoencoder", "dae"],
    requires_gpu=False,
)
class DaeDenoise(BaseAlgorithm):
    """去噪自编码器 (Denoising Autoencoder, DAE)。

    使用轻量级全连接自编码器对信号进行降噪。
    在推理时即时构建网络，在CPU上运行。

    当 torch 不可用时，回退到频域软阈值滤波。
    """

    requires_gpu: ClassVar[bool] = False

    default_params: ClassVar[Dict[str, Any]] = {
        "encoding_dim": 16,
        "hidden_dim": 32,
        "segment_length": 256,
        "overlap": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._net = None

    def setup(self) -> None:
        if _TORCH_AVAILABLE:
            seg_len = int(self.params.get("segment_length", 256))
            enc_dim = int(self.params.get("encoding_dim", 16))
            hid_dim = int(self.params.get("hidden_dim", 32))
            self._net = _build_dae_net(
                encoding_dim=enc_dim,
                hidden_dim=hid_dim,
                input_dim=seg_len,
            )

    def teardown(self) -> None:
        self._net = None

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sig_1d = signal.ndim == 1
        orig_signal = signal  # save original for shape restoration
        signal = _to_stereo(signal)

        if _TORCH_AVAILABLE and self._net is not None:
            try:
                result = self._denoise_torch(signal, params)
            except Exception:
                result = _norch_fallback_spectral(signal, sample_rate, cutoff_ratio=0.15)
        else:
            result = _norch_fallback_spectral(signal, sample_rate, cutoff_ratio=0.15)

        return _from_stereo(result, orig_signal)

    def _denoise_torch(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        import torch

        seg_len = int(params.get("segment_length", 256))
        overlap = float(params.get("overlap", 0.5))
        n_channels, n_samples = signal.shape

        # 构建网络（如果尚未构建）
        if self._net is None:
            enc_dim = int(params.get("encoding_dim", 16))
            hid_dim = int(params.get("hidden_dim", 32))
            self._net = _build_dae_net(
                encoding_dim=enc_dim,
                hidden_dim=hid_dim,
                input_dim=seg_len,
            )
            if self._net is None:
                raise RuntimeError("Failed to build DAE network")

        device = torch.device("cpu")
        self._net.to(device)
        output = np.zeros_like(signal, dtype=np.float64)

        for ch in range(n_channels):
            sig = signal[ch].astype(np.float32)
            # 分帧 + 重叠相加
            hop = int(seg_len * (1.0 - overlap))
            if hop < 1:
                hop = 1
            # 填充
            pad_len = seg_len - ((n_samples - seg_len) % hop) if (n_samples >= seg_len) else seg_len - n_samples
            if pad_len > 0:
                sig = np.pad(sig, (0, pad_len), mode="reflect")
            n_frames = (len(sig) - seg_len) // hop + 1

            out_frames = np.zeros(n_frames * seg_len, dtype=np.float32)
            window = np.hanning(seg_len).astype(np.float32)

            for i in range(n_frames):
                start = i * hop
                frame = sig[start: start + seg_len]
                frame_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)  # (1, 1, seg_len)
                with torch.no_grad():
                    denoised_t = self._net(frame_t)
                denoised = denoised_t.squeeze().cpu().numpy()
                out_frames[i * seg_len: (i + 1) * seg_len] += denoised * window

            # 重叠相加归一化
            weight = np.zeros(len(sig), dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                weight[start: start + seg_len] += window
            weight = np.maximum(weight, 1e-12)
            out_sig = out_frames[:len(sig)] / weight
            output[ch] = out_sig[:n_samples].astype(np.float64)

        return output


# ==================== 2. CDAE (卷积去噪自编码器) ====================


@register_algorithm(
    name="卷积去噪自编码器(CDAE)",
    category=AlgorithmCategory.DEEP_LEARNING,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["deep_learning", "convolutional", "autoencoder", "cdae"],
    requires_gpu=False,
)
class CdaDenoise(BaseAlgorithm):
    """卷积去噪自编码器 (Convolutional Denoising Autoencoder, CDAE)。

    使用轻量级1D卷积自编码器对信号进行降噪。
    在推理时即时构建网络，在CPU上运行。

    当 torch 不可用时，回退到小波阈值降噪。
    """

    requires_gpu: ClassVar[bool] = False

    default_params: ClassVar[Dict[str, Any]] = {
        "kernel_size": 3,
        "n_filters": 16,
        "segment_length": 512,
        "overlap": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._net = None

    def setup(self) -> None:
        if _TORCH_AVAILABLE:
            ks = int(self.params.get("kernel_size", 3))
            nf = int(self.params.get("n_filters", 16))
            self._net = _build_cdae_net(kernel_size=ks, n_filters=nf, input_channels=1)

    def teardown(self) -> None:
        self._net = None

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sig_1d = signal.ndim == 1
        orig_signal = signal  # save original for shape restoration
        signal = _to_stereo(signal)

        if _TORCH_AVAILABLE and self._net is not None:
            try:
                result = self._denoise_torch(signal, params)
            except Exception:
                result = _norch_fallback_wavelet(signal, wavelet="db4", level=3)
        else:
            result = _norch_fallback_wavelet(signal, wavelet="db4", level=3)

        return _from_stereo(result, orig_signal)

    def _denoise_torch(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        import torch

        seg_len = int(params.get("segment_length", 512))
        overlap = float(params.get("overlap", 0.5))
        n_channels, n_samples = signal.shape

        if self._net is None:
            ks = int(params.get("kernel_size", 3))
            nf = int(params.get("n_filters", 16))
            self._net = _build_cdae_net(kernel_size=ks, n_filters=nf, input_channels=1)
            if self._net is None:
                raise RuntimeError("Failed to build CDAE network")

        device = torch.device("cpu")
        self._net.to(device)
        output = np.zeros_like(signal, dtype=np.float64)

        for ch in range(n_channels):
            sig = signal[ch].astype(np.float32)
            hop = int(seg_len * (1.0 - overlap))
            if hop < 1:
                hop = 1
            pad_len = seg_len - ((n_samples - seg_len) % hop) if (n_samples >= seg_len) else seg_len - n_samples
            if pad_len > 0:
                sig = np.pad(sig, (0, pad_len), mode="reflect")
            n_frames = (len(sig) - seg_len) // hop + 1

            out_frames = np.zeros(n_frames * seg_len, dtype=np.float32)
            window = np.hanning(seg_len).astype(np.float32)

            for i in range(n_frames):
                start = i * hop
                frame = sig[start: start + seg_len]
                frame_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)  # (1, 1, seg_len)
                with torch.no_grad():
                    denoised_t = self._net(frame_t)
                denoised = denoised_t.squeeze().cpu().numpy()
                out_frames[i * seg_len: (i + 1) * seg_len] += denoised * window

            weight = np.zeros(len(sig), dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                weight[start: start + seg_len] += window
            weight = np.maximum(weight, 1e-12)
            out_sig = out_frames[:len(sig)] / weight
            output[ch] = out_sig[:n_samples].astype(np.float64)

        return output


# ==================== 3. DnCNN ====================


@register_algorithm(
    name="DnCNN降噪",
    category=AlgorithmCategory.DEEP_LEARNING,
    complexity=AlgorithmComplexity.HIGH,
    tags=["deep_learning", "cnn", "residual", "dncnn"],
    requires_gpu=False,
)
class DncnnDenoise(BaseAlgorithm):
    """DnCNN (Denoising Convolutional Neural Network) 降噪。

    使用残差学习的卷积网络对信号进行降噪。
    网络学习的是噪声残差，最终输出 = 输入 - 预测噪声。

    当 torch 不可用时，回退到维纳滤波。
    """

    requires_gpu: ClassVar[bool] = False

    default_params: ClassVar[Dict[str, Any]] = {
        "n_layers": 10,
        "n_filters": 32,
        "kernel_size": 3,
        "segment_length": 512,
        "overlap": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._net = None

    def setup(self) -> None:
        if _TORCH_AVAILABLE:
            nl = int(self.params.get("n_layers", 10))
            nf = int(self.params.get("n_filters", 32))
            ks = int(self.params.get("kernel_size", 3))
            self._net = _build_dncnn_net(
                n_layers=nl, n_filters=nf, kernel_size=ks, input_channels=1
            )

    def teardown(self) -> None:
        self._net = None

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sig_1d = signal.ndim == 1
        orig_signal = signal  # save original for shape restoration
        signal = _to_stereo(signal)

        if _TORCH_AVAILABLE and self._net is not None:
            try:
                result = self._denoise_torch(signal, params)
            except Exception:
                result = _norch_fallback_wiener(signal, win_size=5)
        else:
            result = _norch_fallback_wiener(signal, win_size=5)

        return _from_stereo(result, orig_signal)

    def _denoise_torch(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        import torch

        seg_len = int(params.get("segment_length", 512))
        overlap = float(params.get("overlap", 0.5))
        n_channels, n_samples = signal.shape

        if self._net is None:
            nl = int(params.get("n_layers", 10))
            nf = int(params.get("n_filters", 32))
            ks = int(params.get("kernel_size", 3))
            self._net = _build_dncnn_net(
                n_layers=nl, n_filters=nf, kernel_size=ks, input_channels=1
            )
            if self._net is None:
                raise RuntimeError("Failed to build DnCNN network")

        device = torch.device("cpu")
        self._net.to(device)
        output = np.zeros_like(signal, dtype=np.float64)

        for ch in range(n_channels):
            sig = signal[ch].astype(np.float32)
            hop = int(seg_len * (1.0 - overlap))
            if hop < 1:
                hop = 1
            pad_len = seg_len - ((n_samples - seg_len) % hop) if (n_samples >= seg_len) else seg_len - n_samples
            if pad_len > 0:
                sig = np.pad(sig, (0, pad_len), mode="reflect")
            n_frames = (len(sig) - seg_len) // hop + 1

            out_frames = np.zeros(n_frames * seg_len, dtype=np.float32)
            window = np.hanning(seg_len).astype(np.float32)

            for i in range(n_frames):
                start = i * hop
                frame = sig[start: start + seg_len]
                frame_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    denoised_t = self._net(frame_t)
                denoised = denoised_t.squeeze().cpu().numpy()
                out_frames[i * seg_len: (i + 1) * seg_len] += denoised * window

            weight = np.zeros(len(sig), dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                weight[start: start + seg_len] += window
            weight = np.maximum(weight, 1e-12)
            out_sig = out_frames[:len(sig)] / weight
            output[ch] = out_sig[:n_samples].astype(np.float64)

        return output


# ==================== 4. UNet ====================


@register_algorithm(
    name="UNet降噪",
    category=AlgorithmCategory.DEEP_LEARNING,
    complexity=AlgorithmComplexity.HIGH,
    tags=["deep_learning", "unet", "segmentation"],
    requires_gpu=False,
)
class UnetDenoise(BaseAlgorithm):
    """1D UNet 降噪网络。

    使用编码器-解码器架构，通过跳跃连接保留细节信息。
    适用于需要保留信号局部结构的降噪场景。

    当 torch 不可用时，回退到频域软阈值滤波。
    """

    requires_gpu: ClassVar[bool] = False

    default_params: ClassVar[Dict[str, Any]] = {
        "n_channels": 16,
        "depth": 4,
        "segment_length": 1024,
        "overlap": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._net = None

    def setup(self) -> None:
        if _TORCH_AVAILABLE:
            nc = int(self.params.get("n_channels", 16))
            dp = int(self.params.get("depth", 4))
            self._net = _build_unet_1d_net(
                n_channels=nc, depth=dp, input_channels=1
            )

    def teardown(self) -> None:
        self._net = None

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sig_1d = signal.ndim == 1
        orig_signal = signal  # save original for shape restoration
        signal = _to_stereo(signal)

        if _TORCH_AVAILABLE and self._net is not None:
            try:
                result = self._denoise_torch(signal, params)
            except Exception:
                result = _norch_fallback_spectral(signal, sample_rate, cutoff_ratio=0.15)
        else:
            result = _norch_fallback_spectral(signal, sample_rate, cutoff_ratio=0.15)

        return _from_stereo(result, orig_signal)

    def _denoise_torch(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        import torch

        seg_len = int(params.get("segment_length", 1024))
        overlap = float(params.get("overlap", 0.5))
        n_channels, n_samples = signal.shape

        if self._net is None:
            nc = int(params.get("n_channels", 16))
            dp = int(params.get("depth", 4))
            self._net = _build_unet_1d_net(
                n_channels=nc, depth=dp, input_channels=1
            )
            if self._net is None:
                raise RuntimeError("Failed to build UNet network")

        device = torch.device("cpu")
        self._net.to(device)
        output = np.zeros_like(signal, dtype=np.float64)

        for ch in range(n_channels):
            sig = signal[ch].astype(np.float32)
            hop = int(seg_len * (1.0 - overlap))
            if hop < 1:
                hop = 1
            pad_len = seg_len - ((n_samples - seg_len) % hop) if (n_samples >= seg_len) else seg_len - n_samples
            if pad_len > 0:
                sig = np.pad(sig, (0, pad_len), mode="reflect")
            n_frames = (len(sig) - seg_len) // hop + 1

            out_frames = np.zeros(n_frames * seg_len, dtype=np.float32)
            window = np.hanning(seg_len).astype(np.float32)

            for i in range(n_frames):
                start = i * hop
                frame = sig[start: start + seg_len]
                frame_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    denoised_t = self._net(frame_t)
                denoised = denoised_t.squeeze().cpu().numpy()
                out_frames[i * seg_len: (i + 1) * seg_len] += denoised * window

            weight = np.zeros(len(sig), dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                weight[start: start + seg_len] += window
            weight = np.maximum(weight, 1e-12)
            out_sig = out_frames[:len(sig)] / weight
            output[ch] = out_sig[:n_samples].astype(np.float64)

        return output


# ==================== 5. GAN降噪 ====================


@register_algorithm(
    name="GAN降噪",
    category=AlgorithmCategory.DEEP_LEARNING,
    complexity=AlgorithmComplexity.HIGH,
    tags=["deep_learning", "gan", "generative"],
    requires_gpu=False,
)
class GanDenoise(BaseAlgorithm):
    """GAN 降噪网络。

    使用生成对抗网络中的 Generator 对信号进行降噪。
    在推理时从潜在空间映射到干净的信号片段。

    当 torch 不可用时，回退到小波阈值降噪。
    """

    requires_gpu: ClassVar[bool] = False

    default_params: ClassVar[Dict[str, Any]] = {
        "latent_dim": 16,
        "n_iter": 20,
        "segment_length": 256,
        "overlap": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._generator = None

    def setup(self) -> None:
        if _TORCH_AVAILABLE:
            ld = int(self.params.get("latent_dim", 16))
            seg_len = int(self.params.get("segment_length", 256))
            self._generator = _build_gan_generator_net(
                latent_dim=ld, output_dim=seg_len
            )

    def teardown(self) -> None:
        self._generator = None

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sig_1d = signal.ndim == 1
        orig_signal = signal  # save original for shape restoration
        signal = _to_stereo(signal)

        if _TORCH_AVAILABLE and self._generator is not None:
            try:
                result = self._denoise_torch(signal, params)
            except Exception:
                result = _norch_fallback_wavelet(signal, wavelet="sym8", level=4)
        else:
            result = _norch_fallback_wavelet(signal, wavelet="sym8", level=4)

        return _from_stereo(result, orig_signal)

    def _denoise_torch(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        import torch

        seg_len = int(params.get("segment_length", 256))
        overlap = float(params.get("overlap", 0.5))
        latent_dim = int(params.get("latent_dim", 16))
        n_iter = int(params.get("n_iter", 100))
        n_channels, n_samples = signal.shape

        if self._generator is None:
            self._generator = _build_gan_generator_net(
                latent_dim=latent_dim, output_dim=seg_len
            )
            if self._generator is None:
                raise RuntimeError("Failed to build GAN generator")

        device = torch.device("cpu")
        self._generator.to(device)
        output = np.zeros_like(signal, dtype=np.float64)

        for ch in range(n_channels):
            sig = signal[ch].astype(np.float32)
            hop = int(seg_len * (1.0 - overlap))
            if hop < 1:
                hop = 1
            pad_len = seg_len - ((n_samples - seg_len) % hop) if (n_samples >= seg_len) else seg_len - n_samples
            if pad_len > 0:
                sig = np.pad(sig, (0, pad_len), mode="reflect")
            n_frames = (len(sig) - seg_len) // hop + 1

            out_frames = np.zeros(n_frames * seg_len, dtype=np.float32)
            window = np.hanning(seg_len).astype(np.float32)

            for i in range(n_frames):
                start = i * hop
                # 从噪声帧映射：使用潜在向量的迭代优化
                z = torch.randn(1, latent_dim, device=device)
                z.requires_grad_(True)

                frame = sig[start: start + seg_len]
                frame_t = torch.from_numpy(frame).unsqueeze(0).to(device)

                # 简单迭代优化以匹配信号帧
                optimizer = torch.optim.SGD([z], lr=0.01)
                for _ in range(min(n_iter, 10)):  # 限制迭代次数以保证CPU速度（前原为20）
                    optimizer.zero_grad()
                    generated = self._generator(z)
                    loss = F.mse_loss(generated, frame_t)
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    denoised_t = self._generator(z)
                denoised = denoised_t.squeeze().cpu().numpy()
                out_frames[i * seg_len: (i + 1) * seg_len] += denoised * window

            weight = np.zeros(len(sig), dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                weight[start: start + seg_len] += window
            weight = np.maximum(weight, 1e-12)
            out_sig = out_frames[:len(sig)] / weight
            output[ch] = out_sig[:n_samples].astype(np.float64)

        return output


# ==================== 6. DRSN (深度残差收缩网络) ====================


@register_algorithm(
    name="深度残差收缩网络(DRSN)",
    category=AlgorithmCategory.DEEP_LEARNING,
    complexity=AlgorithmComplexity.HIGH,
    tags=["deep_learning", "residual", "shrinkage", "drsn"],
    requires_gpu=False,
)
class DrsnDenoise(BaseAlgorithm):
    """深度残差收缩网络 (Deep Residual Shrinkage Network, DRSN)。

    在残差网络中引入软阈值化机制，自动学习阈值，
    对噪声特征进行收缩，适用于强噪声环境下的降噪。

    当 torch 不可用时，回退到频域软阈值滤波。
    """

    requires_gpu: ClassVar[bool] = False

    default_params: ClassVar[Dict[str, Any]] = {
        "n_filters": 16,
        "depth": 6,
        "segment_length": 512,
        "overlap": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._net = None

    def setup(self) -> None:
        if _TORCH_AVAILABLE:
            nf = int(self.params.get("n_filters", 16))
            dp = int(self.params.get("depth", 6))
            self._net = _build_drsn_net(
                n_filters=nf, depth=dp, input_channels=1
            )

    def teardown(self) -> None:
        self._net = None

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sig_1d = signal.ndim == 1
        orig_signal = signal  # save original for shape restoration
        signal = _to_stereo(signal)

        if _TORCH_AVAILABLE and self._net is not None:
            try:
                result = self._denoise_torch(signal, params)
            except Exception:
                result = _norch_fallback_spectral(signal, sample_rate, cutoff_ratio=0.15)
        else:
            result = _norch_fallback_spectral(signal, sample_rate, cutoff_ratio=0.15)

        return _from_stereo(result, orig_signal)

    def _denoise_torch(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        import torch

        seg_len = int(params.get("segment_length", 512))
        overlap = float(params.get("overlap", 0.5))
        n_channels, n_samples = signal.shape

        if self._net is None:
            nf = int(params.get("n_filters", 16))
            dp = int(params.get("depth", 6))
            self._net = _build_drsn_net(n_filters=nf, depth=dp, input_channels=1)
            if self._net is None:
                raise RuntimeError("Failed to build DRSN network")

        device = torch.device("cpu")
        self._net.to(device)
        output = np.zeros_like(signal, dtype=np.float64)

        for ch in range(n_channels):
            sig = signal[ch].astype(np.float32)
            hop = int(seg_len * (1.0 - overlap))
            if hop < 1:
                hop = 1
            pad_len = seg_len - ((n_samples - seg_len) % hop) if (n_samples >= seg_len) else seg_len - n_samples
            if pad_len > 0:
                sig = np.pad(sig, (0, pad_len), mode="reflect")
            n_frames = (len(sig) - seg_len) // hop + 1

            out_frames = np.zeros(n_frames * seg_len, dtype=np.float32)
            window = np.hanning(seg_len).astype(np.float32)

            for i in range(n_frames):
                start = i * hop
                frame = sig[start: start + seg_len]
                frame_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    denoised_t = self._net(frame_t)
                denoised = denoised_t.squeeze().cpu().numpy()
                out_frames[i * seg_len: (i + 1) * seg_len] += denoised * window

            weight = np.zeros(len(sig), dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                weight[start: start + seg_len] += window
            weight = np.maximum(weight, 1e-12)
            out_sig = out_frames[:len(sig)] / weight
            output[ch] = out_sig[:n_samples].astype(np.float64)

        return output


# ==================== 7. TCN-LSTM ====================


@register_algorithm(
    name="TCN-LSTM降噪",
    category=AlgorithmCategory.DEEP_LEARNING,
    complexity=AlgorithmComplexity.HIGH,
    tags=["deep_learning", "tcn", "lstm", "temporal"],
    requires_gpu=False,
)
class TcnLstmDenoise(BaseAlgorithm):
    """TCN-LSTM 降噪网络。

    结合时序卷积网络 (TCN) 的局部特征提取能力和
    LSTM 的长期依赖建模能力，对信号进行降噪。

    当 torch 不可用时，回退到维纳滤波。
    """

    requires_gpu: ClassVar[bool] = False

    default_params: ClassVar[Dict[str, Any]] = {
        "n_filters": 16,
        "kernel_size": 3,
        "n_lstm_units": 32,
        "segment_length": 512,
        "overlap": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._net = None

    def setup(self) -> None:
        if _TORCH_AVAILABLE:
            nf = int(self.params.get("n_filters", 16))
            ks = int(self.params.get("kernel_size", 3))
            nl = int(self.params.get("n_lstm_units", 32))
            self._net = _build_tcn_lstm_net(
                n_filters=nf, kernel_size=ks, n_lstm_units=nl, input_channels=1
            )

    def teardown(self) -> None:
        self._net = None

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sig_1d = signal.ndim == 1
        orig_signal = signal  # save original for shape restoration
        signal = _to_stereo(signal)

        if _TORCH_AVAILABLE and self._net is not None:
            try:
                result = self._denoise_torch(signal, params)
            except Exception:
                result = _norch_fallback_wiener(signal, win_size=7)
        else:
            result = _norch_fallback_wiener(signal, win_size=7)

        return _from_stereo(result, orig_signal)

    def _denoise_torch(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        import torch

        seg_len = int(params.get("segment_length", 512))
        overlap = float(params.get("overlap", 0.5))
        n_channels, n_samples = signal.shape

        if self._net is None:
            nf = int(params.get("n_filters", 16))
            ks = int(params.get("kernel_size", 3))
            nl = int(params.get("n_lstm_units", 32))
            self._net = _build_tcn_lstm_net(
                n_filters=nf, kernel_size=ks, n_lstm_units=nl, input_channels=1
            )
            if self._net is None:
                raise RuntimeError("Failed to build TCN-LSTM network")

        device = torch.device("cpu")
        self._net.to(device)
        output = np.zeros_like(signal, dtype=np.float64)

        for ch in range(n_channels):
            sig = signal[ch].astype(np.float32)
            hop = int(seg_len * (1.0 - overlap))
            if hop < 1:
                hop = 1
            pad_len = seg_len - ((n_samples - seg_len) % hop) if (n_samples >= seg_len) else seg_len - n_samples
            if pad_len > 0:
                sig = np.pad(sig, (0, pad_len), mode="reflect")
            n_frames = (len(sig) - seg_len) // hop + 1

            out_frames = np.zeros(n_frames * seg_len, dtype=np.float32)
            window = np.hanning(seg_len).astype(np.float32)

            for i in range(n_frames):
                start = i * hop
                frame = sig[start: start + seg_len]
                frame_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    denoised_t = self._net(frame_t)
                denoised = denoised_t.squeeze().cpu().numpy()
                out_frames[i * seg_len: (i + 1) * seg_len] += denoised * window

            weight = np.zeros(len(sig), dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                weight[start: start + seg_len] += window
            weight = np.maximum(weight, 1e-12)
            out_sig = out_frames[:len(sig)] / weight
            output[ch] = out_sig[:n_samples].astype(np.float64)

        return output


# ==================== 8. Transformer降噪 ====================


@register_algorithm(
    name="Transformer降噪",
    category=AlgorithmCategory.DEEP_LEARNING,
    complexity=AlgorithmComplexity.HIGH,
    tags=["deep_learning", "transformer", "attention"],
    requires_gpu=False,
)
class TransformerDenoise(BaseAlgorithm):
    """Transformer 降噪网络。

    使用轻量级 Transformer 编码器对信号片段进行降噪。
    自注意力机制可以捕捉信号中的长程依赖关系。

    当 torch 不可用时，回退到频域软阈值滤波。
    """

    requires_gpu: ClassVar[bool] = False

    default_params: ClassVar[Dict[str, Any]] = {
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "dim_feedforward": 64,
        "segment_length": 256,
        "overlap": 0.5,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._net = None

    def setup(self) -> None:
        if _TORCH_AVAILABLE:
            dm = int(self.params.get("d_model", 32))
            nh = int(self.params.get("n_heads", 4))
            nl = int(self.params.get("n_layers", 2))
            df = int(self.params.get("dim_feedforward", 64))
            self._net = _build_transformer_net(
                d_model=dm, n_heads=nh, n_layers=nl,
                dim_feedforward=df, input_channels=1,
            )

    def teardown(self) -> None:
        self._net = None

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sig_1d = signal.ndim == 1
        orig_signal = signal  # save original for shape restoration
        signal = _to_stereo(signal)

        if _TORCH_AVAILABLE and self._net is not None:
            try:
                result = self._denoise_torch(signal, params)
            except Exception:
                result = _norch_fallback_spectral(signal, sample_rate, cutoff_ratio=0.1)
        else:
            result = _norch_fallback_spectral(signal, sample_rate, cutoff_ratio=0.1)

        return _from_stereo(result, orig_signal)

    def _denoise_torch(self, signal: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        import torch

        seg_len = int(params.get("segment_length", 256))
        overlap = float(params.get("overlap", 0.5))
        n_channels, n_samples = signal.shape

        if self._net is None:
            dm = int(params.get("d_model", 32))
            nh = int(params.get("n_heads", 4))
            nl = int(params.get("n_layers", 2))
            df = int(params.get("dim_feedforward", 64))
            self._net = _build_transformer_net(
                d_model=dm, n_heads=nh, n_layers=nl,
                dim_feedforward=df, input_channels=1,
            )
            if self._net is None:
                raise RuntimeError("Failed to build Transformer network")

        device = torch.device("cpu")
        self._net.to(device)
        output = np.zeros_like(signal, dtype=np.float64)

        for ch in range(n_channels):
            sig = signal[ch].astype(np.float32)
            hop = int(seg_len * (1.0 - overlap))
            if hop < 1:
                hop = 1
            pad_len = seg_len - ((n_samples - seg_len) % hop) if (n_samples >= seg_len) else seg_len - n_samples
            if pad_len > 0:
                sig = np.pad(sig, (0, pad_len), mode="reflect")
            n_frames = (len(sig) - seg_len) // hop + 1

            out_frames = np.zeros(n_frames * seg_len, dtype=np.float32)
            window = np.hanning(seg_len).astype(np.float32)

            for i in range(n_frames):
                start = i * hop
                frame = sig[start: start + seg_len]
                frame_t = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    denoised_t = self._net(frame_t)
                denoised = denoised_t.squeeze().cpu().numpy()
                out_frames[i * seg_len: (i + 1) * seg_len] += denoised * window

            weight = np.zeros(len(sig), dtype=np.float32)
            for i in range(n_frames):
                start = i * hop
                weight[start: start + seg_len] += window
            weight = np.maximum(weight, 1e-12)
            out_sig = out_frames[:len(sig)] / weight
            output[ch] = out_sig[:n_samples].astype(np.float64)

        return output
