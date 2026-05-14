"""
盲源分离 (BSS) 降噪算法集合。

包含：
1. PCA降噪：主成分分析降噪
2. ICA降噪：独立成分分析降噪
3. FastICA降噪：FastICA 降噪
4. SOBI降噪：二阶盲辨识降噪

对于单通道信号，先构建延时嵌入矩阵形成伪多通道，
再进行盲源分离，最后重构为1D信号。

所有算法继承自 BaseAlgorithm，通过 @register_algorithm 装饰器注册。
"""

from typing import Any, ClassVar, Dict, Optional

import numpy as np

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


# ============================================================================
# 辅助函数
# ============================================================================

def _delay_embed(signal: np.ndarray, n_lags: int) -> np.ndarray:
    """构建延时嵌入矩阵（轨迹矩阵）。

    将1D信号映射到高维相空间，每一行是一个延时版本。

    Args:
        signal: 1D 输入信号，长度 N
        n_lags: 嵌入维度（延时数）

    Returns:
        延时嵌入矩阵，shape (n_lags, N - n_lags + 1)
    """
    N = len(signal)
    K = N - n_lags + 1
    X = np.zeros((n_lags, K))
    for i in range(n_lags):
        X[i, :] = signal[i : i + K]
    return X


def _reconstruct_from_embed(X: np.ndarray, N: int) -> np.ndarray:
    """从延时嵌入矩阵通过反对角线平均重构1D信号。

    Args:
        X: 延时嵌入矩阵，shape (L, K)
        N: 原始信号长度

    Returns:
        重构的 1D 信号
    """
    L, K = X.shape
    signal = np.zeros(N)
    count = np.zeros(N)
    for i in range(L):
        for j in range(K):
            signal[i + j] += X[i, j]
            count[i + j] += 1
    return signal / count


def _sort_sources_by_variance(S: np.ndarray, W: np.ndarray) -> tuple:
    """按方差（能量）降序排列源信号。

    Args:
        S: 源信号矩阵 (n_sources, n_samples)
        W: 对应的解混矩阵 (n_sources, n_features)

    Returns:
        (S_sorted, W_sorted)
    """
    variances = np.var(S, axis=1)
    order = np.argsort(variances)[::-1]
    return S[order], W[order]


def _compute_whitening(X: np.ndarray) -> tuple:
    """计算白化矩阵和去白化矩阵。

    Args:
        X: 数据矩阵 (n_features, n_samples)，已中心化

    Returns:
        (V, V_inv): 白化矩阵和去白化矩阵
    """
    n_features, n_samples = X.shape
    cov = X @ X.T / n_samples
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)
    # 确保降序排列
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    D = np.diag(1.0 / np.sqrt(eigvals))
    V = D @ eigvecs.T  # (n_features, n_features)
    V_inv = eigvecs @ np.diag(np.sqrt(eigvals))  # (n_features, n_features)
    return V, V_inv


def _fastica_iteration(
    Z: np.ndarray,
    n_components: int,
    fun: str = "logcosh",
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: int = 42,
) -> np.ndarray:
    """FastICA 迭代算法。

    对白化后的数据 Z 求解解混矩阵 W，使得 S = W @ Z 中各分量尽可能独立。

    Args:
        Z: 白化后的数据，shape (n_components, n_samples)
        n_components: 要提取的分量数
        fun: 对比函数，'logcosh' 或 'exp'
        max_iter: 最大迭代次数
        tol: 收敛容差
        random_state: 随机种子（保证可复现）

    Returns:
        解混矩阵 W，shape (n_components, n_components)
    """
    # 使用固定随机数发生器，保证可复现
    rng = np.random.default_rng(random_state)
    W = rng.standard_normal((n_components, n_components))
    W, _ = np.linalg.qr(W)

    for iteration in range(max_iter):
        W_old = W.copy()

        # 计算源信号
        S = W @ Z  # (n_components, n_samples)

        # 对比函数及其导数
        if fun == "logcosh":
            g = np.tanh(S)
            g_prime = 1 - g**2
        elif fun == "exp":
            # 限制 exp 输入避免溢出
            S_clipped = np.clip(S, -30, 30)
            g = S_clipped * np.exp(-(S_clipped**2) / 2)
            g_prime = (1 - S_clipped**2) * np.exp(-(S_clipped**2) / 2)
        else:
            raise ValueError(f"不支持的对比函数: {fun}，可选 'logcosh' 或 'exp'")

        # 更新 W
        W = (g @ Z.T) / Z.shape[1] - np.diag(g_prime.mean(axis=1)) @ W

        # NaN / Inf 检查，防止数值发散
        if not np.all(np.isfinite(W)):
            import warnings
            warnings.warn("FastICA: 检测到数值发散 (NaN/Inf)，回退到上一轮 W")
            return W_old

        # 正交化
        try:
            U, _, Vt = np.linalg.svd(W, full_matrices=False)
            W = U @ Vt
        except np.linalg.LinAlgError:
            import warnings
            warnings.warn("FastICA: SVD 分解失败，使用 W_old")
            return W_old

        # 检查收敛性
        dot_abs = np.abs(np.diag(W @ W_old.T))
        if np.all(dot_abs > 1 - tol):
            break

    return W


def _joint_diagonalization(
    C: list, eps: float = 1e-6, max_iter: int = 200
) -> np.ndarray:
    """联合近似对角化 (JAD)。

    使用Givens旋转迭代地对一组对称矩阵进行联合对角化，
    用于SOBI算法中的酉矩阵估计。

    Args:
        C: 待对角化的对称矩阵列表
        eps: 收敛阈值
        max_iter: 最大迭代轮数

    Returns:
        联合对角化矩阵 V
    """
    m = len(C)  # 矩阵个数
    n = C[0].shape[0]  # 矩阵大小

    # 工作副本
    M = [c.copy() for c in C]
    V = np.eye(n)

    for iteration in range(max_iter):
        total_angle = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                # 计算 Givens 旋转角度
                num = 0.0
                den = 0.0
                for k in range(m):
                    g = M[k][i, i] - M[k][j, j]
                    h = M[k][i, j] + M[k][j, i]
                    num += 2 * g * h
                    den += g * g - h * h

                if abs(num) < eps and abs(den) < eps:
                    continue

                theta = 0.5 * np.arctan2(num, den)
                c = np.cos(theta)
                s = np.sin(theta)

                # 对所有矩阵应用旋转
                for k in range(m):
                    # 更新行 i, j
                    row_i = M[k][i, :].copy()
                    row_j = M[k][j, :].copy()
                    M[k][i, :] = c * row_i + s * row_j
                    M[k][j, :] = -s * row_i + c * row_j
                    # 更新列 i, j
                    col_i = M[k][:, i].copy()
                    col_j = M[k][:, j].copy()
                    M[k][:, i] = c * col_i + s * col_j
                    M[k][:, j] = -s * col_i + c * col_j

                # 累积旋转
                Vi = V[i, :].copy()
                Vj = V[j, :].copy()
                V[i, :] = c * Vi + s * Vj
                V[j, :] = -s * Vi + c * Vj

                total_angle += abs(theta)

        if total_angle < eps:
            break

    return V


# ============================================================================
# ICA 通用流程（供 ICA / FastICA / SOBI 共用）
# ============================================================================

def _ica_denoise_common(
    signal: np.ndarray,
    n_components: Optional[int],
    algorithm: str = "fastica",
    fun: str = "logcosh",
    **kwargs,
) -> np.ndarray:
    """ICA 降噪通用流程。

    Args:
        signal: 输入信号，shape (n_channels, n_samples) 或 (n_samples,)
        n_components: 保留的分量数
        algorithm: 算法类型，'fastica' 或 'sobi'
        fun: FastICA 对比函数
        **kwargs: 额外参数

    Returns:
        降噪后信号
    """
    is_1d = signal.ndim == 1
    if is_1d:
        signal = signal.reshape(1, -1)

    n_channels, n_samples = signal.shape

    if n_channels == 1:
        # 单通道：构建延时嵌入
        sig_1d = signal[0]
        N = len(sig_1d)
        n_lags = min(N // 3, 100)
        n_lags = max(2, n_lags)
        X_emb = _delay_embed(sig_1d, n_lags)  # (n_lags, N-n_lags+1)
        X = X_emb
        feature_name = "lag"
    else:
        # 多通道：直接使用通道
        X = signal
        n_lags = n_channels
        feature_name = "channel"

    n_features, n_samples_bss = X.shape

    # 确定保留分量数
    if n_components is not None:
        k = int(n_components)
    else:
        k = n_features  # 默认保留全部（不降噪）
    k = max(1, min(k, n_features))

    # 中心化
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean

    if algorithm == "fastica":
        # ---- FastICA ----
        # 限制最大分量数以控制计算复杂度
        max_ica_dim = min(40, n_features)
        n_ica = min(k, n_features, max_ica_dim)
        # 白化
        V, V_inv = _compute_whitening(Xc)
        Z = V[:n_ica] @ Xc  # (n_ica, n_samples_bss)

        # FastICA 迭代
        W = _fastica_iteration(Z, n_ica, fun=fun)

        # 源信号
        S = W @ Z  # (n_ica, n_samples_bss)

        # 按方差排序
        S, W = _sort_sources_by_variance(S, W)

        # 保留前 k 个分量（k <= n_ica）
        k_keep = min(k, n_ica)
        S_keep = S[:k_keep]
        W_keep = W[:k_keep]

        # 重构白化空间中的数据
        # W_keep is (k_keep, n_ica), not square if k_keep < n_ica
        # We need to reconstruct Z_denoised in (n_ica, n_samples_bss)
        # S = W @ Z_denoised, so Z_denoised = W.T @ S (since W is orthogonal)
        # Actually W @ W.T = I for square W, but for rectangular W:
        # The best reconstruction: Z_denoised = W_keep.T @ S_keep
        # Wait, W @ Z = S, if W is square and orthogonal, Z = W.T @ S
        # But W_keep is (k_keep, n_ica), not square
        # Z_denoised = W_keep.T @ S_keep gives (n_ica, n_samples_bss) - this is a projection

        Z_denoised = W_keep.T @ S_keep  # (n_ica, n_samples_bss)

        # 去白化
        X_denoised = V_inv[:, :n_ica] @ Z_denoised + mean  # (n_features, n_samples_bss)

    elif algorithm == "sobi":
        # ---- SOBI ----
        n_lags_sobi = kwargs.get("n_lags", 100)
        # 限制最大维度和延时数以控制计算复杂度
        max_sobi_dim = min(20, n_features)
        n_sobi = min(k, n_features, max_sobi_dim)
        n_lags_sobi = min(n_lags_sobi, 20)

        # 白化（降低维数到 n_sobi）
        V, V_inv = _compute_whitening(Xc)
        Z = V[:n_sobi] @ Xc  # (n_sobi, n_samples_bss)

        # 计算延时协方差矩阵
        max_tau = min(n_lags_sobi, n_samples_bss // 3)
        lags = list(range(1, max_tau + 1))
        C = []
        for tau in lags:
            Z_tau = Z[:, tau:]
            Z_0 = Z[:, : n_samples_bss - tau]
            C_tau = (Z_0 @ Z_tau.T) / (n_samples_bss - tau)
            C.append((C_tau + C_tau.T) / 2)  # 对称化

        # 联合对角化
        V_jad = _joint_diagonalization(C)

        # 源信号
        S = V_jad @ Z  # (n_sobi, n_samples_bss)

        # 按方差排序
        S, V_jad = _sort_sources_by_variance(S, V_jad)

        # 保留前 k 个分量
        k_keep = min(k, n_sobi)
        S_keep = S[:k_keep]
        V_keep = V_jad[:k_keep]

        # 重构
        # Z_denoised = V_keep.T @ S_keep (V_jad is orthogonal)
        Z_denoised = V_keep.T @ S_keep

        # 去白化
        X_denoised = V_inv[:, :n_sobi] @ Z_denoised + mean

    else:
        raise ValueError(f"不支持的算法: {algorithm}")

    if feature_name == "lag":
        # 单通道：反对角线平均重构
        sig_denoised = _reconstruct_from_embed(X_denoised, n_samples)
        result = sig_denoised.reshape(1, -1)
    else:
        result = X_denoised

    return result[0] if is_1d else result


# ============================================================================
# 算法类
# ============================================================================


@register_algorithm(
    name="PCA降噪",
    category=AlgorithmCategory.BSS,
    complexity=AlgorithmComplexity.LOW,
    tags=["bss", "pca", "subspace"],
)
class PCADenoise(BaseAlgorithm):
    """PCA 主成分分析降噪。

    对信号进行主成分分解，保留方差最大的主成分，
    去除低方差成分（噪声），实现降噪。

    对于多通道信号，直接对通道做PCA。
    对于单通道信号，先构建延时嵌入矩阵再PCA。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_components": None,
        "retained_variance": 0.9,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        n_components = params.get("n_components")
        retained_variance = float(params.get("retained_variance", 0.9))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape

        if n_channels == 1:
            # 单通道：延时嵌入 + PCA
            sig_1d = signal[0]
            N = len(sig_1d)
            n_lags = min(N // 3, 100)
            n_lags = max(2, n_lags)
            X_emb = _delay_embed(sig_1d, n_lags)

            # 中心化
            mean = X_emb.mean(axis=1, keepdims=True)
            Xc = X_emb - mean

            # SVD
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

            # 确定保留分量
            if n_components is not None:
                k = int(n_components)
            else:
                total_var = np.sum(S**2)
                cumsum = np.cumsum(S**2)
                k = int(np.searchsorted(cumsum, retained_variance * total_var) + 1)
            k = max(1, min(k, len(S)))

            # 截断重构
            X_denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :] + mean

            # 反对角线平均
            result = _reconstruct_from_embed(X_denoised, N)
            result = result.reshape(1, -1)
        else:
            # 多通道：直接PCA
            mean = signal.mean(axis=1, keepdims=True)
            Xc = signal - mean
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

            if n_components is not None:
                k = int(n_components)
            else:
                total_var = np.sum(S**2)
                cumsum = np.cumsum(S**2)
                k = int(np.searchsorted(cumsum, retained_variance * total_var) + 1)
            k = max(1, min(k, len(S)))

            result = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :] + mean

        return result[0] if is_1d else result


@register_algorithm(
    name="ICA降噪",
    category=AlgorithmCategory.BSS,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["bss", "ica", "independence"],
)
class ICADenoise(BaseAlgorithm):
    """ICA 独立成分分析降噪。

    对信号进行独立成分分析，分离出统计独立的源信号，
    去除噪声分量后重构信号。

    对于单通道信号，先构建延时嵌入矩阵再分离。
    使用 FastICA 算法实现。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_components": None,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        n_components = params.get("n_components")

        result = _ica_denoise_common(
            signal,
            n_components=n_components,
            algorithm="fastica",
            fun="logcosh",
        )
        return result


@register_algorithm(
    name="FastICA降噪",
    category=AlgorithmCategory.BSS,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["bss", "fastica", "fixed-point"],
)
class FastICADenoise(BaseAlgorithm):
    """FastICA 快速独立成分分析降噪。

    使用固定点迭代算法进行ICA，支持不同对比函数。
    收敛速度快，适用于大规模信号处理。

    对于单通道信号，先构建延时嵌入矩阵再分离。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_components": None,
        "fun": "logcosh",
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        n_components = params.get("n_components")
        fun = str(params.get("fun", "logcosh"))

        result = _ica_denoise_common(
            signal,
            n_components=n_components,
            algorithm="fastica",
            fun=fun,
        )
        return result


@register_algorithm(
    name="SOBI降噪",
    category=AlgorithmCategory.BSS,
    complexity=AlgorithmComplexity.HIGH,
    tags=["bss", "sobi", "second-order", "diagonalization"],
)
class SOBIDenoise(BaseAlgorithm):
    """SOBI (Second-Order Blind Identification) 降噪。

    利用信号的时间结构（延时协方差矩阵）进行盲源分离。
    通过对多个延时协方差矩阵的联合对角化来估计源信号。

    适用于含有时间相关噪声的信号分离与降噪。
    对于单通道信号，先构建延时嵌入矩阵再分离。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_components": None,
        "n_lags": 100,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        n_components = params.get("n_components")
        n_lags = int(params.get("n_lags", 100))

        result = _ica_denoise_common(
            signal,
            n_components=n_components,
            algorithm="sobi",
            n_lags=n_lags,
        )
        return result


# ============================================================================
# 核PCA降噪 (Kernel PCA)
# ============================================================================

@register_algorithm(
    name="核PCA降噪",
    category=AlgorithmCategory.BSS,
    complexity=AlgorithmComplexity.HIGH,
    tags=["bss", "kernel-pca", "nonlinear", "subspace"],
)
class KernelPCADenoise(BaseAlgorithm):
    """核PCA降噪 (Kernel PCA)。

    通过核技巧将信号隐式映射到高维特征空间后做PCA，
    捕获数据的非线性结构。比线性PCA更适合非线性流形上的降噪。

    使用 scikit-learn 的 KernelPCA 实现。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_components": 3,
        "gamma": 1.0,
        "kernel": "rbf",
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        try:
            from sklearn.decomposition import KernelPCA
        except ImportError:
            # 无 sklearn 时回退到线性 PCA
            from sklearn.decomposition import PCA as FallbackPCA
            params = {**self.params, **kwargs}
            n_comp = int(params.get("n_components", 3))

            # 使用 PCA 近似
            return self._fallback_pca(signal, n_comp)

        params = {**self.params, **kwargs}
        n_components = int(params["n_components"])
        gamma = float(params["gamma"])
        kernel = str(params["kernel"])

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            N = len(sig)
            n_lags = min(N // 3, 100)
            n_lags = max(2, n_lags)

            # 构建延时嵌入矩阵
            X = _delay_embed(sig, n_lags)  # (n_lags, N-n_lags+1)

            # 核 PCA
            kpca = KernelPCA(
                n_components=min(n_components, X.shape[0], X.shape[1]),
                kernel=kernel, gamma=gamma,
                fit_inverse_transform=True,
            )
            X_kpca = kpca.fit_transform(X.T)  # (n_samples_bss, n_components)
            X_denoised = kpca.inverse_transform(X_kpca).T  # (n_lags, n_samples_bss)

            output[ch] = _reconstruct_from_embed(X_denoised, N)

        return output[0] if is_1d else output

    def _fallback_pca(self, signal, n_components):
        from sklearn.decomposition import PCA

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            N = len(sig)
            n_lags = min(N // 3, 100)
            n_lags = max(2, n_lags)

            X = _delay_embed(sig, n_lags).T
            mean = X.mean(axis=0, keepdims=True)
            Xc = X - mean

            pca = PCA(n_components=min(n_components, Xc.shape[1]))
            X_pca = pca.inverse_transform(pca.fit_transform(Xc))
            X_denoised = (X_pca + mean).T

            output[ch] = _reconstruct_from_embed(X_denoised, N)

        return output[0] if is_1d else output


# ============================================================================
# 联合近似对角化 (JADE)
# ============================================================================

@register_algorithm(
    name="联合近似对角化(JADE)",
    category=AlgorithmCategory.BSS,
    complexity=AlgorithmComplexity.HIGH,
    tags=["bss", "jade", "cumulant", "deterministic"],
)
class JadeDenoise(BaseAlgorithm):
    """联合近似对角化 (JADE) 盲源分离降噪。

    通过联合对角化一组四阶累积量矩阵实现更稳定、
    确定性的盲源分离。相比 FastICA 对初始化不敏感，
    不需要迭代优化，适合于短时信号处理。

    使用 scikit-learn 的 FastICA 作为近似实现（JADE 完整实现
    需要专用包如 'jade' 或自行实现四阶累积量对角化）。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_components": None,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        n_components = params.get("n_components")

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape

        if n_channels == 1:
            sig_1d = signal[0]
            N = len(sig_1d)
            n_lags = min(N // 3, 100)
            n_lags = max(2, n_lags)
            X = _delay_embed(sig_1d, n_lags)  # (n_lags, N-n_lags+1)
        else:
            X = signal
            n_lags = n_channels
            N = n_samples

        n_features, n_samples_bss = X.shape

        # 使用 FastICA 近似 JADE
        try:
            from sklearn.decomposition import FastICA

            k = n_features if n_components is None else min(int(n_components), n_features)
            k = max(1, k)

            ica = FastICA(n_components=k, algorithm="deflation",
                          max_iter=200, random_state=42)
            S = ica.fit_transform(X.T).T  # (k, n_samples_bss)
            A = ica.mixing_  # (n_features, k)

            # 按方差排序并去除噪声分量（方差最大的视为噪声）
            variances = np.var(S, axis=1)
            noise_idx = np.argmax(variances)
            keep = [i for i in range(k) if i != noise_idx]

            if len(keep) > 0:
                X_denoised = A[:, keep] @ S[keep, :]
            else:
                X_denoised = X
        except ImportError:
            # 无 sklearn 时回退
            return signal[0] if signal.shape[0] == 1 else signal

        if n_channels == 1:
            return _reconstruct_from_embed(X_denoised, N)
        else:
            return X_denoised
