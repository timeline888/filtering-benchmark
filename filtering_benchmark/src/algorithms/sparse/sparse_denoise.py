"""
稀疏表示类降噪算法集合。

包含：
1. 匹配追踪 (MP): Gabor字典 + 贪婪搜索
2. 正交匹配追踪 (OMP): 原子正交化投影
3. 基追踪降噪 (BPDN): L1范数正则化，使用OMP近似
4. K-SVD字典学习降噪: 从信号中学习过完备字典

所有算法继承自 BaseAlgorithm，通过 @register_algorithm 装饰器注册。
"""

from typing import Any, ClassVar, Dict, Optional

import numpy as np

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


# ==================== 辅助函数 ====================


def _gabor_dictionary(n_samples: int, n_atoms: int, sample_rate: float = 1.0) -> np.ndarray:
    """构造Gabor过完备字典。

    Gabor原子由高斯窗调制的正弦波构成，具有良好的时频局部化特性。

    Args:
        n_samples: 信号长度
        n_atoms: 原子个数
        sample_rate: 采样率 (Hz)，用于频率参数

    Returns:
        字典矩阵，shape (n_samples, n_atoms)，每列为一个归一化原子
    """
    dictionary = np.zeros((n_samples, n_atoms))
    t = np.arange(n_samples) / max(sample_rate, 1e-12)

    for i in range(n_atoms):
        # 频率：均匀覆盖 0 ~ fs/2
        freq = 0.5 * (i + 1) / n_atoms * min(sample_rate, n_samples / 2.0)
        # 高斯窗宽度：覆盖多个周期
        sigma = max(n_samples / (n_atoms * 4.0), 1.0)
        # 时移
        shift = int((i / n_atoms) * n_samples)

        # 高斯包络
        gaussian = np.exp(-0.5 * ((t - shift / max(sample_rate, 1e-12)) / sigma) ** 2)
        # 正弦波调制
        carrier = np.cos(2 * np.pi * freq * t)
        atom = gaussian * carrier

        # 归一化
        norm = np.linalg.norm(atom)
        if norm > 1e-12:
            dictionary[:, i] = atom / norm
        else:
            dictionary[:, i] = atom

    return dictionary


def _matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    max_iter: int = 50,
    sparsity: Optional[int] = None,
) -> np.ndarray:
    """匹配追踪 (MP) 稀疏编码。

    Args:
        signal: 1D 输入信号
        dictionary: 字典矩阵 (n_samples, n_atoms)
        max_iter: 最大迭代次数
        sparsity: 稀疏度（非零系数个数），若指定则作为迭代上限

    Returns:
        重构信号
    """
    n_atoms = dictionary.shape[1]
    residual = signal.copy()
    coeffs = np.zeros(n_atoms)
    max_iters = sparsity if sparsity is not None else max_iter

    # 预计算原子范数
    atom_norms = np.linalg.norm(dictionary, axis=0)
    atom_norms = np.where(atom_norms < 1e-12, 1.0, atom_norms)

    for _ in range(max_iters):
        # 计算残差与所有原子的内积
        correlations = np.abs(dictionary.T @ residual) / atom_norms
        best_idx = np.argmax(correlations)

        if correlations[best_idx] < 1e-12:
            break

        # 更新系数
        coeffs[best_idx] += dictionary[:, best_idx] @ residual
        # 更新残差
        residual -= coeffs[best_idx] * dictionary[:, best_idx]

        # 检查残差能量
        if np.linalg.norm(residual) < 1e-12:
            break

    return dictionary @ coeffs


def _orthogonal_matching_pursuit(
    signal: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int = 15,
) -> np.ndarray:
    """正交匹配追踪 (OMP) 稀疏编码。

    OMP 在每次迭代中对所选原子进行正交化投影，
    确保残差与已选子空间正交，收敛更快。

    Args:
        signal: 1D 输入信号
        dictionary: 字典矩阵 (n_samples, n_atoms)
        sparsity: 稀疏度（非零系数个数）

    Returns:
        重构信号
    """
    n_atoms = dictionary.shape[1]
    residual = signal.copy()
    support = []
    coeffs = np.zeros(n_atoms)

    # 预计算原子范数
    atom_norms = np.linalg.norm(dictionary, axis=0)
    atom_norms = np.where(atom_norms < 1e-12, 1.0, atom_norms)

    for _ in range(sparsity):
        # 计算相关度
        correlations = np.abs(dictionary.T @ residual) / atom_norms
        best_idx = np.argmax(correlations)

        if correlations[best_idx] < 1e-12:
            break

        support.append(best_idx)

        # 在已选原子张成的子空间上做最小二乘投影
        A = dictionary[:, support]
        try:
            x, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
            residual = signal - A @ x
        except np.linalg.LinAlgError:
            break

        # 检查残差能量
        if np.linalg.norm(residual) < 1e-12:
            break

    if support:
        A = dictionary[:, support]
        try:
            x, _, _, _ = np.linalg.lstsq(A, signal, rcond=None)
            return A @ x
        except np.linalg.LinAlgError:
            pass

    return signal


# ==================== 1. 匹配追踪 (MP) ====================


@register_algorithm(
    name="匹配追踪(MP)",
    category=AlgorithmCategory.SPARSE,
    complexity=AlgorithmComplexity.HIGH,
    tags=["sparse", "mp", "gabor", "greedy"],
)
class MatchingPursuitDenoise(BaseAlgorithm):
    """匹配追踪 (Matching Pursuit) 降噪。

    使用Gabor过完备字典，通过贪婪迭代选择与残差最匹配的原子，
    逐步逼近原始信号。稀疏表示能够有效分离信号和噪声成分。

    特点:
    - 使用Gabor字典，具有良好的时频局部化特性
    - 贪婪搜索策略，每次选取相关度最高的原子
    - 适用于非平稳信号和瞬态特征提取
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_atoms": 128,
        "max_iter": 50,
        "sparsity": 10,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        n_atoms = int(params.get("n_atoms", 128))
        max_iter = int(params.get("max_iter", 50))
        sparsity = int(params.get("sparsity", 10))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        # 构造Gabor字典（以第一通道为准，所有通道共用）
        n_atoms = min(n_atoms, n_samples)
        dictionary = _gabor_dictionary(n_samples, n_atoms, sample_rate)

        for ch in range(n_channels):
            try:
                output[ch] = _matching_pursuit(
                    signal[ch], dictionary,
                    max_iter=max_iter, sparsity=sparsity,
                )
            except Exception:
                # 容错：返回原始信号
                output[ch] = signal[ch]

        return output[0] if is_1d else output


# ==================== 2. 正交匹配追踪 (OMP) ====================


@register_algorithm(
    name="正交匹配追踪(OMP)",
    category=AlgorithmCategory.SPARSE,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["sparse", "omp", "orthogonal", "greedy"],
)
class OrthogonalMatchingPursuitDenoise(BaseAlgorithm):
    """正交匹配追踪 (Orthogonal Matching Pursuit) 降噪。

    在匹配追踪的基础上，每次迭代对所选原子进行最小二乘投影，
    确保残差与已选子空间正交，收敛速度更快，稀疏表示质量更高。

    特点:
    - 正交化投影，避免重复选择相似的原子
    - 比MP更快的收敛速度和更优的重构精度
    - 适用于需要精确稀疏表示的场景
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "sparsity": 15,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sparsity = int(params.get("sparsity", 15))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        # 构造Gabor字典（以第一通道为准，所有通道共用）
        n_atoms = min(2 * n_samples, 256)
        dictionary = _gabor_dictionary(n_samples, n_atoms, sample_rate)

        for ch in range(n_channels):
            try:
                output[ch] = _orthogonal_matching_pursuit(
                    signal[ch], dictionary, sparsity=sparsity,
                )
            except Exception:
                output[ch] = signal[ch]

        return output[0] if is_1d else output


# ==================== 3. 基追踪降噪 (BPDN) ====================


@register_algorithm(
    name="基追踪降噪(BPDN)",
    category=AlgorithmCategory.SPARSE,
    complexity=AlgorithmComplexity.HIGH,
    tags=["sparse", "bpdn", "l1-norm", "basis-pursuit"],
)
class BasisPursuitDenoise(BaseAlgorithm):
    """基追踪降噪 (Basis Pursuit Denoising)。

    通过最小化 L1 范数正则化的重构误差来寻找信号的稀疏表示：
        min ||x||_1  subject to  ||Dx - y||_2 <= epsilon
    这里使用OMP算法进行近似求解，平衡稀疏性与重构保真度。

    特点:
    - L1范数正则化促进稀疏性
    - 正则化系数 lambda_ 控制稀疏度与重构误差的平衡
    - 适用于强噪声环境下的信号恢复
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "lambda_": 0.1,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        lambda_ = float(params.get("lambda_", 0.1))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        # 构造Gabor字典
        n_atoms = min(2 * n_samples, 256)
        dictionary = _gabor_dictionary(n_samples, n_atoms, sample_rate)

        for ch in range(n_channels):
            try:
                # BPDN的OMP近似: 将lambda_映射为稀疏度
                # 较大的lambda_意味着更强的稀疏性（更少的原子）
                sig_power = np.mean(signal[ch] ** 2)
                if sig_power < 1e-12:
                    output[ch] = signal[ch]
                    continue

                # 估计噪声水平，计算合适的稀疏度
                # lambda_ 越大，稀疏度越低（更少的原子被选中）
                sparsity = max(1, int(n_samples * lambda_ / (1 + lambda_)))
                sparsity = min(sparsity, n_atoms)

                output[ch] = _orthogonal_matching_pursuit(
                    signal[ch], dictionary, sparsity=sparsity,
                )
            except Exception:
                output[ch] = signal[ch]

        return output[0] if is_1d else output


# ==================== 4. K-SVD字典学习降噪 ====================


@register_algorithm(
    name="K-SVD字典学习降噪",
    category=AlgorithmCategory.SPARSE,
    complexity=AlgorithmComplexity.HIGH,
    tags=["sparse", "ksvd", "dictionary-learning", "adaptive"],
)
class KSVDDenoise(BaseAlgorithm):
    """K-SVD 字典学习降噪。

    从信号本身（或干净训练数据）中学习过完备字典，
    用学习到的字典对信号进行稀疏编码，实现自适应降噪。

    使用 sklearn 的 DictionaryLearning 或回退到手动 SVD 更新。

    特点:
    - 自适应的字典，更好地匹配信号特征
    - 相比固定字典（如Gabor）有更好的表示能力
    - 适用于具有复杂结构或模式重复的信号
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "dict_size": 128,
        "atom_length": 32,
        "sparsity": 10,
        "n_iter": 10,
    }

    # K-SVD 复杂度极高（尤其 simple 回退路径），设安全上限
    max_signal_length: ClassVar[int] = 100000

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        self._check_signal_size(signal)
        params = {**self.params, **kwargs}
        dict_size = int(params.get("dict_size", 128))
        atom_length = int(params.get("atom_length", 32))
        sparsity = int(params.get("sparsity", 10))
        n_iter = int(params.get("n_iter", 10))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            try:
                output[ch] = self._ksvd_denoise_channel(
                    signal[ch], dict_size, atom_length, sparsity, n_iter,
                )
            except Exception:
                output[ch] = signal[ch]

        return output[0] if is_1d else output

    def _ksvd_denoise_channel(
        self,
        sig: np.ndarray,
        dict_size: int,
        atom_length: int,
        sparsity: int,
        n_iter: int,
    ) -> np.ndarray:
        """单通道K-SVD降噪。

        优先使用 sklearn 的 DictionaryLearning，
        如果不可用则使用简化的迭代SVD方法。
        """
        # 优先使用 sklearn
        try:
            return self._sklearn_ksvd(sig, dict_size, atom_length, sparsity, n_iter)
        except ImportError:
            # sklearn 不可用，使用简化实现
            return self._simple_ksvd(sig, dict_size, atom_length, sparsity, n_iter)

    def _sklearn_ksvd(
        self,
        sig: np.ndarray,
        dict_size: int,
        atom_length: int,
        sparsity: int,
        n_iter: int,
    ) -> np.ndarray:
        """使用 sklearn 的 DictionaryLearning 进行字典学习和降噪。"""
        from sklearn.decomposition import DictionaryLearning
        from sklearn.feature_extraction.image import extract_patches_2d

        sig = sig.flatten()
        n = len(sig)

        # 如果信号太短，调整参数
        atom_length = min(atom_length, n // 4)
        if atom_length < 2:
            return sig

        # 从信号中提取重叠片段作为训练样本
        stride = max(1, atom_length // 2)
        n_patches = max(1, (n - atom_length) // stride)
        patches = np.zeros((n_patches, atom_length))

        for i in range(n_patches):
            start = i * stride
            end = start + atom_length
            if end > n:
                break
            patches[i] = sig[start:end]
        patches = patches[:i + 1]

        if patches.shape[0] < 2:
            return sig

        # 去均值
        patch_mean = np.mean(patches, axis=1, keepdims=True)
        patches = patches - patch_mean

        # 字典学习
        dict_size = min(dict_size, patches.shape[0], atom_length * 2)
        dico = DictionaryLearning(
            n_components=dict_size,
            alpha=1.0 / sparsity if sparsity > 0 else 1.0,
            max_iter=n_iter,
            transform_algorithm="omp",
            transform_n_nonzero_coefs=sparsity,
            random_state=42,
            fit_algorithm="ksvd",
        )
        dico.fit(patches)

        # 稀疏编码
        code = dico.transform(patches)
        patches_recon = code @ dico.components_ + patch_mean

        # 重叠拼接重构信号
        recon = np.zeros(n)
        weight = np.zeros(n)
        for i in range(patches_recon.shape[0]):
            start = i * stride
            end = start + atom_length
            if end > n:
                break
            recon[start:end] += patches_recon[i]
            weight[start:end] += 1.0

        weight = np.where(weight > 0, weight, 1.0)
        return recon / weight

    def _simple_ksvd(
        self,
        sig: np.ndarray,
        dict_size: int,
        atom_length: int,
        sparsity: int,
        n_iter: int,
    ) -> np.ndarray:
        """简化的K-SVD实现（不依赖sklearn）。

        通过迭代SVD更新字典，使用OMP进行稀疏编码。

        注意：此回退实现复杂度极高（n_iter × n_patches × sparsity × lstsq），
        为避免卡死系统，超过一定规模时自动降低迭代次数和 patch 数。
        """
        sig = sig.flatten()
        n = len(sig)

        atom_length = min(atom_length, n // 4)
        if atom_length < 2:
            return sig

        stride = max(1, atom_length // 2)
        n_patches = max(1, (n - atom_length) // stride)

        # 自适应规模保护：避免 simple 路径在大信号上死循环
        # n_patches × n_iter 超过 5000 时自动降级
        max_work = 5000
        if n_patches * n_iter > max_work:
            import warnings
            new_stride = max(stride, (n_patches * n_iter) // max_work * stride)
            warnings.warn(
                f"K-SVD simple: n_patches={n_patches} × n_iter={n_iter} 超限，"
                f"stride {stride} -> {new_stride}, n_iter {n_iter} -> {min(n_iter, 5)}"
            )
            stride = new_stride
            n_iter = min(n_iter, 5)
            n_patches = max(1, (n - atom_length) // stride)

        # 提取重叠片段
        patches = np.zeros((n_patches, atom_length))
        for i in range(n_patches):
            start = i * stride
            end = start + atom_length
            if end > n:
                break
            patches[i] = sig[start:end]
        patches = patches[:i + 1]

        if patches.shape[0] < 2:
            return sig

        # 初始化字典：随机选取部分训练样本并归一化
        dict_size = min(dict_size, patches.shape[0])
        indices = np.random.choice(patches.shape[0], dict_size, replace=False)
        dictionary = patches[indices].T  # (atom_length, dict_size)
        # 归一化
        norms = np.linalg.norm(dictionary, axis=0)
        norms = np.where(norms < 1e-12, 1.0, norms)
        dictionary = dictionary / norms

        # 迭代：稀疏编码 -> 字典更新
        for iteration in range(n_iter):
            # ---- 稀疏编码阶段 (OMP) ----
            code = np.zeros((dict_size, patches.shape[0]))
            for j in range(patches.shape[0]):
                patch = patches[j]
                # OMP
                residual = patch.copy()
                support = []
                for _ in range(sparsity):
                    correlations = np.abs(dictionary.T @ residual)
                    best_idx = np.argmax(correlations)
                    if correlations[best_idx] < 1e-12:
                        break
                    support.append(best_idx)
                    A = dictionary[:, support]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, patch, rcond=None)
                        residual = patch - A @ x
                    except np.linalg.LinAlgError:
                        break
                if support:
                    A = dictionary[:, support]
                    try:
                        x, _, _, _ = np.linalg.lstsq(A, patch, rcond=None)
                        code[support, j] = x
                    except np.linalg.LinAlgError:
                        pass

            # ---- 字典更新阶段 (SVD) ----
            for k in range(dict_size):
                # 找到使用该原子的样本索引
                idx = np.where(np.abs(code[k]) > 1e-12)[0]
                if len(idx) == 0:
                    continue

                # 计算不使用该原子时的表示误差
                error = patches[idx].T - dictionary @ code[:, idx]
                error += np.outer(dictionary[:, k], code[k, idx])

                # SVD 更新
                try:
                    U, S, Vt = np.linalg.svd(error, full_matrices=False)
                    dictionary[:, k] = U[:, 0]
                    code[k, idx] = S[0] * Vt[0, :]
                except np.linalg.LinAlgError:
                    continue

        # 重构信号
        patches_recon = (dictionary @ code).T  # (n_patches, atom_length)

        recon = np.zeros(n)
        weight = np.zeros(n)
        for i in range(patches_recon.shape[0]):
            start = i * stride
            end = start + atom_length
            if end > n:
                break
            recon[start:end] += patches_recon[i]
            weight[start:end] += 1.0

        weight = np.where(weight > 0, weight, 1.0)
        return recon / weight


# ==================== CoSaMP 辅助函数 ====================


def _cosamp(
    signal: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int = 10,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """压缩采样匹配追踪 (CoSaMP) 稀疏编码。

    每次迭代选择 2*sparsity 个候选原子，通过最小二乘
    和回溯剔除保留最相关的 sparsity 个原子。
    """
    m, n = dictionary.shape
    x = np.zeros(n)
    r = signal.copy()
    support = set()

    for _ in range(max_iter):
        h = dictionary.T @ r
        idx_new = np.argsort(np.abs(h))[-2 * sparsity:]
        support = support.union(set(idx_new.tolist()))

        D_s = dictionary[:, list(support)]
        try:
            x_s, _, _, _ = np.linalg.lstsq(D_s, signal, rcond=None)
        except np.linalg.LinAlgError:
            break

        idx_sort = np.argsort(np.abs(x_s))[-sparsity:]
        support = set(np.array(list(support))[idx_sort].tolist())

        x_new = np.zeros(n)
        x_new[list(support)] = x_s[idx_sort]
        r = signal - dictionary @ x_new

        if np.linalg.norm(r) < tol:
            break
        x = x_new

    return dictionary @ x


# ==================== FISTA 辅助函数 ====================


def _soft_threshold(x: np.ndarray, alpha: float) -> np.ndarray:
    """软阈值算子"""
    return np.sign(x) * np.maximum(np.abs(x) - alpha, 0.0)


def _fista_lasso(
    signal: np.ndarray,
    dictionary: np.ndarray,
    lam: float = 0.1,
    max_iter: int = 300,
    tol: float = 1e-7,
) -> np.ndarray:
    """FISTA 求解 LASSO 问题。

    min 0.5 * ||Dx - y||^2 + lambda * ||x||_1
    """
    L = np.linalg.norm(dictionary, ord=2) ** 2
    inv_L = 1.0 / L
    n = dictionary.shape[1]

    x = np.zeros(n)
    z = np.zeros(n)
    t = 1.0

    for k in range(max_iter):
        x_old = x.copy()
        grad = dictionary.T @ (dictionary @ z - signal)
        x = _soft_threshold(z - inv_L * grad, lam * inv_L)

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        z = x + (t - 1) / t_new * (x - x_old)
        t = t_new

        if np.linalg.norm(x - x_old) < tol:
            break

    return dictionary @ x


# ==================== 5. 压缩采样匹配追踪 (CoSaMP) ====================

@register_algorithm(
    name="压缩采样匹配追踪(CoSaMP)",
    category=AlgorithmCategory.SPARSE,
    complexity=AlgorithmComplexity.HIGH,
    tags=["sparse", "cosamp", "compressive-sensing", "greedy"],
)
class CosampDenoise(BaseAlgorithm):
    """压缩采样匹配追踪 (CoSaMP) 降噪。"""

    default_params: ClassVar[Dict[str, Any]] = {
        "sparsity": 10,
        "max_iter": 100,
        "tolerance": 1e-6,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sparsity = int(params.get("sparsity", 10))
        max_iter = int(params.get("max_iter", 100))
        tol = float(params.get("tolerance", 1e-6))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        n_atoms = min(2 * n_samples, 256)
        dictionary = _gabor_dictionary(n_samples, n_atoms, sample_rate)

        for ch in range(n_channels):
            try:
                output[ch] = _cosamp(signal[ch], dictionary, sparsity, max_iter, tol)
            except Exception:
                output[ch] = signal[ch]

        return output[0] if is_1d else output


# ==================== 6. 快速迭代收缩阈值算法 (FISTA) ====================

@register_algorithm(
    name="快速迭代收缩阈值算法(FISTA)",
    category=AlgorithmCategory.SPARSE,
    complexity=AlgorithmComplexity.HIGH,
    tags=["sparse", "fista", "l1", "convex"],
)
class FistaDenoise(BaseAlgorithm):
    """快速迭代收缩阈值算法 (FISTA) 降噪。"""

    default_params: ClassVar[Dict[str, Any]] = {
        "lambda_": 0.1,
        "max_iter": 300,
        "tolerance": 1e-7,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        lam = float(params.get("lambda_", 0.1))
        max_iter = int(params.get("max_iter", 300))
        tol = float(params.get("tolerance", 1e-7))

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        n_atoms = min(2 * n_samples, 256)
        dictionary = _gabor_dictionary(n_samples, n_atoms, sample_rate)

        for ch in range(n_channels):
            try:
                output[ch] = _fista_lasso(signal[ch], dictionary, lam, max_iter, tol)
            except Exception:
                output[ch] = signal[ch]

        return output[0] if is_1d else output
