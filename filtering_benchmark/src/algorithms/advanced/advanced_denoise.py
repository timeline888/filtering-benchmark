"""
前沿滤波降噪算法集合。

包含以下前沿方法:
1. 压缩感知降噪 (Compressed Sensing)
2. 随机共振 (Stochastic Resonance)
3. 形态学滤波 (Morphological Filtering)
4. 全变分去噪 (Total Variation)
5. 非局部均值降噪 (Non-Local Means)
6. 图信号处理降噪 (Graph Signal Processing)
7. 分形降噪 (Fractal Denoising)
"""

from typing import Any, ClassVar, Dict

import numpy as np

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


# ========================================================================
# 1. 压缩感知降噪 (Compressed Sensing Denoise)
# ========================================================================

@register_algorithm(
    name="压缩感知降噪",
    category=AlgorithmCategory.ADVANCED,
    complexity=AlgorithmComplexity.HIGH,
    tags=["compressed-sensing", "sparse", "omp", "dct"],
)
class CompressedSensingDenoise(BaseAlgorithm):
    """
    压缩感知降噪算法。

    原理：信号在 DCT 域是稀疏的，通过正交匹配追踪 (OMP) 迭代重建，
    实现从含噪观测中恢复原始稀疏系数，从而达到降噪目的。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "sparsity": 20,       # 稀疏度（OMP 迭代次数）
        "transform": "dct",   # 稀疏变换基: "dct" | "fft"
    }

    # 一次构造 n×n DCT/FFT 矩阵，内存为 O(n²)；n=50000 约耗 20GB。
    max_signal_length: ClassVar[int] = 20000

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        self._check_signal_size(signal)
        params = {**self.params, **kwargs}
        sparsity = int(params["sparsity"])
        transform = params["transform"]

        # 处理单通道/多通道
        if signal.ndim == 1:
            return self._denoise_1d(signal, sparsity, transform)
        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_1d(signal[ch], sparsity, transform)
        return result

    def _denoise_1d(self, sig: np.ndarray, sparsity: int, transform: str) -> np.ndarray:
        n = len(sig)
        # 构建稀疏变换矩阵
        if transform == "dct":
            D = self._dct_matrix(n)
        else:
            D = self._fft_matrix(n)

        # 在变换域中通过 OMP 重建稀疏系数
        y = sig.reshape(-1, 1)
        x_hat = self._omp(D, y, sparsity).ravel()

        # 重建信号
        return D @ x_hat

    def _dct_matrix(self, n: int) -> np.ndarray:
        """构造 DCT-II 正交矩阵 (n x n)"""
        D = np.zeros((n, n))
        for k in range(n):
            if k == 0:
                D[:, k] = 1.0 / np.sqrt(n)
            else:
                D[:, k] = np.sqrt(2.0 / n) * np.cos(
                    np.pi * k * (2 * np.arange(n) + 1) / (2 * n)
                )
        return D

    def _fft_matrix(self, n: int) -> np.ndarray:
        """构造归一化的 FFT 正交矩阵 (n x n)"""
        return np.fft.fft(np.eye(n)) / np.sqrt(n)

    def _omp(self, D: np.ndarray, y: np.ndarray, sparsity: int) -> np.ndarray:
        """
        正交匹配追踪 (Orthogonal Matching Pursuit)。

        Args:
            D: 字典矩阵 (n x n)
            y: 观测信号 (n, 1)
            sparsity: 稀疏度（迭代次数）

        Returns:
            稀疏系数向量 (n,)
        """
        n = D.shape[0]
        # 归一化字典
        norms = np.linalg.norm(D, axis=0)
        norms[norms == 0] = 1.0
        D_norm = D / norms.reshape(1, -1)

        residual = y.astype(np.float64).copy()
        support = []
        x = np.zeros(n)

        for _ in range(min(sparsity, n)):
            # 计算相关性
            correlations = D_norm.T @ residual
            idx = int(np.argmax(np.abs(correlations).ravel()))

            # 更新支撑集
            if idx in support:
                break
            support.append(idx)

            # 在支撑集上最小二乘
            Ds = D[:, support]
            try:
                x_s = np.linalg.lstsq(Ds, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            residual = y - Ds @ x_s
            rr = np.linalg.norm(residual)
            if rr < 1e-12:
                break

        x[support] = x_s.ravel() if len(x_s) > 0 else np.array([])
        return x


# ========================================================================
# 2. 随机共振 (Stochastic Resonance)
# ========================================================================

@register_algorithm(
    name="随机共振降噪",
    category=AlgorithmCategory.ADVANCED,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["stochastic-resonance", "bistable", "nonlinear"],
)
class StochasticResonance(BaseAlgorithm):
    """
    随机共振降噪算法。

    原理：利用适量的噪声可以增强周期性弱信号的现象。
    通过朗之万方程描述的双稳态系统，使信号+噪声在势阱间跃迁，
    利用噪声能量增强信号特征。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "a": 1.0,    # 双稳态势垒参数 a
        "b": 1.0,    # 双稳态势垒参数 b
        "h": 0.3,    # 步长（数值积分步长）
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        a = params["a"]
        b = params["b"]
        h = params["h"]

        if signal.ndim == 1:
            return self._denoise_1d(signal, a, b, h)
        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_1d(signal[ch], a, b, h)
        return result

    def _denoise_1d(self, sig: np.ndarray, a: float, b: float, h: float) -> np.ndarray:
        """
        使用四阶龙格-库塔 (RK4) 求解双稳态系统朗之万方程:
            dx/dt = a*x - b*x^3 + signal(t)

        双稳态势函数: U(x) = -a/2 * x^2 + b/4 * x^4
        势垒位于 x=0，两个势阱位于 x = ±sqrt(a/b)
        """
        n = len(sig)
        x = np.zeros(n)
        x[0] = 0.0  # 初始状态在势垒处

        for i in range(n - 1):
            xn = x[i]
            sn = sig[i]

            # RK4 积分
            k1 = a * xn - b * xn**3 + sn
            k2 = a * (xn + 0.5 * h * k1) - b * (xn + 0.5 * h * k1)**3 + sig[i]
            k3 = a * (xn + 0.5 * h * k2) - b * (xn + 0.5 * h * k2)**3 + sig[i]
            # 对于 k4，使用下一个采样点的信号值（如果存在）
            s_next = sig[i + 1] if i + 1 < n else sig[i]
            k4 = a * (xn + h * k3) - b * (xn + h * k3)**3 + s_next

            x[i + 1] = xn + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 去趋势（双稳态系统的输出可能有偏置）
        x = x - np.mean(x)
        # 将输出缩放到与输入相近的幅度范围
        x = x * (np.std(sig) / (np.std(x) + 1e-12))
        return x


# ========================================================================
# 3. 形态学滤波 (Morphological Filtering)
# ========================================================================

@register_algorithm(
    name="形态学滤波",
    category=AlgorithmCategory.ADVANCED,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["morphological", "structuring-element", "opening-closing"],
)
class MorphologicalFilter(BaseAlgorithm):
    """
    形态学滤波降噪算法。

    原理：利用数学形态学的开运算和闭运算组合来滤除信号中的
    正负脉冲噪声。开运算去除正脉冲（峰值噪声），闭运算填平
    负脉冲（谷值噪声）。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "window_size": 5,          # 结构元素大小（奇数）
        "operation": "opening",    # "opening" | "closing" | "occo"
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        window_size = int(params["window_size"])
        operation = params["operation"]

        if window_size % 2 == 0:
            window_size += 1  # 保证奇数

        if signal.ndim == 1:
            return self._denoise_1d(signal, window_size, operation)
        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_1d(signal[ch], window_size, operation)
        return result

    def _denoise_1d(self, sig: np.ndarray, window_size: int, operation: str) -> np.ndarray:
        try:
            from scipy.ndimage import grey_opening, grey_closing

            if operation == "opening":
                # 开运算：腐蚀 -> 膨胀，去除正脉冲
                return grey_opening(sig, size=window_size)
            elif operation == "closing":
                # 闭运算：膨胀 -> 腐蚀，填平负脉冲
                return grey_closing(sig, size=window_size)
            elif operation == "occo":
                # 开-闭组合: 先开运算再闭运算（综合效果）
                opened = grey_opening(sig, size=window_size)
                return grey_closing(opened, size=window_size)
            else:
                return grey_opening(sig, size=window_size)
        except ImportError:
            # scipy 不可用时回退到简单形态学操作
            return self._morphology_fallback(sig, window_size, operation)

    def _morphology_fallback(self, sig: np.ndarray, window_size: int, operation: str) -> np.ndarray:
        """手动实现一维形态学操作（无 scipy 依赖时的回退方案）"""
        half = window_size // 2
        n = len(sig)
        result = sig.copy()

        def _erode(x: np.ndarray) -> np.ndarray:
            """一维腐蚀（局部最小值）"""
            out = np.zeros_like(x)
            for i in range(n):
                left = max(0, i - half)
                right = min(n, i + half + 1)
                out[i] = np.min(x[left:right])
            return out

        def _dilate(x: np.ndarray) -> np.ndarray:
            """一维膨胀（局部最大值）"""
            out = np.zeros_like(x)
            for i in range(n):
                left = max(0, i - half)
                right = min(n, i + half + 1)
                out[i] = np.max(x[left:right])
            return out

        def _opening(x):
            return _dilate(_erode(x))

        def _closing(x):
            return _erode(_dilate(x))

        if operation == "opening":
            result = _opening(sig)
        elif operation == "closing":
            result = _closing(sig)
        elif operation == "occo":
            result = _closing(_opening(sig))

        return result


# ========================================================================
# 4. 全变分去噪 (Total Variation Denoise)
# ========================================================================

@register_algorithm(
    name="全变分去噪(TV)",
    category=AlgorithmCategory.ADVANCED,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["total-variation", "tv-denoise", "gradient-descent", "regularization"],
)
class TotalVariationDenoise(BaseAlgorithm):
    """
    全变分去噪算法。

    原理：最小化全变分正则化项，在保持边缘的同时去除噪声。
    目标函数: min_x ||x - y||^2 + lambda * TV(x)
    其中 TV(x) = sum(|x[i+1] - x[i]|)
    使用梯度下降法求解。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "lambda_": 0.1,    # 正则化强度
        "n_iter": 100,     # 迭代次数
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        lambda_ = params["lambda_"]
        n_iter = int(params["n_iter"])

        if signal.ndim == 1:
            return self._denoise_1d(signal, lambda_, n_iter)
        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_1d(signal[ch], lambda_, n_iter)
        return result

    def _denoise_1d(self, sig: np.ndarray, lambda_: float, n_iter: int) -> np.ndarray:
        """
        梯度下降求解 TV 去噪。
        TV 的次梯度: div(x/|x|) 的离散形式。
        """
        x = sig.copy().astype(np.float64)
        n = len(x)
        # 自适应步长
        dt = 0.5 / (1.0 + 4.0 * lambda_)

        for _ in range(n_iter):
            # 计算一阶差分
            diff = np.diff(x)
            # 计算 TV 的次梯度: -div(sign(diff)), 即 [sign(diff[0]), -sign(diff[:-1]) - sign(diff[1:]), -sign(diff[-1])]
            # 使用平滑近似避免除零
            eps = 1e-12
            grad_tv = np.zeros(n)
            grad_tv[0] = diff[0] / (np.abs(diff[0]) + eps)
            grad_tv[-1] = -diff[-1] / (np.abs(diff[-1]) + eps)
            for i in range(1, n - 1):
                grad_tv[i] = (diff[i] / (np.abs(diff[i]) + eps)
                              - diff[i - 1] / (np.abs(diff[i - 1]) + eps))

            # 梯度下降
            x = x - dt * ((x - sig) + lambda_ * grad_tv)

        return x


# ========================================================================
# 5. 非局部均值降噪 (Non-Local Means Denoise)
# ========================================================================

@register_algorithm(
    name="非局部均值降噪(NLM)",
    category=AlgorithmCategory.ADVANCED,
    complexity=AlgorithmComplexity.HIGH,
    tags=["non-local-means", "self-similarity", "patch-based", "nlm"],
)
class NonLocalMeansDenoise(BaseAlgorithm):
    """
    非局部均值降噪算法 (NLM)。

    原理：利用信号的自相似性，对每个点的邻域（patch）在整个信号中
    寻找相似 patch，然后对这些相似 patch 进行加权平均，
    权重由 patch 间的欧氏距离决定。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "patch_size": 7,        # 局部 patch 半宽度（实际宽度为 2*patch_size+1）
        "search_radius": 15,    # 搜索窗口半宽度
        "h": 0.1,               # 滤波强度参数（指数衰减系数）
    }

    # NLM 即使向量化后仍为 O(n × search_radius × patch_size)，
    # 并需临时内存约 n × (2*patch_size+1) × 8 字节，设安全上限。
    max_signal_length: ClassVar[int] = 50000

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        self._check_signal_size(signal)
        params = {**self.params, **kwargs}
        patch_size = int(params["patch_size"])
        search_radius = int(params["search_radius"])
        h = params["h"]

        if signal.ndim == 1:
            return self._denoise_1d(signal, patch_size, search_radius, h)
        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_1d(signal[ch], patch_size, search_radius, h)
        return result

    def _denoise_1d(self, sig: np.ndarray, patch_size: int,
                    search_radius: int, h: float) -> np.ndarray:
        n = len(sig)
        if n > 5000:
            scale = min(n / 5000, 4.0)
            patch_size = max(2, int(patch_size / scale))
            search_radius = max(3, int(search_radius / scale))
            import warnings
            warnings.warn(
                f"NLM: n={n} auto-scale patch={patch_size} search={search_radius}"
            )

        # 向量化实现：使用 sliding_window_view 构造所有 patch，
        # 对每个 shift d 在 [-r, r] 范围内一次性计算 n 个距离。
        # 相比原来的 Python 双层循环快 50–100 倍。
        try:
            from numpy.lib.stride_tricks import sliding_window_view
        except ImportError:
            sliding_window_view = None

        pad = patch_size
        padded = np.pad(sig, pad, mode="reflect")
        h2 = (h * h) if h > 0 else 1e-12

        if sliding_window_view is not None:
            # patches 形状: (n, 2*patch_size+1)
            patches = sliding_window_view(padded, 2 * patch_size + 1)
            if patches.shape[0] != n:
                patches = patches[:n]

            values_sum = np.zeros(n, dtype=np.float64)
            weights_sum = np.zeros(n, dtype=np.float64)
            idx = np.arange(n)

            for d in range(-search_radius, search_radius + 1):
                j_idx = idx + d
                valid = (j_idx >= 0) & (j_idx < n)
                j_idx_clip = np.clip(j_idx, 0, n - 1)
                diff = patches - patches[j_idx_clip]
                dist = np.einsum("ij,ij->i", diff, diff)
                w = np.exp(-dist / h2)
                w = np.where(valid, w, 0.0)
                weights_sum += w
                values_sum += w * sig[j_idx_clip]

            result = np.where(
                weights_sum > 1e-12,
                values_sum / np.maximum(weights_sum, 1e-12),
                sig,
            )
            return result.astype(sig.dtype, copy=False)

        # 回退：Python 循环实现（仅当 sliding_window_view 不可用时）
        result = np.zeros_like(sig)
        for i in range(n):
            center = i + pad
            patch_i = padded[center - patch_size: center + patch_size + 1]
            j_start = max(0, i - search_radius)
            j_end = min(n, i + search_radius + 1)
            total_w = 0.0
            acc = 0.0
            for j in range(j_start, j_end):
                j_center = j + pad
                patch_j = padded[j_center - patch_size: j_center + patch_size + 1]
                dist = float(np.sum((patch_i - patch_j) ** 2))
                w = np.exp(-dist / h2) if h > 0 else (1.0 if dist == 0 else 0.0)
                total_w += w
                acc += w * sig[j]
            result[i] = acc / total_w if total_w > 0 else sig[i]
        return result


# ========================================================================
# 6. 图信号处理降噪 (Graph Signal Processing Denoise)
# ========================================================================

@register_algorithm(
    name="图信号处理降噪",
    category=AlgorithmCategory.ADVANCED,
    complexity=AlgorithmComplexity.HIGH,
    tags=["graph-signal", "knn-graph", "laplacian", "regularization"],
)
class GraphSignalDenoise(BaseAlgorithm):
    """
    图信号处理降噪算法。

    原理：将信号建模为图上的函数，构建 kNN 图（每个采样点
    连接到 k 个最相似的邻域点），通过图拉普拉斯正则化实现降噪。
    目标: min_x ||x - y||^2 + alpha * x^T L x
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_neighbors": 10,   # kNN 近邻数量
        "sigma": 1.0,        # 高斯核宽度（计算相似度）
        "alpha": 0.5,        # 图正则化强度
    }

    # kNN 图拉普拉斯矩阵构建为 O(N²)，设安全上限；
    # 超过这个值时算法内部仍会自动降采样，但环境内存需常数
    max_signal_length: ClassVar[int] = 200000

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        self._check_signal_size(signal)
        params = {**self.params, **kwargs}
        n_neighbors = int(params["n_neighbors"])
        sigma = params["sigma"]
        alpha = params["alpha"]

        if signal.ndim == 1:
            return self._denoise_1d(signal, n_neighbors, sigma, alpha)
        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_1d(signal[ch], n_neighbors, sigma, alpha)
        return result

    def _denoise_1d(self, sig: np.ndarray, n_neighbors: int,
                    sigma: float, alpha: float) -> np.ndarray:
        n = len(sig)
        if n > 5000:
            import warnings
            step = max(1, n // 2000)
            sig_down = sig[::step]
            n_down = len(sig_down)
            n_neighbors = min(n_neighbors, n_down - 1)
            warnings.warn(
                f"信号长度 ({n}) 较大，图信号处理自动降采样: "
                f"step={step}, downsampled_size={n_down}"
            )
            L = self._build_graph_laplacian(sig_down, n_neighbors, sigma)
            A = np.eye(n_down) + alpha * L
            b = sig_down.copy()
            try:
                x_down = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x_down = np.linalg.lstsq(A, b, rcond=None)[0]
            # 线性插值回原始网格
            x = np.interp(np.arange(n) / step, np.arange(n_down), x_down)
            return x

        # 构建图拉普拉斯矩阵
        L = self._build_graph_laplacian(sig, n_neighbors, sigma)

        # 求解 (I + alpha * L) x = y
        A = np.eye(n) + alpha * L
        b = sig.copy()

        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 若直接求解失败，使用最小二乘
            x = np.linalg.lstsq(A, b, rcond=None)[0]

        return x

    def _build_graph_laplacian(self, sig: np.ndarray,
                               n_neighbors: int, sigma: float) -> np.ndarray:
        """
        基于信号值相似性构建 kNN 图拉普拉斯矩阵。

        每个节点连接到值最接近的 k 个节点（除自身外），
        边权重由高斯核 exp(-|sig[i]-sig[j]|^2 / sigma^2) 决定。
        """
        n = len(sig)
        n_neighbors = min(n_neighbors, n - 1)

        # 计算所有点对之间的值差异
        # 使用广播 (n,1) - (1,n) -> (n,n)
        diff = sig[:, np.newaxis] - sig[np.newaxis, :]
        dist_sq = diff ** 2

        # 构建邻接矩阵
        adjacency = np.zeros((n, n))

        for i in range(n):
            # 第 i 行：按距离排序（排除自身）
            dists = dist_sq[i]
            # 将自身距离设为无穷大
            dists[i] = np.inf
            # 找到 k 个最近邻
            nearest = np.argpartition(dists, n_neighbors)[:n_neighbors]

            for j in nearest:
                weight = np.exp(-dist_sq[i, j] / (2.0 * sigma ** 2))
                adjacency[i, j] = weight

        # 对称化
        adjacency = (adjacency + adjacency.T) / 2.0

        # 度矩阵
        D = np.diag(np.sum(adjacency, axis=1))

        # 拉普拉斯矩阵 L = D - A
        L = D - adjacency
        return L


# ========================================================================
# 7. 分形降噪 (Fractal Denoising)
# ========================================================================

@register_algorithm(
    name="分形降噪",
    category=AlgorithmCategory.ADVANCED,
    complexity=AlgorithmComplexity.HIGH,
    tags=["fractal", "self-similarity", "hurst-exponent", "multi-scale"],
)
class FractalDenoise(BaseAlgorithm):
    """
    分形降噪算法。

    原理：基于分形维度分析识别并保留信号的自相似结构，滤除非分形噪声。
    通过多尺度分析计算信号的局部赫斯特指数 (Hurst exponent)，
    在变换域中保留具有分形特征的成分。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "threshold": 0.1,   # 分形成分保留阈值
        "max_scale": 10,    # 最大分析尺度
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        threshold = params["threshold"]
        max_scale = int(params["max_scale"])

        if signal.ndim == 1:
            return self._denoise_1d(signal, threshold, max_scale)
        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_1d(signal[ch], threshold, max_scale)
        return result

    def _denoise_1d(self, sig: np.ndarray, threshold: float, max_scale: int) -> np.ndarray:
        n = len(sig)
        # 使用多尺度 DWT 分解
        try:
            import pywt

            # 确定最大分解层数
            max_level = min(max_scale, pywt.dwt_max_level(n, "db4"))

            if max_level < 1:
                return sig

            # 小波分解
            coeffs = list(pywt.wavedec(sig, "db4", level=max_level))
            # coeffs[0] = 近似系数, coeffs[1:] = 细节系数

            # 计算每个尺度（细节层）的赫斯特指数
            for i in range(1, len(coeffs)):
                detail = coeffs[i]
                if len(detail) < 4:
                    continue

                # 计算当前细节系数的局部赫斯特指数
                h = self._hurst_exponent(detail)

                # 分形信号的特征: 赫斯特指数接近 0.5~1.0
                # 白噪声: H ~ 0.5, 分形布朗运动: H ~ 0.5~1.0
                # 当 H 偏离 0.5 越远，越可能是分形成分

                # 分形保留权重: |H - 0.5| / 0.5，归一化到 [0, 1]
                fractal_weight = min(1.0, abs(2.0 * h - 1.0))

                # 根据阈值决定保留多少分形成分
                if fractal_weight < threshold:
                    # 该层主要是噪声，强衰减
                    coeffs[i] = detail * fractal_weight
                else:
                    # 保留大部分分形成分
                    coeffs[i] = detail * (threshold + (1.0 - threshold) * fractal_weight)

            # 重建信号
            result = pywt.waverec(coeffs, "db4")[:n]
            return result

        except ImportError:
            # pywt 不可用时，使用基于 FFT 的分形滤波
            return self._fractal_filter_fft(sig, threshold, max_scale)

    def _hurst_exponent(self, x: np.ndarray) -> float:
        """
        计算序列的赫斯特指数 (Hurst exponent) 使用 R/S 分析。

        H > 0.5: 趋势增强（持续性）
        H = 0.5: 随机游走（白噪声）
        H < 0.5: 均值回复（反持续性）
        """
        n = len(x)
        if n < 4:
            return 0.5

        # R/S 分析
        # 将序列切分为多个子段
        min_seg = 4
        max_seg = max(n // 2, 4)

        scales = []
        rs_values = []

        # 对数均匀采样尺度
        n_scales = min(20, max_seg - min_seg + 1)
        seg_sizes = np.unique(
            np.logspace(np.log10(min_seg), np.log10(max_seg),
                        n_scales, dtype=int)
        )

        for seg_size in seg_sizes:
            n_segs = n // seg_size
            if n_segs < 1:
                continue

            rs_avg = 0.0
            for s in range(n_segs):
                seg = x[s * seg_size: (s + 1) * seg_size]
                mean = np.mean(seg)
                # 累积离差
                deviate = seg - mean
                cum_dev = np.cumsum(deviate)
                # 极差
                R = np.max(cum_dev) - np.min(cum_dev)
                # 标准差
                S = np.std(seg, ddof=1)
                if S > 1e-12:
                    rs_avg += R / S

            rs_avg /= n_segs
            scales.append(np.log(seg_size))
            rs_values.append(np.log(rs_avg))

        if len(scales) < 2:
            return 0.5

        # 线性拟合 log(R/S) ~ log(scale)，斜率 = H
        A = np.vstack([scales, np.ones_like(scales)]).T
        try:
            h, _ = np.linalg.lstsq(A, rs_values, rcond=None)[0]
            # 限制赫斯特指数在合理范围 [0, 1]
            h = max(0.0, min(1.0, h))
        except np.linalg.LinAlgError:
            h = 0.5
        return h

    def _fractal_filter_fft(self, sig: np.ndarray, threshold: float,
                            max_scale: int) -> np.ndarray:
        """
        基于 FFT 的分形滤波回退方案。

        通过频谱的 1/f 特性识别分形成分：
        分形噪声的功率谱 ~ 1/f^beta。
        保留符合 1/f 谱特性的成分。
        """
        n = len(sig)
        # FFT
        spectrum = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n)
        power = np.abs(spectrum) ** 2

        # 在有效频率范围拟合 1/f^beta
        pos = freqs > 0
        log_f = np.log(freqs[pos])
        log_p = np.log(power[pos] + 1e-12)

        # 线性拟合得到 beta
        A = np.vstack([-log_f, np.ones_like(log_f)]).T
        try:
            beta, _ = np.linalg.lstsq(A, log_p, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = 0.0

        # 分形噪声 beta ~ 1 (粉色噪声), 白噪声 beta ~ 0
        # 根据 beta 构造分形成分保留掩码
        fractal_strength = min(1.0, abs(beta))

        # 构造分形滤波器：保留符合 1/f 谱的频率成分
        filt = np.ones(len(spectrum), dtype=np.float64)
        for i in range(len(spectrum)):
            if freqs[i] > 0:
                # 该频点的期望分形功率 ~ 1/f^beta
                expected_power = freqs[i] ** (-beta) if beta > 0 else 1.0
                actual_power = power[i]
                # 比值越接近 1，越符合分形特征
                ratio = expected_power / (actual_power + 1e-12)
                weight = np.exp(-0.5 * ((ratio - 1.0) / threshold) ** 2)
                filt[i] = threshold + (1.0 - threshold) * weight * fractal_strength

        # 应用滤波器
        spectrum_filtered = spectrum * filt
        result = np.fft.irfft(spectrum_filtered, n=n)
        return result
