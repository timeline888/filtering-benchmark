"""
算法计算复杂度量化分析与优化建议模块。

对项目中所有滤波降噪算法进行：
1. 时间复杂度和空间复杂度量化评估
2. 资源消耗等级（CPU、内存）分类
3. 每类算法的性能优化建议
4. 算法参数调优指导
"""

from typing import Dict, List, Optional

from ..core.types import (
    AlgorithmCategory,
    AlgorithmComplexity,
    ComplexityAnalysis,
    ResourceConsumption,
)


# ==================== 全局复杂度数据库 ====================

# 每类算法的通用优化建议
_CATEGORY_TIPS: Dict[str, str] = {
    "经典滤波方法": (
        "• 减小滤波器阶数(order)可显著降低计算量（推荐≤8阶）\n"
        "• 使用IIR代替FIR可获得更低计算开销\n"
        "• 对长信号分段滤波后再拼接，避免大数组操作\n"
        "• 如果实时性要求高，使用sosfilt代替filtfilt（零相位滤波会加倍计算量）"
    ),
    "时域降噪方法": (
        "• 减小窗口大小(windou_size)可降低计算量\n"
        "• 中值滤波窗口保持奇数，推荐3~11\n"
        "• S-G滤波的多项式阶数(polyorder)建议≤5\n"
        "• 使用 scipy.signal.convolve 的 'fft' 模式加速大窗口卷积"
    ),
    "频域降噪方法": (
        "• FFT类算法效率较高，但信号长度非2的幂时会变慢\n"
        "• 建议对信号进行零填充到2的幂长度以加速FFT\n"
        "• 频谱减法中减小噪声估计帧数可提速\n"
        "• 倒谱分析截止点(quefrency_cutoff)影响较小"
    ),
    "小波变换类": (
        "• 减小分解层数(level)可显著降低计算量（推荐3~5层）\n"
        "• 使用 'db' 系列小波(db4/db6)比 'sym'/'coif' 系列更快\n"
        "• 硬阈值(hard)比软阈值(soft)计算稍快\n"
        "• 平稳小波(SWT)比DWT慢，非必要时用DWT\n"
        "• DTCWT需要外部库dtcwt，备选DWT回退方案更高效"
    ),
    "经验模态分解类": (
        "• EMD是最快的分解方法，EEMD/CEEMDAN增加集合次数会大幅增加耗时\n"
        "• 建议将集合次数(n_ensemble)从50降低到10~20\n"
        "• 限制最大IMF数量(max_imfs)可提前终止分解\n"
        "• VMD的alpha参数增大可加快收敛，tol增大可减少迭代\n"
        "• SSA的窗口长度(window_length)越小计算越快（推荐N/4）\n"
        "• 信号较长时(>10000点)，先降采样再分解可大幅提速"
    ),
    "自适应滤波": (
        "• LMS/NLMS是最快的自适应算法，O(N*K)\n"
        "• RLS计算量较大(O(N*K²))，减小filter_order可降维\n"
        "• APA的投影阶数(projection_order)越小越快(推荐2~4)\n"
        "• 卡尔曼滤波计算量很低，适合长序列实时处理\n"
        "• 维纳滤波的矩阵求逆(O(K³))是瓶颈，限制filter_order≤64"
    ),
    "奇异值分解类": (
        "• SVD分解的计算量为O(min(L²K, LK²))，矩阵规模是关键\n"
        "• 减小窗口长度(window_length)可显著降低SVD计算量\n"
        "• 建议window_length=N/4~N/3，避免接近N/2\n"
        "• 保留能量比(retained_energy)从0.9降到0.8可减少分量\n"
        "• 使用scipy.linalg.svd代替np.linalg.svd可能更快"
    ),
    "稀疏表示类": (
        "• MP的计算量O(N*M*iter)，减少迭代次数(sparsity)可大幅提速\n"
        "• Gabor字典大小(n_atoms)越大表示越好但越慢(建议≤128)\n"
        "• OMP每步需最小二乘(O(K³))，比MP更慢但收敛更快\n"
        "• K-SVD迭代(n_iter)建议≤10，字典大小(dict_size)≤64\n"
        "• 对于长信号(>10000)，建议分段处理避免内存爆炸\n"
        "• 提前终止误差阈值设大一些可减少迭代"
    ),
    "盲源分离类": (
        "• PCA计算量O(N²)主要来自SVD，限制延时数可提速\n"
        "• ICA/FastICA的迭代次数(max_iter)从200减到50仍可收敛\n"
        "• SOBI的联合对角化(O(N³))是主要瓶颈，限制分量数≤20\n"
        "• 单通道信号嵌入的延时数(n_lags)越大越慢(建议≤50)\n"
        "• 使用sklearn的FastICA比自实现快且更稳定"
    ),
    "深度学习类": (
        "• ⚠️ 所有深度学习算法都需PyTorch支持，CPU推理较慢\n"
        "• 减小segment_length(建议256~512)可显著加速\n"
        "• 增大overlap(如0.75)会增加帧数但可提高质量\n"
        "• DAE(全连接)比CNN系列(CDAE/DnCNN/UNet)更快\n"
        "• DRSN含软阈值机制，速度最慢\n"
        "• Transformer的自注意力O(L²)是瓶颈，segment_length越小越好\n"
        "• 无GPU时建议仅选1~2个轻量网络(DAE/CDAE)\n"
        "• torch不可用时自动回退到scipy方法，速度更快但效果不同"
    ),
    "旋转机械专用": (
        "• MED迭代优化(O(N*L²*iter))较慢，减小filter_order(建议≤32)\n"
        "• MCKD比MED更慢（需计算相关峭度），减少max_iter(≤20)\n"
        "• MOMEDA无迭代，比MED/MCKD快很多\n"
        "• Kurtogram需计算多层STFT，限制n_levels≤5\n"
        "• 循环平稳分析的FFT计算量大，减小n_cyclic_freqs\n"
        "• 包络解调(共振解调)计算量低，适合实时处理"
    ),
    "语音/声学降噪": (
        "• MMSE-STSA和Log-MMSE逐帧处理STFT，帧移hop越大越快\n"
        "• 减小FFT点数(n_fft)从512到256可加速\n"
        "• 噪声估计帧数(noise_estimation_frames)≤5可减少启动延迟\n"
        "• 子空间降噪的协方差矩阵O(N²)是瓶颈\n"
        "• 噪声门(Noise Gate)计算量极低，适合实时\n"
        "• 对长信号分段处理避免大矩阵运算"
    ),
    "其他前沿方法": (
        "• 压缩感知的OMP迭代(O(N²))是瓶颈，减小sparsity(≤10)\n"
        "• 随机共振的RK4积分O(N)，计算量较低\n"
        "• 形态学滤波中结构元素大小≤11为宜\n"
        "• 全变分(TV)去噪的迭代次数(n_iter)从100减到30仍有效\n"
        "• NLM的搜索半径(search_radius)越大越慢(O(N*R²))，建议≤10\n"
        "• 图信号处理构建kNN图O(N²)，信号>2000点自动降采样\n"
        "• 分形降噪的max_scale≤8以避免过多小波分解层"
    ),
}

# 每个算法ID对应的详细复杂度分析
_ALGORITHM_COMPLEXITY: Dict[str, Dict] = {
    # ==================== 经典滤波方法 ====================
    "low_pass_butterworth": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "高阶数(order>10)可能导致数值不稳定",
    },
    "high_pass_butterworth": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "同低通巴特沃斯",
    },
    "band_pass_butterworth": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "无显著瓶颈",
    },
    "band_stop_butterworth": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "无显著瓶颈",
    },
    "notch_iir": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "无显著瓶颈",
    },
    "fir_band_pass": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N*K)，K为滤波器阶数(约100~500)",
        "space_complexity": "O(K)",
        "estimated_time_per_10k": "0.01~0.05秒",
        "cpu_usage": "低",
        "memory_usage": "低",
        "bottlenecks": "高衰减要求(attenuation>60dB)导致阶数大增",
    },
    "iir_band_pass": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "无显著瓶颈",
    },
    # ==================== 时域降噪方法 ====================
    "moving_average_filter": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.005秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "窗口过大时速度略降",
    },
    "median_filter": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N log K)，K为核大小",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "核大小>31时耗时增加",
    },
    "savitzky_golay_filter": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "窗口很大时(>101)有边际影响",
    },
    "ewma_base": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(1)",
        "estimated_time_per_10k": "< 0.005秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "无显著瓶颈",
    },
    "gaussian_filter": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "sigma很大时核展宽增加计算",
    },
    # ==================== 频域降噪方法 ====================
    "fft_ideal_low_pass": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N log N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "低",
        "bottlenecks": "信号长度非2的幂时FFT变慢",
    },
    "spectral_subtraction": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N log N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "0.01~0.03秒",
        "cpu_usage": "低",
        "memory_usage": "低",
        "bottlenecks": "多次FFT/IFFT运算",
    },
    "cepstral_filter": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N log N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "0.01~0.03秒",
        "cpu_usage": "低",
        "memory_usage": "低",
        "bottlenecks": "两次FFT运算",
    },
    "homomorphic_filter": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "低",
        "memory_usage": "低",
        "bottlenecks": "对数运算+滤波",
    },
    # ==================== 小波变换类 ====================
    "dwt_threshold_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "0.01~0.02秒",
        "cpu_usage": "低",
        "memory_usage": "低",
        "bottlenecks": "SURE阈值计算稍慢；高分解层数(>8)增加开销",
    },
    "wavelet_packet_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N log N)",
        "space_complexity": "O(N log N)",
        "estimated_time_per_10k": "0.03~0.1秒",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "最优基搜索(best_basis)需遍历所有节点",
    },
    "dtcwt_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N)，但常数因子大（双树结构）",
        "space_complexity": "O(N)，且为复数存储",
        "estimated_time_per_10k": "0.1~0.5秒（需dtcwt库，否则回退DWT）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "dtcwt库接口开销；复数运算慢于实数DWT",
    },
    "swt_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N log N)，每层O(N)但N不变",
        "space_complexity": "O(N*level)，存储所有层的近似和细节",
        "estimated_time_per_10k": "0.03~0.08秒",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "信号长度需为2的倍数；存储多尺度系数开销大",
    },
    "ewt_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N log N)，频谱分割+小波包",
        "space_complexity": "O(N * n_modes)",
        "estimated_time_per_10k": "0.1~0.3秒（基于小波包近似）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "频谱峰值检测不稳定；实际使用小波包近似",
    },
    # ==================== 经验模态分解类 ====================
    "emd_standard": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N*K)，K为IMF数量(通常5~15)",
        "space_complexity": "O(N*K)",
        "estimated_time_per_10k": "0.1~0.5秒（需PyEMD库）",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "迭代筛分过程(Fang算法)是主要耗时；IMF数量不确定",
    },
    "eemd_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(N*K*n_ensemble)，n_ensemble=50",
        "space_complexity": "O(N*K*n_ensemble)",
        "estimated_time_per_10k": "2~10秒（50次集合）",
        "cpu_usage": "极高",
        "memory_usage": "高",
        "bottlenecks": "多次EMD叠加极其耗时；集合次数是决定性因素",
    },
    "ceemd_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(N*K*n_ensemble)，约与EEMD相当",
        "space_complexity": "O(N*K*n_ensemble)",
        "estimated_time_per_10k": "2~10秒",
        "cpu_usage": "极高",
        "memory_usage": "高",
        "bottlenecks": "同EEMD",
    },
    "ceemdan_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(N*K*n_ensemble)，略高于EEMD",
        "space_complexity": "O(N*K*n_ensemble)",
        "estimated_time_per_10k": "3~15秒",
        "cpu_usage": "极高",
        "memory_usage": "高",
        "bottlenecks": "比EEMD多了自适应噪声添加步骤",
    },
    "vmd_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N*K*iter)，K=5，iter≈50~200",
        "space_complexity": "O(N*K)",
        "estimated_time_per_10k": "1~5秒（需vmdpy库）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "双上升循环+频域更新；tol越小迭代越多",
    },
    "lmd_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N*K*iter)，内部使用VMD",
        "space_complexity": "O(N*K)",
        "estimated_time_per_10k": "1~5秒（回退到VMD）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "无标准库，回退VMD实现导致高耗时",
    },
    "itd_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N)，仅返回原始信号（占位实现）",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒（当前为占位实现）",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "当前为占位实现，未真正执行ITD分解",
    },
    "ssa_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(L*K²)，L≈N/3，SVD分解O(L²*K)",
        "space_complexity": "O(L*K)，轨迹矩阵存储",
        "estimated_time_per_10k": "0.1~0.5秒",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "窗口长度L越大SVD越慢；信号>10000点内存开销大",
    },
    "alif_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N*K*iter)，回退到VMD实现",
        "space_complexity": "O(N*K)",
        "estimated_time_per_10k": "1~5秒",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "无标准库，回退VMD导致高耗时",
    },
    # ==================== 自适应滤波 ====================
    "lms_adaptive_filter": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N*K)，K为滤波器阶数(默认32)",
        "space_complexity": "O(K)",
        "estimated_time_per_10k": "0.01~0.02秒",
        "cpu_usage": "低",
        "memory_usage": "极低",
        "bottlenecks": "逐样本迭代O(N)无向量化加速",
    },
    "nlms_adaptive_filter": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N*K)，与LMS相近",
        "space_complexity": "O(K)",
        "estimated_time_per_10k": "0.01~0.02秒",
        "cpu_usage": "低",
        "memory_usage": "极低",
        "bottlenecks": "额外的除法运算(O(N))",
    },
    "rls_adaptive_filter": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N*K²)，K²来自矩阵-向量运算",
        "space_complexity": "O(K²)，逆相关矩阵P",
        "estimated_time_per_10k": "0.05~0.2秒（K=32时）",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "矩阵P更新(O(K²))和增益向量计算",
    },
    "apa_adaptive_filter": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N*(K*P²))，P为投影阶数",
        "space_complexity": "O(K*P)",
        "estimated_time_per_10k": "0.05~0.15秒",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "矩阵求逆(O(P³))和矩阵乘法(O(K*P²))",
    },
    "kalman_filter": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N)，每步仅2x2矩阵运算",
        "space_complexity": "O(1)，仅存储2x2状态",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "低",
        "memory_usage": "极低",
        "bottlenecks": "无显著瓶颈；Python循环未向量化",
    },
    "wiener_filter": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(K³ + N*K)，矩阵求解O(K³)为主",
        "space_complexity": "O(K²)，Toeplitz矩阵",
        "estimated_time_per_10k": "0.02~0.1秒",
        "cpu_usage": "中",
        "memory_usage": "低",
        "bottlenecks": "Toeplitz矩阵求解；自相关计算O(N²)",
    },
    # ==================== 奇异值分解类 ====================
    "svd_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(L*K²)，L=N/3，K=2N/3",
        "space_complexity": "O(L*K)，Hankel矩阵",
        "estimated_time_per_10k": "0.05~0.3秒",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "SVD分解是主要瓶颈；矩阵越大越慢",
    },
    "hankel_svd": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(L*K²)，L=N/2，矩阵更大",
        "space_complexity": "O(L*K) ≈ O(N²/4)",
        "estimated_time_per_10k": "0.1~0.5秒（窗口为N/2时）",
        "cpu_usage": "中",
        "memory_usage": "中~高",
        "bottlenecks": "默认窗口N/2使矩阵为原信号一半大小",
    },
    # ==================== 稀疏表示类 ====================
    "matching_pursuit_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N*M*iter)，M为原子数(128)，iter=10",
        "space_complexity": "O(N*M)，Gabor字典",
        "estimated_time_per_10k": "0.5~2秒",
        "cpu_usage": "高",
        "memory_usage": "高",
        "bottlenecks": "Gabor字典构造(O(N*M))；每次迭代计算所有原子内积",
    },
    "orthogonal_matching_pursuit_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N*M + iter³)，iter=15",
        "space_complexity": "O(N*M) + O(iter²)",
        "estimated_time_per_10k": "0.2~0.8秒",
        "cpu_usage": "中",
        "memory_usage": "中~高",
        "bottlenecks": "每步最小二乘(O(iter³))；字典大小影响内存",
    },
    "basis_pursuit_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N*M*sparsity)，内部使用OMP",
        "space_complexity": "O(N*M) + O(N)",
        "estimated_time_per_10k": "0.3~1秒",
        "cpu_usage": "高",
        "memory_usage": "高",
        "bottlenecks": "Gabor字典(2*N大小)；OMP迭代",
    },
    "ksvd_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(n_iter * (M² + N*M))",
        "space_complexity": "O(N + M²)，字典+补丁",
        "estimated_time_per_10k": "2~10秒（sklearn版本）",
        "cpu_usage": "极高",
        "memory_usage": "高",
        "bottlenecks": "多次OMP稀疏编码；SVD字典更新；提取补丁",
    },
    # ==================== 盲源分离类 ====================
    "pca_denoise": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(min(L²K, LK²))，L为延时数",
        "space_complexity": "O(L*K)",
        "estimated_time_per_10k": "0.02~0.08秒",
        "cpu_usage": "低",
        "memory_usage": "低",
        "bottlenecks": "SVD分解；延时数越大越慢",
    },
    "ica_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N*M²*iter)，使用FastICA",
        "space_complexity": "O(M² + N*M)",
        "estimated_time_per_10k": "0.1~0.5秒",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "白化O(M²*N)；FastICA迭代各层O(M²)",
    },
    "fast_ica_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N*M²*iter)，M≤40",
        "space_complexity": "O(M² + N*M)",
        "estimated_time_per_10k": "0.1~0.5秒",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "同ICA降噪",
    },
    "sobi_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N*M² + M³*T + M²*n_lags)",
        "space_complexity": "O(M²*n_lags)",
        "estimated_time_per_10k": "0.5~2秒（n_lags=100时）",
        "cpu_usage": "高",
        "memory_usage": "中~高",
        "bottlenecks": "联合对角化(Givens旋转)O(M³*T)；多个延时协方差矩阵",
    },
    # ==================== 深度学习类 ====================
    "dae_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(n_frames * seg_len²)，seg_len=256",
        "space_complexity": "O(seg_len²)，网络权重",
        "estimated_time_per_10k": "1~3秒（CPU，含网络构建开销）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "PyTorch模型加载+分帧重叠拼接；全连接层逐帧运算",
    },
    "cda_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(n_frames * n_filters * seg_len)",
        "space_complexity": "O(n_filters * kernel_size)",
        "estimated_time_per_10k": "0.5~2秒（CPU）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "卷积运算在CPU上较慢；上采样/下采样操作",
    },
    "dncnn_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(n_frames * n_filters * seg_len * n_layers)",
        "space_complexity": "O(n_filters * kernel_size * n_layers)",
        "estimated_time_per_10k": "1~4秒（CPU，10层卷积）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "10层卷积+BN+ReLU组合运算量大；残差连接",
    },
    "unet_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(n_frames * n_channels * seg_len * depth)",
        "space_complexity": "O(2^(depth)*n_channels*seg_len)，跳跃连接存储",
        "estimated_time_per_10k": "3~10秒（CPU，depth=4）",
        "cpu_usage": "极高",
        "memory_usage": "高",
        "bottlenecks": "多级编解码+跳跃连接；上采样CPU慢；seg_len=1024",
    },
    "gan_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(n_frames * seg_len * seg_len + n_iter * latent_dim)",
        "space_complexity": "O(seg_len * hidden_dim)",
        "estimated_time_per_10k": "1~3秒（CPU，含迭代优化）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "GAN生成器逐帧推理；迭代优化20步冻结语义不符",
    },
    "drsn_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(n_frames * n_filters * seg_len * depth + n_filters²)",
        "space_complexity": "O(n_filters² * depth)，全连接+软阈值",
        "estimated_time_per_10k": "2~6秒（CPU，depth=6）",
        "cpu_usage": "极高",
        "memory_usage": "中~高",
        "bottlenecks": "每个残差块含两个FC层做阈值预测；软阈值化操作",
    },
    "tcn_lstm_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(n_frames * (seg_len + lstm_hidden²))",
        "space_complexity": "O(lstm_hidden² + kernel*dilation)",
        "estimated_time_per_10k": "2~8秒（CPU，LSTM双向）",
        "cpu_usage": "极高",
        "memory_usage": "高",
        "bottlenecks": "LSTM双向计算O(hidden²)；TCN膨胀卷积慢",
    },
    "transformer_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(n_frames * seg_len²)，自注意力O(L²)",
        "space_complexity": "O(seg_len² + d_model² * n_layers)",
        "estimated_time_per_10k": "3~10秒（CPU，自注意力是关键瓶颈）",
        "cpu_usage": "极高",
        "memory_usage": "高",
        "bottlenecks": "自注意力O(L²)时间+空间复杂度；多头注意力拼接开销",
    },
    # ==================== 旋转机械专用 ====================
    "med_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(iter * (N*L + L³))，L=64",
        "space_complexity": "O(N*L)，卷积矩阵",
        "estimated_time_per_10k": "0.5~2秒",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "卷积矩阵构建及求逆；迭代优化30次",
    },
    "mckd_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(iter * (N*L + L³))，比MED多相关峭度",
        "space_complexity": "O(N*L + N)",
        "estimated_time_per_10k": "0.5~3秒",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "相关峭度计算含多次移位和累加；自动周期检测增耗时",
    },
    "momed_adenoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N*L + L³)，无迭代",
        "space_complexity": "O(N*L)",
        "estimated_time_per_10k": "0.1~0.3秒",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "矩阵求逆O(L³)；能量缩放步骤",
    },
    "spectral_kurtosis_filter": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(n_levels * (N log N))，多层STFT",
        "space_complexity": "O(N + n_levels * n_freq_bins)",
        "estimated_time_per_10k": "0.3~1.5秒（n_levels=6）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "多层STFT计算（变窗长）；谱峭度统计计算",
    },
    "envelope_demodulation": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N log N)，Hilbert变换+滤波",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.02秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "无显著瓶颈",
    },
    "cyclostationary_analysis": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(n_freq * (N log N + n_time log n_time))",
        "space_complexity": "O(n_freq * n_time)，STFT谱",
        "estimated_time_per_10k": "0.5~3秒",
        "cpu_usage": "高",
        "memory_usage": "中~高",
        "bottlenecks": "逐频点循环谱计算；多个带通滤波重构",
    },
    # ==================== 语音/声学降噪 ====================
    "mmse_stsa_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(n_frames * n_freq)，每帧逐频点计算增益",
        "space_complexity": "O(n_freq * n_frames)，STFT谱",
        "estimated_time_per_10k": "0.3~1秒（512点FFT）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "逐帧逐频点的增益计算含贝塞尔函数近似",
    },
    "log_mmse_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(n_frames * n_freq)，每帧含指数积分",
        "space_complexity": "O(n_freq * n_frames)",
        "estimated_time_per_10k": "0.3~1秒（512点FFT）",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "指数积分(E1)逐频点计算；更多数学函数调用",
    },
    "subspace_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(L²*K)，L=N/3",
        "space_complexity": "O(L*K + L²)",
        "estimated_time_per_10k": "0.05~0.3秒",
        "cpu_usage": "中",
        "memory_usage": "中",
        "bottlenecks": "协方差矩阵特征值分解O(L³)；Hankel矩阵构建",
    },
    "noise_gate_denoise": {
        "complexity": AlgorithmComplexity.LOW,
        "resource_level": ResourceConsumption.VERY_LOW,
        "time_complexity": "O(N) + O(n_frames * attack/release)",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "< 0.01秒",
        "cpu_usage": "极低",
        "memory_usage": "极低",
        "bottlenecks": "无显著瓶颈",
    },
    # ==================== 其他前沿方法 ====================
    "compressed_sensing_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N²*sparsity)，DCT矩阵O(N²)+OMP迭代",
        "space_complexity": "O(N²)，DCT矩阵",
        "estimated_time_per_10k": "0.5~2秒",
        "cpu_usage": "高",
        "memory_usage": "高",
        "bottlenecks": "N×N变换矩阵构建及存储；OMP最小二乘",
    },
    "stochastic_resonance": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N)，RK4积分逐点计算",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "0.01~0.05秒",
        "cpu_usage": "低",
        "memory_usage": "低",
        "bottlenecks": "Python循环的RK4积分无向量化",
    },
    "morphological_filter": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.LOW,
        "time_complexity": "O(N*K)，K为结构元素大小",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "0.01~0.05秒",
        "cpu_usage": "低",
        "memory_usage": "低",
        "bottlenecks": "scipy grey_* 函数较慢时回退到Python循环实现",
    },
    "total_variation_denoise": {
        "complexity": AlgorithmComplexity.MEDIUM,
        "resource_level": ResourceConsumption.MEDIUM,
        "time_complexity": "O(N*iter)，iter=100",
        "space_complexity": "O(N)",
        "estimated_time_per_10k": "0.1~0.3秒",
        "cpu_usage": "中",
        "memory_usage": "低",
        "bottlenecks": "100次梯度下降迭代；差分次梯度计算",
    },
    "non_local_means_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(N * search_radius)，最坏O(N²)",
        "space_complexity": "O(N * patch_size)",
        "estimated_time_per_10k": "2~10秒（search_radius=15）",
        "cpu_usage": "极高",
        "memory_usage": "中",
        "bottlenecks": "逐点的搜索窗口内patch匹配；信号>5000自动降采样",
    },
    "graph_signal_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.VERY_HIGH,
        "time_complexity": "O(N²)，kNN图构建O(N*K)但距离计算O(N²)",
        "space_complexity": "O(N²)，邻接矩阵",
        "estimated_time_per_10k": "0.5~3秒（N≤2000时）",
        "cpu_usage": "极高",
        "memory_usage": "极高（N²矩阵）",
        "bottlenecks": "距离矩阵计算O(N²)内存和时间双重瓶颈；线性方程组求解",
    },
    "fractal_denoise": {
        "complexity": AlgorithmComplexity.HIGH,
        "resource_level": ResourceConsumption.HIGH,
        "time_complexity": "O(N log N) + O(N_seg * log N_seg)，DWT+R/S分析",
        "space_complexity": "O(N) + O(seg_sizes * n_segs)",
        "estimated_time_per_10k": "0.3~1秒",
        "cpu_usage": "高",
        "memory_usage": "中",
        "bottlenecks": "多次R/S分析分段计算赫斯特指数；pywt不可用时FFT回退",
    },
}


def get_complexity_analysis(algorithm_id: str) -> Optional[ComplexityAnalysis]:
    """获取指定算法的详细复杂度分析。

    Args:
        algorithm_id: 算法ID（蛇形命名）

    Returns:
        ComplexityAnalysis 对象，如果找不到则返回 None
    """
    if algorithm_id not in _ALGORITHM_COMPLEXITY:
        return None

    info = _ALGORITHM_COMPLEXITY[algorithm_id]
    return ComplexityAnalysis(
        algorithm_id=algorithm_id,
        algorithm_name="",
        category=AlgorithmCategory.CLASSICAL_FILTER,  # 占位，由调用者填充
        complexity=info["complexity"],
        resource_level=info["resource_level"],
        time_complexity=info["time_complexity"],
        space_complexity=info["space_complexity"],
        estimated_time_per_10k=info["estimated_time_per_10k"],
        cpu_usage=info["cpu_usage"],
        memory_usage=info["memory_usage"],
        optimization_tips=_get_tips_for_algo(algorithm_id, info),
        bottlenecks=info["bottlenecks"],
    )


def get_optimization_tips(category_value: str) -> str:
    """获取某类算法的通用优化建议。

    Args:
        category_value: 算法类别的中文名（如 "经典滤波方法"）

    Returns:
        优化建议文本
    """
    return _CATEGORY_TIPS.get(category_value, "暂无针对该类别的一般性优化建议。")


def _get_tips_for_algo(algorithm_id: str, info: Dict) -> str:
    """生成针对特定算法的优化建议。"""
    tips: list[str] = []

    complexity = info["complexity"]
    resource_level = info["resource_level"]

    if complexity == AlgorithmComplexity.HIGH:
        tips.append("⚠️ 该算法计算复杂度高，建议在长信号上分段处理。")

    if resource_level in (ResourceConsumption.HIGH, ResourceConsumption.VERY_HIGH):
        tips.append("⚠️ 资源消耗大，请确保系统有足够内存并避免同时运行多个此算法。")

    if "s" in info.get("estimated_time_per_10k", ""):
        tips.append("⏱️ 预计每10000点信号耗时 " + info["estimated_time_per_10k"])

    tips.append("💡 查看类别通用优化建议以获得更多调优指导。")

    # 各算法特有建议
    algo_specific_tips = {
        # EMD族
        "eemd_denoise": "• 将 n_ensemble 从默认50降低到10~20，大幅减少耗时\n• 降低 noise_std 可减少迭代次数",
        "ceemd_denoise": "• 将 n_ensemble 从默认50降低到10~20\n• 信号过长时建议先降采样",
        "ceemdan_denoise": "• 将 max_iter 从1000降低到200~500\n• 推荐先试用普通EMD评估效果",
        "vmd_denoise": "• 增大 tol 从1e-6到1e-4可减少迭代\n• 减小 K 值(从5减到3)\n• alpha 从2000提高到5000可加速收敛",
        # 稀疏类
        "matching_pursuit_denoise": "• 减小 n_atoms(128→64)和 sparsity(10→5)\n• 先对信号降采样到10000点以下",
        "ksvd_denoise": "• 减小 n_iter(10→3~5)和 dict_size(128→64)\n• atom_length 不要超过信号长度的1/8\n• 信号>5000点自动变慢，建议分段",
        # SVD类
        "svd_denoise": "• 信号长度>15000时考虑分段SVD\n• 使用 scipy.linalg.svd 替代 np.linalg.svd",
        "hankel_svd": "• 减小 window_length 到 N/3 或更小",
        # 深度学习
        "dae_denoise": "• 减小 segment_length(256→128)\n• 增大 overlap(0.5→0.75)提高质量但速度不变\n• 无PyTorch时自动回退频域滤波",
        "cda_denoise": "• 减小 n_filters(16→8)和 segment_length(512→256)",
        "dncnn_denoise": "• 减小 n_layers(10→6)大幅降低计算量\n• 减小 n_filters(32→16)",
        "unet_denoise": "• 减小 n_channels(16→8)和 depth(4→3)\n• 减小 segment_length(1024→512)",
        "drsn_denoise": "• 减小 depth(6→4)和 n_filters(16→8)",
        "tcn_lstm_denoise": "• 减小 n_lstm_units(32→16)\n• 使用单向LSTM(当前已双向)",
        "transformer_denoise": "• 减小 d_model(32→16)和 segment_length(256→128)\n• 减小 n_heads(4→2)",
        # BSS类
        "sobi_denoise": "• 减小 n_lags(100→20~30)\n• 限制 n_components 到10以下",
        # 旋转机械
        "med_denoise": "• 减小 filter_order(64→32)和 max_iter(30→15)\n• termination_tol 从1e-3提高到5e-3",
        "mckd_denoise": "• 同MED优化建议\n• 确保 period 参数正确以避免无效迭代",
        "spectral_kurtosis_filter": "• 减小 n_levels(6→4)减少STFT计算层数",
        "cyclostationary_analysis": "• 减小 n_cyclic_freqs(10→5)\n• 增大STFT窗长减少时间帧数",
        # 声学
        "mmse_stsa_denoise": "• 减小 n_fft(512→256)减少频点数\n• 增大 hop(约n_fft/2)减少帧数",
        "log_mmse_denoise": "• 同MMSE-STSA优化建议",
        # 前沿方法
        "non_local_means_denoise": "• 减小 search_radius(15→8~10)和 patch_size(7→3~5)\n• 增大滤波强度 h(0.1→0.2)减少搜索必要\n• 信号>5000点自动降采样",
        "graph_signal_denoise": "• 减小 n_neighbors(10→5)和 alpha(0.5→0.3)\n• 信号>5000点自动降采样到2000点",
        "compressed_sensing_denoise": "• 减小 sparsity(20→10~15)\n• 使用DCT变换(默认)比FFT更高效",
        "total_variation_denoise": "• 减小 n_iter(100→30~50)可大幅加速\n• 增大 lambda_(0.1→0.3)更快收敛",
        "fractal_denoise": "• 减小 max_scale(10→6)减少小波分解层数\n• 提高 threshold(0.1→0.15)减少保留成分",
    }

    if algorithm_id in algo_specific_tips:
        tips.append(algo_specific_tips[algorithm_id])

    return "\n".join(tips)


def get_category_name(category: AlgorithmCategory) -> str:
    """获取类别中文名称。"""
    mapping = {
        AlgorithmCategory.CLASSICAL_FILTER: "经典滤波方法",
        AlgorithmCategory.TIME_DOMAIN: "时域降噪方法",
        AlgorithmCategory.FREQ_DOMAIN: "频域降噪方法",
        AlgorithmCategory.WAVELET: "小波变换类",
        AlgorithmCategory.EMD: "经验模态分解类",
        AlgorithmCategory.ADAPTIVE: "自适应滤波",
        AlgorithmCategory.SVD: "奇异值分解类",
        AlgorithmCategory.SPARSE: "稀疏表示类",
        AlgorithmCategory.BSS: "盲源分离类",
        AlgorithmCategory.DEEP_LEARNING: "深度学习类",
        AlgorithmCategory.ROTATING_SPECIFIC: "旋转机械专用",
        AlgorithmCategory.ACOUSTIC: "语音/声学降噪",
        AlgorithmCategory.ADVANCED: "其他前沿方法",
    }
    return mapping.get(category, category.value if hasattr(category, "value") else str(category))


def get_high_complexity_ids() -> set:
    """获取所有高复杂度算法的ID集合。"""
    return {
        aid for aid, info in _ALGORITHM_COMPLEXITY.items()
        if info["complexity"] == AlgorithmComplexity.HIGH
    }


def get_very_high_resource_ids() -> set:
    """获取所有极高资源消耗算法的ID集合。"""
    return {
        aid for aid, info in _ALGORITHM_COMPLEXITY.items()
        if info["resource_level"] in (ResourceConsumption.VERY_HIGH, ResourceConsumption.HIGH)
    }
