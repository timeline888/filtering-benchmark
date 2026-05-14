"""
梯度下降类优化算法。

实现：
1. 批量梯度下降 (BGD)         - 15.2.1
2. 随机梯度下降 (SGD)         - 15.2.2
3. Adam                      - 15.2.3
4. 牛顿法                    - 15.3.1
5. BFGS                      - 15.3.2
6. 共轭梯度法 (CG)           - 15.4
"""

import numpy as np
from typing import Callable, Optional, Tuple


def batch_gradient_descent(
    grad_func: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """批量梯度下降 (Batch Gradient Descent)。

    沿目标函数梯度的负方向迭代更新参数：
        x_{k+1} = x_k - lr * grad(x_k)

    Args:
        grad_func: 梯度函数 grad(x) -> ndarray
        x0: 初始参数向量
        learning_rate: 学习率（步长）
        max_iter: 最大迭代次数
        tolerance: 梯度模长收敛阈值

    Returns:
        (最优参数, 迭代次数)
    """
    x = x0.copy().astype(np.float64)
    for i in range(max_iter):
        g = grad_func(x)
        if np.linalg.norm(g) < tolerance:
            return x, i
        x -= learning_rate * g
    return x, max_iter


def stochastic_gradient_descent(
    grad_func: Callable[[np.ndarray, int], np.ndarray],
    x0: np.ndarray,
    n_samples: int,
    learning_rate0: float = 0.1,
    decay: float = 0.01,
    epochs: int = 10,
) -> np.ndarray:
    """随机梯度下降 (SGD)。

    每次仅使用一个样本来计算梯度，学习率随迭代衰减：
        lr_k = lr0 / (1 + decay * k)

    Args:
        grad_func: 梯度函数 grad(x, sample_idx) -> ndarray
        x0: 初始参数向量
        n_samples: 总样本数
        learning_rate0: 初始学习率
        decay: 学习率衰减系数
        epochs: 遍历全数据的轮数

    Returns:
        优化后的参数向量
    """
    x = x0.copy().astype(np.float64)
    step = 0
    for _ in range(epochs):
        indices = np.random.permutation(n_samples)
        for i in indices:
            lr = learning_rate0 / (1.0 + decay * step)
            g = grad_func(x, i)
            x -= lr * g
            step += 1
    return x


def adam_optimizer(
    grad_func: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
) -> Tuple[np.ndarray, int]:
    """Adam 优化器 (Adaptive Moment Estimation)。

    结合动量法 (Momentum) 和 RMSProp 的自适应学习率优化器。
        m_t = beta1 * m_{t-1} + (1-beta1) * g_t
        v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
        theta_{t+1} = theta_t - lr * m_hat / (sqrt(v_hat) + eps)

    Args:
        grad_func: 梯度函数 grad(x) -> ndarray
        x0: 初始参数向量
        learning_rate: 学习率
        beta1: 一阶矩衰减率
        beta2: 二阶矩衰减率
        epsilon: 数值稳定项
        max_iter: 最大迭代次数
        tolerance: 梯度模长收敛阈值

    Returns:
        (最优参数, 迭代次数)
    """
    x = x0.copy().astype(np.float64)
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, max_iter + 1):
        g = grad_func(x)
        if np.linalg.norm(g) < tolerance:
            return x, t

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x, max_iter


def newton_method(
    grad_func: Callable[[np.ndarray], np.ndarray],
    hess_func: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 50,
    tolerance: float = 1e-8,
) -> Tuple[np.ndarray, int]:
    """牛顿法。

    利用目标函数的二阶导数（Hessian 矩阵）实现二次收敛速度：
        x_{k+1} = x_k - H^{-1} * grad(x_k)

    适用于维度较低（n < 1000）且 Hessian 矩阵易于计算的场景。

    Args:
        grad_func: 梯度函数 grad(x) -> ndarray
        hess_func: Hessian 矩阵函数 hess(x) -> ndarray (n, n)
        x0: 初始参数向量
        max_iter: 最大迭代次数
        tolerance: 梯度模长收敛阈值

    Returns:
        (最优参数, 迭代次数)
    """
    x = x0.copy().astype(np.float64)
    for i in range(max_iter):
        g = grad_func(x)
        if np.linalg.norm(g) < tolerance:
            return x, i
        H = hess_func(x)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            # Hessian 奇异时回退到梯度下降
            delta = g
        x -= delta
    return x, max_iter


def bfgs_optimize(
    objective_func: Callable[[np.ndarray], float],
    grad_func: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 200,
    tolerance: float = 1e-8,
) -> Tuple[np.ndarray, int]:
    """BFGS 拟牛顿法。

    通过梯度差信息近似 Hessian 矩阵的逆，避免显式计算二阶导数。
    空间复杂度 O(n^2)，适合 n < 10000 的问题。

    Args:
        objective_func: 目标函数 f(x) -> float (用于线搜索)
        grad_func: 梯度函数 grad(x) -> ndarray
        x0: 初始参数向量
        max_iter: 最大迭代次数
        tolerance: 梯度模长收敛阈值

    Returns:
        (最优参数, 迭代次数)
    """
    x = x0.copy().astype(np.float64)
    n = len(x)
    # 初始化 Hessian 逆近似为单位矩阵
    H_inv = np.eye(n)
    g = grad_func(x)

    for i in range(max_iter):
        if np.linalg.norm(g) < tolerance:
            return x, i

        # 搜索方向
        d = -H_inv @ g

        # 简单线搜索（Armijo 准则简化版）
        alpha = 1.0
        f_x = objective_func(x)
        for _ in range(20):
            x_new = x + alpha * d
            if objective_func(x_new) <= f_x + 1e-4 * alpha * g @ d:
                break
            alpha *= 0.5

        s = alpha * d
        x_new = x + s
        g_new = grad_func(x_new)

        # BFGS 更新 Hessian 逆近似
        y = g_new - g
        sy = s @ y
        if sy > 1e-10:
            rho = 1.0 / sy
            I = np.eye(n)
            H_inv = (I - rho * np.outer(s, y)) @ H_inv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x, g = x_new, g_new

    return x, max_iter


def conjugate_gradient(
    A: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: Optional[int] = None,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """共轭梯度法 (Conjugate Gradient) 求解线性系统 Ax = b。

    构造一组关于 A 共轭的方向，最多 n 步内收敛到精确解。
    特别适合求解 Wiener-Hopf 方程 R w = p。

    Args:
        A: 对称正定矩阵 (n, n)
        b: 右侧向量 (n,)
        x0: 初始解（默认为零向量）
        max_iter: 最大迭代次数（默认为 n）
        tolerance: 残差阈值

    Returns:
        (解向量, 迭代次数)
    """
    n = A.shape[0]
    if max_iter is None:
        max_iter = n

    x = np.zeros(n) if x0 is None else x0.copy().astype(np.float64)
    r = b - A @ x
    d = r.copy()
    rr = r @ r

    for i in range(max_iter):
        Ad = A @ d
        alpha = rr / (d @ Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        rr_new = r @ r

        if np.sqrt(rr_new) < tolerance:
            return x, i + 1

        beta = rr_new / rr
        d = r + beta * d
        rr = rr_new

    return x, max_iter


def design_fir_cg(
    X: np.ndarray, d: np.ndarray, max_iter: Optional[int] = None, tol: float = 1e-6
) -> np.ndarray:
    """用共轭梯度法设计 FIR 滤波器（求解 Wiener-Hopf 方程）。

    构造自相关矩阵 R = X^T X 和互相关向量 p = X^T d，
    使用共轭梯度法求解 R w = p。

    Args:
        X: 输入信号矩阵 (n_samples, filter_order)
        d: 期望响应 (n_samples,)
        max_iter: 最大迭代次数
        tol: 残差收敛阈值

    Returns:
        最优 FIR 滤波器系数
    """
    R = X.T @ X
    p = X.T @ d
    w, _ = conjugate_gradient(R, p, max_iter=max_iter, tolerance=tol)
    return w
