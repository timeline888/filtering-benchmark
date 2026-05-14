"""
贝叶斯优化 (Bayesian Optimization) 实现。

基于高斯过程 (GP) 代理模型和 UCB 采集函数，
适用于目标函数评估代价高昂的黑箱优化问题（维度 < 20）。

参考：15.9 贝叶斯优化
"""

from typing import Callable, List, Optional, Tuple

import numpy as np


def gp_ucb_optimize(
    objective_func: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_init: int = 5,
    n_iter: int = 50,
    kappa: float = 2.576,
    random_state: Optional[int] = None,
) -> Tuple[List[float], float]:
    """基于高斯过程上置信界 (GP-UCB) 的贝叶斯优化。

    使用 sklearn 的 GaussianProcessRegressor 作为代理模型，
    UCB 采集函数平衡探索 (exploration) 和利用 (exploitation)。

    Args:
        objective_func: 目标函数 f(x) -> float (最大化)
        bounds: 每维的 (下界, 上界) 列表，长度 = 维度 D
        n_init: 初始随机采样点数
        n_iter: 贝叶斯优化迭代次数
        kappa: UCB 探索系数（越大越倾向探索）
        random_state: 随机种子

    Returns:
        (最优参数列表, 最优值)
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    if random_state is not None:
        np.random.seed(random_state)

    dim = len(bounds)

    # 初始随机采样
    X = np.random.uniform(0, 1, (n_init, dim))
    y = np.array([
        objective_func([b[0] + x * (b[1] - b[0]) for x, b in zip(X[i], bounds)])
        for i in range(n_init)
    ])

    # 高斯过程代理模型
    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state,
    )

    for _ in range(n_iter):
        gp.fit(X, y)

        # 候选点采样（使用大量随机候选）
        n_candidates = 5000
        X_cand = np.random.uniform(0, 1, (n_candidates, dim))
        mu, sigma = gp.predict(X_cand, return_std=True)

        # UCB 采集函数
        ucb = mu + kappa * sigma

        # 选择 UCB 最大的候选点
        x_next = X_cand[int(np.argmax(ucb))]
        y_next = objective_func([
            b[0] + x * (b[1] - b[0]) for x, b in zip(x_next, bounds)
        ])

        X = np.vstack([X, x_next.reshape(1, -1)])
        y = np.append(y, y_next)

    # 返回最优解
    best_idx = int(np.argmax(y))
    best_x = [
        bounds[d][0] + X[best_idx][d] * (bounds[d][1] - bounds[d][0])
        for d in range(dim)
    ]
    return best_x, y[best_idx]
