"""
进化算法与群体智能优化算法。

实现：
1. 遗传算法 (GA)             - 15.5
2. 粒子群优化 (PSO)          - 15.6
3. 差分进化 (DE)             - 15.7
4. 模拟退火 (SA)             - 15.8
"""

import random
from typing import Callable, List, Optional, Tuple

import numpy as np


# ======================== 遗传算法 (GA) ========================


def genetic_algorithm(
    fitness_func: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    pop_size: int = 50,
    max_generations: int = 100,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.1,
    elite_ratio: float = 0.1,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """实数编码遗传算法 (Genetic Algorithm)。

    模拟自然选择和遗传学机制的无梯度全局优化方法。
    通过选择、交叉、变异操作逐代进化种群。

    Args:
        fitness_func: 适应度函数 f(x) -> float (最大化)
        dim: 搜索空间维度
        bounds: 每维的 (下界, 上界) 列表
        pop_size: 种群大小
        max_generations: 最大进化代数
        crossover_prob: 交叉概率
        mutation_prob: 变异概率
        elite_ratio: 精英保留比例
        random_state: 随机种子

    Returns:
        最优解向量
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # 初始化种群
    pop = np.array([
        [random.uniform(b[0], b[1]) for b in bounds]
        for _ in range(pop_size)
    ])
    n_elite = max(1, int(elite_ratio * pop_size))

    for gen in range(max_generations):
        # 适应度评估
        fitness = np.array([fitness_func(ind) for ind in pop])

        # 精英保留
        elite_idx = np.argsort(fitness)[-n_elite:]
        new_pop = [pop[i].copy() for i in elite_idx]

        # 生成后代
        while len(new_pop) < pop_size:
            # 锦标赛选择
            tournament = random.sample(range(pop_size), 3)
            winner = max(tournament, key=lambda i: fitness[i])
            parent = pop[winner].copy()

            if random.random() < crossover_prob:
                # 模拟二进制交叉 (SBX)
                mate_idx = random.choice(tournament)
                mate = pop[mate_idx]
                child = np.where(
                    np.random.random(dim) < 0.5,
                    (parent + mate) / 2.0,
                    parent.copy(),
                )
            else:
                child = parent.copy()

            if random.random() < mutation_prob:
                # 高斯变异
                sigma = np.array([(b[1] - b[0]) * 0.1 for b in bounds])
                child += np.random.randn(dim) * sigma

            # 边界裁剪
            for d in range(dim):
                child[d] = np.clip(child[d], bounds[d][0], bounds[d][1])

            new_pop.append(child)

        pop = np.array(new_pop)

    # 返回最优解
    fitness = np.array([fitness_func(ind) for ind in pop])
    return pop[np.argmax(fitness)]


# ======================== 粒子群优化 (PSO) ========================


def particle_swarm_optimization(
    fitness_func: Callable[[np.ndarray], float],
    dim: int,
    bounds: List[Tuple[float, float]],
    n_particles: int = 30,
    max_iter: int = 200,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """标准粒子群优化 (Particle Swarm Optimization)。

    模拟鸟群觅食行为，通过个体最优和群体最优引导搜索。

    Args:
        fitness_func: 适应度函数 f(x) -> float (最小化)
        dim: 搜索空间维度
        bounds: 每维的 (下界, 上界) 列表
        n_particles: 粒子数量
        max_iter: 最大迭代次数
        w: 惯性权重
        c1: 个体学习因子
        c2: 社会学习因子
        random_state: 随机种子

    Returns:
        (最优解, 最优值)
    """
    if random_state is not None:
        np.random.seed(random_state)

    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)
    scale = ub - lb

    # 初始化位置和速度
    x = np.random.uniform(lb, ub, (n_particles, dim))
    v = np.random.uniform(-scale * 0.1, scale * 0.1, (n_particles, dim))

    p_best = x.copy()
    p_best_val = np.array([fitness_func(x[i]) for i in range(n_particles)])
    g_best_idx = int(np.argmin(p_best_val))
    g_best = p_best[g_best_idx].copy()
    g_best_val = p_best_val[g_best_idx]

    for _ in range(max_iter):
        r1, r2 = np.random.rand(2)
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        x = np.clip(x + v, lb, ub)

        for i in range(n_particles):
            val = fitness_func(x[i])
            if val < p_best_val[i]:
                p_best[i] = x[i].copy()
                p_best_val[i] = val
                if val < g_best_val:
                    g_best = x[i].copy()
                    g_best_val = val

    return g_best, g_best_val


# ======================== 差分进化 (DE) ========================


def differential_evolution(
    objective_func: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    pop_size: int = 50,
    F: float = 0.8,
    CR: float = 0.7,
    max_iter: int = 200,
    tol: float = 1e-7,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """差分进化 (Differential Evolution)。

    使用 DE/rand/1/bin 策略进行连续空间全局优化。
    通过向量差分实现变异，二项式交叉生成试验向量。

    Args:
        objective_func: 目标函数 f(x) -> float (最小化)
        bounds: 每维的 (下界, 上界) 列表
        pop_size: 种群大小
        F: 缩放因子 [0, 2]
        CR: 交叉概率 [0, 1]
        max_iter: 最大迭代次数
        tol: 最优值变化收敛阈值
        random_state: 随机种子

    Returns:
        (最优解, 最优值)
    """
    if random_state is not None:
        np.random.seed(random_state)

    dim = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)

    # 初始化种群
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([objective_func(pop[i]) for i in range(pop_size)])
    best_idx = int(np.argmin(fitness))
    best_val = fitness[best_idx]
    stagnant = 0

    for gen in range(max_iter):
        for i in range(pop_size):
            # 随机选择三个不同的个体
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)

            # 变异: v = x_r1 + F * (x_r2 - x_r3)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lb, ub)

            # 二项式交叉
            j_rand = np.random.randint(dim)
            trial = np.where(
                np.random.random(dim) < CR,
                mutant,
                pop[i],
            )
            trial[j_rand] = mutant[j_rand]

            # 贪婪选择
            f_trial = objective_func(trial)
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < best_val:
                    best_val = f_trial
                    stagnant = 0
                else:
                    stagnant += 1
            else:
                stagnant += 1

        if stagnant > pop_size * 3 and gen > 50:
            break
        if best_val < tol:
            break

    best_idx = int(np.argmin(fitness))
    return pop[best_idx], fitness[best_idx]


# ======================== 模拟退火 (SA) ========================


def simulated_annealing(
    objective_func: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
    min_temp: float = 1e-6,
    max_iter_per_temp: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """模拟退火 (Simulated Annealing)。

    模拟物理退火过程，通过概率性接受劣解来逃离局部最优。
    理论上以概率 1 收敛到全局最优（无限时间）。

    Args:
        objective_func: 目标函数 f(x) -> float (最小化)
        x0: 初始解
        bounds: 每维的 (下界, 上界) 列表
        initial_temp: 初始温度
        cooling_rate: 降温速率 (0, 1)
        min_temp: 最低温度
        max_iter_per_temp: 每温度迭代次数
        random_state: 随机种子

    Returns:
        (最优解, 最优值)
    """
    if random_state is not None:
        np.random.seed(random_state)

    dim = len(x0)
    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)
    scale = ub - lb

    x_current = x0.copy().astype(np.float64)
    f_current = objective_func(x_current)
    x_best = x_current.copy()
    f_best = f_current
    temp = initial_temp

    while temp > min_temp:
        for _ in range(max_iter_per_temp):
            # 在当前解附近生成新解
            step = scale * 0.1 * np.random.randn(dim)
            x_new = np.clip(x_current + step, lb, ub)
            f_new = objective_func(x_new)

            delta = f_new - f_current
            if delta < 0 or random.random() < np.exp(-delta / temp):
                x_current = x_new
                f_current = f_new
                if f_new < f_best:
                    x_best = x_new.copy()
                    f_best = f_new

        temp *= cooling_rate

    return x_best, f_best
