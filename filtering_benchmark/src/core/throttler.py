"""
计算资源控制工具（节流器）。

提供进程优先级调整、CPU 亲和性设置、并发度控制等能力，
用于在高复杂度算法执行时限制资源消耗，避免系统卡顿。

支持 Windows（ctypes 调用 Windows API）和 Linux。
"""

import os
import platform
import threading
from contextlib import contextmanager
from typing import Optional

from loguru import logger


def set_process_priority(level: str = "below_normal") -> bool:
    """设置当前进程的优先级。

    Args:
        level: 优先级等级
            - "idle": 空闲，仅在 CPU 空闲时运行（对后台任务最友好）
            - "below_normal": 低于正常（推荐，用户不会感到卡顿）
            - "normal": 正常（默认）

    Returns:
        是否设置成功
    """
    system = platform.system()

    if system == "Windows":
        return _set_priority_windows(level)
    elif system == "Linux":
        return _set_priority_linux(level)
    else:
        logger.warning(f"不支持的操作系统: {system}，无法设置进程优先级")
        return False


def _set_priority_windows(level: str) -> bool:
    """Windows 下通过 ctypes 调用 kernel32 设置进程优先级。"""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # 优先级类常量
        PRIORITY_MAP = {
            "idle": 0x00000040,           # IDLE_PRIORITY_CLASS
            "below_normal": 0x00004000,   # BELOW_NORMAL_PRIORITY_CLASS
            "normal": 0x00000020,          # NORMAL_PRIORITY_CLASS
        }

        priority_class = PRIORITY_MAP.get(level)
        if priority_class is None:
            logger.warning(f"未知的优先级等级: {level}")
            return False

        # GetCurrentProcess() returns -1 (pseudo-handle)
        handle = ctypes.c_void_p(-1)
        result = kernel32.SetPriorityClass(handle, priority_class)

        if result:
            logger.info(f"进程优先级已设为: {level}")
        else:
            err = ctypes.get_last_error()
            logger.warning(f"设置进程优先级失败，错误码: {err}")

        return bool(result)

    except Exception as e:
        logger.warning(f"设置进程优先级时出错: {e}")
        return False


def _set_priority_linux(level: str) -> bool:
    """Linux 下通过 os.nice 调整优先级。"""
    try:
        # nice 值范围: -20(最高) ~ 19(最低)
        NICE_MAP = {
            "idle": 19,
            "below_normal": 10,
            "normal": 0,
        }
        target_nice = NICE_MAP.get(level, 0)
        current_nice = os.nice(0)
        delta = target_nice - current_nice
        os.nice(delta)
        logger.info(f"进程 nice 值已设为: {target_nice} (level={level})")
        return True
    except Exception as e:
        logger.warning(f"设置进程优先级时出错: {e}")
        return False


def set_cpu_affinity(max_cores: int = 0) -> bool:
    """限制当前进程可用的 CPU 核心数。

    Args:
        max_cores: 最大 CPU 核心数。
            0 = 不限制（使用所有核心）
            >0 = 只使用前 N 个核心

    Returns:
        是否设置成功
    """
    if max_cores <= 0:
        return True  # 不限核心 = 无操作

    system = platform.system()
    total_cores = os.cpu_count() or 1
    actual_cores = min(max_cores, total_cores)

    if system == "Windows":
        return _set_affinity_windows(actual_cores)
    elif system == "Linux":
        return _set_affinity_linux(actual_cores)
    else:
        logger.warning(f"不支持的操作系统: {system}，无法设置 CPU 亲和性")
        return False


def _set_affinity_windows(num_cores: int) -> bool:
    """Windows 下通过 ctypes 限制 CPU 核心数。"""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        # 构建亲和性掩码：使用前 num_cores 个核心
        mask = (1 << num_cores) - 1

        # GetCurrentProcess
        handle = ctypes.c_void_p(-1)

        # SetProcessAffinityMask
        kernel32.SetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL

        result = kernel32.SetProcessAffinityMask(handle, mask)
        if result:
            logger.info(f"CPU 亲和性已限制为前 {num_cores} 个核心 (掩码: {mask:#x})")
        else:
            err = ctypes.get_last_error()
            logger.warning(f"设置 CPU 亲和性失败，错误码: {err}")

        return bool(result)

    except Exception as e:
        logger.warning(f"设置 CPU 亲和性时出错: {e}")
        return False


def _set_affinity_linux(num_cores: int) -> bool:
    """Linux 下通过 os.sched_setaffinity 限制 CPU 核心数。"""
    try:
        import os
        pid = os.getpid()
        cpu_list = list(range(num_cores))
        os.sched_setaffinity(pid, cpu_list)
        logger.info(f"CPU 亲和性已限制为核心 0~{num_cores - 1}")
        return True
    except Exception as e:
        logger.warning(f"设置 CPU 亲和性时出错: {e}")
        return False


def get_optimal_workers(max_workers: int = 0, max_cpu_cores: int = 0) -> int:
    """计算最佳并发工作线程数。

    Args:
        max_workers: 用户设置的最大工作线程数，0=自动
        max_cpu_cores: 用户设置的 CPU 核心限制，0=不限

    Returns:
        建议的工作线程数
    """
    total_cores = os.cpu_count() or 1

    # 如果限制了 CPU 核心数，以限制为准
    available_cores = total_cores
    if max_cpu_cores > 0:
        available_cores = min(max_cpu_cores, total_cores)

    # 自动决定：保留 1 个核心给系统
    auto_workers = max(1, available_cores - 1)

    if max_workers <= 0:
        return auto_workers

    return min(max_workers, available_cores)


def set_compute_thread_limit(max_threads: int) -> None:
    """限制 BLAS (OpenBLAS/MKL) 和 PyTorch 内部计算线程数。

    实际系统卡顿的根本原因之一：即使进程 affinity 限制了核心，
    numpy/scipy 的底层 BLAS 仍默认启用物理核数个线程，造成严重的
    上下文切换。此函数业外暴露以便在程序入口处调用。

    Args:
        max_threads: 最大线程数，<=0 表示不限制
    """
    if max_threads <= 0:
        return

    n = str(int(max_threads))
    # 环境变量需在 BLAS 初始化前设置才能完全生效，
    # 但在运行时设置对后续新建的 BLAS 线程池仍有效。
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", n)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", n)
    # 强制覆盖（对于运行时调用）
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n

    # PyTorch 运行时线程数
    try:
        import torch
        torch.set_num_threads(int(max_threads))
        try:
            torch.set_num_interop_threads(int(max_threads))
        except Exception:
            # set_num_interop_threads 只能在首次并行前调用，忽略异常
            pass
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"设置 PyTorch 线程数失败: {e}")

    # MKL 直接 API
    try:
        import mkl
        mkl.set_num_threads(int(max_threads))
    except ImportError:
        pass
    except Exception:
        pass

    # threadpoolctl（若安装则能更细粒度控制）
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(max_threads))
    except ImportError:
        pass
    except Exception:
        pass

    logger.info(f"计算线程数已限制为: {max_threads}")


class ComputationThrottle:
    """计算资源节流控制器。

    提供 contextmanager，在执行耗时计算时自动调整资源使用。

    用法:
        throttle = ComputationThrottle(
            process_priority="below_normal",
            max_workers=2
        )
        with throttle.apply():
            pipeline.run()
    """

    def __init__(
        self,
        process_priority: str = "normal",
        max_cpu_cores: int = 0,
        max_workers: int = 0,
    ):
        self.process_priority = process_priority
        self.max_cpu_cores = max_cpu_cores
        self.max_workers = max_workers

    def get_workers(self) -> int:
        """获取建议的工作线程数。"""
        return get_optimal_workers(self.max_workers, self.max_cpu_cores)

    @contextmanager
    def apply(self):
        """上下文管理器：进入时应用节流设置，退出时自动清理（如有需要）。"""
        has_set_priority = False
        has_set_affinity = False
        workers = self.get_workers()

        logger.info(f"应用资源节流: 优先级={self.process_priority}, "
                    f"CPU核心={self.max_cpu_cores or '不限'}, "
                    f"工作线程={workers}")

        try:
            # 设置进程优先级
            if self.process_priority != "normal":
                has_set_priority = set_process_priority(self.process_priority)

            # 限制 CPU 核心数
            if self.max_cpu_cores > 0:
                has_set_affinity = set_cpu_affinity(self.max_cpu_cores)

            # 同步限制 BLAS / PyTorch 内部线程数，避免线程穿透亲和性
            # 使用 workers 作为逻辑线程上限（包括并行算法数）
            effective_threads = self.max_cpu_cores if self.max_cpu_cores > 0 else 0
            if effective_threads > 0:
                # 每个 worker 共享剩余 BLAS 线程
                per_worker_threads = max(1, effective_threads // max(1, workers))
                set_compute_thread_limit(per_worker_threads)

            yield workers

        finally:
            # 当前架构下，进程优先级和亲和性在整个进程生命周期有效
            # 退出后无需恢复（如果真需要恢复，可以在这里实现）
            pass
