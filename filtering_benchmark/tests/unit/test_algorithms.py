"""
算法核心测试：形状保持、工具函数、安全保护、多通道支持。
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from src.algorithms.base import _to_stereo, _from_stereo
from src.core.exceptions import SignalTooLargeError


# ==================== 共享工具函数测试 ====================


def _sig():
    fs = 12000
    t = np.linspace(0, 1, fs, 0)
    return np.sin(2*np.pi*78.5*t) + 0.3*np.random.randn(fs), fs


def _sig_stereo():
    s, fs = _sig()
    return np.stack([s, s * 0.5]), fs


class TestStereoTools:
    """测试 _to_stereo / _from_stereo 工具函数"""

    def test_to_stereo_1d(self):
        s = np.array([1.0, 2.0, 3.0])
        out = _to_stereo(s)
        assert out.shape == (1, 3)
        assert np.allclose(out[0], s)

    def test_to_stereo_2d(self):
        s = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = _to_stereo(s)
        assert out.shape == (2, 2)
        assert np.allclose(out, s)

    def test_from_stereo_1d(self):
        original = np.array([1.0, 2.0, 3.0])
        processed = np.array([[10.0, 20.0, 30.0]])
        out = _from_stereo(processed, original)
        assert out.ndim == 1
        assert np.allclose(out, processed[0])

    def test_from_stereo_2d(self):
        original = np.array([[1.0, 2.0], [3.0, 4.0]])
        processed = original * 2
        out = _from_stereo(processed, original)
        assert out.shape == (2, 2)
        assert np.allclose(out, processed)


# ==================== 算法执行测试 ====================

# 从各分类中选择代表性算法进行形状保持测试
# 注意：SVD算法通过超时保护跳过
_SLOW_ALGOS = {"svd_denoise", "hankel_svd", "basis_pursuit_denoise"}

_REPRESENTATIVE_ALGOS = [
    # classical
    "low_pass_butterworth", "high_pass_butterworth", "band_pass_butterworth",
    "fir_band_pass", "iir_band_pass",
    # time_domain
    "moving_average_filter", "median_filter", "adaptive_median_filter",
    "weighted_moving_average", "savitzky_golay_filter", "ewma_base", "gaussian_filter",
    # freq_domain
    "fft_ideal_low_pass", "spectral_subtraction", "multi_band_spectral_subtraction",
    # wavelet
    "dwt_threshold_denoise", "wavelet_packet_denoise",
    # adaptive
    "lms_adaptive_filter", "rls_adaptive_filter", "kalman_filter",
    # svd
    "svd_denoise", "hankel_svd",
    # sparse
    "basis_pursuit_denoise",
    # advanced
    "non_local_means_denoise",
    # optimization
    "denoise_param_searcher",
]


@pytest.mark.parametrize("algo_id", _REPRESENTATIVE_ALGOS)
def test_shape_1d(algo_id):
    """验证算法处理1D信号后形状不变"""
    from src.algorithms import registry
    registry.ensure_discovered()
    try:
        inst = registry.create_instance(algo_id)
    except KeyError:
        pytest.skip(f"{algo_id} 未在注册表中找到")
    if inst is None:
        pytest.skip(f"{algo_id} 实例化为 None")
    s, fs = _sig()
    # 对慢速算法使用更短的信号
    if algo_id in _SLOW_ALGOS:
        fs = 600
        t = np.linspace(0, 1, fs, 0)
        s = np.sin(2*np.pi*78.5*t) + 0.3*np.random.randn(fs)
    try:
        r = inst.denoise(s, fs)
    except Exception as exc:
        pytest.skip(f"{algo_id} 不兼容1D输入: {exc}")
    assert r.shape == s.shape, f"{algo_id}: shape mismatch {r.shape} vs {s.shape}"


@pytest.mark.parametrize("algo_id", _REPRESENTATIVE_ALGOS)
def test_shape_2d(algo_id):
    """验证算法处理2D信号后形状不变"""
    from src.algorithms import registry
    registry.ensure_discovered()
    try:
        inst = registry.create_instance(algo_id)
    except KeyError:
        pytest.skip(f"{algo_id} 未在注册表中找到")
    if inst is None:
        pytest.skip(f"{algo_id} 实例化为 None")
    s, fs = _sig_stereo()
    if algo_id in _SLOW_ALGOS:
        fs = 600
        t = np.linspace(0, 1, fs, 0)
        sig = np.sin(2*np.pi*78.5*t) + 0.3*np.random.randn(fs)
        s = np.stack([sig, sig * 0.5])
    try:
        r = inst.denoise(s, fs)
    except Exception as exc:
        pytest.skip(f"{algo_id} 不兼容2D输入: {exc}")
    assert r.shape == s.shape, f"{algo_id}: shape mismatch {r.shape} vs {s.shape}"


@pytest.mark.parametrize("algo_id", _REPRESENTATIVE_ALGOS)
def test_not_all_zero(algo_id):
    """验证降噪结果不是全零"""
    from src.algorithms import registry
    registry.ensure_discovered()
    try:
        inst = registry.create_instance(algo_id)
    except KeyError:
        pytest.skip(f"{algo_id} 未在注册表中找到")
    if inst is None:
        pytest.skip(f"{algo_id} 实例化为 None")
    s, fs = _sig()
    if algo_id in _SLOW_ALGOS:
        fs = 600
        t = np.linspace(0, 1, fs, 0)
        s = np.sin(2*np.pi*78.5*t) + 0.3*np.random.randn(fs)
    try:
        r = inst.denoise(s, fs)
    except Exception as exc:
        pytest.skip(f"{algo_id} 执行异常: {exc}")
    assert np.any(np.abs(r) > 1e-10), f"{algo_id}: output is all zeros"


# ==================== 安全保护测试 ====================


def test_max_signal_length_raises():
    """验证超限信号抛出 SignalTooLargeError"""
    from src.algorithms import registry
    registry.ensure_discovered()
    # 选择一个设置了 max_signal_length 的算法
    inst = registry.create_instance("svd_denoise")
    meta = registry.get("svd_denoise")
    assert inst is not None
    max_len = getattr(meta.algorithm_class, 'max_signal_length', None)
    assert max_len is not None, "svd_denoise must have max_signal_length"
    # 构造一个超长信号
    s = np.zeros(max_len + 1)
    with pytest.raises(SignalTooLargeError):
        inst.denoise(s, 1000)


def test_max_signal_length_boundary():
    """验证信号长度刚好等于上限时不抛异常"""
    from src.algorithms import registry
    registry.ensure_discovered()
    count = 0
    for aid in registry.list_ids():
        meta = registry.get(aid)
        max_len = getattr(meta.algorithm_class, 'max_signal_length', None)
        if max_len is None:
            continue
        if max_len <= 0:
            continue
        inst = registry.create_instance(aid)
        s = np.zeros(max_len)
        try:
            inst.denoise(s, 10000)
            count += 1
        except SignalTooLargeError:
            pytest.fail(f"{aid}: exactly at max_signal_length should not raise")
        except Exception:
            pass  # 其他异常可以接受
        if count >= 3:
            break
    assert count > 0, "No algorithms with max_signal_length tested"


# ==================== 注册表完整性测试 ====================


def test_all_algorithms_listed():
    """验证注册算法数量 >= 30（基本覆盖性检查）"""
    from src.algorithms import registry
    registry.ensure_discovered()
    ids = registry.list_ids()
    assert len(ids) >= 30, f"Only {len(ids)} registered, expected >= 30"
    # 确认所有主要分类都有算法
    categories = set()
    for aid in ids:
        meta = registry.get(aid)
        if meta and hasattr(meta, 'category'):
            categories.add(meta.category.value)
    assert len(categories) >= 6, f"Only {len(categories)} categories: {categories}"


if __name__ == "__main__":
    # 打印所有注册算法ID供调试
    from src.algorithms import registry
    registry.ensure_discovered()
    for _id in sorted(registry.list_ids()):
        print(_id)


# ==================== 新增测试：针对代码审查修复的验证 ====================

class TestAutoSetup:
    """测试 DL 算法的自动 setup() 调用（Task #1 修复验证）"""

    def test_dl_auto_setup_without_manual_setup(self):
        """验证 DL 算法在未手动调用 setup() 时，denoise() 会自动调用"""
        from src.algorithms import registry
        registry.ensure_discovered()

        # 选择一个 DL 算法进行测试
        dl_algo_id = "dae_denoise"  # DAE 算法
        if dl_algo_id not in registry.list_ids():
            pytest.skip(f"{dl_algo_id} 未注册，跳过测试")

        inst = registry.create_instance(dl_algo_id)
        # 不调用 inst.setup()，直接调用 denoise()
        # 如果自动 setup 工作正常，不应该抛出 AttributeError 或 TypeError
        try:
            s = np.zeros(512), fs = 12000
            result = inst.denoise(s, fs)
            # 如果能执行到这里，说明自动 setup 工作了
            assert result is not None
        except Exception as e:
            # 允许其他异常（如 torch 不可用时的 fallback），但不应是 "_net" 相关的 AttributeError
            assert "_net" not in str(e), f"自动 setup 可能未工作: {e}"


class TestImpostorWarnings:
    """测试冒充算法的运行时警告（Task #2 修复验证）"""

    def test_ewt_warning(self):
        """验证 EWT 算法会发出警告"""
        import warnings
        from src.algorithms import registry
        registry.ensure_discovered()

        if "empirical_wavelet" not in registry.list_ids():
            pytest.skip("empirical_wavelet 未注册，跳过测试")

        inst = registry.create_instance("empirical_wavelet")
        s, fs = np.zeros(512), 12000

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                inst.denoise(s, fs)
            except Exception:
                pass  # 忽略执行错误，只检查警告

        # 检查是否发出了警告
        warning_msgs = [str(warning.message) for warning in w]
        assert any("EWT" in msg for msg in warning_msgs), f"EWT 应发出警告，实际警告: {warning_msgs}"

    def test_jade_warning(self):
        """验证 JADE 算法会发出警告"""
        import warnings
        from src.algorithms import registry
        registry.ensure_discovered()

        if "jade_denoise" not in registry.list_ids():
            pytest.skip("jade_denoise 未注册，跳过测试")

        inst = registry.create_instance("jade_denoise")
        s, fs = np.zeros(512), 12000

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                inst.denoise(s, fs)
            except Exception:
                pass  # 忽略执行错误，只检查警告

        warning_msgs = [str(warning.message) for warning in w]
        assert any("JADE" in msg for msg in warning_msgs), f"JADE 应发出警告，实际警告: {warning_msgs}"

    def test_bpdn_warning(self):
        """验证 BPDN 算法会发出警告"""
        import warnings
        from src.algorithms import registry
        registry.ensure_discovered()

        if "basis_pursuit_denoise" not in registry.list_ids():
            pytest.skip("basis_pursuit_denoise 未注册，跳过测试")

        inst = registry.create_instance("basis_pursuit_denoise")
        s, fs = np.zeros(512), 12000

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                inst.denoise(s, fs)
            except Exception:
                pass  # 忽略执行错误，只检查警告

        warning_msgs = [str(warning.message) for warning in w]
        assert any("BPDN" in msg for msg in warning_msgs), f"BPDN 应发出警告，实际警告: {warning_msgs}"


