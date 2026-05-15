
import sys, numpy as np
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.join('e:/Qoder项目/滤波基石系统设计/filtering_benchmark', "src"))

def _sig():
    fs = 12000
    t = np.linspace(0, 1, fs, 0)
    c = np.sin(2*np.pi*78.5*t)
    n = c + 0.3*np.random.randn(fs)
    return c, n, fs

def test_all():
    from src.metrics import metric_registry
    c, n, fs = _sig()
    for name in metric_registry.list_names():
        m = metric_registry.get(name)
        r = m.compute(denoised=n, original=c, sample_rate=fs)
        assert r is not None

def test_compute_all():
    from src.metrics import metric_registry
    c, n, fs = _sig()
    names = metric_registry.list_names()
    r = metric_registry.compute_all(denoised=n, reference=c, sample_rate=fs, enabled=names)
    assert len(r) == len(names)

if __name__ == "__main__":
    test_all()
    test_compute_all()
    print("OK")
