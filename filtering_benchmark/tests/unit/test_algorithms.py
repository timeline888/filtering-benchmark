
import sys, numpy as np
import os, sys; sys.path.insert(0, 'e:/Qoder项目/滤波基石系统设计/filtering_benchmark'); sys.path.insert(0, os.path.join('e:/Qoder项目/滤波基石系统设计/filtering_benchmark', "src"))

def _sig():
    fs = 12000
    t = np.linspace(0, 1, fs, 0)
    return np.sin(2*np.pi*78.5*t) + 0.3*np.random.randn(fs), fs

def test_execute():
    from src.algorithms import registry
    registry.ensure_discovered()
    s, fs = _sig()
    for aid in registry.list_ids()[:3]:
        inst = registry.create_instance(aid)
        r = inst.denoise(s, fs)
        assert r.shape == s.shape

def test_envelope():
    from src.envelope.analyzer import EnvelopeAnalyzer
    s, fs = _sig()
    r = EnvelopeAnalyzer({}).analyze(s, fs)
    assert r.envelope_spectrum is not None

if __name__ == "__main__":
    test_execute()
    test_envelope()
    print("OK")
