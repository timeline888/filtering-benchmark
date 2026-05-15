
import sys
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); sys.path.insert(0, os.path.join('e:/Qoder项目/滤波基石系统设计/filtering_benchmark', "src"))

def test_singleton():
    from src.algorithms.registry import AlgorithmRegistry, registry
    assert AlgorithmRegistry() is registry

def test_discovery():
    from src.algorithms import registry
    registry.ensure_discovered()
    assert registry.count() > 0

def test_filter():
    from src.algorithms import registry
    from src.core.types import AlgorithmCategory
    registry.ensure_discovered()
    r = registry.filter(category=[AlgorithmCategory.TIME_DOMAIN])
    assert len(r) > 0

def test_filter_str():
    from src.algorithms import registry
    registry.ensure_discovered()
    r = registry.filter(category=["time_domain"])
    assert len(r) > 0

def test_create():
    from src.algorithms import registry
    registry.ensure_discovered()
    ids = registry.list_ids()
    inst = registry.create_instance(ids[0])
    assert inst is not None

if __name__ == "__main__":
    test_singleton()
    test_discovery()
    test_filter()
    test_filter_str()
    test_create()
    print("OK")
