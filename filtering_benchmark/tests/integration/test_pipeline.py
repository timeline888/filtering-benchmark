
import sys, math
import os, sys; sys.path.insert(0, 'e:/Qoder项目/滤波基石系统设计/filtering_benchmark'); sys.path.insert(0, os.path.join('e:/Qoder项目/滤波基石系统设计/filtering_benchmark', "src"))

def test_minimal():
    from src.config.loader import ConfigLoader
    from src.pipeline.orchestrator import PipelineOrchestrator
    c = ConfigLoader().load(user_config_path=None, cli_overrides={
        "signal": {"source": {"simulate_reference": True, "sample_rate": 12000}},
        "algorithms": {"selection": {"mode": "category", "categories": ["time_domain"]}},
        "scoring": {"top_n": 3},
        "metrics": {"enabled": ["ffr", "lsnr", "esk", "kurtosis", "sk"]},
    })
    r = PipelineOrchestrator(c).run()
    assert len(r.rankings) > 0
    for rank in r.rankings:
        assert not math.isnan(rank.overall_score)

if __name__ == "__main__":
    test_minimal()
    print("OK")
