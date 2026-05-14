Test Suite for Filtering Benchmark
================================

Structure:
- unit/        : Unit tests for individual components
- integration/ : Integration tests for the full pipeline

Run tests:
    python -m pytest tests/ -v
    python tests/unit/test_registry.py
    python tests/unit/test_algorithms.py
    python tests/unit/test_metrics.py
    python tests/integration/test_pipeline.py
