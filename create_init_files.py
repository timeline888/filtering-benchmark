import os
import sys

base_dir = os.path.join(os.path.dirname(__file__), "src")

subdirs = [
    "core", "signal_io", "signal_io/readers",
    "algorithms", "algorithms/classical", "algorithms/time_domain", "algorithms/freq_domain",
    "algorithms/wavelet", "algorithms/emd_family", "algorithms/adaptive",
    "algorithms/svd", "algorithms/sparse", "algorithms/bss",
    "algorithms/deep_learning", "algorithms/rotating_specific", "algorithms/acoustic",
    "algorithms/advanced",
    "envelope", "signal_simulation",
    "metrics", "scoring", "scoring/methods",
    "pipeline", "execution",
    "output", "output/report", "output/report/templates", "output/report/styles", "output/charts",
    "storage", "config", "cli", "cli/commands",
]

init_content = ""

for d in subdirs:
    path = os.path.join(base_dir, d, "__init__.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(init_content)

print(f"Created {len(subdirs)} __init__.py files")
