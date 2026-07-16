from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# Tiny test networks are faster and more deterministic without intra-op
# fan-out; CUDA kernels are unaffected by this CPU setting.
import torch
torch.set_num_threads(1)
