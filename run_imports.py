#quick smoke test
import sys
print("Python executable:", sys.executable)

import numpy as np
print("numpy:", np.__version__)

try:
    import torch
    print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)

try:
    import rasterio
    print("rasterio:", rasterio.__version__)
except Exception as e:
    print("rasterio import failed:", e)
