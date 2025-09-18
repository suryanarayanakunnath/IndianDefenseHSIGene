import numpy as np

def compute_ndvi(patch, red_band_idx, nir_band_idx):
    red = patch[red_band_idx].astype(np.float32)
    nir = patch[nir_band_idx].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

# Example usage:
# ndvi = compute_ndvi(patch, red_band_idx=3, nir_band_idx=7)
