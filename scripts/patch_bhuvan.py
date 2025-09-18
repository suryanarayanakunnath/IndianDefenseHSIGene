import os
import numpy as np
import rasterio
from skimage.util import view_as_windows

def create_patches_from_geotiff(geotiff_path, patch_size, stride, out_dir):
    with rasterio.open(geotiff_path) as src:
        img = src.read()  # shape: (bands, H, W)
    patches = view_as_windows(img, (img.shape[0], patch_size, patch_size), step=(1, stride, stride))
    patches = patches.reshape(-1, img.shape[0], patch_size, patch_size)
    base = os.path.splitext(os.path.basename(geotiff_path))[0]
    for i, patch in enumerate(patches):
        np.save(os.path.join(out_dir, f"{base}_patch_{i:04d}.npy"), patch)

# Example usage
# create_patches_from_geotiff("bhuvan_scene.tif", patch_size=64, stride=32, out_dir="bhuvan_patches")
