import os
import numpy as np
from skimage.util import view_as_windows

def create_patches(tile_path, patch_size, stride, out_dir):
    tile = np.load(tile_path)  # shape: (bands, H, W)
    patches = view_as_windows(tile, (tile.shape[0], patch_size, patch_size), step=(1, stride, stride))
    patches = patches.reshape(-1, tile.shape[0], patch_size, patch_size)
    base = os.path.splitext(os.path.basename(tile_path))[0]
    for i, patch in enumerate(patches):
        np.save(os.path.join(out_dir, f"{base}_patch_{i:04d}.npy"), patch)

# Example usage
# create_patches("bigearthnet_tile.npy", patch_size=64, stride=32, out_dir="bigearthnet_patches")
