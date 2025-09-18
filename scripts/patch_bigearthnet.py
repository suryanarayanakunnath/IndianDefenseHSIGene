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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_path", type=str, required=True, help="Path to a BigEarthNet .npy tile")
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    create_patches(args.tile_path, args.patch_size, args.stride, args.out_dir)

# Example usage
# create_patches("bigearthnet_tile.npy", patch_size=64, stride=32, out_dir="bigearthnet_patches")
