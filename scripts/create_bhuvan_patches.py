# scripts/create_bhuvan_patches.py
import os, glob, numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

IN_DIR = "data/Bhuvan"
OUT_DIR = "data/Bhuvan/patches"
PATCH = 64
STRIDE = 32
os.makedirs(OUT_DIR, exist_ok=True)

tifs = glob.glob(os.path.join(IN_DIR, "*.tif")) + glob.glob(os.path.join(IN_DIR, "*.tiff"))
if len(tifs) == 0:
    print("No Bhuvan TIFFs found in", IN_DIR)

count = 0
for t in tifs:
    print("Reading", t)
    with rasterio.open(t) as src:
        arr = src.read().astype('float32')  # (bands,H,W)
        # rescale if needed
        if arr.max() > 1000:
            arr /= 10000.0
        _, H, W = arr.shape
        for y in range(0, H-PATCH+1, STRIDE):
            for x in range(0, W-PATCH+1, STRIDE):
                patch = arr[:, y:y+PATCH, x:x+PATCH]
                np.save(os.path.join(OUT_DIR, f"patch_{count:06d}_msi.npy"), patch)
                # placeholder HSI
                hsi = np.zeros((200, PATCH, PATCH), dtype='float32')
                np.save(os.path.join(OUT_DIR, f"patch_{count:06d}_hsi.npy"), hsi)
                count += 1
print("Wrote", count, "patches to", OUT_DIR)
