# scripts/create_bigearth_patches.py
import os, glob, numpy as np
import rasterio
from rasterio.windows import Window

IN_DIR = "data/BigEarthNet-S2/tiles"  # update if your tiles are in a different path
OUT_DIR = "data/BigEarthNet-S2/patches"
PATCH = 64
STRIDE = 32
os.makedirs(OUT_DIR, exist_ok=True)

def sliding_windows(img, patch=PATCH, stride=STRIDE):
    _, H, W = img.shape
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            yield y, x

count = 0
tifs = glob.glob(os.path.join(IN_DIR, "*.tif"))
if len(tifs) == 0:
    print("No GeoTIFFs found in", IN_DIR)
for t in tifs:
    print("Processing", t)
    with rasterio.open(t) as src:
        arr = src.read().astype('float32')  # (bands,H,W)
        # optional scale if values very large
        if arr.max() > 1000:
            arr /= 10000.0
        for y, x in sliding_windows(arr):
            patch = arr[:, y:y+PATCH, x:x+PATCH]
            np.save(os.path.join(OUT_DIR, f"patch_{count:06d}_msi.npy"), patch)
            # placeholder HSI: zeros (or load real HSI if you have)
            hsi = np.zeros((200, PATCH, PATCH), dtype='float32')
            np.save(os.path.join(OUT_DIR, f"patch_{count:06d}_hsi.npy"), hsi)
            count += 1
print("Wrote", count, "patches to", OUT_DIR)
