# src/data/datasets.py
"""
Dataset classes for EuroSAT, BigEarthNet (patch-based) and Bhuvan (GeoTIFF -> patches).
Each dataset returns (msi, hsi) as torch.FloatTensor:
 - msi: (C_msi, H, W)
 - hsi: (C_hsi, H, W)  (if no true HSI available, a zero tensor placeholder is returned)

Usage:
  from src.data.datasets import EuroSATDataset, PatchFolderDataset
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import rasterio

# -----------------------
# EuroSAT loader (on-the-fly)
# -----------------------
class EuroSATDataset(Dataset):
    """
    EuroSAT expects a folder tree like:
      data/EuroSAT/<class_name>/*.jpg
    This loader returns single-image MSI (multispectral or RGB). We optionally expand single image
    to have a fake HSI target (zeros) unless you provide a mapping to true HSI.
    """
    def __init__(self, root_dir, transform=None, to_float=True):
        self.root_dir = root_dir
        self.transform = transform
        # find all image files recursively
        patterns = ('**/*.jpg', '**/*.png', '**/*.tif', '**/*.tiff')
        self.files = []
        for p in patterns:
            self.files += glob.glob(os.path.join(root_dir, p), recursive=True)
        self.files = sorted(self.files)
        if len(self.files) == 0:
            raise RuntimeError(f"No images found under {root_dir}")
        self.to_float = to_float

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')  # EuroSAT RGB images -> convert to 3-channel
        arr = np.asarray(img, dtype=np.float32) / 255.0 if self.to_float else np.asarray(img, dtype=np.float32)
        # convert to C,H,W
        arr = np.transpose(arr, (2,0,1)).astype('float32')
        msi = torch.from_numpy(arr)
        # placeholder HSI: if you have true HSI, change this to load it
        hsi = torch.zeros((200, msi.shape[1], msi.shape[2]), dtype=torch.float32)
        if self.transform:
            msi = self.transform(msi)
            hsi = self.transform(hsi)
        return msi, hsi

# -----------------------
# Generic patch folder loader (for BigEarthNet & Bhuvan patches)
# -----------------------
class PatchFolderDataset(Dataset):
    """
    Loads .npy patches from a folder. Expects MSI patch files named *_msi.npy and optional
    HSI patch files named *_hsi.npy with the same index.
    Example:
      patches/
        patch_00000_msi.npy
        patch_00000_hsi.npy   # optional
    """
    def __init__(self, patch_dir, transform=None):
        self.patch_dir = patch_dir
        self.transform = transform
        self.msi_files = sorted(glob.glob(os.path.join(patch_dir, '*_msi.npy')))
        if len(self.msi_files) == 0:
            raise RuntimeError(f"No MSI patch files found in {patch_dir} (expected *_msi.npy)")
        # map to corresponding hsi files if they exist
        self.hsi_files = []
        for m in self.msi_files:
            base = os.path.basename(m)
            idx = base.replace('_msi.npy', '')
            hpath = os.path.join(patch_dir, f'{idx}_hsi.npy')
            self.hsi_files.append(hpath if os.path.exists(hpath) else None)

    def __len__(self):
        return len(self.msi_files)

    def __getitem__(self, idx):
        mpath = self.msi_files[idx]
        msi = np.load(mpath).astype('float32')   # shape (C,H,W)
        hpath = self.hsi_files[idx]
        if hpath:
            hsi = np.load(hpath).astype('float32')
        else:
            # placeholder HSI (same H,W as MSI, default 200 bands)
            H = msi.shape[1]; W = msi.shape[2]
            hsi = np.zeros((200, H, W), dtype='float32')
        msi_t = torch.from_numpy(msi)
        hsi_t = torch.from_numpy(hsi)
        if self.transform:
            msi_t = self.transform(msi_t)
            hsi_t = self.transform(hsi_t)
        return msi_t, hsi_t

# -----------------------
# Small helper to load a single GeoTIFF into memory (for debugging)
# -----------------------
def read_geotiff(path):
    with rasterio.open(path) as src:
        arr = src.read().astype('float32')  # bands, H, W
        meta = src.meta.copy()
    return arr, meta
