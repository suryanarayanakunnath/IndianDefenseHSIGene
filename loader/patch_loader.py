import os
import numpy as np
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, patch_dirs, labels_dict=None, transform=None, add_ndvi=False, red_idx=None, nir_idx=None):
        self.files = []
        self.labels = []
        for patch_dir in patch_dirs:
            for fname in os.listdir(patch_dir):
                if fname.endswith('.npy'):
                    self.files.append(os.path.join(patch_dir, fname))
                    cls = os.path.basename(patch_dir)
                    self.labels.append(cls)
        self.transform = transform
        self.add_ndvi = add_ndvi
        self.red_idx = red_idx
        self.nir_idx = nir_idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(self.files[idx])
        label = self.labels[idx]
        if self.add_ndvi and self.red_idx is not None and self.nir_idx is not None:
            from utils.spectral_indices import compute_ndvi
            ndvi = compute_ndvi(patch, self.red_idx, self.nir_idx)
            patch = np.concatenate([patch, ndvi[None]], axis=0)
        if self.transform:
            patch = self.transform(patch)
        return patch, label
