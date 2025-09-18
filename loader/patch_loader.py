import os
import numpy as np
from torch.utils.data import Dataset
from loader.patch_loader import PatchDataset
from torch.utils.data import DataLoader
from utils.spectral_indices import compute_ndvi

train_dirs = ["data/eurosat", "data/bigearthnet_patches", "data/bhuvan_patches"]
dataset = PatchDataset(train_dirs)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class PatchDataset(Dataset):
    def __init__(self, patch_dirs, labels_dict=None, transform=None):
        self.files = []
        self.labels = []
        for patch_dir in patch_dirs:
            for fname in os.listdir(patch_dir):
                if fname.endswith('.npy'):
                    self.files.append(os.path.join(patch_dir, fname))
                    # Infer class from path or labels_dict if available
                    cls = None
                    if labels_dict:
                        cls = labels_dict.get(fname, -1)
                    else:
                        # Example: folder name == class
                        cls = os.path.basename(patch_dir)
                    self.labels.append(cls)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch = np.load(self.files[idx])
        label = self.labels[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch, label
        ndvi = compute_ndvi(patch, red_band_idx=3, nir_band_idx=7)
        patch = np.concatenate([patch, ndvi[None]], axis=0)
