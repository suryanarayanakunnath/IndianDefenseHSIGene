import os
import numpy as np
from torch.utils.data import Dataset

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
