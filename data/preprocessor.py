"""
Data Preprocessing Module
Handles unified preprocessing for all HSI datasets
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
import cv2
from pathlib import Path
from tqdm import tqdm

class HSIPreprocessor:
    """Unified HSI preprocessing pipeline"""

    def __init__(self, target_bands=48, target_range=(400, 1000), target_size=256):
        self.target_bands = target_bands
        self.target_range = target_range
        self.target_size = target_size
        self.target_wavelengths = np.linspace(target_range[0], target_range[1], target_bands)

    def create_synthetic_data(self, num_samples=1000):
        """Create synthetic HSI data for testing"""
        print(f"Creating {num_samples} synthetic HSI samples...")
        samples = []

        for _ in tqdm(range(num_samples), desc="Generating synthetic data"):
            base_spectrum = self.generate_realistic_spectrum()

            # Allocate as float32 to reduce memory
            hsi_sample = np.zeros(
                (self.target_bands, self.target_size, self.target_size),
                dtype=np.float32
            )

            for band in range(self.target_bands):
                noise = np.random.normal(0, 0.1, (self.target_size, self.target_size)).astype(np.float32)
                spatial_pattern = self.generate_spatial_pattern().astype(np.float32)
                hsi_sample[band] = (
                    base_spectrum[band].astype(np.float32)
                    + 0.1 * spatial_pattern
                    + noise
                )

            # Clip and ensure float32
            hsi_sample = np.clip(hsi_sample, 0, 1).astype(np.float32)
            samples.append(hsi_sample)

        # Return array with dtype float32
        return np.array(samples, dtype=np.float32)

    def generate_realistic_spectrum(self):
        """Generate realistic hyperspectral signature"""
        spectrum = np.ones(self.target_bands, dtype=np.float32) * 0.1

        # Green peak (~550nm)
        green_center = int(self.target_bands * 0.3)
        spectrum[green_center-5:green_center+5] = 0.3

        # Red-edge (~700nm)
        red_edge = int(self.target_bands * 0.7)
        spectrum[red_edge:] = 0.6

        # NIR plateau (~750nm+)
        nir_start = int(self.target_bands * 0.75)
        spectrum[nir_start:] = 0.8

        # Add noise
        spectrum += np.random.normal(0, 0.05, self.target_bands).astype(np.float32)
        return np.clip(spectrum, 0, 1).astype(np.float32)

    def generate_spatial_pattern(self):
        """Generate spatial patterns in image"""
        pattern = np.random.random((self.target_size, self.target_size)).astype(np.float32)
        pattern = cv2.GaussianBlur(pattern, (15, 15), 0)
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        return pattern.astype(np.float32)

class HSIDataset(Dataset):
    """PyTorch Dataset for HSI data"""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return torch.FloatTensor(sample)

def create_dataloaders(data_root=None, batch_size=4, num_workers=4,
                       train_samples=20, val_samples=5):
    """Create train and validation dataloaders with small synthetic sets."""
    print("Using small synthetic data for quick testing.")

    preprocessor = HSIPreprocessor()

    # Generate reduced synthetic data
    train_data = preprocessor.create_synthetic_data(train_samples)
    val_data   = preprocessor.create_synthetic_data(val_samples)

    train_dataset = HSIDataset(train_data)
    val_dataset   = HSIDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Created dataloaders: {len(train_loader)} train, {len(val_loader)} val batches")
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders(batch_size=2)
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch.shape}")  # (batch_size, bands, height, width)
