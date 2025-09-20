import os
import glob
import numpy as np
from PIL import Image, ImageDraw
import rasterio
import logging
from typing import List, Tuple, Optional

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------
# Configurable paths and params
# ------------------------------
class Config:
    DATA_ROOT = r"D:/Desktop/CU/Research/5th sem/IndianDefenseHSIGene/data"
    EUROSAT_PATH = os.path.join(DATA_ROOT, "EuroSAT", "2750")
    BHUVAN_PATH = os.path.join(DATA_ROOT, "bhuvan_patches")
    BIGEARTHNET_PATH = os.path.join(DATA_ROOT, "bigearthnet_patches")
    PATCH_SIZE = 64  # Resize all patches to this size (HxW)
    ANOMALY_FRACTION = 0.2  # Fraction of patches to augment with anomalies
    
# ---------------------------------
# Utility functions for normalization
# ---------------------------------
def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype('float32')
    arr /= 255.0
    return arr

# ---------------------------------
# Dataset loading functions
# ---------------------------------
def load_eurosat_patches(base_folder: str) -> Tuple[np.ndarray, List[str]]:
    patches = []
    labels = []
    try:
        class_folders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
        if not class_folders:
            logging.warning(f"No class folders found in EuroSAT base folder: {base_folder}")
        for class_dir in class_folders:
            class_name = os.path.basename(class_dir)
            img_paths = glob.glob(os.path.join(class_dir, "*.jpg"))
            if not img_paths:
                logging.warning(f"No images found in {class_dir}")
            for img_path in img_paths:
                try:
                    img = Image.open(img_path).resize((Config.PATCH_SIZE, Config.PATCH_SIZE))
                    patches.append(np.array(img))
                    labels.append(class_name)
                except Exception as e:
                    logging.error(f"Failed to load image: {img_path}, error: {e}")
    except Exception as e:
        logging.error(f"Error scanning EuroSAT base folder: {e}")
    
    patches = np.stack(patches) if patches else np.zeros((0, Config.PATCH_SIZE, Config.PATCH_SIZE, 3))
    return patches, labels

def load_tiff_patches(folder: str) -> np.ndarray:
    patches = []
    try:
        patch_files = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff"))
        if not patch_files:
            logging.warning(f"No .tif files found in {folder}")
        for tif_path in patch_files:
            try:
                with rasterio.open(tif_path) as src:
                    arr = src.read(out_shape=(src.count, Config.PATCH_SIZE, Config.PATCH_SIZE))
                    arr = np.moveaxis(arr, 0, -1)  # (bands, H, W) -> (H, W, bands)
                    patches.append(arr)
            except Exception as e:
                logging.error(f"Error loading tiff: {tif_path}, error: {e}")
    except Exception as e:
        logging.error(f"Error scanning folder {folder}: {e}")
    return np.stack(patches) if patches else np.zeros((0, Config.PATCH_SIZE, Config.PATCH_SIZE, 3))

# ---------------------------------
# Anomaly embedding with mask generation
# ---------------------------------
def embed_anomaly_with_mask(patch: np.ndarray, anomaly_type: str = "vehicle") -> Tuple[np.ndarray, np.ndarray]:
    img = Image.fromarray((patch * 255).astype(np.uint8))
    mask = Image.new("L", img.size, 0)
    draw_patch = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    
    if anomaly_type == "vehicle":
        x0, y0 = np.random.randint(5, 40), np.random.randint(5, 40)
        x1, y1 = x0 + np.random.randint(10, 20), y0 + np.random.randint(6, 12)
        draw_patch.rectangle([x0, y0, x1, y1], fill=(45, 63, 45))
        draw_mask.rectangle([x0, y0, x1, y1], fill=255)
    elif anomaly_type == "troop":
        x0, y0 = np.random.randint(10, 54), np.random.randint(10, 54)
        x1, y1 = x0 + np.random.randint(5, 10), y0 + np.random.randint(5, 10)
        draw_patch.ellipse([x0, y0, x1, y1], fill=(80, 120, 80))
        draw_mask.ellipse([x0, y0, x1, y1], fill=255)
    else:
        logging.warning(f"Unknown anomaly type: {anomaly_type}, returning original patch without anomaly.")
        return patch, np.zeros(patch.shape[:2])

    return np.array(img) / 255.0, np.array(mask) / 255.0

def augment_with_anomalies_masks(patches: np.ndarray, fraction: Optional[float] = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
    if fraction is None:
        fraction = Config.ANOMALY_FRACTION

    patches_aug = []
    masks = []
    for patch in patches:
        if np.random.rand() < fraction:
            anomaly_type = np.random.choice(["vehicle", "troop"])
            patch_anom, mask = embed_anomaly_with_mask(patch, anomaly_type)
            masks.append(mask)
            patches_aug.append(patch_anom)
        else:
            patches_aug.append(patch)
            masks.append(np.zeros(patch.shape[:2]))
    return np.stack(patches_aug), np.stack(masks)

# ---------------------------------
# Data augmentation functions
# ---------------------------------
def random_augment(patch: np.ndarray) -> np.ndarray:
    img = Image.fromarray((patch * 255).astype(np.uint8))
    if np.random.rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.rand() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if np.random.rand() < 0.5:
        angle = np.random.choice([90, 180, 270])
        img = img.rotate(angle)
    
    arr = np.array(img)
    if np.random.rand() < 0.3:
        noise = np.random.normal(0, 5, arr.shape)
        arr = arr + noise
        arr = np.clip(arr, 0, 255)
    return arr / 255.0

def augment_dataset(patches: np.ndarray) -> np.ndarray:
    return np.stack([random_augment(p) for p in patches])

# ---------------------------------
# Main preprocessing pipeline
# ---------------------------------
def preprocess_pipeline() -> Tuple[np.ndarray, List[str], np.ndarray]:
    eurosat_patches, eurosat_labels = load_eurosat_patches(Config.EUROSAT_PATH)
    bhuvan_patches = load_tiff_patches(Config.BHUVAN_PATH)
    bigearthnet_patches = load_tiff_patches(Config.BIGEARTHNET_PATH)
    
    eurosat_patches = normalize(eurosat_patches)
    bhuvan_patches = normalize(bhuvan_patches)
    bigearthnet_patches = normalize(bigearthnet_patches)
    
    # Aggregate all patches
    all_patches = np.concatenate([eurosat_patches, bhuvan_patches, bigearthnet_patches], axis=0)
    
    # For EuroSAT labels, pad with "Unknown" for non-EuroSAT data
    unknown_labels = ["Unknown"] * (bhuvan_patches.shape[0] + bigearthnet_patches.shape[0])
    all_labels = eurosat_labels + unknown_labels

    # Embed anomalies and get corresponding masks
    patches_with_anomalies, anomaly_masks = augment_with_anomalies_masks(all_patches)
    
    # Optionally perform data augmentations here (uncomment if needed)
    # patches_with_anomalies = augment_dataset(patches_with_anomalies)

    logging.info(f"Total patches loaded: {all_patches.shape[0]}, with {np.sum(np.max(anomaly_masks, axis=(1, 2)) > 0)} anomaly-embedded patches")
    
    return patches_with_anomalies, all_labels, anomaly_masks

# ---------------------------------
# Visualization utilities
# ---------------------------------
import matplotlib.pyplot as plt

def show_anomaly_examples(patches: np.ndarray, masks: np.ndarray, n: int = 5, alpha: float = 0.4) -> None:
    plt.figure(figsize=(15, 3))
    for i in range(min(n, patches.shape[0])):
        plt.subplot(1, n, i + 1)
        patch = patches[i]
        mask = masks[i]
        
        display_patch = patch[:, :, :3] if patch.shape[-1] > 3 else patch
        plt.imshow((display_patch * 255).astype(np.uint8))
        plt.imshow(mask, cmap='Reds', alpha=alpha)
        plt.axis('off')
    plt.suptitle(f"Anomaly-Embedded Patches (Showing {n} samples)")
    plt.show()

def show_only_anomalies(patches: np.ndarray, masks: np.ndarray, n: int = 5, alpha: float = 0.4) -> None:
    """Visualize only patches that have anomalies."""
    anomaly_indices = [i for i, m in enumerate(masks) if np.max(m) > 0]
    if not anomaly_indices:
        logging.warning("No anomaly patches found to display.")
        return
    plt.figure(figsize=(15, 3))
    for idx, i in enumerate(anomaly_indices[:n]):
        plt.subplot(1, n, idx + 1)
        patch = patches[i]
        mask = masks[i]
        
        display_patch = patch[:, :, :3] if patch.shape[-1] > 3 else patch
        plt.imshow((display_patch * 255).astype(np.uint8))
        plt.imshow(mask, cmap='Reds', alpha=alpha)
        plt.axis('off')
    plt.suptitle(f"Only Anomaly-Embedded Patches (Showing {min(n, len(anomaly_indices))} samples)")
    plt.show()

# ---------------------------------
# Example main usage block
# ---------------------------------
if __name__ == "__main__":
    patches, labels, masks = preprocess_pipeline()
    # Visualize some examples with anomalies highlighted
    show_anomaly_examples(patches, masks, n=5)
    # Visualize only patches with anomalies
    show_only_anomalies(patches, masks, n=5)
