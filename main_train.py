# main_train.py - entry point
import os, torch, yaml, argparse
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from src.training.trainer import HSIGeneTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# -------------------------
# Dummy Dataset
# -------------------------
class DummyDataset(Dataset):
    def __init__(self, split, config):
        self.split = split
        self.size = config.get('dataset_size', 100)
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        msi_bhuvan = torch.randn(self.in_channels, 64, 64)
        target = torch.randn(self.out_channels, 64, 64)
        return msi_bhuvan, target

# -------------------------
# EuroSAT Dataset Wrapper
# -------------------------
class EuroSATWrapper(Dataset):
    def __init__(self, root, split, config):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.dataset = datasets.ImageFolder(root=os.path.join(root, split), transform=transform)
        self.out_channels = config['out_channels']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # EuroSAT RGB (3 channels)
        target = torch.randn(self.out_channels, 64, 64)  # dummy HSI target
        return img, target

# -------------------------
# BigEarthNet-S2 Wrapper
# -------------------------
class BigEarthNetWrapper(Dataset):
    def __init__(self, root, split, config):
        self.root = os.path.join(root, split)
        self.files = [os.path.join(self.root, f) for f in os.listdir(self.root) if f.endswith('.pt')]
        self.out_channels = config['out_channels']

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx])  # (bands, H, W)
        target = torch.randn(self.out_channels, sample.shape[1], sample.shape[2])  # dummy
        return sample, target

# -------------------------
# Dataloader Factory
# -------------------------
def create_dataloader(config, split):
    dataset_name = config.get("dataset", "dummy").lower()
    if dataset_name == "dummy":
        dataset = DummyDataset(split, config)
    elif dataset_name == "eurosat":
        dataset = EuroSATWrapper(config['eurosat_root'], split, config)
    elif dataset_name == "bigearthnet":
        dataset = BigEarthNetWrapper(config['bigearthnet_root'], split, config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 0)
    )

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/hsigene_config.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    torch.manual_seed(config['seed'])
    trainer = HSIGeneTrainer(config)

    train_loader = create_dataloader(config, 'train')
    val_loader = create_dataloader(config, 'val')

    trainer.train(train_loader, val_loader, config['epochs'])
    print("Training done")

if __name__ == '__main__':
    main()
