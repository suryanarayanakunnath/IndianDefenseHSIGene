import os
import torch
from torch.utils.data import DataLoader
from loader.patch_loader import PatchDataset
from models.simple_cnn import SimpleCNN

# ---- SETTINGS ----
PATCH_DIRS = [
    "data/eurosat",
    "data/bigearthnet_patches",
    "data/bhuvan_patches"
]
PATCH_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = len(PATCH_DIRS)  # Change if more classes/labels
IN_CHANNELS = 13  # Example: Sentinel-2 bands; +1 if NDVI added

# ---- DATA LOADER ----
# Update red_idx and nir_idx according to your dataset's band order!
RED_IDX = 3  # Example: Sentinel-2 Red band index
NIR_IDX = 7  # Example: Sentinel-2 NIR band index

dataset = PatchDataset(
    patch_dirs=PATCH_DIRS,
    add_ndvi=True,
    red_idx=RED_IDX,
    nir_idx=NIR_IDX
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- MODEL ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(in_channels=IN_CHANNELS+1, num_classes=NUM_CLASSES).to(device)  # +1 for NDVI

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# ---- TRAINING LOOP ----
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for patches, labels in dataloader:
        patches = torch.tensor(patches, dtype=torch.float32).to(device)
        # Encode labels as integers if needed
        if isinstance(labels[0], str):
            label_set = list(sorted(set(labels)))
            labels = torch.tensor([label_set.index(l) for l in labels], dtype=torch.long).to(device)
        else:
            labels = torch.tensor(labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(dataloader):.4f}")

print("Training complete.")
