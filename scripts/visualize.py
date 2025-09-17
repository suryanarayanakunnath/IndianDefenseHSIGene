# scripts/visualize.py
#Quick visualization (one-band) — create & run script
import torch
import matplotlib.pyplot as plt
p = 'generated/sample_0.pt'   # change if needed
d = torch.load(p)
hsi = d['generated'].squeeze(0)   # (C,H,W)
print("HSI shape:", hsi.shape)
band = 30
plt.imshow(hsi[band].numpy(), cmap='gray')
plt.title(f'Generated sample - band {band}')
plt.axis('off')
plt.show()
