# scripts/convert_npy_to_pt.py
import os, glob, torch, numpy as np

src_dir = "data/BigEarthNet-S2/patches"   # folder with *_msi.npy
out_train = "data/BigEarthNet-S2/train"
out_val = "data/BigEarthNet-S2/val"
os.makedirs(out_train, exist_ok=True)
os.makedirs(out_val, exist_ok=True)

files = sorted(glob.glob(os.path.join(src_dir, "*_msi.npy")))
if not files:
    print("No npy files found in", src_dir)
else:
    # simple split: 90% train, 10% val
    n = len(files)
    split = int(n * 0.9)
    for i, f in enumerate(files):
        arr = np.load(f).astype('float32')        # shape (C,H,W)
        tensor = torch.from_numpy(arr)
        if i < split:
            target = os.path.join(out_train, os.path.basename(f).replace('_msi.npy', '.pt'))
        else:
            target = os.path.join(out_val, os.path.basename(f).replace('_msi.npy', '.pt'))
        torch.save(tensor, target)
    print(f"Converted {len(files)} files -> {out_train} + {out_val}")
