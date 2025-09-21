<<<<<<< HEAD
# IndianDefenseHSIGene
Enhanced HSIGene hyperspectral image generation project
=======
# Enhanced HSIGene: Physics-Informed Hyperspectral Image Generation

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python scripts/download_datasets.py

# 3. Quick test (2 epochs)
python scripts/train_model.py --quick-test

# 4. Full training
python scripts/train_model.py --epochs 50
```

## Project Structure

```
enhanced_hsigene/
├── config/           # Configuration files
├── data/            # Data processing modules
├── models/          # Model implementations
├── training/        # Training pipeline
├── scripts/         # Executable scripts
├── outputs/         # Generated outputs
└── requirements.txt # Dependencies
```

## Features

- Enhanced HSIGene Architecture: Multi-condition ControlNet with physics constraints
- Defense Applications: Indian terrain anomaly detection
- Memory Optimized: Works with 8GB+ GPU memory
- Publication Ready: Comprehensive evaluation metrics

## Expected Results

- SAM < 0.08 (Spectral Angle Mapper)
- SSIM > 0.80 (Structural Similarity)
- SRF Compliance > 0.85 (Physics constraint)

## Usage

### Training
```bash
# Basic training
python scripts/train_model.py

# Custom parameters
python scripts/train_model.py --epochs 100 --batch-size 4

# Quick test
python scripts/train_model.py --quick-test
```

### Dataset Setup
1. Run `python scripts/download_datasets.py`
2. Follow instructions for manual downloads
3. Place datasets in `data/raw_datasets/`

## Citation

Based on HSIGene foundation model with novel physics-informed constraints for Indian defense applications.

## Support

Check console output for detailed error messages. Reduce batch size if encountering memory issues.
>>>>>>> 6c629d4 (Initial commit of Enhanced HSIGene project structure)
