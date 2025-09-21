#!/usr/bin/env python3
"""
Main Training Script for Enhanced HSIGene (Patched)
Disables checkpointing during quick tests to avoid pickling errors.
"""

import sys
import argparse
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from training.trainer import HSIGeneTrainer
from data.preprocessor import create_dataloaders
from config.model_config import ModelConfig
from config.training_config import TrainingConfig

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced HSIGene Model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--quick-test', action='store_true', help='Quick test run')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb')
    
    args = parser.parse_args()
    
    print("Enhanced HSIGene Training")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Update config
    train_config = TrainingConfig()
    model_config = ModelConfig()
    
    if args.quick_test:
        train_config.MAX_EPOCHS = 2
        train_config.USE_WANDB = False
        model_config.BATCH_SIZE = 1
        print("Quick test mode: 2 epochs")
    else:
        train_config.MAX_EPOCHS = args.epochs
        model_config.BATCH_SIZE = args.batch_size
    
    if args.no_wandb:
        train_config.USE_WANDB = False
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(
        batch_size=model_config.BATCH_SIZE, 
        num_workers=0
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = HSIGeneTrainer(train_config)
    
    # Disable checkpointing during quick-test to avoid pickling issues
    #if args.quick_test:
    #    trainer.save_checkpoint = lambda *a, **k: None
    
    print(f"Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # Start training
    try:
        trainer.train(train_loader, val_loader)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()