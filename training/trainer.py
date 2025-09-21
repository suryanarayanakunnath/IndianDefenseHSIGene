#!/usr/bin/env python3
"""
Training Pipeline for Enhanced HSIGene
Simplified version for quick setup
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

from models.enhanced_hsigene import EnhancedHSIGene
from data.preprocessor import create_dataloaders
from config.model_config import ModelConfig
from config.training_config import TrainingConfig

class HSIGeneTrainer:
    """Trainer class for Enhanced HSIGene"""
    
    def __init__(self, config=None):
        self.model_config = ModelConfig()
        self.train_config = TrainingConfig() if config is None else config
        
        # Initialize device
        self.device = torch.device(self.train_config.DEVICE)
        print(f"Using device: {self.device}")
        
        # Create directories
        self.train_config.create_directories()
        
        # Initialize model
        self.model = EnhancedHSIGene(self.model_config)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.LEARNING_RATE,
            weight_decay=self.train_config.WEIGHT_DECAY
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        # Mixed precision scaler
        if self.train_config.MIXED_PRECISION:
            self.scaler = GradScaler()
        
        # Tracking variables
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def compute_losses(self, model_output, target):
        """Compute loss components"""
        pred_hsi = model_output['sample']
        mu = model_output['mu']
        logvar = model_output['logvar']
        
        # Reconstruction loss
        recon_loss = self.mse_loss(pred_hsi, target)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.numel()
        
        # Physics loss
        physics_loss, _ = self.model.physics_loss(pred_hsi, target)
        
        # Total loss
        total_loss = (
            self.train_config.RECONSTRUCTION_WEIGHT * recon_loss +
            self.train_config.KL_WEIGHT * kl_loss +
            self.train_config.PHYSICS_WEIGHT * physics_loss
        )
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss,
            'physics': physics_loss
        }
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        batch = batch.to(self.device)
        
        # Forward pass
        timesteps = torch.randint(
            0, self.model_config.NUM_TIMESTEPS, 
            (batch.shape[0],), device=self.device
        )
        
        outputs = self.model(batch, timesteps=timesteps)
        losses = self.compute_losses(outputs, batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient clipping
        if self.train_config.GRADIENT_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.train_config.GRADIENT_CLIP_NORM
            )
        
        self.optimizer.step()
        
        return losses
    
    def validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                timesteps = torch.randint(
                    0, self.model_config.NUM_TIMESTEPS,
                    (batch.shape[0],), device=self.device
                )
                
                outputs = self.model(batch, timesteps=timesteps)
                losses = self.compute_losses(outputs, batch)
                val_losses.append(losses['total'].item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint only (avoid pickling config)"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        checkpoint_path = Path(self.train_config.MODEL_SAVE_DIR) / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = Path(self.train_config.MODEL_SAVE_DIR) / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
        
        # Cleanup old checkpoints
        checkpoints = glob.glob(str(Path(self.train_config.MODEL_SAVE_DIR) / "checkpoint_epoch_*.pth"))
        if len(checkpoints) > self.train_config.MAX_CHECKPOINTS:
            epochs = sorted(int(Path(cp).stem.split('_')[-1]) for cp in checkpoints)
            for old_epoch in epochs[:-self.train_config.MAX_CHECKPOINTS]:
                old_ckpt = Path(self.train_config.MODEL_SAVE_DIR) / f"checkpoint_epoch_{old_epoch}.pth"
                old_ckpt.unlink(missing_ok=True)
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"Starting training for {self.train_config.MAX_EPOCHS} epochs")
        
        for epoch in range(self.train_config.MAX_EPOCHS):
            epoch_start_time = time.time()
            train_losses = []
            
            # Training
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.train_config.MAX_EPOCHS}"
            )
            
            for batch in progress_bar:
                self.global_step += 1
                losses = self.train_step(batch)
                train_losses.append(losses['total'].item())
                
                progress_bar.set_postfix({
                    'Loss': f"{losses['total'].item():.4f}"
                })
            
            # Validation
            avg_val_loss = self.validate(val_loader)
            
            # Save checkpoint
            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = np.mean(train_losses)
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}, Time={epoch_time:.1f}s")
        
        print("Training completed!")

# No top-level imports of HSIGeneTrainer here, so no circular import
