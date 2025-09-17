# src/training/trainer.py
#training loop with checkpointing
import torch
import torch.optim as optim
from tqdm import tqdm
import os

from src.models.generator import HSIGeneGenerator
from src.models.discriminator import HSIGeneDiscriminator
from src.losses.physics_losses import CombinedLoss, WGANGPLoss

class HSIGeneTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('device','cpu')=='cuda' else 'cpu')
        self.generator = HSIGeneGenerator(in_channels=config['in_channels'], out_channels=config['out_channels']).to(self.device)
        self.discriminator = HSIGeneDiscriminator(in_channels=config['out_channels']).to(self.device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=config['g_lr'], betas=(0.5,0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=config['d_lr'], betas=(0.5,0.999))
        self.combined_loss = CombinedLoss(config.get('loss_weights'))
        self.wgan_loss = WGANGPLoss(config.get('lambda_gp',10.0))
        self.current_epoch = 0
        self.best_loss = float('inf')
        os.makedirs('models/checkpoints', exist_ok=True)

    def train_epoch(self, dataloader):
        self.generator.train(); self.discriminator.train()
        epoch_g_loss = 0.0; epoch_d_loss = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, (msi_bhuvan, real_hsi) in enumerate(pbar):
            batch_size = msi_bhuvan.size(0)
            msi_bhuvan = msi_bhuvan.to(self.device)
            real_hsi = real_hsi.to(self.device) if real_hsi is not None else None

            # Train discriminator
            for _ in range(self.config.get('d_steps',1)):
                self.d_optimizer.zero_grad()
                with torch.no_grad():
                    fake_hsi = self.generator(msi_bhuvan)
                real_valid = self.discriminator(real_hsi) if real_hsi is not None else torch.zeros(batch_size, device=self.device)
                fake_valid = self.discriminator(fake_hsi.detach())
                gp = self.wgan_loss.gradient_penalty(self.discriminator, real_hsi.detach() if real_hsi is not None else fake_hsi.detach(), fake_hsi.detach(), self.device)
                d_loss = self.wgan_loss.discriminator_loss(real_valid, fake_valid, gp)
                d_loss.backward()
                self.d_optimizer.step()

            # Train generator
            for _ in range(self.config.get('g_steps',1)):
                self.g_optimizer.zero_grad()
                gen_hsi = self.generator(msi_bhuvan)
                fake_validity = self.discriminator(gen_hsi)
                total_loss, loss_components = self.combined_loss(gen_hsi, msi_bhuvan, fake_validity, reference_spectra=None)
                total_loss.backward()
                self.g_optimizer.step()

            epoch_g_loss += total_loss.item()
            epoch_d_loss += d_loss.item()
            pbar.set_postfix({'G_Loss': f'{total_loss.item():.4f}', 'D_Loss': f'{d_loss.item():.4f}'})
        avg_g = epoch_g_loss / len(dataloader)
        avg_d = epoch_d_loss / len(dataloader)
        return avg_g, avg_d

    def validate(self, val_dataloader):
        self.generator.eval(); self.discriminator.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for msi_bhuvan, real_hsi in val_dataloader:
                msi_bhuvan = msi_bhuvan.to(self.device)
                real_hsi = real_hsi.to(self.device)
                generated_hsi = self.generator(msi_bhuvan)
                fake_validity = self.discriminator(generated_hsi)
                val_loss, _ = self.combined_loss(generated_hsi, msi_bhuvan, fake_validity)
                total_val_loss += val_loss.item()
        return total_val_loss / len(val_dataloader)

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        torch.save(checkpoint, f'models/checkpoints/hsigene_epoch_{epoch}.pth')
        if is_best:
            torch.save(checkpoint, 'models/hsigene_best.pth')
            self.best_loss = loss

    def train(self, train_dataloader, val_dataloader, epochs):
        print(f"Starting training for {epochs} epochs on device {self.device}")
        for epoch in range(epochs):
            self.current_epoch = epoch
            avg_g, avg_d = self.train_epoch(train_dataloader)
            avg_val = self.validate(val_dataloader)
            is_best = avg_val < self.best_loss
            if epoch % self.config.get('save_freq', 10) == 0 or is_best:
                self.save_checkpoint(epoch, avg_val, is_best)
            print(f"Epoch {epoch}: G_loss={avg_g:.4f}, D_loss={avg_d:.4f}, Val_loss={avg_val:.4f}")
