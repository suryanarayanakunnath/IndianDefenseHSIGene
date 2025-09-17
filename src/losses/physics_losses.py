# src/losses/physics_losses.py
#combined & WGAN-GP losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import spectral_angle_mapper

class PhysicsConsistencyLoss(nn.Module):
    def __init__(self, sentinel2_bands=[2,3,4,8]):
        super().__init__()
        self.s2_bands = sentinel2_bands

    def forward(self, generated_hsi, input_msi):
        # pick a few HSI band indices as pseudo-simulated MSI (small example)
        hsi_bands = [30, 60, 90, 120]
        generated_pseudo_msi = generated_hsi[:, hsi_bands, :, :]
        target_msi = input_msi[:, :len(hsi_bands), :, :]   # assume input_msi arranged similarly in dummy
        return F.l1_loss(generated_pseudo_msi, target_msi)

class SAMLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, generated, target):
        # torchmetrics spectral_angle_mapper expects (B, C, H, W)
        return spectral_angle_mapper(generated, target, reduction=self.reduction)

class SpectralSmoothnessloss(nn.Module):
    def __init__(self, spectral_weight=1.0, spatial_weight=1.0):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.spatial_weight = spatial_weight
    def forward(self, x):
        spectral_tv = torch.mean(torch.abs(x[:, :-1] - x[:, 1:]))
        spatial_tv_h = torch.mean(torch.abs(x[:, :, :-1] - x[:, :, 1:]))
        spatial_tv_w = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        total_loss = (self.spectral_weight * spectral_tv +
                      self.spatial_weight * (spatial_tv_h + spatial_tv_w))
        return total_loss

class WGANGPLoss(nn.Module):
    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    def gradient_penalty(self, discriminator, real_data, fake_data, device):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        d_interpolated = discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.lambda_gp * gradient_penalty
    def discriminator_loss(self, real_validity, fake_validity, gradient_penalty):
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
        return d_loss
    def generator_loss(self, fake_validity):
        return -torch.mean(fake_validity)

class CombinedLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = {'adversarial':1.0,'physics':10.0,'sam':5.0,'smoothness':1.0}
        self.weights = weights
        self.wgan = WGANGPLoss()
        self.physics = PhysicsConsistencyLoss()
        self.sam = SAMLoss()
        self.smooth = SpectralSmoothnessloss()
    def forward(self, generated_hsi, input_msi, discriminator_output, reference_spectra=None):
        losses = {}
        losses['adversarial'] = self.wgan.generator_loss(discriminator_output)
        losses['physics'] = self.physics(generated_hsi, input_msi)
        losses['smoothness'] = self.smooth(generated_hsi)
        if reference_spectra is not None:
            losses['sam'] = self.sam(generated_hsi, reference_spectra)
        else:
            losses['sam'] = torch.tensor(0.0, device=generated_hsi.device)
        total = sum(self.weights[k] * losses[k] for k in losses)
        return total, losses
