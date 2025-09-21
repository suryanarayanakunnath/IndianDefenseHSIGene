"""
Model Configuration for Enhanced HSIGene
"""

import torch

class ModelConfig:
    # Model Architecture
    MODEL_NAME = "enhanced_hsigene"
    SPECTRAL_BANDS = 48
    SPATIAL_SIZE = 256
    LATENT_CHANNELS = 8
    BATCH_SIZE = 4  # Reduced for memory efficiency
    
    # U-Net Configuration
    UNET_CHANNELS = [128, 256, 512, 1024]
    ATTENTION_HEAD_DIM = 64
    DROPOUT_RATE = 0.1
    
    # VAE Configuration
    VAE_LATENT_DIM = 8
    VAE_ENCODER_LAYERS = [32, 64, 128, 256]
    VAE_DECODER_LAYERS = [256, 128, 64, 32]
    
    # ControlNet Configuration
    CONDITION_TYPES = [
        'sentinel2', 'bhuvan', 'hed', 'segmentation', 
        'sketch', 'mlsd', 'content', 'text'
    ]
    CONDITION_CHANNELS = 8
    
    # Physics Constraints
    PHYSICS_WEIGHT = 0.1
    SMOOTHNESS_WEIGHT = 0.05
    SRF_WEIGHT = 0.1
    
    # Defense-specific
    TERRAIN_TYPES = ['desert', 'forest', 'coastal', 'himalayan']
    ANOMALY_TYPES = ['camouflaged_vehicle', 'troop_movement', 
                    'hidden_tunnel', 'intrusion']
    
    # Diffusion Parameters
    NUM_TIMESTEPS = 1000
    BETA_START = 0.0001
    BETA_END = 0.02
    BETA_SCHEDULE = "scaled_linear"
    
    # Memory Optimization
    GRADIENT_CHECKPOINTING = True
    MIXED_PRECISION = True
    USE_XFORMERS = True  # If available
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') and not callable(v)}