"""
Training Configuration for Enhanced HSIGene (Fixed)
Adds missing MIXED_PRECISION attribute
"""

import os
import torch
from datetime import datetime

class TrainingConfig:
    # Training Parameters
    MAX_EPOCHS = 50  # Reduced for 1-week timeline
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 1000

    # Optimization
    OPTIMIZER = "adamw"
    SCHEDULER = "cosine"
    GRADIENT_CLIP_NORM = 1.0
    ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS

    # Loss Weights
    RECONSTRUCTION_WEIGHT = 1.0
    PHYSICS_WEIGHT = 0.1
    DEFENSE_WEIGHT = 0.15
    KL_WEIGHT = 0.001

    # Validation
    VALIDATION_FREQUENCY = 500  # Every N steps
    SAVE_FREQUENCY = 1000  # Every N steps
    MAX_CHECKPOINTS = 5

    # Data Loading
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2

    # Augmentation
    USE_AUGMENTATION = True
    AUGMENTATION_PROBABILITY = 0.7
    DEFENSE_SCENARIO_PROBABILITY = 0.5  # 50% of batches include defense scenarios

    # Logging & Monitoring
    USE_WANDB = False  # Disabled for quick test
    WANDB_PROJECT = "enhanced-hsigene"
    LOG_FREQUENCY = 100  # Every N steps

    # Paths
    DATA_ROOT = "data/raw_datasets"
    OUTPUT_DIR = "outputs"
    MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "models")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

    # Resume Training
    RESUME_FROM_CHECKPOINT = None
    AUTO_RESUME = True

    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MULTI_GPU = False  # Set to True if multiple GPUs available

    # Mixed Precision setting (added to avoid attribute error)
    MIXED_PRECISION = False

    # Experiment Tracking
    EXPERIMENT_NAME = f"enhanced_hsigene_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        dirs = [cls.OUTPUT_DIR, cls.MODEL_SAVE_DIR, cls.LOG_DIR, cls.FIGURE_DIR]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith('__') and not callable(v)}