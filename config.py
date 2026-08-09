"""
Central configuration for backdoor attack experiments.
"""

import os
import torch


class Config:
    # ---------------- General ----------------
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # ---------------- Dataset ----------------
    dataset_name = "cifar10"           # "cifar10" or "gtsrb"
    data_root = "./data"
    num_classes = 10
    image_size = 32
    batch_size = 128

    # ---------------- Model ----------------
    model_name = "resnet18"

    # ---------------- Training ----------------
    epochs = 100
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    lr_schedule = [50, 75]             # epochs at which lr is divided by 10
    lr_gamma = 0.1

    # ---------------- Backdoor attack ----------------
    attack_name = "blend"              # "blend", "sig", "wanet"
    target_label = 0                   # single-target attack
    poison_rate = 0.05                 # fraction of TRAIN set poisoned (0.01 / 0.05 / 0.10)

    # Blend attack
    blend_alpha = 0.15
    blend_trigger_path = "./triggers/hello_kitty.png"   # any RGB image, resized to image_size

    # SIG attack (sinusoidal signal)
    sig_delta = 20                     # amplitude
    sig_freq = 6                       # frequency

    # WaNet attack
    wanet_s = 0.5                      # warping strength
    wanet_k = 4                        # control grid size
    wanet_grid_rescale = 1.0
    wanet_cross_ratio = 2.0            # ratio of "noise-mode" (cover) samples to poison samples

    # ---------------- Paths ----------------
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"

    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def tag(self):
        """Unique identifier for a given experiment configuration."""
        return f"{self.dataset_name}_{self.model_name}_{self.attack_name}_pr{self.poison_rate}"


cfg = Config()
