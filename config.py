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
    attack_name = "blend"              # "clean", "blend", "sig", "wanet",
                                        # or hybrid: "blend_sig", "blend_wanet",
                                        # "sig_wanet", "blend_sig_wanet"
    target_label = 0                   # single-target attack
    poison_rate = 0.05                 # fraction of TRAIN set poisoned (0.01 / 0.05 / 0.10)

    # Blend attack
    blend_alpha = 0.15
    blend_trigger_path = "./triggers/hello_kitty.png"   # any RGB image, resized to image_size
    blend_clean_label = False          # False (default, ORIGINAL behaviour) = dirty-label:
                                        #   poisons any class, flips label to target.
                                        # True = clean-label variant: only poisons images
                                        #   already in the target class, label unchanged.
                                        #   Toggle via --blend_clean_label.

    # SIG attack (sinusoidal signal)
    sig_delta = 20                     # amplitude
    sig_freq = 6                       # frequency
    sig_clean_label = True             # True  = clean-label SIG (original Barni et al. design):
                                        #         only poisons images already in the target class,
                                        #         label is NEVER changed. More stealthy to label
                                        #         audits, but less sample-efficient (lower ASR at
                                        #         the same poison_rate).
                                        # False = dirty-label SIG: poisons images from ANY class
                                        #         and flips the label to target, same mechanism as
                                        #         Blend/WaNet. Less stealthy (label flips exist) but
                                        #         much higher ASR at the same poison_rate.
                                        # Toggle via --sig_dirty_label on the command line, or set
                                        # this directly.

    # WaNet attack
    wanet_s = 0.5                      # warping strength
    wanet_k = 4                        # control grid size
    wanet_grid_rescale = 1.0
    wanet_cross_ratio = 2.0            # ratio of "noise-mode" (cover) samples to poison samples
    wanet_clean_label = False          # False (default, ORIGINAL behaviour) = dirty-label:
                                        #   poisons any class, flips label to target.
                                        # True = clean-label variant: only poisons images
                                        #   already in the target class, label unchanged.
                                        #   Toggle via --wanet_clean_label.

    # ---------------- Hybrid attack tuning (all optional, off by default) ----------------
    # By default every component in a hybrid attack is scaled to 1/N of its solo intensity
    # (N = number of components combined) -- this is the ORIGINAL, backward-compatible
    # behaviour and is what every hybrid checkpoint you've already trained used.
    #
    # If you want to rebalance a hybrid (e.g. WaNet is the weakest solo attacker, so giving
    # it relatively more intensity than Blend/SIG within the hybrid may raise the hybrid's
    # ASR), set hybrid_component_scale to a dict overriding specific components' scale
    # factor (as a fraction of that attack's FULL solo intensity, not of 1/N). Any component
    # NOT listed still falls back to the default uniform 1/N scale.
    #
    # Example: hybrid_component_scale = {"wanet": 0.75}  -> in EVERY hybrid that includes
    # WaNet, WaNet runs at 75% of its full solo warping strength (much stronger than the
    # default 1/N split), while Blend/SIG (if present) still use the default 1/N each.
    #
    # Leave as None to reproduce the exact original uniform-scaling behaviour.
    hybrid_component_scale = None      # None, or e.g. {"wanet": 0.75}. Also settable via
                                        # --hybrid_wanet_scale FLOAT on the command line
                                        # (shortcut for the common case of just boosting WaNet).

    # ---------------- Paths ----------------
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"

    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def tag(self):
        """Unique identifier for a given experiment configuration."""
        name = self.attack_name.lower()
        suffix = ""
        if name == "sig":
            suffix = "_clean" if self.sig_clean_label else "_dirty"
        elif name == "blend" and self.blend_clean_label:
            suffix = "_clean"
        elif name == "wanet" and self.wanet_clean_label:
            suffix = "_clean"

        hybrid_suffix = "_customw" if self.hybrid_component_scale else ""

        return f"{self.dataset_name}_{self.model_name}_{self.attack_name}{suffix}{hybrid_suffix}_pr{self.poison_rate}"


cfg = Config()
