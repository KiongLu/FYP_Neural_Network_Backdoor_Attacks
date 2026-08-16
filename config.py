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
    dataset_name = "cifar10"           # "cifar10"
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

    # ---------------- Hybrid attack tuning (optional) ----------------
    # By default, each attack in a hybrid is scaled equally.
    # For example:
    #   - 2 attacks -> each uses 50% of its solo strength
    #   - 3 attacks -> each uses 33% of its solo strength
    #
    # To make one attack stronger or weaker in a hybrid, use this dictionary.
    # The value is the percentage of that attack's FULL solo strength.
    #
    # Example:
    #   hybrid_component_scale = {"wanet": 0.75}
    #
    # This makes WaNet use 75% of its normal solo strength whenever it is
    # included in a hybrid attack. Other attacks still use the default 1/N scale.
    # Set to None to use the default equal scaling.
    hybrid_component_scale = None      # None, or e.g. {"wanet": 0.75}. Also settable via
                                        # --hybrid_wanet_scale FLOAT on the command line
                                        # (shortcut for the common case of just boosting WaNet).
    
    # ---------------- Adaptive Hybrid attack (PROPOSED) ----------------
    # See attacks/adaptive_hybrid.py for the full implementation.
    # Attack options:
    #   - "adaptive_hybrid"  -> uses both spatial warping and frequency signal
    #   - "adaptive_spatial" -> uses spatial warping only (ablation)
    #   - "adaptive_freq"    -> uses frequency signal only (ablation)
    #
    # The attack strength changes for each image based on its image statistics.
    #   - alpha(x) -> controls spatial warping strength
    #   - beta(x)  -> controls sinusoidal signal strength
    #
    # Each value is scaled between its minimum and maximum strength using
    # a normalized image statistic. The calibration range is obtained from
    # calibrate_adaptive_stats.py.
    #
    # IMPORTANT:
    # adaptive_alpha_min should be well above adaptive_cover_alpha.
    #
    # The cover samples use adaptive_cover_alpha as their warping strength.
    # If poisoned samples use a similar strength, the model may not be able
    # to distinguish poisoned images from cover images, resulting in low ASR.
    #
    # Therefore, keep alpha_min and alpha_max clearly above cover_alpha.
     
    adaptive_alpha_min = 0.45    # min spatial-warp strength (same units as wanet_s)
    adaptive_alpha_max = 0.80    # max spatial-warp strength
    adaptive_beta_min = 6.0      # min sinusoidal-signal amplitude (same units as sig_delta)
    adaptive_beta_max = 22.0     # max sinusoidal-signal amplitude

    # Calibration constants for the two per-image statistics (variance, edge strength)
    # used to compute alpha(x)/beta(x). These are APPROXIMATE, UNVERIFIED defaults for
    # CIFAR-10 images normalized to [0,1] -- NOT measured from the actual dataset. Run
    # `python calibrate_adaptive_stats.py` once and replace these four values before
    # treating adaptive-attack results as final; using un-calibrated defaults is fine for
    # a first smoke-test run but should be flagged as a limitation if left uncalibrated in
    # your report.
    adaptive_v_min = 0.1256        # approx. lower bound of per-image pixel std
    adaptive_v_max = 0.3025        # approx. upper bound of per-image pixel std
    adaptive_e_min = 0.2672        # approx. lower bound of mean Sobel edge magnitude
    adaptive_e_max = 0.7900        # approx. upper bound of mean Sobel edge magnitude

    adaptive_cover_alpha = 0.20  # FIXED (non-adaptive) spatial-warp strength used for this
                                  # attack's WaNet-style "noise mode" cover samples -- kept
                                  # well BELOW adaptive_alpha_min (see note above) so poison
                                  # and cover samples are always clearly separated in
                                  # magnitude, not just in field pattern. See
                                  # attacks/adaptive_hybrid.py for why this must be fixed
                                  # rather than adaptive.

    # ---------------- Paths ----------------
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"

    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def tag(self):
        """Unique identifier for a given experiment configuration."""
        suffix = ""
        if self.attack_name.lower() == "sig":
            suffix = "_clean" if self.sig_clean_label else "_dirty"

        hybrid_suffix = "_customw" if self.hybrid_component_scale else ""

        return f"{self.dataset_name}_{self.model_name}_{self.attack_name}{suffix}{hybrid_suffix}_pr{self.poison_rate}"


cfg = Config()