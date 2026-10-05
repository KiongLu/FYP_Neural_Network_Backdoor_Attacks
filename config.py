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
    wanet_s = 0.8                      # warping strength
    wanet_k = 4                        # control grid size
    wanet_grid_rescale = 1.0
    wanet_cross_ratio = 2.0            # ratio of "noise-mode" (cover) samples to poison samples

    # Fixed hybrid attacks ("blend_sig", "blend_wanet", "sig_wanet", "blend_sig_wanet") reuse
    # the settings above, with every component scaled to 1/N of its solo intensity
    # (N = number of components combined) -- see attacks/hybrid.py.

    # ---------------- Adaptive Hybrid attack (PROPOSED/EXPERIMENTAL) ----------------
    # See attacks/adaptive_hybrid.py for the full mathematical formulation, status, and
    # label-semantics discussion. Attack names: "adaptive_hybrid" (both components),
    # "adaptive_spatial" (warp only, ablation), "adaptive_freq" (signal only, ablation).
    #
    # alpha(x) [spatial warp strength] and beta(x) [sinusoidal signal amplitude] are each
    # linearly interpolated between a min/max bound according to a per-image statistic,
    # normalized against a calibration range (see calibrate_adaptive_stats.py).
    #
    # IMPORTANT: adaptive_alpha_min must stay comfortably ABOVE adaptive_cover_alpha (below).
    # If the poison strength range overlaps the cover-sample strength, cover and poisoned
    # samples become nearly indistinguishable in MAGNITUDE for images whose alpha(x) lands
    # near the overlap, leaving only the (much harder to learn) field PATTERN as a
    # distinguishing signal -- this was diagnosed as the likely cause of a very low ASR for
    # adaptive_spatial (see FYP report, Discussion 6.7) even after recalibrating the
    # variance/edge statistics, and is why alpha_min/alpha_max sit well above cover_alpha
    # here, unlike the first version of this config.
    adaptive_alpha_min = 0.45    # min spatial-warp strength (same units as wanet_s)
    adaptive_alpha_max = 0.80    # max spatial-warp strength
    adaptive_beta_min = 6.0      # min sinusoidal-signal amplitude (same units as sig_delta)
    adaptive_beta_max = 22.0     # max sinusoidal-signal amplitude

    # Calibration constants for the two per-image statistics (pixel std, edge strength)
    # used to compute alpha(x)/beta(x), for CIFAR-10 images normalized to [0,1]. These are
    # the 5th/95th percentiles measured by `python calibrate_adaptive_stats.py` on 2,000
    # training images; re-run it and replace these four values if you change dataset.
    adaptive_v_min = 0.1256        # lower bound of per-image pixel std (5th percentile)
    adaptive_v_max = 0.3025        # upper bound of per-image pixel std (95th percentile)
    adaptive_e_min = 0.2672        # lower bound of mean Sobel edge magnitude (5th percentile)
    adaptive_e_max = 0.7900        # upper bound of mean Sobel edge magnitude (95th percentile)

    adaptive_cover_alpha = 0.20  # FIXED (non-adaptive) spatial-warp strength used for this
                                  # attack's WaNet-style "noise mode" cover samples -- kept
                                  # well BELOW adaptive_alpha_min (see note above) so poison
                                  # and cover samples are always clearly separated in
                                  # magnitude, not just in field pattern. See
                                  # attacks/adaptive_hybrid.py for why this must be fixed
                                  # rather than adaptive.

    # ---------------- Adaptive SIG+WaNet v2 ("adaptive_hybrid" mode ONLY) ----------------
    # Stealth-improved settings used ONLY by the full "adaptive_hybrid" attack. The ablation
    # variants ("adaptive_spatial", "adaptive_freq") deliberately keep using the original
    # adaptive_alpha_*/adaptive_beta_*/adaptive_cover_alpha values above, so their existing
    # checkpoints and results are unaffected.
    #
    # Two changes vs. the original adaptive_hybrid (which had ~0.887 SSIM / ~25.9 dB PSNR,
    # worse than Fixed SIG+WaNet's ~0.938 / ~29.1 dB):
    #   1. Rebalanced per-image strength ranges: the spatial warp is reduced (it costs a lot
    #      of PSNR but contributes little to ASR -- adaptive_spatial alone only reaches ~7%
    #      ASR at 0.5%) and the budget is moved to the sinusoidal signal, which drives ASR.
    #   2. Per-PIXEL texture masking of the sinusoidal signal: the per-image amplitude
    #      beta(x) is further multiplied by a local-texture mask m(x) in [floor, 1], so the
    #      signal is strongest in busy regions (where it is masked by existing texture) and
    #      weakest in flat regions (where SSIM/human vision are most sensitive to it).
    #      See attacks/adaptive_hybrid.py ("Texture masking") for the formulation.
    adaptive_hybrid_alpha_min = 0.20
    adaptive_hybrid_alpha_max = 0.35
    adaptive_hybrid_beta_min = 10.0
    adaptive_hybrid_beta_max = 18.0
    adaptive_hybrid_cover_alpha = 0.08  # kept well below adaptive_hybrid_alpha_min (see above)

    adaptive_texture_mask = True        # False = no per-pixel masking (uniform signal)
    adaptive_mask_ref = 0.10            # local std (gray, [0,1] scale) at which mask reaches 1.0
    adaptive_mask_floor = 0.55          # minimum mask value -- signal is never fully removed,
                                        # so the same sinusoidal pattern is present everywhere
    adaptive_mask_window = 5            # Gaussian window size for the local std statistic
    adaptive_mask_sigma = 1.5           # Gaussian window std-dev

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

        return f"{self.dataset_name}_{self.model_name}_{self.attack_name}{suffix}_pr{self.poison_rate}"


cfg = Config()