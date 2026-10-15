"""
All experiment settings live here. Most of them can be changed from the command line
(see train.py / evaluate.py), so this file mainly holds the defaults.
"""

import os
import torch


class Config:
    # general
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4

    # data: "cifar10" (default) or "gtsrb" (resized to 32x32, see datasets/gtsrb.py) - use set_dataset()
    dataset_name = "cifar10"
    data_root = "./data"
    num_classes = 10
    image_size = 32
    batch_size = 128
    NUM_CLASSES = {"cifar10": 10, "gtsrb": 43}
    # per-channel mean and std used for normalisation (GTSRB values measured on its 32x32 training set)
    NORMALIZATION = {"cifar10": ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
                     "gtsrb": ([0.3417, 0.3126, 0.3216], [0.2737, 0.2607, 0.2662])}

    # model ("resnet18" or "vgg16")
    model_name = "resnet18"

    # training
    epochs = 100
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    lr_schedule = [50, 75]             # lr is divided by 10 at these epochs
    lr_gamma = 0.1

    # backdoor attack - see train.py --attack for the full list of names
    attack_name = "blend"
    target_label = 0                   # airplane on CIFAR-10, the 20 km/h sign on GTSRB
    poison_rate = 0.05                 # fraction of the whole training set that gets poisoned

    # Blend
    blend_alpha = 0.15
    blend_trigger_path = "./triggers/hello_kitty.png"   # any RGB image works, it gets resized to 32x32

    # SIG
    sig_delta = 20                     # amplitude on the 0-255 scale
    sig_freq = 6
    # True: clean-label, as in the original paper - only target-class images are poisoned and their
    # labels stay the same. False: dirty-label - any image can be poisoned and is relabelled as the target,
    # like Blend and WaNet. Use --sig_dirty_label to switch.
    sig_clean_label = True

    # WaNet
    wanet_s = 0.8                      # warp strength
    wanet_k = 4                        # size of the random control grid
    wanet_grid_rescale = 1.0
    wanet_cross_ratio = 2.0            # number of cover (noise-mode) samples per poisoned sample

    # The fixed hybrids (blend_sig, blend_wanet, sig_wanet, blend_sig_wanet) use the settings above,
    # with each part scaled down by 1/N where N is the number of parts.

    # "sig_wanet_matched": the same fixed SIG+WaNet hybrid, but with each part set to v2's average
    # strength instead of 1/N. The averages (warp 0.27, signal 10.6 after masking) were measured in
    # notebooks/data_exploration.ipynb. Comparing it with v2 tells us how much of v2's result comes
    # from the adaptive part and how much just from moving strength from the warp to the signal.
    sig_wanet_matched_s = 0.27
    sig_wanet_matched_delta = 10.6

    # Adaptive hybrid - my proposed attack (attacks/adaptive_hybrid.py).
    # Attack names: "adaptive_hybrid" (warp + signal), and the two ablations "adaptive_spatial"
    # (warp only) and "adaptive_freq" (signal only).
    # Each image gets its own warp strength alpha(x) and signal amplitude beta(x), interpolated
    # between the min/max values below using the image's pixel std and edge strength.
    #
    # These first values are the original version (v1). The two ablations still use them.
    # alpha_min has to stay well above the cover strength further down, otherwise some poisoned
    # images are warped about as much as the cover images and the model can only tell them apart
    # by the warp pattern. That was the reason for the very low adaptive_spatial ASR in my first
    # config, which had the ranges overlapping.
    adaptive_alpha_min = 0.45
    adaptive_alpha_max = 0.80
    adaptive_beta_min = 6.0            # same units as sig_delta
    adaptive_beta_max = 22.0

    # Range of the two image statistics, used to normalise them to [0, 1].
    # 5th and 95th percentiles from calibrate_adaptive_stats.py (2,000 training images).
    # For another dataset, re-run it and pass the values with --set (see README).
    adaptive_v_min = 0.1256            # pixel std
    adaptive_v_max = 0.3025
    adaptive_e_min = 0.2672            # mean Sobel edge magnitude
    adaptive_e_max = 0.7900

    # Warp strength for the cover samples. It is the same for every cover image on purpose -
    # see the noise_mode comment in attacks/adaptive_hybrid.py.
    adaptive_cover_alpha = 0.20

    # v2 settings - only used by the full "adaptive_hybrid" attack, so the ablation models are
    # not affected. v1 was worse than the fixed hybrid on stealth (about 0.900 SSIM / 26.6 dB
    # against 0.943 / 29.6 dB), so v2 changes two things:
    #   1. weaker warp, stronger signal - the warp costs a lot of image quality but adds very
    #      little ASR (adaptive_spatial alone only gets ~7% ASR at 0.5% poisoning)
    #   2. a per-pixel texture mask on the signal, so it is full strength in busy areas and
    #      weaker in flat areas where it would be easy to see
    adaptive_hybrid_alpha_min = 0.20
    adaptive_hybrid_alpha_max = 0.35
    adaptive_hybrid_beta_min = 10.0
    adaptive_hybrid_beta_max = 18.0
    adaptive_hybrid_cover_alpha = 0.08  # again kept well below alpha_min

    adaptive_texture_mask = True        # set to False to switch the mask off
    adaptive_mask_ref = 0.10            # local std at which the mask reaches 1
    adaptive_mask_floor = 0.55          # the mask never goes below this, so the signal is never removed
    adaptive_mask_window = 5            # Gaussian window used for the local std
    adaptive_mask_sigma = 1.5

    # output folders
    checkpoint_dir = "./checkpoints"
    results_dir = "./results"

    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    # Run variants. They are all off by default and only add something to the run name when they
    # are used, so the names of the normal runs stay the same.
    DEFAULT_SEED = 42
    augment = False             # random crop + flip after poisoning                 -> "_aug"
    augment_mode = "crop_flip"  # or "crop" / "flip" to use only one of them         -> "_augcrop" / "_augflip"
    holdout = 0                 # number of training images kept back for validation -> "_val<N>"
    tag_suffix = ""             # extra text for the run name, e.g. to label --set candidates

    def set_dataset(self, name):
        self.dataset_name = name
        self.num_classes = self.NUM_CLASSES[name]

    def tag(self):
        """Name used for the checkpoint and log files of this run."""
        suffix = ""
        if self.attack_name.lower() == "sig":
            suffix = "_clean" if self.sig_clean_label else "_dirty"
        if self.seed != self.DEFAULT_SEED:
            suffix += f"_s{self.seed}"
        if self.augment:
            suffix += "_aug" if self.augment_mode == "crop_flip" else f"_aug{self.augment_mode}"
        if self.holdout:
            suffix += f"_val{self.holdout}"
        suffix += self.tag_suffix

        return f"{self.dataset_name}_{self.model_name}_{self.attack_name}{suffix}_pr{self.poison_rate}"

    def apply_overrides(self, assignments):
        """Apply --set name=value options. Values are read as Python literals when possible."""
        import ast
        for item in assignments or []:
            key, value = item.split("=", 1)
            if not hasattr(self, key):
                raise ValueError(f"Unknown config setting: {key}")
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass          # not a literal, keep it as a string
            setattr(self, key, value)


cfg = Config()
