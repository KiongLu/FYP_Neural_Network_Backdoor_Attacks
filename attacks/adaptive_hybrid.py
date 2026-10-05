"""
Adaptive Hybrid Invisible Backdoor Attack (PROPOSED / EXPERIMENTAL)
====================================================================
STATUS: This is NOT a published, peer-reviewed attack. It is an original
experimental combination implemented for this FYP, built from two existing
published techniques:
  - WaNet (Nguyen & Tran, "WaNet -- Imperceptible Warping-based Backdoor
    Attack", ICLR 2021)                                  -- spatial warp
  - SIG   (Barni et al., "A New Backdoor Attack in CNNs by Training Set
    Corruption Without Label Poisoning", ICIP 2019)       -- sinusoidal signal

Do NOT present "combining SIG and WaNet" as this project's novel
contribution -- that is already covered by the existing FIXED-strength
Blend/SIG/WaNet hybrid in attacks/hybrid.py (attack name "sig_wanet").
The thing under test here, and the only part of this file that is not a
direct reproduction of prior work, is INPUT-ADAPTIVE weighting: instead of
applying both triggers at one fixed strength to every poisoned image, each
image's own spatial-warp strength alpha(x) and sinusoidal-signal strength
beta(x) are computed from simple, cheap image statistics and bounded to a
configurable range. The research question this attack exists to test:

    Does input-adaptive trigger-strength weighting improve the
    stealth/effectiveness trade-off compared to the existing FIXED-strength
    hybrid (attack "sig_wanet"), at the same nominal poison_rate?

See the FYP report (Methodology, "Adaptive Hybrid" section) for the full
write-up, and calibrate_adaptive_stats.py for recalibrating the statistics
below on a different dataset.

Label semantics (see also datasets/poison_dataset.py, config.py)
------------------------------------------------------------------
Dirty-label, unconditionally: ANY class may be poisoned, and the label is
always flipped to target_label. This mirrors the existing HybridAttack's
design decision (attacks/hybrid.py) for the same reason: alpha(x)/beta(x)
are computed from generic image statistics unrelated to class identity, so
restricting to target-class-only would conflate "clean-label" with
"adaptive" and make it impossible to isolate which property drives any
observed effect. This does NOT change the semantics of the existing solo
SIG or WaNet attacks, which are untouched by this file.

Mathematical formulation
-------------------------
Let x be a clean image tensor in [0,1], shape (C,H,W).

Statistic 1 -- global busyness (drives the spatial-warp strength):
    v(x)      = std(x) over all pixels                          (scalar)
    v_norm(x) = clip((v(x) - V_MIN) / (V_MAX - V_MIN), 0, 1)

Statistic 2 -- edge strength (drives the sinusoidal-signal strength):
    e(x)      = mean( sqrt(Gx(gray(x))^2 + Gy(gray(x))^2) )
                using fixed 3x3 Sobel kernels                    (scalar)
    e_norm(x) = clip((e(x) - E_MIN) / (E_MAX - E_MIN), 0, 1)

V_MIN/V_MAX, E_MIN/E_MAX are calibration constants (config.py), measured on
CIFAR-10 by calibrate_adaptive_stats.py; recalibrate if you change dataset.

Adaptive strengths:
    alpha(x) = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * v_norm(x)
    beta(x)  = BETA_MIN  + (BETA_MAX  - BETA_MIN)  * e_norm(x)

Trigger application (spatial warp first, then additive signal -- matches
the component ordering already used by the project's Fixed hybrid):
    grid(x)  = clip((identity_grid + alpha(x) * noise_grid / image_size)
                     * grid_rescale, -1, 1)
    x_warped = grid_sample(x, grid(x))
    x_hybrid = clip(x_warped + (beta(x) / 255) * sig_pattern, 0, 1)

Texture masking (v2, "adaptive_hybrid" mode only)
--------------------------------------------------
The original per-image strengths (alpha in [0.45,0.80], beta in [6,22])
were on average stronger than the Fixed SIG+WaNet hybrid (s=0.40,
delta=10), giving worse SSIM/PSNR. The full "adaptive_hybrid" mode now
uses rebalanced ranges (config.adaptive_hybrid_*): a weaker warp, which
costs PSNR but adds little ASR, and a stronger signal, which drives ASR.
It also masks the sinusoidal signal
PER PIXEL by local texture:
    s(x)      = local std of gray(x) over a Gaussian window (5x5, sigma 1.5)
    m(x)      = clip(s(x) / MASK_REF, MASK_FLOOR, 1)               (1,H,W)
    x_hybrid  = clip(x_warped + (beta(x) / 255) * m(x) * sig_pattern, 0, 1)
So the per-image adaptation (beta(x)) is refined to a per-pixel one: the
signal keeps its full amplitude in textured regions, where it is masked by
existing content, and is attenuated (never below MASK_FLOOR) in flat
regions, where SSIM and human vision are most sensitive to an added
periodic pattern. The sinusoid's frequency/phase are unchanged, so the
trigger pattern the model learns is the same. The ablation modes
("adaptive_spatial", "adaptive_freq") keep the original settings and no
mask, so their results are unchanged.

identity_grid, noise_grid, and sig_pattern are each FIXED ONCE at
construction (the same field/pattern is reused for every poisoned image --
required for the backdoor to be learnable at all). Only the per-image
STRENGTHS alpha(x)/beta(x) vary; the trigger's spatial/frequency PATTERN
does not.

Rationale (perceptual masking -- an assumption this project's ablation
tests, not an established fact for this application): human perceptual
sensitivity to an added perturbation is reduced in visually "busy" content.
Higher v(x) is used as a proxy for how much a geometric warp can be masked;
higher e(x) is used as a proxy for how much an additive periodic signal can
be masked. This masking intuition is standard in image-watermarking
literature (not a contribution of this project); applying it to backdoor
trigger strength, and measuring whether it helps, is what this project
investigates.
"""

import torch
import torch.nn.functional as F


class AdaptiveHybridAttack:
    MODES = {
        "adaptive_hybrid": "both",
        "adaptive_spatial": "spatial_only",
        "adaptive_freq": "freq_only",
    }

    def __init__(self, cfg, mode: str = "both"):
        assert mode in ("both", "spatial_only", "freq_only"), f"Unknown adaptive mode: {mode}"
        self.mode = mode
        self.image_size = cfg.image_size
        self.target_label = cfg.target_label
        self.k = cfg.wanet_k
        self.grid_rescale = cfg.wanet_grid_rescale
        self.freq = cfg.sig_freq
        self.device = "cpu"  # see attacks/wanet.py: must stay CPU (DataLoader workers)

        self.v_min, self.v_max = cfg.adaptive_v_min, cfg.adaptive_v_max
        self.e_min, self.e_max = cfg.adaptive_e_min, cfg.adaptive_e_max

        if mode == "both":
            self.alpha_min, self.alpha_max = cfg.adaptive_hybrid_alpha_min, cfg.adaptive_hybrid_alpha_max
            self.beta_min, self.beta_max = cfg.adaptive_hybrid_beta_min, cfg.adaptive_hybrid_beta_max
            self.cover_alpha = cfg.adaptive_hybrid_cover_alpha
            self.use_texture_mask = cfg.adaptive_texture_mask
        else:
            # Ablation variants keep the original settings so their results are unchanged.
            self.alpha_min, self.alpha_max = cfg.adaptive_alpha_min, cfg.adaptive_alpha_max
            self.beta_min, self.beta_max = cfg.adaptive_beta_min, cfg.adaptive_beta_max
            self.cover_alpha = cfg.adaptive_cover_alpha
            self.use_texture_mask = False

        self.mask_ref = cfg.adaptive_mask_ref
        self.mask_floor = cfg.adaptive_mask_floor
        self.mask_window = cfg.adaptive_mask_window
        self.mask_kernel = self._make_gaussian_kernel(cfg.adaptive_mask_window, cfg.adaptive_mask_sigma)

        self.identity_grid, self.noise_grid = self._make_grids()

        # Fixed sinusoidal PATTERN at unit amplitude; actual amplitude
        # beta(x) is applied per-image at poison() time.
        self.sig_pattern_unit = self._make_unit_sig_pattern()

        # Fixed Sobel kernels for the edge-strength statistic (not learned)
        self.sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

    # ---------------------------------------------------------------- #
    #  Fixed trigger components (constructed once)
    # ---------------------------------------------------------------- #
    def _make_grids(self):
        xs = torch.linspace(-1, 1, self.image_size)
        ys = torch.linspace(-1, 1, self.image_size)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        identity_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,H,W,2)

        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))  # matches the corrected WaNet normalization
        noise_grid = F.interpolate(ins, size=(self.image_size, self.image_size),
                                    mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)  # (1,H,W,2)
        return identity_grid, noise_grid

    def _make_unit_sig_pattern(self):
        j = torch.arange(self.image_size).float()
        pattern_1d = torch.sin(2 * torch.pi * j * self.freq / self.image_size)
        pattern = pattern_1d.unsqueeze(0).repeat(self.image_size, 1)   # (H,W)
        pattern = pattern.unsqueeze(0).repeat(3, 1, 1)                  # (C,H,W), amplitude 1.0
        return pattern

    @staticmethod
    def _make_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(size).float() - size // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        return (g.unsqueeze(1) @ g.unsqueeze(0)).view(1, 1, size, size)

    # ---------------------------------------------------------------- #
    #  Per-image adaptive statistics
    # ---------------------------------------------------------------- #
    def _variance_stat(self, img: torch.Tensor) -> float:
        return img.std().item()

    def _edge_stat(self, img: torch.Tensor) -> float:
        """
        NOTE (known limitation, not a bug): Sobel-based gradient magnitude,
        like any local finite-difference edge detector, has near-zero
        response to content at exactly the Nyquist frequency (e.g. an exact
        single-pixel checkerboard) due to cancellation in the kernel -- see
        docs/adaptive_hybrid_notes.md. This does not meaningfully affect
        natural photographic images (CIFAR-10), which essentially never
        contain single-pixel-period patterns, but is worth stating as a
        limitation of the edge-strength statistic specifically.
        """
        gray = img.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
        # Replicate-pad (not zero-pad) before convolving: zero-padding would
        # introduce a spurious strong "edge" at the image border for any
        # image whose border pixels aren't already ~0, which dominates the
        # statistic for otherwise-flat/low-texture images.
        gray_padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(gray_padded, self.sobel_x, padding=0)
        gy = F.conv2d(gray_padded, self.sobel_y, padding=0)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return mag.mean().item()

    def compute_alpha_beta(self, img: torch.Tensor):
        """Public helper (used by inspect scripts / calibration) -- returns (alpha, beta)."""
        v = self._variance_stat(img)
        e = self._edge_stat(img)
        v_norm = min(max((v - self.v_min) / (self.v_max - self.v_min), 0.0), 1.0)
        e_norm = min(max((e - self.e_min) / (self.e_max - self.e_min), 0.0), 1.0)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * v_norm
        beta = self.beta_min + (self.beta_max - self.beta_min) * e_norm
        return alpha, beta

    def texture_mask(self, img: torch.Tensor) -> torch.Tensor:
        """
        Per-pixel texture mask m(x) in [mask_floor, 1], shape (1,H,W).
        Local std of the grayscale image over a small Gaussian window,
        divided by mask_ref and clipped. Replicate-padded for the same
        reason as _edge_stat (no spurious border response).
        """
        gray = img.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
        p = self.mask_window // 2
        gray_padded = F.pad(gray, (p, p, p, p), mode="replicate")
        mu = F.conv2d(gray_padded, self.mask_kernel)
        mu_sq = F.conv2d(gray_padded ** 2, self.mask_kernel)
        local_std = (mu_sq - mu ** 2).clamp_min(0).sqrt().squeeze(0)
        return (local_std / self.mask_ref).clamp(self.mask_floor, 1.0)

    # ---------------------------------------------------------------- #
    #  Attack interface (matches BlendAttack / SIGAttack / WaNetAttack)
    # ---------------------------------------------------------------- #
    def poison(self, img: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.compute_alpha_beta(img)
        out = img

        if self.mode in ("both", "spatial_only"):
            grid = self.identity_grid + alpha * self.noise_grid / self.image_size
            grid = torch.clamp(grid * self.grid_rescale, -1, 1)
            out = F.grid_sample(out.unsqueeze(0), grid, align_corners=True,
                                 mode="bilinear", padding_mode="reflection").squeeze(0)

        if self.mode in ("both", "freq_only"):
            signal = (beta / 255.0) * self.sig_pattern_unit
            if self.use_texture_mask:
                signal = signal * self.texture_mask(img)
            out = out + signal

        return torch.clamp(out, 0.0, 1.0)

    def noise_mode(self, img: torch.Tensor) -> torch.Tensor:
        """
        WaNet-style cover sample: a FRESH random spatial field at a FIXED
        (non-adaptive) cover strength, no sinusoidal component, label always
        unchanged. Fixed strength (not alpha(x)) is deliberate: cover
        samples must not correlate with any per-image statistic, since their
        only purpose is teaching the model that generic warping alone
        (independent of image content) does not trigger the backdoor.
        Only relevant when this attack's spatial component is active
        ("both" or "spatial_only" mode) -- "freq_only" has no warp to cover.
        """
        if self.mode == "freq_only":
            return img
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        random_field = F.interpolate(ins, size=(self.image_size, self.image_size),
                                      mode="bicubic", align_corners=True).permute(0, 2, 3, 1)
        grid = self.identity_grid + self.cover_alpha * random_field / self.image_size
        grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        warped = F.grid_sample(img.unsqueeze(0), grid, align_corners=True,
                                mode="bilinear", padding_mode="reflection").squeeze(0)
        return torch.clamp(warped, 0.0, 1.0)

    def is_poisonable(self, label: int) -> bool:
        return True  # dirty-label -- see module docstring for the explicit design rationale

    def get_target_label(self, original_label: int) -> int:
        return self.target_label

    def describe(self) -> str:
        mask = (f"texture mask ref={self.mask_ref}, floor={self.mask_floor}"
                if self.use_texture_mask else "no texture mask")
        return (f"AdaptiveHybrid(mode={self.mode}, "
                f"alpha in [{self.alpha_min},{self.alpha_max}], "
                f"beta in [{self.beta_min},{self.beta_max}], "
                f"cover_alpha={self.cover_alpha}, {mask}, "
                f"calibration: v in [{self.v_min},{self.v_max}], "
                f"e in [{self.e_min},{self.e_max}])")
