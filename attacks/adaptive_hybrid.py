"""
Adaptive Hybrid Invisible Backdoor Attack
"""

import torch
import torch.nn.functional as F


class AdaptiveHybridAttack:
    # Maps a CLI-facing attack name to which components are active.
    # "adaptive_hybrid" = both components (the actual proposed method).
    # "adaptive_spatial" / "adaptive_freq" = ablation-only variants (STEP:
    # ablation items 5-6) that isolate whether the adaptive MECHANISM itself
    # (as opposed to which components it's applied to) contributes anything.
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

        self.alpha_min, self.alpha_max = cfg.adaptive_alpha_min, cfg.adaptive_alpha_max
        self.beta_min, self.beta_max = cfg.adaptive_beta_min, cfg.adaptive_beta_max
        self.v_min, self.v_max = cfg.adaptive_v_min, cfg.adaptive_v_max
        self.e_min, self.e_max = cfg.adaptive_e_min, cfg.adaptive_e_max
        self.cover_alpha = cfg.adaptive_cover_alpha

        # Fixed spatial-warp field -- SAME field reused for every poisoned
        # image; only its STRENGTH alpha(x) varies per image, not the field
        # itself (the backdoor needs one consistent pattern to be learnable).
        self.identity_grid, self.noise_grid = self._make_grids()

        # Fixed sinusoidal PATTERN at unit amplitude; actual amplitude
        # beta(x) is applied per-image at poison() time.
        self.sig_pattern_unit = self._make_unit_sig_pattern()

        # Fixed Sobel kernels for the edge-strength statistic (not learned)
        self.sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

    #  Fixed trigger components (constructed once)
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

    #  Per-image adaptive statistics
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

    #  Attack interface (matches BlendAttack / SIGAttack / WaNetAttack)
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
        return (f"AdaptiveHybrid(mode={self.mode}, "
                f"alpha in [{self.alpha_min},{self.alpha_max}], "
                f"beta in [{self.beta_min},{self.beta_max}], "
                f"calibration: v in [{self.v_min},{self.v_max}], "
                f"e in [{self.e_min},{self.e_max}])")
