"""
Adaptive hybrid backdoor attack (my proposed extension, not a published attack).

It uses the same two triggers as the fixed SIG+WaNet hybrid - a WaNet warp (Nguyen & Tran, 2021)
and a SIG sinusoidal signal (Barni et al., 2019). The difference is that every image gets its own
strengths instead of one fixed strength:

    v(x) = std of all pixel values                  -> warp strength alpha(x)
    e(x) = mean Sobel edge magnitude of the gray image -> signal amplitude beta(x)

Each statistic is normalised with the calibration range in config.py (clipped to [0, 1]) and then
mapped linearly onto [ALPHA_MIN, ALPHA_MAX] or [BETA_MIN, BETA_MAX]. The warp is applied first and
then the signal, in the same order as the fixed hybrid:

    x_warped = grid_sample(x, identity_grid + alpha(x) * noise_grid / image_size)
    x_hybrid = clip(x_warped + beta(x) / 255 * sig_pattern, 0, 1)

The warp field and the sine pattern are created once and reused for every image - only their
strength changes - otherwise the model would have no consistent trigger to learn.

v2 (the full "adaptive_hybrid" mode only) also multiplies the signal by a per-pixel texture mask:

    m(x) = clip(local std of gray(x) / MASK_REF, MASK_FLOOR, 1)

so the signal stays at full strength in busy regions and is turned down (but never removed) in flat
regions, where a stripe pattern is easiest to see. The idea comes from perceptual masking in image
watermarking; whether it actually helps here is what the experiments test.

The attack is always dirty-label, like the fixed hybrids: the strengths depend on image content,
not on the class, so restricting it to airplanes would mix up "clean-label" with "adaptive".
The full maths is in Appendix A of the report.
"""

import torch
import torch.nn.functional as F


class AdaptiveHybridAttack:
    # attack name -> which parts are switched on. The two single-part modes are the ablations.
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
        self.device = "cpu"  # same as WaNet: this runs inside the DataLoader workers

        self.v_min, self.v_max = cfg.adaptive_v_min, cfg.adaptive_v_max
        self.e_min, self.e_max = cfg.adaptive_e_min, cfg.adaptive_e_max

        if mode == "both":
            # the full attack uses the v2 settings
            self.alpha_min, self.alpha_max = cfg.adaptive_hybrid_alpha_min, cfg.adaptive_hybrid_alpha_max
            self.beta_min, self.beta_max = cfg.adaptive_hybrid_beta_min, cfg.adaptive_hybrid_beta_max
            self.cover_alpha = cfg.adaptive_hybrid_cover_alpha
            self.use_texture_mask = cfg.adaptive_texture_mask
        else:
            # the ablations keep the v1 settings so their results can be compared with v1
            self.alpha_min, self.alpha_max = cfg.adaptive_alpha_min, cfg.adaptive_alpha_max
            self.beta_min, self.beta_max = cfg.adaptive_beta_min, cfg.adaptive_beta_max
            self.cover_alpha = cfg.adaptive_cover_alpha
            self.use_texture_mask = False

        self.mask_ref = cfg.adaptive_mask_ref
        self.mask_floor = cfg.adaptive_mask_floor
        self.mask_window = cfg.adaptive_mask_window
        self.mask_kernel = self._make_gaussian_kernel(cfg.adaptive_mask_window, cfg.adaptive_mask_sigma)

        # one warp field for all poisoned images - only its strength changes per image
        self.identity_grid, self.noise_grid = self._make_grids()

        # sine pattern with amplitude 1; beta(x) is applied in poison()
        self.sig_pattern_unit = self._make_unit_sig_pattern()

        self.sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

    # --- fixed parts of the trigger, built once ---

    def _make_grids(self):
        xs = torch.linspace(-1, 1, self.image_size)
        ys = torch.linspace(-1, 1, self.image_size)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        identity_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,H,W,2)

        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))  # same normalisation as attacks/wanet.py
        noise_grid = F.interpolate(ins, size=(self.image_size, self.image_size),
                                    mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)  # (1,H,W,2)
        return identity_grid, noise_grid

    def _make_unit_sig_pattern(self):
        j = torch.arange(self.image_size).float()
        pattern_1d = torch.sin(2 * torch.pi * j * self.freq / self.image_size)
        pattern = pattern_1d.unsqueeze(0).repeat(self.image_size, 1)   # (H,W)
        pattern = pattern.unsqueeze(0).repeat(3, 1, 1)                  # (C,H,W)
        return pattern

    @staticmethod
    def _make_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        x = torch.arange(size).float() - size // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        return (g.unsqueeze(1) @ g.unsqueeze(0)).view(1, 1, size, size)

    # --- per-image statistics ---

    def _variance_stat(self, img: torch.Tensor) -> float:
        return img.std().item()

    def _edge_stat(self, img: torch.Tensor) -> float:
        """
        Mean Sobel edge magnitude of the gray image. (Like any small edge filter it misses
        patterns that alternate every single pixel, but real photos don't have those.)
        """
        gray = img.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
        # replicate padding - zero padding would create a fake edge all around the border,
        # which would dominate the value for smooth images
        gray_padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(gray_padded, self.sobel_x, padding=0)
        gy = F.conv2d(gray_padded, self.sobel_y, padding=0)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        return mag.mean().item()

    def compute_alpha_beta(self, img: torch.Tensor):
        """Warp strength and signal amplitude for this image (also used by the notebooks)."""
        v = self._variance_stat(img)
        e = self._edge_stat(img)
        v_norm = min(max((v - self.v_min) / (self.v_max - self.v_min), 0.0), 1.0)
        e_norm = min(max((e - self.e_min) / (self.e_max - self.e_min), 0.0), 1.0)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * v_norm
        beta = self.beta_min + (self.beta_max - self.beta_min) * e_norm
        return alpha, beta

    def texture_mask(self, img: torch.Tensor) -> torch.Tensor:
        """
        Per-pixel mask in [mask_floor, 1], shape (1,H,W): the local std of the gray image
        (Gaussian window) divided by mask_ref. Replicate padding for the same reason as above.
        """
        gray = img.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
        p = self.mask_window // 2
        gray_padded = F.pad(gray, (p, p, p, p), mode="replicate")
        mu = F.conv2d(gray_padded, self.mask_kernel)
        mu_sq = F.conv2d(gray_padded ** 2, self.mask_kernel)
        local_std = (mu_sq - mu ** 2).clamp_min(0).sqrt().squeeze(0)
        return (local_std / self.mask_ref).clamp(self.mask_floor, 1.0)

    # --- same interface as the other attacks ---

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        return self.poison_components(img, spatial=self.mode in ("both", "spatial_only"),
                                      freq=self.mode in ("both", "freq_only"))

    def poison_components(self, img: torch.Tensor, spatial: bool = True, freq: bool = True) -> torch.Tensor:
        """
        Apply the warp and/or the (masked) signal at this image's strengths. evaluate_components.py
        calls this directly to test one part of the trigger at a time.
        """
        alpha, beta = self.compute_alpha_beta(img)
        out = img

        if spatial:
            grid = self.identity_grid + alpha * self.noise_grid / self.image_size
            grid = torch.clamp(grid * self.grid_rescale, -1, 1)
            out = F.grid_sample(out.unsqueeze(0), grid, align_corners=True,
                                 mode="bilinear", padding_mode="reflection").squeeze(0)

        if freq:
            signal = (beta / 255.0) * self.sig_pattern_unit
            if self.use_texture_mask:
                signal = signal * self.texture_mask(img)
            out = out + signal

        return torch.clamp(out, 0.0, 1.0)

    def noise_mode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Cover sample: a new random warp at a fixed strength, no signal, label kept.
        The strength is fixed on purpose. If it followed alpha(x), the cover samples would depend on
        the image content, but they are only there to teach the model that "some warp" on its own
        does not mean airplane. Not needed in freq_only mode since there is no warp to cover.
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
        return True  # dirty-label, see the top of the file

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
