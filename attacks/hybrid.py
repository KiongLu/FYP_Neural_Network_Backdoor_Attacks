"""
Fixed hybrid attacks: two or three of Blend, SIG and WaNet combined into one trigger.

Each part is scaled down by 1/N (N = number of parts), so the combined change to the image stays
about as large as a single attack, but no single trigger is there at full strength.

- Always dirty-label. Clean-label SIG only poisons airplanes, which doesn't fit with Blend and WaNet
  poisoning any class, so I kept all hybrids dirty-label.
- The parts are always applied in the order blend -> sig -> wanet.
- If WaNet is one of the parts, its cover samples are kept too (at the reduced warp strength).
"""

import torch

from attacks.blend import BlendAttack
from attacks.sig import SIGAttack
from attacks.wanet import WaNetAttack

_ORDER = ["blend", "sig", "wanet"]

HYBRID_COMPONENTS = {
    "blend_sig": ["blend", "sig"],
    "blend_wanet": ["blend", "wanet"],
    "sig_wanet": ["sig", "wanet"],
    "blend_sig_wanet": ["blend", "sig", "wanet"],
    # same parts as sig_wanet, but with the fixed strengths from config.py instead of 1/N
    "sig_wanet_matched": ["sig", "wanet"],
}


def get_components(attack_name: str):
    """Which base attacks make up this attack, e.g. "blend_sig" -> ["blend", "sig"]."""
    name = attack_name.lower()
    if name in HYBRID_COMPONENTS:
        return HYBRID_COMPONENTS[name]
    return [name]  # a single attack


class HybridAttack:
    def __init__(self, cfg, components, strengths=None):
        self.target_label = cfg.target_label
        self.components = [c for c in _ORDER if c in components]
        n = len(self.components)
        if n < 2:
            raise ValueError("HybridAttack requires at least 2 component attacks.")

        # Normally every part is scaled by 1/N. `strengths` can give the actual values instead,
        # e.g. {"sig": 10.6, "wanet": 0.27} for the matched baseline.
        self.scale = 1.0 / n
        self.strengths = strengths

        self.sub_attacks = {}
        if "blend" in self.components:
            reduced_alpha = strengths["blend"] if strengths else cfg.blend_alpha * self.scale
            self.sub_attacks["blend"] = BlendAttack(
                cfg.blend_trigger_path, cfg.image_size, reduced_alpha, cfg.target_label
            )
        if "sig" in self.components:
            reduced_delta = strengths["sig"] if strengths else cfg.sig_delta * self.scale
            self.sub_attacks["sig"] = SIGAttack(
                cfg.image_size, reduced_delta, cfg.sig_freq, cfg.target_label, clean_label=False
            )
        if "wanet" in self.components:
            reduced_s = strengths["wanet"] if strengths else cfg.wanet_s * self.scale
            # keep WaNet on the CPU, because poisoning happens in the DataLoader workers
            self.sub_attacks["wanet"] = WaNetAttack(
                cfg.image_size, reduced_s, cfg.wanet_k, cfg.wanet_grid_rescale,
                cfg.target_label, device="cpu"
            )

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        """Apply all the parts, one after the other."""
        return self.poison_components(img, self.components)

    def poison_components(self, img: torch.Tensor, components) -> torch.Tensor:
        """
        Apply only some of the parts (still in the usual order). evaluate_components.py uses this to
        see how much each part of a trained model's trigger works on its own.
        """
        out = img
        for name in self.components:
            if name in components:
                out = self.sub_attacks[name].poison(out)
        return torch.clamp(out, 0.0, 1.0)

    def noise_mode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Cover sample: only the WaNet part, with a random warp field. Blend and SIG are left out, the
        same as for plain WaNet - cover samples should never trigger the backdoor.
        """
        if "wanet" not in self.sub_attacks:
            return img
        return self.sub_attacks["wanet"].noise_mode(img)

    def is_poisonable(self, label: int) -> bool:
        return True  # always dirty-label

    def get_target_label(self, original_label: int) -> int:
        return self.target_label

    def describe(self) -> str:
        if self.strengths:
            return " + ".join(f"{name} (strength {self.strengths[name]})" for name in self.components)
        return " + ".join(f"{name} (scaled x{self.scale:.3f})" for name in self.components)
