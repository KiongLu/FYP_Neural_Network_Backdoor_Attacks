"""
Hybrid Backdoor Attacks: combine multiple invisible trigger mechanisms (Blend, SIG, WaNet) into a single trigger.
"""

import torch

from attacks.blend import BlendAttack
from attacks.sig import SIGAttack
from attacks.wanet import WaNetAttack

# Fixed application order regardless of how components are specified
_ORDER = ["blend", "sig", "wanet"]

HYBRID_COMPONENTS = {
    "blend_sig": ["blend", "sig"],
    "blend_wanet": ["blend", "wanet"],
    "sig_wanet": ["sig", "wanet"],
    "blend_sig_wanet": ["blend", "sig", "wanet"],
}


def get_components(attack_name: str):
    """Return the list of base-attack names that make up `attack_name`."""
    name = attack_name.lower()
    if name in HYBRID_COMPONENTS:
        return HYBRID_COMPONENTS[name]
    return [name]  # single attack: "blend", "sig", "wanet", "clean"


class HybridAttack:
    def __init__(self, cfg, components):
        self.target_label = cfg.target_label
        self.components = [c for c in _ORDER if c in components]
        n = len(self.components)
        if n < 2:
            raise ValueError("HybridAttack requires at least 2 component attacks.")

        # Each component's intensity is scaled down by 1/N by default so the
        # combined perturbation stays comparable in magnitude to a single
        # attack. This can be overridden per-component via
        # cfg.hybrid_component_scale (e.g. {"wanet": 0.75}) to rebalance a
        # hybrid toward a stronger component -- see config.py for details.
        # Leaving cfg.hybrid_component_scale as None reproduces the exact
        # original, uniform-scaling behaviour.
        self.scale = 1.0 / n
        overrides = getattr(cfg, "hybrid_component_scale", None) or {}

        def scale_for(component_name):
            return overrides.get(component_name, self.scale)

        self.sub_attacks = {}
        if "blend" in self.components:
            reduced_alpha = cfg.blend_alpha * scale_for("blend")
            self.sub_attacks["blend"] = BlendAttack(
                cfg.blend_trigger_path, cfg.image_size, reduced_alpha, cfg.target_label
            )
        if "sig" in self.components:
            reduced_delta = cfg.sig_delta * scale_for("sig")
            self.sub_attacks["sig"] = SIGAttack(
                cfg.image_size, reduced_delta, cfg.sig_freq, cfg.target_label, clean_label=False
            )
        if "wanet" in self.components:
            reduced_s = cfg.wanet_s * scale_for("wanet")
            # Same reasoning as in datasets/poison_dataset.py: keep WaNet on CPU
            # since poisoning runs inside DataLoader worker processes.
            self.sub_attacks["wanet"] = WaNetAttack(
                cfg.image_size, reduced_s, cfg.wanet_k, cfg.wanet_grid_rescale,
                cfg.target_label, device="cpu"
            )

        self.overrides = overrides  # kept for describe()

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        """Apply each component attack in sequence, each at reduced intensity."""
        out = img
        for name in self.components:
            out = self.sub_attacks[name].poison(out)
        return torch.clamp(out, 0.0, 1.0)

    def noise_mode(self, img: torch.Tensor) -> torch.Tensor:
        if "wanet" not in self.sub_attacks:
            return img
        return self.sub_attacks["wanet"].noise_mode(img)

    def is_poisonable(self, label: int) -> bool:
        return True  # hybrid attacks are always dirty-label

    def get_target_label(self, original_label: int) -> int:
        return self.target_label

    def describe(self) -> str:
        parts = []
        for name in self.components:
            actual_scale = self.overrides.get(name, self.scale)
            parts.append(f"{name} (scaled x{actual_scale:.3f})")
        return " + ".join(parts)
