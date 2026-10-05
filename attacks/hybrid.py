"""
Hybrid Backdoor Attacks: combine multiple invisible trigger mechanisms
(Blend, SIG, WaNet) into a single trigger.

Rationale
---------
Each individual attack (Blend, SIG, WaNet) must perturb the image by enough
to be reliably learned as a trigger. When N techniques are combined, each
individual technique only needs to contribute a fraction of the total signal
the model learns to associate with the target label — so each component's
intensity is scaled down by 1/N. This keeps the *combined* perturbation
visually comparable to (or smaller than) a single full-strength attack,
while potentially being harder to reverse-engineer/detect since no single
attack's signature (additive blend pattern, sinusoidal ramp, or warping
field) is present at full strength.

Design notes / simplifications
-------------------------------
- Hybrid attacks are always DIRTY-LABEL: any poisoned sample has its label
  flipped to the target class, regardless of its original class. This is a
  deliberate simplification because SIG's clean-label property (only poison
  samples already in the target class) doesn't compose cleanly with
  dirty-label Blend/WaNet poisoning of arbitrary classes.
- Components are applied sequentially: img -> blend -> sig -> wanet (only
  the selected subset, in this fixed order) so results are reproducible
  regardless of how components are listed.
- If WaNet is one of the components, its "noise mode" cover-sample
  mechanism is preserved (only the WaNet sub-attack is used for cover
  samples, at its scaled-down warping strength) since that mechanism is
  specific to defeating warping-based trigger detection.
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

        # Each component's intensity is scaled down by 1/N so the combined
        # perturbation stays comparable in magnitude to a single attack.
        self.scale = 1.0 / n

        self.sub_attacks = {}
        if "blend" in self.components:
            reduced_alpha = cfg.blend_alpha * self.scale
            self.sub_attacks["blend"] = BlendAttack(
                cfg.blend_trigger_path, cfg.image_size, reduced_alpha, cfg.target_label
            )
        if "sig" in self.components:
            reduced_delta = cfg.sig_delta * self.scale
            self.sub_attacks["sig"] = SIGAttack(
                cfg.image_size, reduced_delta, cfg.sig_freq, cfg.target_label, clean_label=False
            )
        if "wanet" in self.components:
            reduced_s = cfg.wanet_s * self.scale
            # Same reasoning as in datasets/poison_dataset.py: keep WaNet on CPU
            # since poisoning runs inside DataLoader worker processes.
            self.sub_attacks["wanet"] = WaNetAttack(
                cfg.image_size, reduced_s, cfg.wanet_k, cfg.wanet_grid_rescale,
                cfg.target_label, device="cpu"
            )

    def poison(self, img: torch.Tensor) -> torch.Tensor:
        """Apply each component attack in sequence, each at reduced intensity."""
        out = img
        for name in self.components:
            out = self.sub_attacks[name].poison(out)
        return torch.clamp(out, 0.0, 1.0)

    def noise_mode(self, img: torch.Tensor) -> torch.Tensor:
        """
        Cover-sample generation for the WaNet component only (random field,
        reduced strength). Other components are not applied to cover samples,
        matching solo-WaNet behaviour: cover samples must NOT trigger the
        backdoor, only exercise "generic warping" so the model learns to
        require the exact fixed configuration.
        """
        if "wanet" not in self.sub_attacks:
            return img
        return self.sub_attacks["wanet"].noise_mode(img)

    def is_poisonable(self, label: int) -> bool:
        return True  # hybrid attacks are always dirty-label

    def get_target_label(self, original_label: int) -> int:
        return self.target_label

    def describe(self) -> str:
        return " + ".join(f"{name} (scaled x{self.scale:.3f})" for name in self.components)
