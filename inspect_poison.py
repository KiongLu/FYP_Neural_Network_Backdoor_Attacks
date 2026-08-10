"""
Inspect which training samples were poisoned by a given attack config.

Usage:
    python inspect_poison.py --attack blend --poison_rate 0.05
"""
import argparse
import torchvision
import matplotlib.pyplot as plt

from config import cfg
from datasets.poison_dataset import PoisonedDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=str, default=cfg.attack_name, choices=["clean", "blend", "sig", "wanet", "blend_sig", "blend_wanet", "sig_wanet", "blend_sig_wanet"])
    p.add_argument("--poison_rate", type=float, default=cfg.poison_rate)
    p.add_argument("--n_show", type=int, default=6, help="how many poisoned samples to visualize")
    p.add_argument("--sig_dirty_label", action="store_true")
    p.add_argument("--blend_clean_label", action="store_true")
    p.add_argument("--wanet_clean_label", action="store_true")
    p.add_argument("--hybrid_wanet_scale", type=float, default=None)
    return p.parse_args()


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]


def main():
    args = parse_args()
    cfg.attack_name = args.attack
    cfg.poison_rate = args.poison_rate
    cfg.sig_clean_label = not args.sig_dirty_label
    cfg.blend_clean_label = args.blend_clean_label
    cfg.wanet_clean_label = args.wanet_clean_label
    cfg.hybrid_component_scale = {"wanet": args.hybrid_wanet_scale} if args.hybrid_wanet_scale is not None else None

    train_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True)
    train_set = PoisonedDataset(train_base, cfg, train=True)

    poison_idx = sorted(train_set.poison_indices)
    print(f"Attack: {cfg.attack_name} | poison_rate: {cfg.poison_rate}")
    print(f"Total training samples: {len(train_set)}")
    print(f"Poisoned samples: {len(poison_idx)} ({len(poison_idx)/len(train_set)*100:.2f}%)")
    if hasattr(train_set, "noise_indices") and train_set.noise_indices:
        print(f"WaNet 'noise mode' cover samples: {len(train_set.noise_indices)}")

    print(f"\nFirst 10 poisoned indices: {poison_idx[:10]}")

    # Show original label -> target label for a few poisoned samples
    print("\nSample | original_label -> new_label | is_poison")
    for i in poison_idx[:10]:
        orig_img, orig_label = train_base[i]
        _, new_label, is_poison = train_set[i]
        print(f"  idx {i:5d} | {CIFAR10_CLASSES[orig_label]:>10s} -> {CIFAR10_CLASSES[new_label]:<10s} | {is_poison}")

    # Visualize a handful: clean vs poisoned side by side
    n = min(args.n_show, len(poison_idx))
    fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(poison_idx[:n]):
        orig_img, orig_label = train_base[idx]
        poisoned_img, new_label, _ = train_set[idx]

        axes[row, 0].imshow(orig_img)
        axes[row, 0].set_title(f"clean: {CIFAR10_CLASSES[orig_label]}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(poisoned_img.permute(1, 2, 0).clamp(0, 1))
        axes[row, 1].set_title(f"poisoned -> {CIFAR10_CLASSES[new_label]}", fontsize=9)
        axes[row, 1].axis("off")

    plt.tight_layout()
    out_path = f"{cfg.results_dir}/{cfg.tag()}_poison_samples.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization to: {out_path}")


if __name__ == "__main__":
    main()
