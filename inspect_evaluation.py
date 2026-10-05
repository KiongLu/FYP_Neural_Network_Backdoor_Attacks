"""
Visualize exactly which images evaluate.py uses when computing the Attack
Success Rate (ASR), and — if you give it a trained checkpoint — show what
the model actually predicts for each one (success = trigger fooled the
model into predicting the target label; fail = it didn't).

This uses the SAME TestPoisonedDataset class that evaluate.py's ASR
calculation runs on, so what you see here is exactly what's being measured.

Usage:
    # Just look at the poisoned test images (no model needed)
    python inspect_evaluation.py --attack wanet --poison_rate 0.01

    # Also show the model's predictions and mark success/fail per image
    python inspect_evaluation.py --attack wanet --poison_rate 0.01 \
        --checkpoint checkpoints/cifar10_resnet18_wanet_pr0.01_final.pt
"""

import argparse
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from config import cfg
from utils import set_seed, load_checkpoint
from models.resnet import get_model
from datasets.poison_dataset import TestPoisonedDataset, build_attack


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=str, default=cfg.attack_name,
                    choices=["blend", "sig", "wanet", "blend_sig", "blend_wanet",
                             "sig_wanet", "blend_sig_wanet",
                             "adaptive_hybrid", "adaptive_spatial", "adaptive_freq"])
    p.add_argument("--poison_rate", type=float, default=cfg.poison_rate)
    p.add_argument("--target_label", type=int, default=cfg.target_label)
    p.add_argument("--sig_dirty_label", action="store_true",
                    help="Must match whichever mode the checkpoint was TRAINED with, if given.")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Optional. If given, also runs the model on shown images and marks "
                         "success/fail. If omitted, only the images themselves are shown.")
    p.add_argument("--n_show", type=int, default=8, help="how many ASR test images to visualize")
    p.add_argument("--seed", type=int, default=None,
                    help="Optional. Set for a reproducible sample of shown images (e.g. same "
                         "images every run for consistent report figures).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg.attack_name = args.attack
    cfg.poison_rate = args.poison_rate
    cfg.target_label = args.target_label
    cfg.sig_clean_label = not args.sig_dirty_label

    if args.seed is not None:
        set_seed(args.seed)

    device = cfg.device
    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    test_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True)

    # This is EXACTLY the dataset evaluate.py builds for its ASR calculation:
    # every test image that is NOT already the target class, with the trigger applied.
    asr_set = TestPoisonedDataset(test_base, cfg)
    print(f"Attack: {cfg.attack_name} | poison_rate: {cfg.poison_rate}")
    print(f"Target label: {cfg.target_label} ({CIFAR10_CLASSES[cfg.target_label]})")
    print(f"Total ASR test images (test set minus target-class images): {len(asr_set)}")

    # Load model if a checkpoint was given
    model = None
    if args.checkpoint:
        model = get_model(cfg.model_name, cfg.num_classes).to(device)
        load_checkpoint(model, None, args.checkpoint, device)
        model.eval()
        print(f"Loaded checkpoint: {args.checkpoint}")

    n = min(args.n_show, len(asr_set))
    fig, axes = plt.subplots(n, 2, figsize=(4.5, 2.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    n_correct_shown = 0
    print(f"\n{'#':>3} | {'orig_label':>10} | {'target':>10} | {'model_pred':>10} | attack result")
    print("-" * 65)

    for row in range(n):
        # asr_set.eval_indices maps position -> original test-set index, so we
        # can recover the true original label to display alongside.
        orig_idx = asr_set.eval_indices[row]
        orig_img, orig_label = test_base[orig_idx]
        poisoned_img, target_label = asr_set[row]

        axes[row, 0].imshow(orig_img)
        axes[row, 0].set_title(f"clean: {CIFAR10_CLASSES[orig_label]}", fontsize=9)
        axes[row, 0].axis("off")

        title_right = f"poisoned (target: {CIFAR10_CLASSES[target_label]})"
        pred_str = "n/a"
        result_str = "n/a (no checkpoint given)"

        if model is not None:
            with torch.no_grad():
                inp = normalize(poisoned_img).unsqueeze(0).to(device)
                pred = model(inp).argmax(dim=1).item()
            pred_str = CIFAR10_CLASSES[pred]
            success = (pred == target_label)
            n_correct_shown += int(success)
            result_str = "SUCCESS (fooled model)" if success else "failed (model resisted trigger)"
            title_right += f"\npred: {CIFAR10_CLASSES[pred]} {'✓' if success else '✗'}"

        axes[row, 1].imshow(poisoned_img.permute(1, 2, 0).clamp(0, 1))
        axes[row, 1].set_title(title_right, fontsize=8,
                                color=("green" if model is not None and pred_str == CIFAR10_CLASSES[target_label]
                                       else ("red" if model is not None else "black")))
        axes[row, 1].axis("off")

        print(f"{row:>3} | {CIFAR10_CLASSES[orig_label]:>10} | {CIFAR10_CLASSES[target_label]:>10} | "
              f"{pred_str:>10} | {result_str}")

    if model is not None:
        print(f"\nShown-sample success rate: {n_correct_shown}/{n} = {100*n_correct_shown/n:.1f}%")
        print("(This is just the small visualized sample, not the full ASR — "
              "run evaluate.py for the ASR over the whole test set.)")

    plt.tight_layout()
    suffix = "_with_preds" if model is not None else ""
    out_path = f"{cfg.results_dir}/{cfg.tag()}_eval_samples{suffix}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved visualization to: {out_path}")


if __name__ == "__main__":
    main()
