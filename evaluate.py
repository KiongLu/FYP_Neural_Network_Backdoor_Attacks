"""
Evaluate a trained (possibly backdoored) model on:
1. Clean Accuracy (CA)        -- accuracy on the untouched clean test set
2. Attack Success Rate (ASR)  -- fraction of triggered non-target samples classified as the target label
3. SSIM / PSNR                -- stealthiness of the trigger vs. clean images

Usage: python evaluate.py --attack blend --poison_rate 0.05 --checkpoint checkpoints/cifar10_resnet18_blend_pr0.05_final.pt
"""

import argparse
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from utils import set_seed, load_checkpoint, ssim, psnr, AverageMeter, CSVLogger
from models.resnet import get_model
from datasets.poison_dataset import TestPoisonedDataset, build_attack


class CleanCollate:
    """Module-level (picklable) collate for the clean-test DataLoader."""
    def __init__(self, normalize, to_tensor):
        self.normalize = normalize
        self.to_tensor = to_tensor

    def __call__(self, batch):
        imgs, labels = zip(*[(self.to_tensor(img), label) for img, label in batch])
        imgs = torch.stack([self.normalize(im) for im in imgs])
        return imgs, torch.tensor(labels, dtype=torch.long)


class ASRCollate:
    """Module-level (picklable) collate for the ASR DataLoader."""
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack([self.normalize(im) for im in imgs])
        return imgs, torch.tensor(labels, dtype=torch.long)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=str, default=cfg.attack_name, choices=["clean", "blend", "sig", "wanet", "blend_sig", "blend_wanet", "sig_wanet", "blend_sig_wanet"])
    p.add_argument("--poison_rate", type=float, default=cfg.poison_rate)
    p.add_argument("--target_label", type=int, default=cfg.target_label)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sig_dirty_label", action="store_true",
                    help="Evaluate against dirty-label SIG instead of the default clean-label SIG. "
                         "Must match whichever mode the checkpoint was TRAINED with.")
    p.add_argument("--blend_clean_label", action="store_true",
                    help="Evaluate against clean-label Blend. Must match training mode.")
    p.add_argument("--wanet_clean_label", action="store_true",
                    help="Evaluate against clean-label WaNet. Must match training mode.")
    p.add_argument("--hybrid_wanet_scale", type=float, default=None,
                    help="Must match whatever value (if any) was used during training for this "
                         "checkpoint's hybrid attack.")
    return p.parse_args()


@torch.no_grad()
def clean_accuracy(model, loader, device):
    model.eval()
    acc_meter = AverageMeter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(dim=1)
        acc_meter.update((preds == labels).float().mean().item(), imgs.size(0))
    return acc_meter.avg


@torch.no_grad()
def attack_success_rate(model, loader, device):
    model.eval()
    acc_meter = AverageMeter()
    for imgs, target_labels in loader:
        imgs, target_labels = imgs.to(device), target_labels.to(device)
        preds = model(imgs).argmax(dim=1)
        acc_meter.update((preds == target_labels).float().mean().item(), imgs.size(0))
    return acc_meter.avg  # this IS the ASR


def stealthiness(test_base, cfg, n_samples: int = 200):
    """Average SSIM / PSNR between clean and poisoned images over a sample."""
    attack = build_attack(cfg)
    to_tensor = T.ToTensor()

    ssim_meter, psnr_meter = AverageMeter(), AverageMeter()
    for i in range(min(n_samples, len(test_base))):
        img, _ = test_base[i]
        img = to_tensor(img)
        poisoned = attack.poison(img)
        s = ssim(img.unsqueeze(0), poisoned.unsqueeze(0))
        p = psnr(img, poisoned)
        ssim_meter.update(s)
        if p != float("inf"):
            psnr_meter.update(p)
    return ssim_meter.avg, psnr_meter.avg


def main():
    args = parse_args()
    cfg.attack_name = args.attack
    cfg.poison_rate = args.poison_rate
    cfg.target_label = args.target_label
    cfg.sig_clean_label = not args.sig_dirty_label
    cfg.blend_clean_label = args.blend_clean_label
    cfg.wanet_clean_label = args.wanet_clean_label
    cfg.hybrid_component_scale = {"wanet": args.hybrid_wanet_scale} if args.hybrid_wanet_scale is not None else None

    set_seed(cfg.seed)
    device = cfg.device
    if cfg.attack_name.lower() == "sig":
        mode = "CLEAN-LABEL" if cfg.sig_clean_label else "DIRTY-LABEL"
        print(f"[INFO] SIG mode: {mode}")
    if cfg.attack_name.lower() == "blend" and cfg.blend_clean_label:
        print("[INFO] Blend mode: CLEAN-LABEL")
    if cfg.attack_name.lower() == "wanet" and cfg.wanet_clean_label:
        print("[INFO] WaNet mode: CLEAN-LABEL")
    if cfg.hybrid_component_scale:
        print(f"[INFO] Hybrid component scale override: {cfg.hybrid_component_scale}")

    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    to_tensor = T.ToTensor()

    test_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True)

    # ---- Clean accuracy loader ----
    clean_loader = DataLoader(test_base, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=CleanCollate(normalize, to_tensor))

    # ---- ASR loader ---- (skip for the clean baseline: there's no trigger to test)
    asr_loader = None
    if cfg.attack_name != "clean":
        asr_set = TestPoisonedDataset(test_base, cfg)
        asr_loader = DataLoader(asr_set, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, collate_fn=ASRCollate(normalize))

    # ---- Load model ----
    model = get_model(cfg.model_name, cfg.num_classes).to(device)
    load_checkpoint(model, None, args.checkpoint, device)

    ca = clean_accuracy(model, clean_loader, device)

    if cfg.attack_name == "clean":
        asr, avg_ssim, avg_psnr = float("nan"), float("nan"), float("nan")
    else:
        asr = attack_success_rate(model, asr_loader, device)
        avg_ssim, avg_psnr = stealthiness(test_base, cfg)

    print(f"\n===== Evaluation Results ({cfg.attack_name}, poison_rate={cfg.poison_rate}) =====")
    print(f"Clean Accuracy (CA):        {ca*100:.2f}%")
    if cfg.attack_name == "clean":
        print("Attack Success Rate (ASR):  N/A (clean baseline, no trigger)")
    else:
        print(f"Attack Success Rate (ASR):  {asr*100:.2f}%")
        print(f"Avg SSIM (clean vs poison): {avg_ssim:.4f}")
        print(f"Avg PSNR (clean vs poison): {avg_psnr:.2f} dB")

    logger = CSVLogger(f"{cfg.results_dir}/summary.csv",
                        fieldnames=["attack", "poison_rate", "clean_acc", "asr", "ssim", "psnr"])
    logger.log({
        "attack": cfg.attack_name, "poison_rate": cfg.poison_rate,
        "clean_acc": ca, "asr": asr, "ssim": avg_ssim, "psnr": avg_psnr,
    })
    logger.close()


if __name__ == "__main__":
    main()
