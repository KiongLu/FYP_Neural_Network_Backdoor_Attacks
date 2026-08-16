"""
Evaluate a trained (possibly backdoored) model on:
  1. Clean Accuracy (CA)        -- accuracy on the untouched clean test set
  2. Attack Success Rate (ASR)  -- fraction of triggered non-target samples classified as the target label
  3. SSIM / PSNR                -- stealthiness of the trigger vs. clean images
"""

import argparse
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from utils import set_seed, load_checkpoint, ssim, psnr, l2_norm, linf_norm, lpips_distance, AverageMeter, CSVLogger
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
    p.add_argument("--attack", type=str, default=cfg.attack_name, choices=["clean", "blend", "sig", "wanet", "blend_sig", "blend_wanet", "sig_wanet", "blend_sig_wanet", "adaptive_hybrid", "adaptive_spatial", "adaptive_freq"])
    p.add_argument("--poison_rate", type=float, default=cfg.poison_rate)
    p.add_argument("--target_label", type=int, default=cfg.target_label)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sig_dirty_label", action="store_true",
                    help="Evaluate against dirty-label SIG instead of the default clean-label SIG. "
                         "Must match whichever mode the checkpoint was TRAINED with.")
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
    """Average SSIM / PSNR / L2 / L_inf (and LPIPS if available) between clean and poisoned images."""
    attack = build_attack(cfg)
    to_tensor = T.ToTensor()

    ssim_meter, psnr_meter = AverageMeter(), AverageMeter()
    l2_meter, linf_meter, lpips_meter = AverageMeter(), AverageMeter(), AverageMeter()
    lpips_seen_none = False

    for i in range(min(n_samples, len(test_base))):
        img, _ = test_base[i]
        img = to_tensor(img)
        poisoned = attack.poison(img)

        s = ssim(img.unsqueeze(0), poisoned.unsqueeze(0))
        p = psnr(img, poisoned)
        l2 = l2_norm(img, poisoned)
        linf = linf_norm(img, poisoned)

        ssim_meter.update(s)
        if p != float("inf"):
            psnr_meter.update(p)
        l2_meter.update(l2)
        linf_meter.update(linf)

        lp = lpips_distance(img, poisoned)
        if lp is None:
            lpips_seen_none = True
        else:
            lpips_meter.update(lp)

    avg_lpips = None if lpips_seen_none else lpips_meter.avg
    return ssim_meter.avg, psnr_meter.avg, l2_meter.avg, linf_meter.avg, avg_lpips


def main():
    args = parse_args()
    cfg.attack_name = args.attack
    cfg.poison_rate = args.poison_rate
    cfg.target_label = args.target_label
    cfg.sig_clean_label = not args.sig_dirty_label
    cfg.hybrid_component_scale = {"wanet": args.hybrid_wanet_scale} if args.hybrid_wanet_scale is not None else None

    set_seed(cfg.seed)
    device = cfg.device
    if cfg.attack_name.lower() == "sig":
        mode = "CLEAN-LABEL" if cfg.sig_clean_label else "DIRTY-LABEL"
        print(f"[INFO] SIG mode: {mode}")
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
        asr = float("nan")
        avg_ssim = avg_psnr = avg_l2 = avg_linf = avg_lpips = float("nan")
    else:
        asr = attack_success_rate(model, asr_loader, device)
        avg_ssim, avg_psnr, avg_l2, avg_linf, avg_lpips = stealthiness(test_base, cfg)

    print(f"\n===== Evaluation Results ({cfg.attack_name}, poison_rate={cfg.poison_rate}) =====")
    print(f"Clean Accuracy (CA):        {ca*100:.2f}%")
    if cfg.attack_name == "clean":
        print("Attack Success Rate (ASR):  N/A (clean baseline, no trigger)")
    else:
        print(f"Attack Success Rate (ASR):  {asr*100:.2f}%")
        print(f"Avg SSIM (clean vs poison): {avg_ssim:.4f}")
        print(f"Avg PSNR (clean vs poison): {avg_psnr:.2f} dB")
        print(f"Avg L2 norm:                {avg_l2:.4f}")
        print(f"Avg L_inf norm:             {avg_linf:.4f}")
        if avg_lpips is not None:
            print(f"Avg LPIPS:                  {avg_lpips:.4f}")
        else:
            print("Avg LPIPS:                  N/A (pip install lpips to enable)")

    logger = CSVLogger(f"{cfg.results_dir}/summary.csv",
                        fieldnames=["attack", "poison_rate", "clean_acc", "asr", "ssim", "psnr",
                                    "l2", "linf", "lpips"])
    logger.log({
        "attack": cfg.attack_name, "poison_rate": cfg.poison_rate,
        "clean_acc": ca, "asr": asr, "ssim": avg_ssim, "psnr": avg_psnr,
        "l2": avg_l2, "linf": avg_linf, "lpips": avg_lpips if avg_lpips is not None else "",
    })
    logger.close()


if __name__ == "__main__":
    main()
