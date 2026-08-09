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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=str, default=cfg.attack_name, choices=["blend", "sig", "wanet"])
    p.add_argument("--poison_rate", type=float, default=cfg.poison_rate)
    p.add_argument("--target_label", type=int, default=cfg.target_label)
    p.add_argument("--checkpoint", type=str, required=True)
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

    set_seed(cfg.seed)
    device = cfg.device

    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    to_tensor = T.ToTensor()

    test_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True)

    # ---- Clean accuracy loader ----
    def clean_collate(batch):
        imgs, labels = zip(*[(to_tensor(img), label) for img, label in batch])
        imgs = torch.stack([normalize(im) for im in imgs])
        return imgs, torch.tensor(labels, dtype=torch.long)

    clean_loader = DataLoader(test_base, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=clean_collate)

    # ---- ASR loader ----
    asr_set = TestPoisonedDataset(test_base, cfg)

    def asr_collate(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack([normalize(im) for im in imgs])
        return imgs, torch.tensor(labels, dtype=torch.long)

    asr_loader = DataLoader(asr_set, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, collate_fn=asr_collate)

    # ---- Load model ----
    model = get_model(cfg.model_name, cfg.num_classes).to(device)
    load_checkpoint(model, None, args.checkpoint, device)

    ca = clean_accuracy(model, clean_loader, device)
    asr = attack_success_rate(model, asr_loader, device)
    avg_ssim, avg_psnr = stealthiness(test_base, cfg)

    print(f"\n===== Evaluation Results ({cfg.attack_name}, poison_rate={cfg.poison_rate}) =====")
    print(f"Clean Accuracy (CA):        {ca*100:.2f}%")
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
