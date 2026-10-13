"""
Evaluate a trained model:
  - clean accuracy (CA) on the normal test set
  - attack success rate (ASR): how many triggered non-airplane test images are classified as airplane
  - how visible the trigger is: SSIM, PSNR, L2, L-inf and LPIPS between clean and triggered images

The results are also appended to results/summary.csv.

Example:
    python evaluate.py --attack blend --poison_rate 0.05 --checkpoint checkpoints/cifar10_resnet18_blend_pr0.05_final.pt
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
    """Batches the clean test images (a class so it can be pickled for the worker processes)."""
    def __init__(self, normalize, to_tensor):
        self.normalize = normalize
        self.to_tensor = to_tensor

    def __call__(self, batch):
        imgs, labels = zip(*[(self.to_tensor(img), label) for img, label in batch])
        imgs = torch.stack([self.normalize(im) for im in imgs])
        return imgs, torch.tensor(labels, dtype=torch.long)


class ASRCollate:
    """Batches the triggered test images."""
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack([self.normalize(im) for im in imgs])
        return imgs, torch.tensor(labels, dtype=torch.long)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=str, default=cfg.attack_name, choices=["clean", "badnets", "blend", "sig", "wanet", "blend_sig", "blend_wanet", "sig_wanet", "sig_wanet_matched", "blend_sig_wanet", "adaptive_hybrid", "adaptive_spatial", "adaptive_freq"])
    p.add_argument("--poison_rate", type=float, default=cfg.poison_rate)
    p.add_argument("--target_label", type=int, default=cfg.target_label)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sig_dirty_label", action="store_true",
                    help="Evaluate against dirty-label SIG instead of the default clean-label SIG. "
                         "Must match whichever mode the checkpoint was TRAINED with.")
    p.add_argument("--seed", type=int, default=cfg.seed,
                    help="Must match the seed the checkpoint was trained with (it fixes the trigger pattern).")
    p.add_argument("--augment", action="store_true", help="Checkpoint was trained with --augment (label only).")
    p.add_argument("--augment_mode", choices=["crop_flip", "crop", "flip"], default="crop_flip",
                    help="Augmentation mode the checkpoint was trained with (label only).")
    p.add_argument("--model", choices=["resnet18", "vgg16"], default=cfg.model_name,
                    help="Network architecture of the checkpoint.")
    p.add_argument("--holdout", type=int, default=0,
                    help="Checkpoint was trained with --holdout N (label only; evaluation always uses the test set).")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="Same config overrides as used in training.")
    p.add_argument("--tag_suffix", type=str, default="")
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
    return acc_meter.avg  # accuracy against the target label = ASR


def stealthiness(test_base, cfg, attack, n_samples: int = 200):
    """
    Average SSIM, PSNR, L2, L-inf (and LPIPS if it is installed) between clean and triggered images.

    Pass in the attack object that was built right after set_seed, i.e. the trigger the model was
    trained with. I used to build a new one in here, but by then creating the model had used up
    random numbers, so anything with a random warp got a different warp field than the trained one.
    """
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
    cfg.seed = args.seed
    cfg.augment = args.augment
    cfg.augment_mode = args.augment_mode
    cfg.model_name = args.model
    cfg.holdout = args.holdout
    cfg.tag_suffix = args.tag_suffix
    cfg.apply_overrides(args.set)

    set_seed(cfg.seed)
    device = cfg.device
    if cfg.attack_name.lower() == "sig":
        mode = "CLEAN-LABEL" if cfg.sig_clean_label else "DIRTY-LABEL"
        print(f"[INFO] SIG mode: {mode}")

    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    to_tensor = T.ToTensor()

    test_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True)

    # clean test set
    clean_loader = DataLoader(test_base, batch_size=cfg.batch_size, shuffle=False,
                               num_workers=cfg.num_workers, collate_fn=CleanCollate(normalize, to_tensor))

    # triggered test set (not needed for the clean model, which has no trigger)
    asr_loader = None
    if cfg.attack_name != "clean":
        asr_set = TestPoisonedDataset(test_base, cfg)
        asr_loader = DataLoader(asr_set, batch_size=cfg.batch_size, shuffle=False,
                                 num_workers=cfg.num_workers, collate_fn=ASRCollate(normalize))

    # model
    model = get_model(cfg.model_name, cfg.num_classes).to(device)
    load_checkpoint(model, None, args.checkpoint, device)

    ca = clean_accuracy(model, clean_loader, device)

    if cfg.attack_name == "clean":
        asr = float("nan")
        avg_ssim = avg_psnr = avg_l2 = avg_linf = avg_lpips = float("nan")
    else:
        asr = attack_success_rate(model, asr_loader, device)
        avg_ssim, avg_psnr, avg_l2, avg_linf, avg_lpips = stealthiness(test_base, cfg, asr_set.attack)

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
    name = cfg.tag().split(f"{cfg.model_name}_", 1)[1].rsplit("_pr", 1)[0].replace("sig_clean", "sig").replace("sig_dirty", "sig")
    if cfg.model_name != "resnet18":          # e.g. "vgg16_sig_wanet", so the ResNet-18 rows keep their names
        name = f"{cfg.model_name}_{name}"
    logger.log({
        "attack": name,
        "poison_rate": cfg.poison_rate,
        "clean_acc": ca, "asr": asr, "ssim": avg_ssim, "psnr": avg_psnr,
        "l2": avg_l2, "linf": avg_linf, "lpips": avg_lpips if avg_lpips is not None else "",
    })
    logger.close()


if __name__ == "__main__":
    main()
