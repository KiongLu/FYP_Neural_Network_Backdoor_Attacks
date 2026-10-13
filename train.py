"""
Train a model on CIFAR-10 with one of the backdoor attacks applied to part of the training set.

Examples:
    python train.py --attack blend --poison_rate 0.05
    python train.py --attack sig --sig_dirty_label --poison_rate 0.01
    python train.py --attack adaptive_hybrid --poison_rate 0.005 --seed 1
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from utils import set_seed, save_checkpoint, CSVLogger, AverageMeter
from models.resnet import get_model
from datasets.poison_dataset import PoisonedDataset, split_holdout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=str, default=cfg.attack_name, choices=["clean", "badnets", "blend", "sig", "wanet", "blend_sig", "blend_wanet", "sig_wanet", "sig_wanet_matched", "blend_sig_wanet", "adaptive_hybrid", "adaptive_spatial", "adaptive_freq"])
    p.add_argument("--poison_rate", type=float, default=cfg.poison_rate)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--target_label", type=int, default=cfg.target_label)
    p.add_argument("--sig_dirty_label", action="store_true",
                    help="Use dirty-label SIG instead of the default clean-label SIG "
                         "(poisons any class and flips label to target, for comparison).")
    p.add_argument("--seed", type=int, default=cfg.seed,
                    help="Random seed (poison selection, trigger patterns, initialisation). Non-default "
                         "seeds add '_s<seed>' to the checkpoint name.")
    p.add_argument("--augment", action="store_true",
                    help="Train with random crop (4-px padding) and horizontal flip, applied after poisoning.")
    p.add_argument("--augment_mode", choices=["crop_flip", "crop", "flip"], default="crop_flip",
                    help="Which augmentation --augment applies (non-default modes add '_augcrop' / '_augflip').")
    p.add_argument("--model", choices=["resnet18", "vgg16"], default=cfg.model_name,
                    help="Network architecture (part of the checkpoint name).")
    p.add_argument("--holdout", type=int, default=0,
                    help="Keep this many training images aside as a validation set (fixed split).")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="Override a config.py setting, e.g. --set adaptive_mask_floor=0.4 (repeatable).")
    p.add_argument("--tag_suffix", type=str, default="",
                    help="Extra suffix for checkpoint/log names (e.g. to tell --set candidates apart).")
    p.add_argument("--save_every", type=int, default=20,
                    help="Save an intermediate checkpoint every N epochs (0 = only the final model).")
    return p.parse_args()


class PoisonCollate:
    """
    Turns a list of samples into a batch (optionally with augmentation) and normalises it.
    It is a class rather than a nested function so that Windows can pickle it for the worker
    processes.
    """
    def __init__(self, normalize, augment=False, mode="crop_flip"):
        self.normalize = normalize
        self.augment = augment
        self.mode = mode

    @staticmethod
    def _crop_flip(img, pad=4, crop=True, flip=True):
        """The usual CIFAR augmentation: random crop from a zero-padded image, then a random flip."""
        if crop:
            h, w = img.shape[-2:]
            padded = torch.nn.functional.pad(img, (pad, pad, pad, pad))
            top, left = torch.randint(0, 2 * pad + 1, (2,)).tolist()
            img = padded[:, top:top + h, left:left + w]
        if flip:
            return img.flip(-1) if torch.rand(1).item() < 0.5 else img
        return img

    def __call__(self, batch):
        imgs, labels, is_poison = zip(*batch)
        if self.augment:
            imgs = [self._crop_flip(im, crop=self.mode != "flip", flip=self.mode != "crop") for im in imgs]
        imgs = torch.stack([self.normalize(im) for im in imgs])
        labels = torch.tensor(labels, dtype=torch.long)
        is_poison = torch.tensor(is_poison, dtype=torch.long)
        return imgs, labels, is_poison


def get_dataloaders(cfg):
    # The triggers are added to the raw [0, 1] images; normalisation only happens afterwards,
    # in PoisonCollate.
    train_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True)
    test_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True)

    if cfg.holdout:
        train_subset, _ = split_holdout(train_base, cfg.holdout)
        train_set = PoisonedDataset(train_subset, cfg, train=True)
    else:
        train_set = PoisonedDataset(train_base, cfg, train=True)

    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    collate = PoisonCollate(normalize, augment=cfg.augment, mode=cfg.augment_mode)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate, drop_last=True,
                               pin_memory=True, persistent_workers=(cfg.num_workers > 0))

    return train_loader, train_base, test_base, normalize


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        loss_meter.update(loss.item(), imgs.size(0))
        acc_meter.update(acc, imgs.size(0))
    return loss_meter.avg, acc_meter.avg


def main():
    args = parse_args()
    cfg.attack_name = args.attack
    cfg.poison_rate = args.poison_rate
    cfg.epochs = args.epochs
    cfg.target_label = args.target_label
    cfg.sig_clean_label = not args.sig_dirty_label
    cfg.seed = args.seed
    cfg.augment = args.augment
    cfg.augment_mode = args.augment_mode
    cfg.model_name = args.model
    cfg.holdout = args.holdout
    cfg.tag_suffix = args.tag_suffix
    cfg.apply_overrides(args.set)

    if cfg.attack_name == "clean":
        cfg.poison_rate = 0.0  # the clean model never has poisoned images

    set_seed(cfg.seed)
    device = cfg.device
    print(f"[INFO] Training on device: {device}, attack={cfg.attack_name}, poison_rate={cfg.poison_rate}")
    if cfg.attack_name.lower() == "sig":
        mode = "CLEAN-LABEL (poisons target-class images only, label unchanged)" if cfg.sig_clean_label \
            else "DIRTY-LABEL (poisons any class, label flipped to target)"
        print(f"[INFO] SIG mode: {mode}")
    if args.set or cfg.augment or cfg.holdout or cfg.seed != cfg.DEFAULT_SEED:
        print(f"[INFO] seed={cfg.seed} augment={cfg.augment} holdout={cfg.holdout} overrides={args.set} -> {cfg.tag()}")

    train_loader, train_base, test_base, normalize = get_dataloaders(cfg)

    model = get_model(cfg.model_name, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                           weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_schedule, gamma=cfg.lr_gamma)

    logger = CSVLogger(f"{cfg.results_dir}/{cfg.tag()}_train_log.csv",
                        fieldnames=["epoch", "loss", "train_acc", "lr", "time_sec"])

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:3d}/{cfg.epochs}] loss={loss:.4f} acc={acc:.4f} lr={current_lr:.5f} time={dt:.1f}s")
        logger.log({"epoch": epoch, "loss": loss, "train_acc": acc, "lr": current_lr, "time_sec": dt})

        if args.save_every and (epoch % args.save_every == 0 or epoch == cfg.epochs):
            ckpt_path = f"{cfg.checkpoint_dir}/{cfg.tag()}_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    logger.close()
    final_path = f"{cfg.checkpoint_dir}/{cfg.tag()}_final.pt"
    save_checkpoint(model, optimizer, cfg.epochs, final_path)
    print(f"[INFO] Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
