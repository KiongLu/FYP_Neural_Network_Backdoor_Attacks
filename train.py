"""
Train a ResNet-18 on CIFAR-10 with a chosen backdoor attack injected into a
fraction of the training set.

Usage:
    python train.py --attack blend --poison_rate 0.05
    python train.py --attack sig   --poison_rate 0.10
    python train.py --attack wanet --poison_rate 0.01
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
from datasets.poison_dataset import PoisonedDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", type=str, default=cfg.attack_name, choices=["clean", "blend", "sig", "wanet", "blend_sig", "blend_wanet", "sig_wanet", "blend_sig_wanet", "adaptive_hybrid", "adaptive_spatial", "adaptive_freq"])
    p.add_argument("--poison_rate", type=float, default=cfg.poison_rate)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--target_label", type=int, default=cfg.target_label)
    p.add_argument("--sig_dirty_label", action="store_true",
                    help="Use dirty-label SIG instead of the default clean-label SIG "
                         "(poisons any class and flips label to target, for comparison).")
    return p.parse_args()


class PoisonCollate:
    """
    Module-level (not a closure) so it is picklable by Windows' 'spawn'
    multiprocessing start method when num_workers > 0.
    """
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, batch):
        imgs, labels, is_poison = zip(*batch)
        imgs = torch.stack([self.normalize(im) for im in imgs])
        labels = torch.tensor(labels, dtype=torch.long)
        is_poison = torch.tensor(is_poison, dtype=torch.long)
        return imgs, labels, is_poison


def get_dataloaders(cfg):
    # NOTE: normalization deliberately left out of the *poisoning* transform chain
    # so triggers are applied in [0,1] pixel space; normalization is applied after.
    train_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True)
    test_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True)

    train_set = PoisonedDataset(train_base, cfg, train=True)

    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    collate = PoisonCollate(normalize)

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

    if cfg.attack_name == "clean":
        cfg.poison_rate = 0.0  # baseline: no poisoning regardless of what was passed

    set_seed(cfg.seed)
    device = cfg.device
    print(f"[INFO] Training on device: {device}, attack={cfg.attack_name}, poison_rate={cfg.poison_rate}")
    if cfg.attack_name.lower() == "sig":
        mode = "CLEAN-LABEL (poisons target-class images only, label unchanged)" if cfg.sig_clean_label \
            else "DIRTY-LABEL (poisons any class, label flipped to target)"
        print(f"[INFO] SIG mode: {mode}")

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

        if epoch % 20 == 0 or epoch == cfg.epochs:
            ckpt_path = f"{cfg.checkpoint_dir}/{cfg.tag()}_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    logger.close()
    final_path = f"{cfg.checkpoint_dir}/{cfg.tag()}_final.pt"
    save_checkpoint(model, optimizer, cfg.epochs, final_path)
    print(f"[INFO] Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
