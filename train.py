"""
Train a ResNet-18 on CIFAR-10 with a chosen backdoor attack injected into a
fraction of the training set.
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
    p.add_argument("--attack", type=str, default=cfg.attack_name, choices=["blend", "sig", "wanet"])
    p.add_argument("--poison_rate", type=float, default=cfg.poison_rate)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--target_label", type=int, default=cfg.target_label)
    return p.parse_args()


def get_dataloaders(cfg):
    # NOTE: normalization deliberately left out of the *poisoning* transform chain
    # so triggers are applied in [0,1] pixel space; normalization is applied after.
    train_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True)
    test_base = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True)

    train_set = PoisonedDataset(train_base, cfg, train=True)

    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    def collate(batch):
        imgs, labels, is_poison = zip(*batch)
        imgs = torch.stack([normalize(im) for im in imgs])
        labels = torch.tensor(labels, dtype=torch.long)
        is_poison = torch.tensor(is_poison, dtype=torch.long)
        return imgs, labels, is_poison

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, collate_fn=collate, drop_last=True)

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

    set_seed(cfg.seed)
    device = cfg.device
    print(f"[INFO] Training on device: {device}, attack={cfg.attack_name}, poison_rate={cfg.poison_rate}")

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
