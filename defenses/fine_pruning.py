"""
Fine-Pruning (Liu et al., 2018): try to remove the backdoor from the model.

1. Pruning: rank the 512 channels of ResNet-18's layer4 by how active they are on clean images and
   switch off the least active ones, 5% at a time, until clean accuracy would drop by more than
   max_ca_drop. The idea is that the backdoor often lives in channels that clean images hardly use.
2. Fine-tuning: train the pruned model for a few epochs on a small clean set (the defender's data).

Works on a copy, so the model passed in is left as it is.
"""

import copy

import torch
import torch.nn.functional as F


class _ChannelMask:
    def __init__(self, layer, n_channels, device):
        self.mask = torch.ones(n_channels, device=device)
        self.handle = layer.register_forward_hook(lambda mod, inp, out: out * self.mask[None, :, None, None])


@torch.no_grad()
def _mean_activation(model, layer, images, normalize, device, batch=500):
    acts = []
    h = layer.register_forward_hook(lambda mod, inp, out: acts.append(out.mean(dim=(0, 2, 3)).cpu()))
    for i in range(0, len(images), batch):
        model(normalize(images[i:i + batch]).to(device))
    h.remove()
    return torch.stack(acts).mean(0)


@torch.no_grad()
def accuracy(model, images, labels, normalize, device, batch=500):
    model.eval()
    hits = 0
    for i in range(0, len(images), batch):
        hits += (model(normalize(images[i:i + batch]).to(device)).argmax(1).cpu() == labels[i:i + batch]).sum().item()
    return hits / len(images)


def fine_prune(model, layer_name, defender_x, defender_y, eval_clean, eval_labels, eval_poison, target,
               normalize, device="cuda", max_ca_drop=0.05, step=0.05, ft_epochs=5, ft_lr=0.01, batch=128, seed=0):
    """CA and ASR before the defence, after pruning and after fine-tuning."""
    model = copy.deepcopy(model).to(device).eval()
    layer = dict(model.named_modules())[layer_name]
    tgt = torch.full((len(eval_poison),), target)
    ca0 = accuracy(model, eval_clean, eval_labels, normalize, device)
    asr0 = accuracy(model, eval_poison, tgt, normalize, device)

    order = torch.argsort(_mean_activation(model, layer, defender_x, normalize, device))   # least active first
    n = len(order)
    cm = _ChannelMask(layer, n, device)
    # the defender only has its own clean data, so the stopping rule is based on defender_x
    base_def = accuracy(model, defender_x, defender_y, normalize, device)
    pruned = 0
    for frac in [step * k for k in range(1, int(1 / step))]:
        k = int(frac * n)
        cm.mask.fill_(1.0)
        cm.mask[order[:k].to(device)] = 0.0
        if base_def - accuracy(model, defender_x, defender_y, normalize, device) > max_ca_drop:
            break
        pruned = k
    cm.mask.fill_(1.0)
    cm.mask[order[:pruned].to(device)] = 0.0
    ca_p = accuracy(model, eval_clean, eval_labels, normalize, device)
    asr_p = accuracy(model, eval_poison, tgt, normalize, device)

    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=ft_lr, momentum=0.9, weight_decay=5e-4)
    for _ in range(ft_epochs):
        model.train()
        perm = torch.randperm(len(defender_x))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(normalize(defender_x[idx]).to(device)), defender_y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    ca_f = accuracy(model, eval_clean, eval_labels, normalize, device)
    asr_f = accuracy(model, eval_poison, tgt, normalize, device)
    cm.handle.remove()
    return {"fp_pruned_fraction": pruned / n, "ca_before": ca0, "asr_before": asr0, "ca_pruned": ca_p,
            "asr_pruned": asr_p, "ca_finetuned": ca_f, "asr_finetuned": asr_f}
