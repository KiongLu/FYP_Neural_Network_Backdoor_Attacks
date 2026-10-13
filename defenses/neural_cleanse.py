"""
Neural Cleanse (Wang et al., 2019): look for a backdoor by reverse-engineering a trigger per class.

For every class t it finds a mask m (1 x H x W) and pattern p (3 x H x W), both in [0, 1], so that
clean images stamped as x' = (1 - m) * x + m * p get classified as t, while keeping the mask as
small as possible (L1 penalty with the adaptive weight from the paper). A backdoored class should
need a much smaller mask than the rest. Each class gets an anomaly index
|L1_t - median| / (1.4826 * MAD), and a class with a below-median L1 and an index above 2 is flagged.
"""

import numpy as np
import torch
import torch.nn.functional as F


def reverse_engineer(model, images, target, normalize, device="cuda", epochs=30, batch=128, lr=0.1,
                     init_cost=1e-3, asr_threshold=0.99, patience=5, seed=0):
    """Best (mask L1, ASR, mask, pattern) found for one class."""
    g = torch.Generator().manual_seed(seed)
    m_raw = torch.full((1, 1, *images.shape[-2:]), -2.0, device=device, requires_grad=True)   # sigmoid(-2) ~ 0.12
    p_raw = torch.zeros((1, 3, *images.shape[-2:]), device=device).normal_(0, 0.1).requires_grad_(True)
    opt = torch.optim.Adam([m_raw, p_raw], lr=lr, betas=(0.5, 0.9))
    cost, up, down = init_cost, 0, 0
    best = (float("inf"), 0.0, None, None)
    y = torch.full((batch,), target, device=device, dtype=torch.long)
    for _ in range(epochs):
        order = torch.randperm(len(images), generator=g)
        hits, seen = 0, 0
        for i in range(0, len(images), batch):
            x = images[order[i:i + batch]].to(device)
            m, p = torch.sigmoid(m_raw), torch.sigmoid(p_raw)
            logits = model(normalize((1 - m) * x + m * p))
            loss = F.cross_entropy(logits, y[:len(x)]) + cost * m.abs().sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            hits += (logits.argmax(1) == target).sum().item()
            seen += len(x)
        asr, l1 = hits / seen, torch.sigmoid(m_raw).sum().item()
        if asr >= asr_threshold and l1 < best[0]:
            best = (l1, asr, torch.sigmoid(m_raw).detach().cpu(), torch.sigmoid(p_raw).detach().cpu())
        # adaptive L1 weight as in the paper: push it up while the trigger works, down when it doesn't
        if asr >= asr_threshold:
            up, down = up + 1, 0
        else:
            up, down = 0, down + 1
        if up >= patience:
            cost, up = cost * 1.5, 0
        elif down >= patience:
            cost, down = cost / 1.5 ** 1.5, 0
    if best[2] is None:                     # never got to the threshold, so just report the last mask
        best = (torch.sigmoid(m_raw).sum().item(), asr, torch.sigmoid(m_raw).detach().cpu(),
                torch.sigmoid(p_raw).detach().cpu())
    return best


def anomaly_indices(l1_norms):
    l1 = np.asarray(l1_norms, dtype=float)
    med = np.median(l1)
    mad = 1.4826 * np.median(np.abs(l1 - med))
    return (l1 - med) / max(mad, 1e-12)      # negative = smaller than the median, which is the suspicious side


def neural_cleanse(model, images, normalize, num_classes=10, target_label=0, device="cuda", **kw):
    """Run Neural Cleanse on every class and summarise it for the actual target class."""
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    results = [reverse_engineer(model, images, t, normalize, device=device, **kw) for t in range(num_classes)]
    for prm in model.parameters():
        prm.requires_grad_(True)
    l1 = [r[0] for r in results]
    ai = anomaly_indices(l1)
    flagged = [t for t in range(num_classes) if ai[t] < -2]
    others = [l1[t] for t in range(num_classes) if t != target_label]
    return {
        "nc_target_l1": l1[target_label],
        "nc_median_l1": float(np.median(l1)),
        "nc_min_other_l1": float(min(others)),
        "nc_target_anomaly": float(-ai[target_label]),          # > 2 means the target class is flagged
        "nc_max_anomaly": float(-ai.min()),                     # strongest outlier over all classes
        "nc_flagged": ",".join(map(str, flagged)),
        "nc_detected": target_label in flagged,
        "nc_l1_all": l1,
        "nc_masks": [r[2] for r in results],
        "nc_patterns": [r[3] for r in results],
    }
