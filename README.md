# Implementation and Evaluation of Invisible Backdoor Attacks in DNNs,
# with a Proposed Adaptive Hybrid Extension

## Project structure
```
FYP/
├── attacks/
│   ├── blend.py             # Blended trigger attack (Chen et al., 2017) — dirty-label
│   ├── sig.py                # Sinusoidal signal attack (Barni et al., 2019) — toggleable
│   │                          # clean-label (default) / dirty-label
│   ├── wanet.py                # Warping-based invisible attack (Nguyen & Tran, ICLR 2021)
│   │                            # — dirty-label
│   ├── clean.py                  # No-op attack, used for the clean baseline model
│   ├── hybrid.py                   # Fixed-strength hybrids: combines 2-3 of the above,
│   │                                # each at 1/N intensity. Always dirty-label.
│   └── adaptive_hybrid.py            # PROPOSED/EXPERIMENTAL: SIG+WaNet hybrid with
│                                       # per-image ADAPTIVE trigger strength (see below)
├── datasets/
│   └── poison_dataset.py       # Poisoned dataset wrappers (train / test / ASR eval)
├── models/
│   └── resnet.py                 # CIFAR-style ResNet-18
├── train.py                        # Training loop (all attacks, all toggles)
├── evaluate.py                       # CA / ASR / SSIM / PSNR / L2 / L_inf / (LPIPS) evaluation
├── inspect_poison.py                   # Visualize which TRAINING samples got poisoned
├── inspect_evaluation.py                 # Visualize ASR TEST samples + model predictions
├── calibrate_adaptive_stats.py             # Recompute Adaptive Hybrid's calibration bounds
│                                             # from real data (see Appendix A of the report)
├── config.py                                # Central experiment configuration
├── utils.py                                   # Seeding, metrics, checkpointing, logging
└── notebooks/
    ├── data_exploration.ipynb
    └── result_visualization.ipynb
```

## Setup
```bash
pip install -r requirements.txt
```
CIFAR-10 downloads automatically into `./data` on first run. On Windows,
make sure you're using a CUDA-enabled PyTorch build if you have a GPU (see
`nvidia-smi` to check your CUDA version, then install from
https://pytorch.org/get-started/locally/ with the matching `--index-url`).

For the Blend attack, drop any RGB image at `./triggers/hello_kitty.png`
(or update `cfg.blend_trigger_path`). If the file is missing, a fixed
pseudo-random static trigger is used instead so the code still runs.

LPIPS (optional stealthiness metric) requires `pip install lpips`; if not
installed, `evaluate.py` reports it as N/A rather than failing.

## Running experiments

Train the clean baseline (no backdoor, for comparison):
```bash
python train.py --attack clean --poison_rate 0.0
```

Train a single-technique backdoored model:
```bash
python train.py --attack blend --poison_rate 0.05
python train.py --attack sig   --poison_rate 0.10
python train.py --attack wanet --poison_rate 0.01
```

SIG defaults to **clean-label** (only poisons target-class images, label
unchanged). In both modes the number of poisoned images is `poison_rate` x
the whole training set (e.g. 0.01 -> 500 images); in clean-label mode they
are all drawn from the target class. Add `--sig_dirty_label` to instead
poison any class with the label flipped to target (directly comparable to
Blend/WaNet):
```bash
python train.py --attack sig --poison_rate 0.05 --sig_dirty_label
```
Whichever mode you train with, evaluate with the **same** flag — the mode
isn't stored anywhere else, and checkpoints get distinct filenames
(`..._sig_clean_...` vs `..._sig_dirty_...`) so the two don't collide.

Train a **fixed-strength hybrid** (2 or 3 attacks combined, each at reduced intensity):
```bash
python train.py --attack blend_sig       --poison_rate 0.05
python train.py --attack blend_wanet     --poison_rate 0.05
python train.py --attack sig_wanet       --poison_rate 0.05
python train.py --attack blend_sig_wanet --poison_rate 0.05
```
Each component's individual intensity (Blend's blend ratio, SIG's signal
amplitude, WaNet's warping strength) is scaled by `1/N`, where `N` is the
number of attacks combined.

Train the **Adaptive Hybrid** (proposed/experimental — see `attacks/adaptive_hybrid.py`
and Appendix A of the report for the full write-up):
```bash
# Recommended first step: calibrate the adaptive statistics on real data
python calibrate_adaptive_stats.py
# then copy the 4 printed values into config.py (already done for CIFAR-10)

python train.py --attack adaptive_hybrid --poison_rate 0.01   # both components adaptive
python train.py --attack adaptive_spatial --poison_rate 0.01  # ablation: spatial only
python train.py --attack adaptive_freq --poison_rate 0.01     # ablation: frequency only
```

Evaluate a trained checkpoint:
```bash
python evaluate.py --attack blend --poison_rate 0.05 \
    --checkpoint checkpoints/cifar10_resnet18_blend_pr0.05_final.pt
```

This reports:
- **Clean Accuracy (CA)** — accuracy on the untouched test set
- **Attack Success Rate (ASR)** — % of triggered non-target samples classified as the target label
- **SSIM / PSNR** — stealthiness of the trigger vs. clean images
- **L2 / L∞ norm** — raw perturbation magnitude (average / worst-case pixel change)
- **LPIPS** — learned perceptual similarity, if the `lpips` package is installed

Results append to `results/summary.csv`, ready to be pulled into
`notebooks/result_visualization.ipynb` for comparison plots across attacks
and poisoning rates.

`results/summary.csv` is an append-only log (repeated runs, and clean- vs dirty-label SIG
both logged as `sig`). `results/report_results.csv` is the curated one-row-per-configuration
table that matches the report; `notebooks/result_visualization.ipynb` reads it (and
cross-checks it against `summary.csv`), and `notebooks/data_exploration.ipynb` visualises the
triggers. Both notebooks save their figures to `results/figures/`.

Visualize what actually happened:
```bash
# Which training images got poisoned, and what did their labels become?
python inspect_poison.py --attack sig --poison_rate 0.01

# Which test images were used for ASR, and did the trained model actually
# get fooled by each one? (pass --checkpoint to see model predictions)
python inspect_evaluation.py --attack wanet --poison_rate 0.01 \
    --checkpoint checkpoints/cifar10_resnet18_wanet_pr0.01_final.pt
```

## Suggested experiment matrix

**Core comparison** (poison rates 1% / 5% / 10%):

| Attack | Metric focus |
|--------|--------------|
| Clean (baseline) | CA reference point |
| Blend | CA, ASR, SSIM/PSNR, training time |
| SIG (clean-label, default) | CA, ASR, SSIM/PSNR — lower ASR expected, trade-off for no label flips |
| SIG (dirty-label, `--sig_dirty_label`) | Direct comparison to Blend/WaNet under the same convention |
| WaNet | CA, ASR, SSIM/PSNR, robustness to noise-mode covers |
| Blend+SIG / Blend+WaNet / SIG+WaNet / Blend+SIG+WaNet (fixed hybrids) | Same as above; each component at reduced intensity |

**Adaptive Hybrid study** (same poison rates: 0.5% / 1% / 5%):

| Attack | Purpose |
|--------|---------|
| Fixed SIG+WaNet | Baseline for the adaptive comparison |
| Adaptive SIG+WaNet | The proposed method — does adaptive strength beat fixed strength? |
| Adaptive spatial-only / Adaptive freq-only | Ablation — which component's adaptivity (if either) drives any observed benefit? |

## Notes for the report
- All attacks share the same `poison()` / `is_poisonable()` / `get_target_label()`
  interface, so `train.py`/`evaluate.py` stay attack-agnostic — swapping
  attacks is just a `--attack` flag, no code changes.
- Fixed hybrids and the Adaptive Hybrid are always **dirty-label**, even
  when SIG (clean-label by default) is a component — clean-label's
  target-class-only restriction doesn't compose with poisoning arbitrary
  classes across multiple mechanisms. This is a deliberate simplification,
  documented in the code and worth restating in your methodology/limitations
  section.
- WaNet (and the spatial component of the Adaptive Hybrid, when active)
  uses a "noise mode" (`cross_ratio` in `config.py`) — cover samples warped
  with a *different random* field each time, which is what makes it
  resistant to defenses like Neural Cleanse/STRIP.
- **WaNet must run on CPU inside the DataLoader**, even if you're training
  on GPU — `attacks/wanet.py` and `attacks/adaptive_hybrid.py` are
  hard-coded to `device="cpu"` internally regardless of `cfg.device`. This
  isn't a performance choice; it avoids a genuine CUDA/multiprocessing crash
  when `num_workers > 0` on Windows. The model itself still trains on GPU as
  normal — only trigger generation stays on CPU.
- **Adaptive Hybrid's `ALPHA_MIN` must stay clearly above `ADAPTIVE_COVER_ALPHA`**
  in `config.py` — if the poison-strength range overlaps the cover-sample
  strength, the model can't reliably distinguish real triggers from cover
  samples by magnitude alone, which measurably hurts ASR. See Appendix A.8
  of the report for the full explanation; the shipped defaults
  (`adaptive_alpha_min=0.45`, `adaptive_cover_alpha=0.20`) already keep a
  safe margin.
- **Adaptive SIG+WaNet v2** (`adaptive_hybrid` only): uses its own, narrower
  strength ranges (`adaptive_hybrid_*` in `config.py`) plus per-pixel
  texture masking of the sinusoidal signal (`adaptive_texture_mask`), which
  improves SSIM/PSNR beyond the Fixed SIG+WaNet hybrid. The ablations
  (`adaptive_spatial`, `adaptive_freq`) still use the original settings.
  Checkpoints from the original version are kept in
  `checkpoints/adaptive_hybrid_v1_backup/`; set `adaptive_texture_mask = False`
  and the `adaptive_hybrid_*` values back to 0.45/0.80/6/22/0.20 to
  reproduce them.
- The Adaptive Hybrid's `adaptive_v_min/max`/`adaptive_e_min/max`
  calibration constants in `config.py` are the values measured by
  `calibrate_adaptive_stats.py` on CIFAR-10; re-run it if you change dataset.
- GTSRB support: swap `torchvision.datasets.CIFAR10` for
  `torchvision.datasets.GTSRB` in `train.py`/`evaluate.py`, update
  `cfg.num_classes` accordingly, and recalibrate the Adaptive Hybrid's
  statistics for the new dataset.