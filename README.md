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
│   ├── badnets.py                # 3x3 patch attack, ONLY a positive control for the defences
│   ├── hybrid.py                   # Fixed-strength hybrids: combines 2-3 of the above,
│   │                                # each at 1/N intensity (or, for the budget-matched
│   │                                # SIG+WaNet baseline, fixed absolute strengths).
│   │                                # Always dirty-label.
│   └── adaptive_hybrid.py            # PROPOSED/EXPERIMENTAL: SIG+WaNet hybrid with
│                                       # per-image ADAPTIVE trigger strength (see below)
├── datasets/
│   ├── poison_dataset.py       # Poisoned dataset wrappers (train / test / ASR eval)
│   └── gtsrb.py                # GTSRB resized to 32x32 (cached as .npz) + dataset loader
├── models/
│   ├── resnet.py                 # CIFAR-style ResNet-18 (+ get_model)
│   └── vgg.py                    # CIFAR-style VGG-16 with batch norm (--model vgg16)
├── train.py                        # Training loop (all attacks, all toggles)
├── evaluate.py                       # CA / ASR / SSIM / PSNR / L2 / L_inf / (LPIPS) evaluation
├── evaluate_components.py            # Test-time ASR with only the SIG or only the warp
│                                     # component of a trained SIG+WaNet model
├── defenses/
│   ├── strip.py                      # STRIP entropy detection + AUROC / FAR@1%FRR
│   ├── gradcam.py                    # Grad-CAM for the ResNet-18 (+ map spread)
│   ├── spectral.py                   # Spectral Signatures + Activation Clustering
│   ├── frequency.py                  # DCT energy concentration + DCT-based detector
│   ├── neural_cleanse.py             # Neural Cleanse trigger reverse-engineering + anomaly index
│   └── fine_pruning.py               # Fine-Pruning (prune dormant layer4 channels + fine-tune)
├── select_adaptive_params.py         # Picks the Adaptive v2 setting on a held-out
│                                     # validation split (needs the --holdout runs)
├── inspect_poison.py                   # Visualize which TRAINING samples got poisoned
├── inspect_evaluation.py                 # Visualize ASR TEST samples + model predictions
├── calibrate_adaptive_stats.py             # Recompute Adaptive Hybrid's calibration bounds
│                                             # from real data (see Appendix A of the report)
├── config.py                                # Central experiment configuration
├── utils.py                                   # Seeding, metrics, checkpointing, logging
└── notebooks/
    ├── data_exploration.ipynb       # triggers, Adaptive Hybrid behaviour, per-image stealth
    ├── result_visualization.ipynb   # all result figures (incl. component ablation)
    └── defense_analysis.ipynb       # STRIP, Grad-CAM, data-level and DCT analyses (1% models)
```

## Setup
```bash
pip install -r requirements.txt
```
CIFAR-10 downloads automatically into `./data` on first run (GTSRB too, when `--dataset gtsrb`
is used). On Windows,
make sure you're using a CUDA-enabled PyTorch build if you have a GPU (see
`nvidia-smi` to check your CUDA version, then install from
https://pytorch.org/get-started/locally/ with the matching `--index-url`).

**Blend trigger image (not included).** The Blend attack blends a trigger image into the poisoned
images. The results in the report used a cartoon image (`triggers/hello_kitty.png`), which is not in
this repository because of copyright. To run Blend, put any RGB image at `./triggers/hello_kitty.png`
(or change `blend_trigger_path` in `config.py`); it is resized to 32x32. Without an image, the code
prints a warning and uses a fixed random-noise trigger instead, so everything still runs.

Because the trigger is different, Blend and the hybrids that contain it (Blend+SIG, Blend+WaNet,
Blend+SIG+WaNet) will not reproduce the exact numbers in the report. Every other attack (SIG, WaNet,
SIG+WaNet, the matched baseline and the adaptive attacks) does not use the image and is unaffected.

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

`results/summary.csv` is an append-only log (repeated runs; older rows log clean- and dirty-label
SIG both as `sig`, newer ones as `sig_clean` / `sig_dirty`). `results/report_results.csv` is the curated one-row-per-configuration
table that matches the report; `notebooks/result_visualization.ipynb` reads it (and
cross-checks it against `summary.csv`), and `notebooks/data_exploration.ipynb` visualises the
triggers. Both notebooks save their figures to `results/figures/`.

Train the **budget-matched Fixed SIG+WaNet baseline**: the same fixed, non-adaptive
hybrid with each component at Adaptive v2's *average* strength (warp 0.27, signal
amplitude 10.6; `sig_wanet_matched_*` in `config.py`). It separates the effect of the
adaptive strengths and texture mask from the effect of moving budget from the warp
to the signal:
```bash
python train.py --attack sig_wanet_matched --poison_rate 0.01
```

Which trigger component does each trained SIG+WaNet model rely on? Re-measure the ASR
at test time with only the SIG or only the warp component (no retraining):
```bash
python evaluate_components.py          # writes results/component_asr.csv
```

Defence analyses on the 1% models (STRIP, Grad-CAM, Spectral Signatures, Activation
Clustering, a DCT frequency-domain detector tested on seen and unseen triggers, Neural Cleanse
and Fine-Pruning; the last two also on the SIG+WaNet variants at 0.5% and 5%): run
`notebooks/defense_analysis.ipynb`. It writes `results/strip_results.csv`,
`results/gradcam_results.csv`, `results/data_level_results.csv`, `results/frequency_results.csv`,
`results/frequency_unseen_results.csv`, `results/neural_cleanse_results.csv`,
`results/fine_pruning_results.csv`, the sensitivity checks
`results/neural_cleanse_l1_all.csv`, `results/neural_cleanse_threshold_sensitivity.csv` and
`results/fine_pruning_budget_sensitivity.csv`, and figures in `results/figures/`. It needs the BadNets
positive-control model:
```bash
python train.py --attack badnets --poison_rate 0.01
```

### Run variants (seeds, augmentation, validation split, config overrides)

All of these are off by default, and each one only adds a suffix to the checkpoint/log name
when it is used, so the default runs keep their original names. `evaluate.py` takes the
same options and must be given the ones the checkpoint was trained with.
```bash
# another random seed (changes the poisoned images AND the trigger pattern) -> ..._s1_pr0.01
python train.py --attack sig_wanet --poison_rate 0.01 --seed 1
python evaluate.py --attack sig_wanet --poison_rate 0.01 --seed 1 --checkpoint checkpoints/cifar10_resnet18_sig_wanet_s1_pr0.01_final.pt

# standard augmentation (random crop with 4-px padding + horizontal flip, after poisoning) -> ..._aug_
python train.py --attack wanet --poison_rate 0.01 --augment
# only one of the two augmentations -> ..._augcrop_ / ..._augflip_
python train.py --attack wanet --poison_rate 0.01 --augment --augment_mode crop

# another architecture -> cifar10_vgg16_...
python train.py --attack adaptive_hybrid --poison_rate 0.01 --model vgg16

# another dataset: GTSRB, resized to 32x32 and cached in data/ on first use -> gtsrb_resnet18_...
# The adaptive attack needs GTSRB's own calibration range (run the calibration script and pass the
# four printed values with --set, the same in train.py and evaluate.py):
python calibrate_adaptive_stats.py --dataset gtsrb
python train.py --dataset gtsrb --attack adaptive_hybrid --poison_rate 0.01 \
    --set adaptive_v_min=0.0300 --set adaptive_v_max=0.3190 --set adaptive_e_min=0.0771 --set adaptive_e_max=0.8142

# keep 5,000 training images aside as a validation set -> ..._val5000_
python train.py --attack adaptive_hybrid --poison_rate 0.005 --holdout 5000 --tag_suffix _candC1

# override any config.py setting for one run (e.g. a candidate parameter setting)
python train.py --attack adaptive_hybrid --poison_rate 0.005 --holdout 5000 --tag_suffix _candC2 --set adaptive_hybrid_beta_min=12.0 --set adaptive_hybrid_beta_max=20.0

# then choose the Adaptive v2 setting on the validation images (writes results/validation_selection.csv)
python select_adaptive_params.py
```
`--save_every N` saves an intermediate checkpoint every N epochs (default 20; 0 = final only).
`--epochs` and `--target_label` override `config.py` (defaults 100 and 0). The helper scripts also take
`--n_show` (`inspect_poison.py` / `inspect_evaluation.py`: how many images to show) and `--rates` / `--out`
(`evaluate_components.py`: which poisoning rates to test and where to write the CSV). Every script
prints its full list of options with `--help`.

Visualize what actually happened:
```bash
# Which training images got poisoned, and what did their labels become?
python inspect_poison.py --attack sig --poison_rate 0.01

# Which test images were used for ASR, and did the trained model actually
# get fooled by each one? (pass --checkpoint to see model predictions)
python inspect_evaluation.py --attack wanet --poison_rate 0.01 --checkpoint checkpoints/cifar10_resnet18_wanet_pr0.01_final.pt
```

## Suggested experiment matrix

**Core comparison** (poison rates 0.5% / 1% / 5%):

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
| Fixed SIG+WaNet (matched) | Fixed baseline at Adaptive v2's average strengths — is the gain from adaptivity or from the budget split? |
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
  harder for defences like Neural Cleanse/STRIP to detect.
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
  `calibrate_adaptive_stats.py` on CIFAR-10. For another dataset, re-run it with `--dataset`
  and pass the printed values with `--set` (see the GTSRB example above).
- GTSRB is supported through `--dataset gtsrb` in `train.py`, `evaluate.py` and
  `calibrate_adaptive_stats.py` (43 classes, target class 0 = the 20 km/h sign). The helper
  scripts (`inspect_poison.py`, `inspect_evaluation.py`, `evaluate_components.py`,
  `select_adaptive_params.py`) are CIFAR-10 only.