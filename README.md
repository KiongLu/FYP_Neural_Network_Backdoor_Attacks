# FYP_Neural_Network_Backdoor_Attacks
Final Year Project investigating invisible backdoor attacks in deep neural networks through the implementation and evaluation of Blend, SIG, and WaNet attacks.

## Project structure
```
FYP/
├── attacks/
│   ├── blend.py
│   ├── sig.py
│   ├── wanet.py
│   ├── clean.py
│   └── hybrid.py
├── datasets/
│   └── poison_dataset.py
├── models/
│   └── resnet.py   
├── train.py
├── evaluate.py
├── config.py
├── utils.py
└── notebooks/
    ├── data_exploration.ipynb
    └── result_visualization.ipynb
```

## Setup
```bash
pip install pip install -r requirements.txt
```
or

```bash
pip install torch torchvision numpy pillow matplotlib pandas ipykernel
```
CIFAR-10 downloads automatically into `./data` on first run.

## Running experiments
Train a backdoored model:
```bash
python train.py --attack blend --poison_rate 0.05
python train.py --attack sig   --poison_rate 0.10
python train.py --attack wanet --poison_rate 0.01
```

Evaluate a trained checkpoint:
```bash
python evaluate.py --attack blend --poison_rate 0.05 --checkpoint checkpoints/cifar10_resnet18_blend_pr0.05_final.pt
```

Train a **hybrid** backdoored model (2 or 3 attacks combined, each at reduced intensity):
```bash
python train.py --attack blend_sig      --poison_rate 0.05
python train.py --attack blend_wanet    --poison_rate 0.05
python train.py --attack sig_wanet      --poison_rate 0.05
python train.py --attack blend_sig_wanet --poison_rate 0.05
```

Each component's individual intensity (Blend's blend ratio, SIG's signal
amplitude, WaNet's warping strength) is automatically scaled by `1/N` where
`N` is the number of attacks combined - so a 2-way hybrid uses each
component at half strength, and the 3-way hybrid uses each at a third
strength. This keeps the combined trigger's overall visual perturbation
comparable to (or smaller than) any single full-strength attack. See
`attacks/hybrid.py` for details and rationale.

Evaluate a trained checkpoint:
```bash
python evaluate.py --attack blend --poison_rate 0.05 \
    --checkpoint checkpoints/cifar10_resnet18_blend_pr0.05_final.pt
```

This reports:
- **Clean Accuracy (CA)** — accuracy on the untouched test set
- **Attack Success Rate (ASR)** — % of triggered non-target samples classified as the target label
- **SSIM / PSNR** — stealthiness of the trigger vs. clean images

Results append to `results/summary.csv`, ready to be pulled into
`notebooks/result_visualization.ipynb` for comparison plots across attacks
and poisoning rates
