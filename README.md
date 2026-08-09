# FYP_Neural_Network_Backdoor_Attacks
Final Year Project investigating invisible backdoor attacks in deep neural networks through the implementation and evaluation of Blend, SIG, and WaNet attacks.

## Project structure
```
FYP/
├── attacks/
│   ├── blend.py      # Blended trigger attack (Chen et al., 2017)
│   ├── sig.py         # Sinusoidal signal / clean-label attack (Barni et al., 2019)
│   └── wanet.py        # Warping-based invisible attack (Nguyen & Tran, ICLR 2021)
├── datasets/
│   └── poison_dataset.py   # Poisoned dataset wrappers (train / test / ASR eval)
├── models/
│   └── resnet.py            # CIFAR-style ResNet-18
├── train.py                 # Training loop (clean-label & dirty-label attacks)
├── evaluate.py               # Clean Accuracy / ASR / SSIM / PSNR evaluation
├── config.py                  # Central experiment configuration
├── utils.py                    # Seeding, metrics, checkpointing, logging
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
pip install torch torchvision numpy pillow
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
This reports:
- **Clean Accuracy (CA)** — accuracy on the untouched test set
- **Attack Success Rate (ASR)** — % of triggered non-target samples classified as the target label
- **SSIM / PSNR** — stealthiness of the trigger vs. clean images

Results append to `results/summary.csv`, ready to be pulled into
`notebooks/result_visualization.ipynb` for comparison plots across attacks
and poisoning rates
