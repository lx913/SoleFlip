# SoleFlip

This repository provides the code for the paper "Backdoor Attacks on Neural Networks via One-Bit Flip", including training benign models, backdoor injection simulation, and performance testing.

## Environment

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── models/
│   └── *.py                       # Model templates
├── bash/
│   └── *.sh                       # Training bash scripts for different datasets/models
├── saved_model/                   # Directory to store trained and backdoored models
├── train_clean_model.py           # Clean training with various models
├── inject_backdoor.py             # Backdoor injection via bit-level weight manipulation
└── test_attack_performance.py     # Evaluate attack success rate and model robustness
```

##  Training Benign Models

Example: Train multiple ResNets on CIFAR-10

```bash
bash resnet_cifar10.sh
```

Or use `train_clean_model.py` directly:

```bash
python train_clean_model.py -dataset CIFAR10 -backbone resnet -device 0 -batch_size 512 -epochs 200 -lr 0.1 -weight_decay 1e-3 -model_num 1 -optimizer SGD
```

## Backdoor Injection

Example: Find exploitable weights and generate triggers for backdoor injection on CIFAR10/ResNet with a mini batch of the dataset

```bash
python inject_backdoor.py -dataset CIFAR10 -backbone resnet -device 0
```

## Performance Evaluation

Example: Test the benign accuracy degradation and the attack success rate on the whole test dataset.

```bash
python test_attack_performance.py -dataset CIFAR10 -backbone resnet -device 0
```
