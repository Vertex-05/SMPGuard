<img src="resource/pyg_logo.png" style="height: 60px;" align="right">

# SMPGuard: An Adaptive Backdoor Defense Framework

![PyTorch 1.11](https://img.shields.io/badge/PyTorch-1.11-brightgreen) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

SMPGuard is an adaptive backdoor defense framework that does **not** require a pre-collected trusted clean training dataset. It screens a potentially poisoned training set for high-confidence reference samples, then repairs the compromised model by pruning neurons whose activation distributions deviate from the clean reference. SMPGuard is built on top of [BackdoorBench](https://github.com/SCLBD/BackdoorBench).

---

## Features

- **Trusted-data-free** — no externally collected clean training set is needed.
- **Two-stage pipeline** — data-level sample screening followed by model-level neuron repair.
- **Two pruning strategies** — global and layer-wise threshold search, both constrained by clean accuracy.
- **Model- and attack-agnostic** — evaluated across standard vision backbones and backdoor attacks.

---

## Installation

```bash
git clone https://github.com/Vertex-05/SMPGuard.git
cd SMPGuard
conda create -n smpguard python=3.8
conda activate smpguard
sh ./sh/install.sh
sh ./sh/init_folders.sh
```

---

## Quick Start

### Step 1 — Train a backdoored model

SMPGuard operates on the output of a backdoor attack. Example below uses **BadNets on CIFAR-10**:

```bash
# generate the trigger
python ./resource/badnet/generate_white_square.py \
    --image_size 32 --square_size 3 \
    --distance_to_right 0 --distance_to_bottom 0 \
    --output_path ./resource/badnet/trigger_image.png

# train the backdoored model
python ./attack/badnet.py \
    --yaml_path ../config/attack/prototype/cifar10.yaml \
    --patch_mask_path ../resource/badnet/trigger_image.png \
    --save_folder_name badnet_cifar10_0_1
```

Other attacks follow the same pattern — swap `badnet.py` for any script under `./attack` (e.g. `blended.py`, `wanet.py`, `inputaware.py`), and adjust the YAML path. Outputs are stored under `./record/<save_folder_name>/`, including `attack_result.pt` used by the next step.

### Step 2 — Apply SMPGuard defense

SMPGuard provides two pruning-parameter search strategies:

- **Global search** — `./defense/bnp1.py`
- **Layer-wise search** — `./defense/bnp.py`

Example (global search):

```bash
python ./defense/bnp1.py \
    --result_file badnet_cifar10_0_1 \
    --yaml_path ./config/defense/bnp/config.yaml \
    --dataset cifar10 \
    --u_min 1.0 --u_max 5.0 --u_num 41 \
    --acc_drop_tolerance 0.015 \
    --val_ratio 0.2
```

To use the layer-wise variant, replace `bnp1.py` with `bnp.py`. Results (repaired model, search log, and final ACC / ASR / RA summary) are written under `./record/<result_file>/defense/bnp/`.

---

## Method Overview

SMPGuard operates in two stages:

1. **Sample Screening** — assigns a confidence score to each training sample via bi-level optimization, and retains the most trustworthy subset as reference data. Coarse-to-fine screening keeps the cost manageable.
2. **Neuron Repair** — compares clean-reference activation statistics with the model's trained normalization statistics using **KL divergence**, and prunes neurons whose divergence exceeds a robust threshold `T = Median(S) + u · MAD(S)`. The threshold parameter `u` is selected under a **clean-accuracy constraint** so that the repaired model does not trade away too much clean performance.

The two pruning-parameter strategies above differ only in how `u` is chosen: a single global `u` shared across all normalization layers, or a per-layer set `{u_l}` found by coordinate search.

---

## Model and Data Downloading

Pre-trained backdoor models and their corresponding poisoned data are available via the link below. Each archive follows the BackdoorBench format:

- **bd_train_dataset** — poisoned training data
- **bd_test_dataset** — backdoor test data
- **attack_result.py** — backdoor model + data loader
- **cross_test_dataset** — cross-mode data (for attacks such as WaNet, Input-aware)

Use `load_attack_result` in [`utils/save_load_attack.py`](./utils/save_load_attack.py) to load them.

**Download:** [Backdoor Model Sharepoint](https://cuhko365.sharepoint.com/:f:/s/SDSbackdoorbench/EmYD8BoPY8hAqNCV_Rb_zwsBFdqf88Yx01xi0V8tc4whvw?e=d7oJNc)

---

## Citation

If you find SMPGuard useful, please cite:

**SMPGuard** — *add your paper citation here once published.*

SMPGuard is built on [BackdoorBench](https://github.com/SCLBD/BackdoorBench):

```bibtex
@article{backdoorbench2025,
  title={BackdoorBench: A Comprehensive Benchmark and Analysis of Backdoor Learning},
  author={Wu, Baoyuan and Chen, Hongrui and Zhang, Mingda and Zhu, Zihao and Wei, Shaokui and Yuan, Danni and Zhu, Mingli and Wang, Ruotong and Liu, Li and Shen, Chao},
  journal={International Journal of Computer Vision (IJCV)},
  year={2025}
}

@inproceedings{backdoorbench,
  title={BackdoorBench: A Comprehensive Benchmark of Backdoor Learning},
  author={Wu, Baoyuan and Chen, Hongrui and Zhang, Mingda and Zhu, Zihao and Wei, Shaokui and Yuan, Danni and Shen, Chao},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```

---

## Copyright

This repository is built on top of [BackdoorBench](https://github.com/SCLBD/BackdoorBench), which is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under the Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0, see [LICENSE](./LICENSE)). SMPGuard inherits this license.