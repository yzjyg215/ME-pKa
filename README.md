# ME-pKa: A deep learning method with multimodal learning for protein pKa prediction

## Introduction

This repository provides codes and materials associated with the manuscript [ME-pKa: A deep learning method with multimodal learning for protein pKa prediction].

- **ME-pKa** is a novel pka prediction method which takes advantages of multi-fidelity learning, multimodal information.
The prediction process consisted of four steps: local features generation, FASTA feature generation, multimodal information learning ,and proteion pKa prediction.

We acknowledge the paper [Cai et al (2023). Basis for Accurate Protein p K a Prediction with Machine Learning. Journal of Chemical Information and Modeling 2023, 63 (10), 2936-2947.](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00254) and the [ESM-2](https://github.com/facebookresearch/esm) repository.

## Overview 

- ```pka_predict/```: the source codes of ME-pKa.
- ```data/```: the pre-training dataset, fine-tuning dataset, PE-pKa dataset used in ME-pKa.
- ```pka_process/```: the code for generating local features of amino acids.

## Dependencies

```
cuda >= 8.0 + cuDNN
python>=3.6
flask>=1.1.2
gunicorn>=20.0.4
hyperopt>=0.2.3
matplotlib>=3.1.3
numpy>=1.18.1
pandas>=1.0.3
pandas-flavor>=0.2.0
pip>=20.0.2
pytorch>=1.4.0
rdkit>=2020.03.1.0
scipy>=1.4.1
tensorboardX>=2.0
torchvision>=0.5.0
tqdm>=4.45.0
einops>=0.3.2
seaborn>=0.11.1
```
## pka_process
## Installation

Install Anaconda, Charmm, Chimera, then

```
cd pka_process
conda env create -f evironment.yml
```

## Usage

Using functions preprocess_expt_data() in 'pka_process/main.py' can create input data.

## pka_predict

The main functions of the program is used to train models and evaluate models.

## Usage

Using 'pka_predict/train.py' to train models.
Using function evaluate_single_model() in 'pka_predict/evaluate.py' to evaluate models.
