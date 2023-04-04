# Flipper: Benchmarks for Protein Engineering

This repository is a reproduction of the [Fitness Landscape Inference for Proteins (FLIP) benchmark](https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1.full), built with [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/1.9.1/) and [Hydra](https://github.com/ashleve/lightning-hydra-template).

<!-- * a few modular parts, configured through yaml files - most important are data, model, callbacks, trainer -->

### Quick start
Instructions to get a baseline model running on a GB1 split are included below.
1. Initialize conda environment
2. Unzip data files with splits
3. Run main.py
```bash
conda env create -n flipper --file environment.yml
cd data/aav && unzip splits.zip && cd ../..
python src/train.py
```
