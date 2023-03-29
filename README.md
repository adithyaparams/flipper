# Flipper: Benchmarks for Protein Engineering

This repository is a reproduction of the [Fitness Landscape Inference for Proteins (FLIP) benchmark](https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1.full), built with Pytorch Lightning. It separates the benchmark into three modular parts; model params (LightningModules), data extraction and preprocessing (DataModule), and model training (Trainer). This structure affords users the ability to plug in their own models to test on splits, add their own datasets, or experiment with training regularization with minimal overhead.

## Overview
General comments on folder breakup and structural changes compared to the [FLIP repo](https://github.com/J-SNACKKB/FLIP) included below.
* Includes functionality of baseline CNN and ESM models, benchmarked on GB1 and AAV datasets (`/models`)
* Sets up DataModule to be agnostic across datasets, with injectable preprocessing to speed up inference based on data source (`data.py`)
* Handles hyperparam initialization through command-line args and model training/inference through the Lightning Trainer (`main.py`)

### Quick start
Instructions to get a baseline model running on a GB1 split are included below.
1. Initialize conda environment
2. Unzip data files with splits
3. Run main.py
```bash
conda env create -n flipper --file environment.yml
cd splits/gb1 && unzip splits.zip && cd ../..
python main.py gb1 two_vs_rest cnn 100
```

### Setting hyperparameters
Hyperparameters are set through an argument parser in main.py; options can be viewed by checking the flags on `main.py`.
```bash
(flipper) jupyter@flip-repro-test-1:~/src/didactic-robot$ python main.py -h
usage: main.py [-h] {aav,gb1} split {cnn,esm1b,esm1v} max_epochs batch_size

Train a flipper model

options:
  -h, --help         show this help message and exit

Required arguments:
  {aav,gb1}          Choose the type of dataset
  split              Choose the split within the dataset
  {cnn,esm1b,esm1v}  Choose the model
  max_epochs         Choose the max num of epochs
  batch_size         Choose the batch size
```

## Plug in a model
A new model requires a few components to plug into a dataset offered through the DataModule and run training or inference on the Trainer: a LightningModule, Tokenizer, and (optionally) EarlyStopping callbacks. All objects can be set up in a single file under `/models`.

### LightningModule
Required methods are listed below, inheriting from `pl.LightningModule`; many are similar to `nn.Module`, with a few added to simplify optimization and training/inference.
* `__init__`, `forward` (like `nn.Module`)
* `training_step`, `validation_step`, `test_step` (identify behavior for a single step)
* `configure_optimizers` (set optimizers as model attribute)

```
TODO: add example of Lightning module broken down into the above pieces
```

### Tokenizer
Tokenizers vary by model, passed into the `DataModule` to tokenize sequences while generating mini-batches. An abstract Tokenizer class can be found in `data.py`, and included below for convenience.

```python
class Tokenizer(ABC):
    @property
    @abstractmethod
    def pad_tok(self) -> int:
        pass

    @abstractmethod
    def tokenize(self, seq: str) -> list[int]:
        pass
```

### EarlyStopping callbacks
Although optional, EarlyStopping callbacks are often passed to the `Trainer` for regularization during training, utilizing metrics logged during the `training_step` or `validation_step` of a model. An example is included below, used by the baseline CNN model.

```python
early_stop = EarlyStopping(monitor='val_spearman', mode='max', patience=20) # in cnn.py

trainer = Trainer(callbacks=[early_stop], accelerator='gpu', devices=[0], max_epochs=args.max_epochs) # in main.py
```

## Add a dataset
```
TODO: add details (format of csv to be compatible with DataModule, set and validation fields)
```