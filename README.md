# Flipper: Benchmarks for Protein Engineering

This repository is a reproduction of the [Fitness Landscape Inference for Proteins (FLIP) benchmark](https://www.biorxiv.org/content/10.1101/2021.11.09.467890v1.full), built with [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/1.9.1/) and [Hydra](https://github.com/ashleve/lightning-hydra-template).

![](/performance.png)


An extended description of the context behind flipper can be found [here](https://adithyaparams.com/posts/2023/06/flipper/). It leverages Lightning to allow users to easily extend the FLIP benchmark with new models and datasets and uses Hydra to simplify configuration and hyperparameter search.

## Overview
General comments on folder breakup and structural changes compared to the [FLIP repo](https://github.com/J-SNACKKB/FLIP) included below.
* `/src` includes model logic (`/models`), data processing logic (`/data`), and general utilities for logging, error handling and other tasks (`/utils`)
* `/data` contains the compressed data sources
* `/configs` includes `.yaml` files that allow users to explicitly train on configurations of models, data, hyperparams, and other fields

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

### Setting hyperparameters
Hyperparam configuration is handled through hydra, which sets configs through `.yaml` files that can also be overridden through the command line.

Overall settings are accessed through `configs/train.yaml`. A portion of the file is shown below.

```yaml
# @package _global_

# specify here default configuration
# order of defaults determines the order in which configs override each other
defaults:
  - _self_
  - data: default.yaml
  - model: cnn.yaml
  - callbacks: spearman_early_stopping.yaml
  - logger: csv.yaml # set logger here or use command line (e.g. `python train.py logger=tensorboard`)
  - trainer: gpu.yaml
  - paths: default.yaml
  - extras: default.yaml
  # - hydra: default.yaml

...

# set False to skip model training
train: True

# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on the metric specified in checkpoint callback
test: True

# compile model for faster training with pytorch 2.0
compile: False

# simply provide checkpoint path to resume training
ckpt_path: null

# seed for random number generators in pytorch, numpy and python.random
seed: null

```

Hydra allows `configs/train.yaml` to [extend](https://hydra.cc/docs/patterns/extending_configs/) other `.yaml` files (listed under defaults) that control different aspects of the training run. Hyperparams are mostly set through `data`, `model`, and `callbacks`. The structure of `model` and `callbacks` is discussed in 'Adding a new model', so `data` is used as an example below.

The config file referenced in `data: default.yaml` can be found in `configs/data/default.yaml`.
```yaml
defaults:
  - base.yaml

dataset: "aav"
split: "two_vs_many"
```

`default.yaml` extends `configs/data/base.yaml` (all subconfigs can be found in their respective folder).

```yaml
_target_: src.data.datamodule.DataModule
dataset: ???
split: ???
batch_size: 256
preprocess: True
data_dir: ${paths.data_dir}
```

Attributes assigned a value of `???` in `base.yaml` are set in the extension config `default.yaml`, which are added to `train.yaml` under the `data` attribute.

Users may also override configs through the command line, i.e.
```bash
python src/train.py seed=2 data.batch_size=64
```
The `seed` and `batch_size` arguments can be found in `train.yaml` and `base.yaml`, respectively.

## Adding a model
A new model requires a few components to plug into a dataset offered through the DataModule and run training or inference on the Trainer: a LightningModule, Tokenizer, and (optionally) EarlyStopping callbacks. All components sit inside a single file in `/src/models` (ie `cnn_module.py`) and the respective configuration file in `/configs`.

### LightningModule
Required methods are listed below, inheriting from `pl.LightningModule`; many are similar to `nn.Module`, with a few added to simplify optimization and training/inference.
* `__init__`, `forward` (like `nn.Module`)
* `training_step`, `validation_step`, `test_step` (identify behavior for a single step)
* `configure_optimizers` (set optimizers as model attribute)

An empty `LightningModule` is shown below; a full example [can be found in](https://github.com/an1lam/flipper/blob/hydra/src/models/cnn_module.py) `src/models/cnn_module.py`, under `CNN`.

```python
class NewModel(LightningModule):

    def __init__():
        ...

    def forward():
        ...

    def training_step():
        ...

    def validation_step():
        ...

    def test_step():
        ...

    def configure_optimizers():
        ...
```

Once the model is written, a respective config file needs to be added in `configs/models`. An example can be found in `configs/models/cnn.yaml`.

```yaml
_target_: src.models.cnn_module.CNN
kernel_size: 5
input_size: 43
dropout: 0.0
```

The config file identifies the class from which the model can be instantiated and arguments provided during initialization.

```python
class CNN(pl.LightningModule):
    def __init__(self, kernel_size, input_size, dropout):
```

The config file should then be added to `train.yaml` (ie. `model: cnn.yaml`) to include those attributes.

### Tokenizer
Tokenizers convert strings (amino-acid sequences) to the input models require; they often vary by model, passed into the `DataModule` to tokenize sequences while generating mini-batches. An abstract Tokenizer class can be found in `src/models/components/tokenizer.py` (included below for convenience).
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
A full example [can be found in](https://github.com/an1lam/flipper/blob/1d90ab3569275fd4101dd1e8a32fd8adb936be4b/src/models/cnn_module.py#L88) `src/models/cnn_module.py`, under `CNNTokenizer`.

The tokenizer's configuration is taken care of internally once `model._target_` is set in the config. To do this for a new set of model/tokenizer, a new field needs to be added to the `model_to_tok` dictionary in `utils/utils.py`.
```python
model_to_tok = {
    'src.models.cnn_module.CNN': 'src.models.cnn_module.CNNTokenizer',
    'src.models.esm_module.ESM': 'src.models.esm_module.ESMTokenizer'
}
```

### EarlyStopping
Early stopping is a method of regularization to avoid overfitting during training. If a model requires early stopping, it can be configured under `configs/callbacks`. An example of early stopping used by the CNN (found in `spearman_early_stopping`) is shown below.

```yaml
defaults:
  - base_early_stopping.yaml

early_stopping:
  monitor: "val_spearman"
  patience: 20
  mode: "max"
```

This configuration stops training if the Spearman coefficient calculated on the validation set has not improved in the last 20 epochs.

## Adding a dataset
### Adding data
Splits for a new dataset should be provided in a `.csv` file, under `data/{dataset_name}/{split_name}.csv`. All splits should be compressed into a `data/{dataset_name}/splits.zip`, which is the only file that should be pushed to the remote.

On the structure of a single split:
* `sequences` column should contain the amino acid sequences
* `set` column should == `'test'` || `'train'`
* `validation` column should == `True` || `NaN`
* `target` column should contain a fitness value

### Adjusting config
Once the data has been added to `/data`, it can be accessed in a training run by editing `configs/data/default.yaml`.

```yaml
defaults:
  - base.yaml

dataset: {dataset_name}
split: {split_name}
```