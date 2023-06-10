# Flipper: extending protein sequence-function benchmarks

Over the past few years, the number of benchmarks comparing protein sequence-to-function models has grown a lot. Benchmarks vary on the models and datasets they choose to survey, but they share limitations around reproducing performance and extending models and datasets. To do this, we started with the Fitness Landscape Inference for Proteins (FLIP) benchmark due to its use of sanitized datasets, variety of train/test splits, and models spanning across vanilla CNNs to pretrained language models. Building on top of the benchmark with an emphasis on extensibility, we used Pytorch Lightning and Hydra to allow users to easily add models and perform hyperparam optimization.

[FIGURE COMPARING FLIP/FLIPPER PERFORMANCE]

## What are sequence-function models?
Sequence-function models act as proxies for experimental assays that probe the fitness of proteins. Experimental assays are often low throughput, which means that it can be infeasible to assess the fitness of a long list of protein candidates within practical time constraints. By approximating these assays, sequence-function models can predict proteins’ fitness levels, winnowing down a list of candidates to an approachable size for assays.

Let’s illustrate this with an example. Suppose that we’re biofuel engineers, trying to find a protein that breaks down cellulose into simple sugars; this could then be converted into ethanol or other fuels. We already have a protein (represented by an amino acid sequence of length 35) that does this, but it needs to be 10x more efficient to generate biofuel at a profitable rate. The first place we look for a better protein are sequences that are one mutation ‘away’ from the wild-type; there are 20^35 (nearly 1e45) options available. Our assay might only be able to process two million proteins per week, but sequence-function models operate at a higher throughput of 1e10 - 1e15. This higher rate allows us to examine a longer list of potential proteins before passing off the highest quality candidates to the assay.

Sequence-function models can work in many different ways, but they often take the amino acid sequence for a protein as input and generate a higher dimensional embedding. They then predict a fitness level from this representation. Embeddings can be generated through very large models trained through self-supervised learning, on the entire corpus of protein sequence data (like ESM) or through supervised learning on smaller models (ie. CNNs, transformers)

## What is FLIP?
Essentially, FLIP compares the performance of models on fitness prediction tasks. This is done across a few different datasets. The benchmark compares three classes of models; ordered in decreasing size, they are language models (ESM), CNNs and ridge regression. ESM models generate individual embeddings for each amino acid in the sequence, so there are subcategories of models that merge these individual embeddings in different ways to generate an overall protein embedding. These methods include averaging embeddings over the entire protein sequence, averaging over just the mutation window, and calculating a weighted average based learned parameters.
There are three fitness prediction tasks in the benchmark: fluorescence, AAV, and thermostability. Fluorescence includes the smallest dataset, limited to mutations at four positions of the GFP protein. The AAV dataset is larger, due to a larger search space opened up from a longer mutation window in the protein. And thermostability is the largest, sampled from a diverse set of proteins that share this property. Due to compute constraints, we selectively worked on reproducing the results of the ESM models and CNNs on the fluorescence and AAV datasets.

The benchmark also includes a variety of train-test splits for each dataset, simulating realistic data availability for model development during the design cycle. Often, training data only includes a local ‘neighborhood’ of sequences clustered a few mutations away from the wild-type protein, while the sequences we’d like to assay sit outside of this neighborhood. By splitting the dataset to train and test sets that are within and outside of, for example, two mutations of the WT, the benchmark ensures that results on the test set will transfer over to a laboratory setting.

## What did we do differently?
Improvements to the pipeline begin with the use of PyTorch Lightning to reproduce models. By moving the models into `LightningModule`s, training and validation logic is moved outside of for loops and into more legible step functions. Training loop logic for the CNN, with FLIP and Flipper, is included below.

```python
def epoch(model, train, current_step=0):
    if train:
        model = model.train()
        loader = train_iterator
        t = 'Training'
        n_total = len(train_iterator)
    else:
        model = model.eval()
        loader = val_iterator
        t = 'Validating'
        n_total = len(val_iterator) 
    
    losses = []
    outputs = []
    tgts = []
    n_seen = 0
    for i, batch in enumerate(loader):
        loss, output, tgt = step(model, batch, train)
        losses.append(loss)
        outputs.append(output)
        tgts.append(tgt)

        n_seen += len(batch[0])
        if train:
            nsteps = current_step + i + 1
        else:
            nsteps = i
        
    outputs = torch.cat(outputs).numpy()
    tgts = torch.cat(tgts).cpu().numpy()

    if train:
        with torch.no_grad():
            _, val_rho = epoch(model, False, current_step=nsteps)
        print('epoch: %d loss: %.3f val loss: %.3f' % (e + 1, loss, val_rho))
    
    if not train:
        val_rho = spearmanr(tgts, outputs).correlation
        mse = mean_squared_error(tgts, outputs)
        
    return i, val_rho
```

Epoch level code branches off at multiple points based on a `train` variable. It identifies whether training or validation is being run during that epoch. Meanwhile, the CNN's LightningModule includes separate functions for training and validation steps/epochs.

```python
class CNN(pl.LightningModule):
    ...

    def training_step(self, batch, batch_idx):
        src, tgt, mask, _ = batch
        ohe = self.generate_ohe(src).float()
        output = self(ohe, mask)
        
        self.training_loss.update(output, tgt)
        self.log("training_loss", self.training_loss, on_step=False, on_epoch=True)
        return F.mse_loss(output, tgt)
    
    def validation_step(self, batch, batch_idx):
        src, tgt, mask, _ = batch
        output = self(self.generate_ohe(src).float(), mask)
        
        output = output.flatten()
        tgt = tgt.flatten()
        
        self.val_spearman.update(output, tgt)
        self.log("val_spearman", self.val_spearman, on_step=False, on_epoch=True)
        self.val_loss.update(output, tgt)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True)
```

Metrics tracking loss and relevant early stopping variables are also explicitly initialized in the `__init__` and regularly saved to a logger for future debugging. Tokenizers are passed into models during initialization, allowing them to be defined outside models and shared according to a user’s discretion. Datasets are all processed through a single `DataModule`, with dataset specific processing injected via functions.

Setup for training runs and inference is abstracted with Hydra; all config options related to models, datasets, logging and other options is explicitly defined in config files, circumventing the need for a long list of command line args. To simplify scripts for hyperparam search, users can override specific configs via the command line as well. Training commands for FLIP and Flipper are included below for comparison.

```bash
python train_all.py aav_4 cnn 
```

FLIP hardcodes a dictionary that references specific splits in a dataset according to the first argument. With Flipper, the user just runs:

```bash
python train.py
```

This starts a training run based on the configs set in `train.yaml` and its extensions, included below.

```yaml
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

  # experiment configs allow for version control of specific hyperparameters
  # e.g. best hyperparameters for given model and datamodule
  - experiment: null

  # config for hyperparameter optimization
  - hparams_search: null

  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default.yaml

  # debugging config (enable through command line, e.g. `python train.py debug=default)
  - debug: null
```

Specific config options from `default.yaml` can also be overridden.

```bash
python train.py data.split=one_vs_many debug=default
```

Embedding generation for ESM models was also moved to inside the training cycle, rather than a single pass done during data pre-processing, to enable fine-tuning during the training process.

## Acknowledgements
I entered the field after a brief internship at a biotech startup in the summer of 2022, as a newcomer to both ML and protein engineering. Stephen kindly volunteered to take me under his wing, offering up inspiration for Flipper as a project and providing valuable technical mentorship whenever I hit roadblocks.
