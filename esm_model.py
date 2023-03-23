from typing import Tuple
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import MeanSquaredError, SpearmanCorrCoef
from esm import pretrained, data
from modules import Attention1d
from data import Tokenizer

class ESM(pl.LightningModule):
    name_to_model = {
        'esm1v': 'esm1v_t33_650M_UR90S_1', # use first of 5 models 
        'esm1b': 'esm1b_t33_650M_UR50S'
    }

    """Outputs of the ESM model with the attention1d"""
    def __init__(self, model_name, d_embedding, pooling="per_aa", lr=0.001): # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(ESM, self).__init__()
        if model_name not in ESM.name_to_model.keys():
            raise ValueError("Invalid model name provided")
        if pooling not in ["per_aa", "mean", "mut_mean"]:
            raise AttributeError("Provided invalid pooling method")

        self.model, self.alphabet = pretrained.load_model_and_alphabet(ESM.name_to_model[model_name])
        self.model.eval()
        self.pooling = pooling
        if self.pooling == "per_aa":
            self.attention1d = Attention1d(in_dim=d_embedding) # ???
        self.linear = torch.nn.Linear(d_embedding, d_embedding)
        self.relu = torch.nn.ReLU()
        self.final = torch.nn.Linear(d_embedding, 1)
        self.lr = lr
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()
        self.test_spearman = SpearmanCorrCoef()
        
    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x

    def training_step(self, batch, batch_idx):
        src, tgt, mask, dataset = batch
        pooled = self.pool(src, mask, dataset)
        output = self(pooled)

        output = output.flatten()
        tgt = tgt.flatten()
        
        return F.mse_loss(output, tgt)

    def validation_step(self, batch, batch_idx):
        src, tgt, mask, dataset = batch
        pooled = self.pool(src, mask, dataset)
        output = self(pooled)
        
        output = output.flatten()
        tgt = tgt.flatten()
        
        self.val_loss.update(output, tgt)
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        src, tgt, mask, dataset = batch
        pooled = self.pool(src, mask, dataset)
        output = self(pooled)
        
        output = output.flatten()
        tgt = tgt.flatten()
        
        self.test_spearman.update(output, tgt)
        self.log("test_spearman", self.test_spearman, on_step=False, on_epoch=True)
        self.test_loss.update(output, tgt)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.attention1d.parameters(), 'lr': 1e-3},
            {'params': self.linear.parameters(), 'lr': 1e-3},
            {'params': self.final.parameters(), 'lr': 1e-3}
        ])
        return optimizer

    def pool(self, x: torch.Tensor, mask: torch.Tensor, dataset: str) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x, repr_layers=[33])["representations"][33]
        if self.pooling == "per_aa":
            x = self.attention1d(x, input_mask=mask) #TODO: mask will look different w non padding=1, will it still work the same
        elif self.pooling == "mean": #TODO: check that output matches mean pooled seqs generated from extract.py
            seq_lens = [m.argmin() for m in mask] # get index of first 0 in mask (where padding starts)
            seq_lens = [s if s else x.shape[1] for s in seq_lens] # set seq_len to full length if argmin==0 (no padding)
            x = torch.stack(seq[:seq_len].mean(0) for seq, seq_len in zip(x, seq_lens))
        elif self.pooling == "mut_mean":
            if dataset == "gb1":
                x = torch.mean(x[:, [38, 39, 40, 53], :], 1)
            elif dataset == "aav":
                x = torch.mean(x[:, 560:590, :], 1)
            else:
                raise AttributeError("Invalid dataset for mut_mean pooling method")
        else:
            raise AttributeError("Invalid pooling method")

        return x # shape = num_samples * d_embedding


class ESMTokenizer(Tokenizer):
    def __init__(self, model_name):
        if model_name not in ESM.name_to_model.keys():
            raise ValueError("Invalid model name provided")
        # https://github.com/facebookresearch/esm/blob/main/esm/data.py#L151, both esm1v/1b use this alphabet
        self.alphabet = data.Alphabet.from_architecture('ESM-1b')
        self._pad_tok = self.alphabet.padding_idx

    @property
    def pad_tok(self) -> int:
        return self._pad_tok

    def tokenize(self, seq: str) -> list[int]:
        tokenized = self.alphabet.encode(seq)

        if self.alphabet.prepend_bos:
            tokenized.insert(0, self.alphabet.cls_idx)
        if self.alphabet.append_eos:
            tokenized.append(self.alphabet.eos_idx)
        
        return tokenized
