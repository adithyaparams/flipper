import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics import MeanSquaredError, SpearmanCorrCoef
from esm import pretrained, data
from .components.modules import Attention1d
from .components.tokenizer import Tokenizer

class ESM(pl.LightningModule):
    name_to_model = {
        'esm1v': 'esm1v_t33_650M_UR90S_1', # use first of 5 models 
        'esm1b': 'esm1b_t33_650M_UR50S'
    }

    """Outputs of the ESM model with the attention1d"""
    def __init__(self, model_name, d_embedding, pooling="per_aa", lr=1e-3): # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(ESM, self).__init__()
        if model_name not in ESM.name_to_model.keys():
            raise ValueError("Invalid model name provided")
        if pooling not in ["per_aa", "mean"]:
            raise AttributeError("Provided invalid pooling method")

        self.model, self.alphabet = pretrained.load_model_and_alphabet(ESM.name_to_model[model_name])
        self.model.eval()
        self.pooling = pooling
        if self.pooling == "per_aa":
            self.attention1d = Attention1d(in_dim=d_embedding)
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
        params = [
            {'params': self.linear.parameters(), 'lr': self.lr},
            {'params': self.final.parameters(), 'lr': self.lr}
        ]
        
        if hasattr(self, 'attention1d'):
            params.append({'params': self.attention1d.parameters(), 'lr': self.lr})
            
        return torch.optim.Adam(params)

    def pool(self, x: torch.Tensor, mask: torch.Tensor, dataset: str) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x, repr_layers=[33])["representations"][33]
        if self.pooling == "per_aa":
            x = self.attention1d(x, input_mask=mask) #TODO: mask will look different w non padding=1, will it still work the same
        elif self.pooling == "mean": #TODO: check that output matches mean pooled seqs generated from extract.py
            # select non-padding positions with mask and average the value of every position in the sequence
            x = torch.stack([torch.sum(seq * m.view(-1, 1), dim=0) / torch.sum(m) for seq, m in zip(x, mask)])
        else:
            raise AttributeError("Invalid pooling method")

        return x # shape = num_samples * d_embedding


class ESMTokenizer(Tokenizer):
    def __init__(self):
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
    
    def mask(self, seq_len: int, max_len: int) -> torch.Tensor:
        m = super().mask(seq_len, max_len)
        
        if self.alphabet.prepend_bos:
            m[0] = 0
        if self.alphabet.append_eos:
            m[seq_len-1] = 0
            
        return m
