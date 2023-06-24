import pytorch_lightning as pl
import torch
from ankh import load_model
from ankh.models.ankh_transformers import AvailableModels
from torch.nn import functional as F
from torchmetrics import MeanSquaredError, SpearmanCorrCoef
from transformers import AutoTokenizer

from .components.modules import Attention1d
from .components.tokenizer import Tokenizer


class AnkhPredictor(pl.LightningModule):
    """Train a top model using Ankh embeddings"""
    def __init__(self, model_name, d_embedding, pooling="per_aa", lr=1e-3):
        super(AnkhPredictor, self).__init__()

        self.model, _ = load_model(model_name)
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
            x = self.model(input_ids=x)[0]
        if self.pooling == "per_aa":
            x = self.attention1d(x, input_mask=mask)
        elif self.pooling == "mean":
            # select non-padding positions with mask and average the value of every position in the sequence
            x = torch.stack([torch.sum(seq * m.view(-1, 1), dim=0) / torch.sum(m) for seq, m in zip(x, mask)])
        else:
            raise AttributeError("Invalid pooling method")

        return x # shape = num_samples * d_embedding


model_names_to_tokenizers = {"base": AvailableModels.ANKH_BASE, "large": AvailableModels.ANKH_LARGE}


class AnkhTokenizer(Tokenizer):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_names_to_tokenizers[model_name].value)
        self._pad_tok = self.tokenizer.pad_token_id

    @property
    def pad_tok(self) -> int:
        return self._pad_tok

    def tokenize(self, seq: str) -> list[int]:
        ids = self.tokenizer.batch_encode_plus(
            [seq], add_special_tokens=True, 
            padding=True, is_split_into_words=True, 
            return_tensors="pt"
        )
        return ids['input_ids'].tolist()