import torch
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import SpearmanCorrCoef, MeanSquaredError, SumMetric
from data import Tokenizer
from modules import MaskedConv1d, LengthMaxPool1D
from scipy.stats import spearmanr

class CNN(pl.LightningModule):
    def __init__(self, kernel_size, input_size, dropout):
        super(CNN, self).__init__()
        self.encoder = MaskedConv1d(CNNTokenizer.n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(linear=True, in_dim=input_size, out_dim=input_size*2)
        self.decoder = nn.Linear(input_size*2, 1)
        self.dropout = nn.Dropout(dropout) # TODO: actually add this to model
        self.input_size = input_size
        self.val_spearman = SpearmanCorrCoef()
        self.training_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.test_spearman = SpearmanCorrCoef()
        self.test_loss = MeanSquaredError()

    def forward(self, x, mask):
        # encoder
        x = F.relu(self.encoder(x, input_mask=mask.repeat(CNNTokenizer.n_tokens, 1, 1).permute(1, 2, 0)))
        x = x * mask.repeat(self.input_size, 1, 1).permute(1, 2, 0)
        # embed
        x = self.embedding(x)
        x = self.dropout(x)
        # decoder
        output = self.decoder(x)
        return output

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

    def test_step(self, batch, batch_idx):
        src, tgt, mask, _ = batch
        output = self(self.generate_ohe(src).float(), mask)

        output = output.flatten()
        tgt = tgt.flatten()
        
        self.test_spearman.update(output, tgt)
        self.log("test_spearman", self.test_spearman, on_step=False, on_epoch=True)
        self.test_loss.update(output, tgt)
        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 1e-3, 'weight_decay': 0},
            {'params': self.embedding.parameters(), 'lr': 5e-5, 'weight_decay': 0.05},
            {'params': self.decoder.parameters(), 'lr': 5e-6, 'weight_decay': 0.05}
        ])
        return optimizer

    def generate_ohe(self, sequences: torch.Tensor) -> torch.Tensor:
        max_len = sequences.shape[1]
        seq_transposed = [seq.view(-1,1) for seq in sequences]

        ohe = []
        for seq in seq_transposed:
            onehot = torch.cuda.FloatTensor(max_len, len(CNNTokenizer.alphabet))
            onehot.zero_()
            onehot.scatter_(1, seq, 1)
            ohe.append(onehot)

        return torch.stack(ohe)

class CNNTokenizer(Tokenizer):
    alphabet = 'ARNDCQEGHILKMFPSTWYVXU'
    a_to_t = {a: i for i, a in enumerate(alphabet)}
    n_tokens = len(alphabet)

    def __init__(self, pad_tok: int = 0) -> None:
        self._pad_tok = pad_tok

    @property
    def vocab_size(self) -> int:
        return len(CNNTokenizer.alphabet)

    @property
    def pad_tok(self) -> int:
        return self._pad_tok

    def tokenize(self, seq: str) -> list[int]:
        return [CNNTokenizer.a_to_t[a] for a in seq]