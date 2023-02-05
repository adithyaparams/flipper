import os
import torch
from torch import optim, nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import SpearmanCorrCoef, MeanSquaredError

class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.linear:
            x = F.relu(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x


class CNN(pl.LightningModule):
    def __init__(self, n_tokens, kernel_size, input_size, dropout):
        super(CNN, self).__init__()
        self.encoder = MaskedConv1d(n_tokens, input_size, kernel_size=kernel_size)
        self.embedding = LengthMaxPool1D(linear=True, in_dim=input_size, out_dim=input_size*2)
        self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout) # TODO: actually add this to model
        self.input_size = input_size
        self.val_spearman = SpearmanCorrCoef()
        self.val_loss = MeanSquaredError()
        self.test_spearman = SpearmanCorrCoef()
        self.test_loss = MeanSquaredError()


    def forward(self, x, mask):
        # encoder
        x = F.relu(self.encoder(x, input_mask=mask.repeat(self.n_tokens, 1, 1).permute(1, 2, 0)))
        x = x * mask.repeat(self.input_size, 1, 1).permute(1, 2, 0)
        # embed
        x = self.embedding(x)
        x = self.dropout(x)
        # decoder
        output = self.decoder(x)
        return output

    def training_step(self, batch, batch_idx):
        src, tgt, mask = batch
        src = src.float()
        tgt = tgt.float()
        mask = mask.float()
        output = self(src, mask)
        return F.mse_loss(output, tgt)
        
    def validation_step(self, batch, batch_idx):
        src, tgt, mask = batch
        src = src.float()
        tgt = tgt.float()
        mask = mask.float()
        output = self(src, mask)
        
        output = output.flatten()
        tgt = tgt.flatten()
        self.log("val_spearman", self.val_spearman(output, tgt), on_step=False, on_epoch=True)
        self.log("val_loss", self.val_loss(output, tgt), on_step=False, on_epoch=True)
        # self.val_spearman.update(output, tgt)
        return F.mse_loss(output, tgt)
    
    # def validation_epoch_end(self, outputs):
    #     self.log("spearmanr", self.val_spearman.compute())
    #     self.val_spearman.reset()

    def test_step(self, batch, batch_idx):
        src, tgt, mask = batch
        src = src.float()
        tgt = tgt.float()
        mask = mask.float()
        output = self(src, mask)

        output = output.flatten()
        tgt = tgt.flatten()
        self.log("test_spearman", self.test_spearman(output, tgt), on_step=False, on_epoch=True)
        self.log("test_loss", self.test_loss(output, tgt), on_step=False, on_epoch=True)
        # self.test_spearman.update(output, tgt)
        return F.mse_loss(output, tgt, reduction='none')

    # def test_epoch_end(self, outputs):
    #     s = self.test_spearman.compute()
    #     print("Spearman" + str(s))
    #     print("Average loss" + str(torch.cat(outputs).mean()))

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': 1e-3, 'weight_decay': 0},
            {'params': self.embedding.parameters(), 'lr': 5e-5, 'weight_decay': 0.05},
            {'params': self.decoder.parameters(), 'lr': 5e-6, 'weight_decay': 0.05}
        ])
        return optimizer