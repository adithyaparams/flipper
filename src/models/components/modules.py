import torch
from torch import nn
from torch.nn import functional as F

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
            x = x * input_mask # n * seq_len * d_emb
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False):
        """
        :param in_dim: input channels
        :param out_dim: output channels
        :param linear: add linear layer
        """
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.linear:
            x = F.relu(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x

class Attention1d(nn.Module):
    def __init__(self, in_dim):
        """
        :param in_dim: input channels
        """
        super().__init__()
        self.layer = MaskedConv1d(in_dim, 1, 1)

    def forward(self, x, input_mask=None):
        """
        Aggregate sequence from n * seq_len * d_emb to n * d_emb, using
        the MaskedConv1d layer to generate a linear layer (seq_len, 1) 
        with softmaxed weights.

        MaskedConv1d takes a tensor of size (n * seq_len * d_emb), 
        transposes dim 1, 2 to (n * d_emb * seq_len) then convolves 
        from d_emb to 1 with kernel size 1 to create a tensor of size
        (n * seq_len). Weighted sum of this 'attention' layer with x 
        generates output of size (n * d_emb).
        """
        n, ell, _ = x.shape # n * seq_len * d_emb
        attn = self.layer(x) # attn.shape = n * seq_len
        attn = attn.view(n, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(n, -1).bool(), float('-inf'))
        attn = F.softmax(attn, dim=-1).view(n, -1, 1) # softmax across seq_len
        out = (attn * x).sum(dim=1) # weighted sum of embeddings across seq_len
        return out # out.shape = n * d_emb