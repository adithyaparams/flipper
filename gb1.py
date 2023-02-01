import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import List, Any

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, split, batch_size, start=0, end=None):
        super().__init__()
        self.data_dir = Path('splits/')
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.start = start
        self.end = end

    def prepare_data(self):
        PATH = self.data_dir / self.dataset / 'splits' / self.split 
        print('reading dataset:', self.split)
            
        df = pd.read_csv(PATH)

        if self.end is not None:
            print('shortening gb1 to first 56 AAs')
            df.sequence = df.sequence.apply(lambda s: s[self.start : self.end])
        
        df.sequence = df.sequence.apply(lambda s: re.sub(r'[^A-Z]', '', s.upper())) #remove special characters
        max_length = max(df.sequence.str.len())
        
        self.test = df[df.set == 'test']
        self.train = df[(df.set == 'train')&(df.validation.isna())] # change False for meltome 
        self.val = df[df.validation == True]

        print('loaded train/val/test:', len(self.train), len(self.val), len(self.test))

        self.collate = ASCollater(vocab, Tokenizer(vocab), pad=True)

    def train_dataloader(self):
        return DataLoader(
            SequenceDataset(self.train), 
            collate_fn=self.collate, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            SequenceDataset(self.val), 
            collate_fn=self.collate, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            SequenceDataset(self.test), 
            collate_fn=self.collate, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )

vocab = 'ARNDCQEGHILKMFPSTWYVXU'

class Tokenizer(object):
    """convert between strings and their one-hot representations"""
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    def tokenize(self, seq: str) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in seq])

    def untokenize(self, x) -> str:
        return ''.join([self.t_to_a[t] for t in x])


class ASCollater(object):
    def __init__(self, alphabet: str, tokenizer: object, pad=False, pad_tok=0., backwards=False):
        self.pad = pad
        self.pad_tok = pad_tok
        self.tokenizer = tokenizer
        self.backwards = backwards
        self.alphabet = alphabet

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        sequences = [torch.tensor(self.tokenizer.tokenize(s)) for s in sequences]
        sequences = [i.view(-1,1) for i in sequences]
        maxlen = max([i.shape[0] for i in sequences])
        padded = [F.pad(i, (0, 0, 0, maxlen - i.shape[0]),"constant", self.pad_tok) for i in sequences]
        padded = torch.stack(padded)
        mask = [torch.ones(i.shape[0]) for i in sequences]
        mask = [F.pad(i, (0, maxlen - i.shape[0])) for i in mask]
        mask = torch.stack(mask)
        y = data[1]
        y = torch.tensor(y).unsqueeze(-1)
        ohe = []
        for i in padded:
            i_onehot = torch.FloatTensor(maxlen, len(self.alphabet))
            i_onehot.zero_()
            i_onehot.scatter_(1, i, 1)
            ohe.append(i_onehot)
        padded = torch.stack(ohe)
            
        return padded, y, mask

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        row = self.data.iloc[index]
        return row['sequence'], row['target']