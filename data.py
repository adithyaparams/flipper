import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import List, Any
from abc import ABC, abstractmethod

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, split, batch_size, encoder, start=0, end=None):
        super().__init__()
        self.data_dir = Path('splits/')
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.start = start
        self.end = end

        self.collate = Collator(encoder, dataset)

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
            # shuffle=True, 
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            SequenceDataset(self.test), 
            collate_fn=self.collate, 
            batch_size=self.batch_size, 
            # shuffle=True, 
            num_workers=4
        )

class Tokenizer(ABC):
    @property
    @abstractmethod
    def pad_tok(self) -> int:
        pass

    @abstractmethod
    def tokenize(self, seq: str) -> list[int]:
        pass

class Collator(object):
    def __init__(self, tokenizer: Tokenizer, dataset: str):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def __call__(self, batch) -> tuple[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        seq_tokenized = [torch.tensor(self.tokenizer.tokenize(seq)) for seq in sequences]
        max_len = max([len(seq) for seq in seq_tokenized])
        seq_padded = torch.empty((len(seq_tokenized), max_len), dtype=torch.int64)
        seq_padded.fill_(self.tokenizer.pad_tok)

        masks = []
        for i, seq in enumerate(seq_tokenized):
            seq_len = len(seq)
            seq_padded[i, :seq_len] = seq
            mask = F.pad(torch.ones(seq_len), (0, max_len - seq_len))
            masks.append(mask)

        y = torch.tensor(data[1]).unsqueeze(-1)

        return seq_padded, y.float(), torch.stack(masks).float(), self.dataset

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        row = self.data.iloc[index]
        return row['sequence'], row['target']
