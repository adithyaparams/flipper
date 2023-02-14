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
    def __init__(self, dataset, split, batch_size, encoder, start=0, end=None):
        super().__init__()
        self.data_dir = Path('splits/')
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.start = start
        self.end = end

        print("setting collator")

        self.collate = Collator(encoder)

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

        # self.collate = ASCollater(vocab, Tokenizer(vocab), pad=True)

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

    # def collate_fn(self, batch):
    #     print("got to first collate")
    #     data = tuple(zip(*batch))
    #     sequences = data[0]
    #     max_len = max([len(seq) for seq in sequences]) #TODO: this will be inaccurate if len(encoder(seq)) != len(seq)
    #     all_encoded, all_masks = [], []

    #     for seq in sequences:
    #         seq_encoded, seq_len = self.encoder(seq, max_len)
    #         mask = F.pad(torch.ones(seq_len), (0, max_len - seq_len))

    #         all_encoded.append(seq_encoded)
    #         all_masks.append(mask)

    #     y = torch.tensor(data[1]).unsqueeze(-1)
            
    #     return torch.stack(all_encoded), y, torch.stack(all_masks)

class Collator(object):
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, batch) -> tuple[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        max_len = max([len(seq) for seq in sequences]) #TODO: this will be inaccurate if len(encoder(seq)) != len(seq)
        all_encoded, all_masks = [], []

        for seq in sequences:
            seq_encoded, seq_len = self.encoder.encode(seq, max_len)
            mask = F.pad(torch.ones(seq_len), (0, max_len - seq_len))

            all_encoded.append(seq_encoded)
            all_masks.append(mask)

        y = torch.tensor(data[1]).unsqueeze(-1)

        return torch.stack(all_encoded), y.float(), torch.stack(all_masks).float()

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        row = self.data.iloc[index]
        return row['sequence'], row['target']