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


def preprocess_aav_data(df):
    """
    Cut down aav sequence to only include mutation window (begins at hardcoded mutation start position, with 
    length of hardcoded region + max diff bw mutated and WT sequence (accounting for insertions)
    """
    # Source: https://www.uniprot.org/uniparc/UPI00000F124C/entry
    wild_type_sequence = "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL"
    wild_type_mutation_start, region_length = 560, 28 


    df["length_difference"] = df["sequence"].map(lambda s: len(s) - len(wild_type_sequence))
    df["full_length_sequence"] = df["sequence"].copy()
    df["sequence"] = df.apply(
        lambda row: row.sequence[wild_type_mutation_start: wild_type_mutation_start + region_length + row.length_difference],
        axis=1
    )
    return df


preprocessors = {
    "aav": preprocess_aav_data,
}


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, split, batch_size, encoder, preprocess):
        super().__init__()
        self.data_dir = Path('splits/')
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.preprocess = preprocess

        self.collate = Collator(encoder, dataset)

    def prepare_data(self):
        PATH = self.data_dir / self.dataset / 'splits' / self.split 
        print('reading dataset:', self.split)
            
        df = pd.read_csv(PATH)

        
        df.sequence = df.sequence.apply(lambda s: re.sub(r'[^A-Z]', '', s.upper())) #remove special characters
        if self.dataset in preprocessors and self.preprocess:
            df = preprocessors[self.dataset](df)
        max_length = max(df.sequence.str.len())
        
        self.test = df[df.set == 'test']
        self.train = df[(df.set == 'train')&(df.validation.isna())] # change False for meltome 
        self.val = df[df.validation == True]

        print('loaded train/val/test:', len(self.train), len(self.val), len(self.test), "with max length:", max_length)

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
    
    def mask(self, seq_len: int, max_len: int) -> torch.Tensor:
        return F.pad(torch.ones(seq_len), (0, max_len - seq_len))

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
            mask = self.tokenizer.mask(seq_len, max_len)
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
