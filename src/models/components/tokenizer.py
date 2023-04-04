from abc import ABC, abstractmethod
import torch.nn.functional as F
import torch

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