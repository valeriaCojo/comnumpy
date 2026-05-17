import torch 
import torch.nn as nn 

class SymbolMapper(nn.Module):
    r"""
    Symbol Mapper for converting integer indices to complex symbols.

    Parameters
    ----------
    alphabet : torch.Tensor
        Complex tensor containing the modulation alphabet, shape (M,).
    """
    def __init__(self, alphabet: torch.Tensor):
        super().__init__()
        self.alphabet = alphabet
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.alphabet[X]

class SymbolDemapper(nn.Module):
    r"""
    Symbol Demapper: finds the nearest symbol in the alphabet for each input symbol.

    Parameters
    ----------
    alphabet : torch.Tensor
        Complex tensor containing the modulation alphabet, shape (M,).
    """

    def __init__(self, alphabet: torch.Tensor):
        super().__init__()
        self.alphabet = alphabet

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        distances = torch.abs(X.unsqueeze(-1) - self.alphabet.unsqueeze(0))**2
        return torch.argmin(distances, dim=-1)