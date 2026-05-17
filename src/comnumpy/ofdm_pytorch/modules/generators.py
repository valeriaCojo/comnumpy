import torch
import torch.nn as nn

class SymbolGenerator(nn.Module):
    r"""
    Generator for IID symbols uniform in {0, 1, ..., M-1}.

    Parameters
    ----------
    M : int
        Alphabet size.
    seed : int, optional
        Random seed for reproducibility.
    
    """

    def __init__(self, M: int, seed: int = None):
        super().__init__()
        self.M = M
        self.seed = seed
        
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

    def forward(self, X) -> torch.Tensor:
        if isinstance(X, int):
            size = (X,)
        elif isinstance(X, (tuple, list)):
            size = tuple(X)
        else:
            raise ValueError("X must be an int, tuple, or list.")

        Y = torch.randint(0, self.M, size=size, generator=self.generator)
        return Y
    