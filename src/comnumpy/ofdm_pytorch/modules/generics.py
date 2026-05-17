import torch
import torch.nn as nn
from typing import List


class Sequential(nn.Module):
    r"""
    Sequential container for processing modules.

    Executes modules in order, passing the output of each as input to the next.
    Modules can be accessed by name using the [] operator.

    Parameters
    ----------
    module_list : list
        Ordered list of nn.Module instances to execute sequentially.
    """

    def __init__(self, module_list: List[nn.Module]):
        super().__init__()
        self.module_list = nn.ModuleList(module_list)

    def forward(self, X):
        Y = X
        for module in self.module_list:
            Y = module(Y)
        return Y

    def __getitem__(self, key):
        if isinstance(key, str):
            for module in self.module_list:
                if hasattr(module, 'name') and module.name == key:
                    return module
            raise AttributeError(f"Module '{key}' not found.")
        elif isinstance(key, int):
            return self.module_list[key]
        else:
            raise TypeError("Key must be a string or an integer.")
        
class Recorder(nn.Module):
    r"""
    Records the input signal without modifying it.

    Stores the last input tensor for later retrieval (e.g., for plotting or metric computation).

    Parameters
    ----------
    name : str, optional
        Name of the recorder instance. Default is 'recorder'.
    """

    def __init__(self, name: str = "recorder"):
        super().__init__()
        self.name = name
        self.data = None

    def get_data(self):
        return self.data

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.data = X
        return X