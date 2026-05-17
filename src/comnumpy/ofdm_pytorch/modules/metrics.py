import torch

def compute_ber(x_target: torch.Tensor, x_detected: torch.Tensor, width: int) -> float:
    """
    Compute the Bit Error Rate (BER) between target and detected symbols.

    Parameters
    ----------
    x_target : torch.Tensor
        Target symbols, shape (N,), integer values in {0, ..., M-1}.
    x_detected : torch.Tensor
        Detected symbols, shape (N,), integer values in {0, ..., M-1}.
    width : int
        Number of bits per symbol (e.g. 4 for 16-QAM).

    Returns
    -------
    float
        Bit Error Rate.
    """
    x_target = x_target.reshape(-1)
    x_detected = x_detected.reshape(-1)

    # Convertim fiecare simbol in biti
    # exemplu: simbol 5, width=4 -> [0, 1, 0, 1]
    bits_target = torch.zeros(len(x_target) * width, dtype=torch.int32)
    bits_detected = torch.zeros(len(x_detected) * width, dtype=torch.int32)

    for i in range(width):
        bits_target[i::width] = (x_target >> (width - 1 - i)) & 1
        bits_detected[i::width] = (x_detected >> (width - 1 - i)) & 1

    nb_errors = (bits_target != bits_detected).sum().item()
    return nb_errors / len(bits_target)


def compute_ser(x_target: torch.Tensor, x_detected: torch.Tensor) -> float:
    """
    Compute the Symbol Error Rate (SER) between target and detected symbols.

    Parameters
    ----------
    x_target : torch.Tensor
        Target symbols, shape (N,).
    x_detected : torch.Tensor
        Detected symbols, shape (N,).

    Returns
    -------
    float
        Symbol Error Rate.
    """
    x_target = x_target.reshape(-1)
    x_detected = x_detected.reshape(-1)
    N = min(len(x_target), len(x_detected))
    nb_errors = (x_target[:N] != x_detected[:N]).sum().item()
    return nb_errors / N


def compute_evm(x_target: torch.Tensor, x_estimated: torch.Tensor) -> float:
    """
    Compute the Error Vector Magnitude (EVM).

    Parameters
    ----------
    x_target : torch.Tensor
        Target signal, complex tensor.
    x_estimated : torch.Tensor
        Estimated signal, complex tensor.

    Returns
    -------
    float
        EVM value.
    """
    error = torch.abs(x_target - x_estimated) ** 2
    power = torch.abs(x_target) ** 2
    return torch.sqrt(error.mean() / power.mean()).item()


def compute_mse(x_target: torch.Tensor, x_estimated: torch.Tensor) -> float:
    """
    Compute the Mean Squared Error (MSE).

    Parameters
    ----------
    x_target : torch.Tensor
        Target signal.
    x_estimated : torch.Tensor
        Estimated signal.

    Returns
    -------
    float
        MSE value.
    """
    return torch.mean(torch.abs(x_target.reshape(-1) - x_estimated.reshape(-1)) ** 2).item()
