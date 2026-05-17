import torch
import torch.nn as nn

class Serial2Parallel(nn.Module):
    r"""
    Converts a serial data stream into parallel data streams.

    Reshapes the last dimension of the input tensor into (N_sub, M),
    where N_sub is the number of subcarriers and M is the number of OFDM symbols.
    If necessary, the input is zero-padded or truncated.

    Parameters
    ----------
    N_sub : int
        Number of subcarriers (size of second-to-last dimension in output).
    method : str, optional
        How to handle data that does not fit perfectly: 'zero-padding' or 'truncate'.
        Default is 'zero-padding'.
    """
    def __init__(self, N_sub: int, method: str = "zero-padding"):
        super().__init__()
        if N_sub <= 0:
            raise ValueError("N_sub must be a positive integer.")
        self.N_sub = N_sub
        self.method = method

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N_sub = self.N_sub
        N = X.shape[-1]
        M = N // N_sub

        if N % N_sub != 0:
            if self.method == "zero-padding":
                M += 1
                pad_size = N_sub * M - N
                # Pad zeros at the end of the last dimension
                X = torch.nn.functional.pad(X, (0, pad_size))
            elif self.method == "truncate":
                X = X[..., :M * N_sub]

        # Reshape last dim N -> (N_sub, M) in Fortran order
        # torch nu are order='F', deci facem reshape si transpose manual
        new_shape = X.shape[:-1] + (M, N_sub)
        Y = X.reshape(new_shape)        # (..., M, N_sub)
        Y = Y.transpose(-1, -2)         # (..., N_sub, M)
        return Y
    
class Parallel2Serial(nn.Module):
    r"""
    Converts parallel data streams into a serial data stream.

    Flattens the last two dimensions of the input tensor using Fortran-style ordering.

    Parameters
    ----------
    None
    """

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_t = X.transpose(-1, -2)                               # (..., M, N_sub)
        new_shape = X.shape[:-2] + (X.shape[-2] * X.shape[-1],)
        return X_t.reshape(new_shape)    

class FFTProcessor(nn.Module):
    r"""
    Performs Fast Fourier Transform (FFT) on the input tensor.

    Parameters
    ----------
    axis : int, optional
        Axis along which to perform the FFT. Default is 0.
    norm : str, optional
        Normalization mode. Default is 'ortho'.
    """

    def __init__(self, axis: int = 0, norm: str = "ortho"):
        super().__init__()
        self.axis = axis
        self.norm = norm

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft(X, norm=self.norm, dim=self.axis)


class IFFTProcessor(nn.Module):
    r"""
    Performs Inverse Fast Fourier Transform (IFFT) on the input tensor.

    Parameters
    ----------
    axis : int, optional
        Axis along which to perform the IFFT. Default is 0.
    norm : str, optional
        Normalization mode. Default is 'ortho'.
    """

    def __init__(self, axis: int = 0, norm: str = "ortho"):
        super().__init__()
        self.axis = axis
        self.norm = norm

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft(X, norm=self.norm, dim=self.axis)


class CarrierAllocator(nn.Module):
    r"""
    Allocates data to specific subcarriers based on a carrier type vector.

    Parameters
    ----------
    carrier_type : torch.Tensor
        1D tensor specifying the type of each subcarrier (1=data, -1=conjugate, 0=null).
    hermitian_sym : bool, optional
        If True, applies Hermitian symmetry. Default is False.
    """

    def __init__(self, carrier_type: torch.Tensor, hermitian_sym: bool = False):
        super().__init__()
        self.carrier_type = carrier_type
        self.hermitian_sym = hermitian_sym
        self.N = len(carrier_type)

        # Precompute masks
        self.index_data = (carrier_type == 1)
        if hermitian_sym:
            self.index_conj = (carrier_type == -1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Initialize output with zeros
        new_shape = list(X.shape)
        new_shape[0] = self.N
        Y = torch.zeros(new_shape, dtype=X.dtype)

        # Assign data subcarriers
        Y[self.index_data] = X

        # Hermitian symmetry
        if self.hermitian_sym:
            data = Y[self.index_data]
            conj_data = torch.conj(data.flip(0))
            Y[self.index_conj] = conj_data

        return Y


class CarrierExtractor(nn.Module):
    r"""
    Extracts data from specific subcarriers based on a carrier type vector.

    Parameters
    ----------
    carrier_type : torch.Tensor
        1D tensor specifying the type of each subcarrier (1=data, 0=null, -1=conjugate).
    """

    def __init__(self, carrier_type: torch.Tensor):
        super().__init__()
        self.carrier_type = carrier_type
        self.index_data = (carrier_type == 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X[self.index_data]