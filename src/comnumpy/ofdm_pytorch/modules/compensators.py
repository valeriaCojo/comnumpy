import torch
import torch.nn as nn
import numpy as np


class PhaseModulator(nn.Module):
    r"""
    Phase Modulator for Constant Envelope OFDM (CE-OFDM) in PyTorch.

    Signal Model
    ------------

    .. math::

        y[n] = e^{j 2 \pi h x[n]}

    Parameters
    ----------
    h : float
        Modulation index.
    """
    def __init__(self, h: float):
        super().__init__()
        self.h = h
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(2j * torch.pi * self.h * x)

class PhaseDemodulator(nn.Module):
    r"""
    Phase Demodulator for Constant Envelope OFDM (CE-OFDM) in PyTorch.

    Signal Model
    ------------
    This processor performs phase demodulation to recover the original
    OFDM time-domain signal from the constant envelope signal.

    The received CE-OFDM signal has the form:

    .. math::

        y[n] = e^{j 2 \pi h x[n]}

    The demodulation consists in extracting the phase of the received signal:

    .. math::

        z[n] = \frac{\angle y[n]}{2 \pi h}

    where:

    * :math:`\angle y[n]` is the phase of the complex signal,
    * :math:`h` is the modulation index,
    * :math:`z[n]` is the recovered estimate of the original signal :math:`x[n]`.

    Differentiable Unwrap
    ---------------------
    Unlike the NumPy implementation which uses ``np.unwrap``, this implementation
    uses a differentiable unwrap based on ``torch.cumsum``, allowing gradient
    backpropagation through the unwrap operation.

    The differentiable unwrap works as follows:

    1. Compute consecutive differences of the phase.
    2. Wrap differences into :math:`[-\pi, \pi]` to detect and correct phase jumps.
    3. Reconstruct the continuous phase via cumulative sum.

    Parameters
    ----------
    h : float
        Modulation index. Default is 1.0.
    unwrap : bool
        If True, applies differentiable unwrap to the phase. Default is False.

    Example
    -------
    >>> demod = PhaseDemodulator(h=0.3, unwrap=True)
    >>> y = torch.exp(2j * torch.pi * 0.3 * x)
    >>> z = demod(y)  # recovers x
    """
    def __init__(self, h: float, unwrap: bool = False):
        super().__init__()
        self.h = h
        self.unwrap = unwrap

    @staticmethod
    def differentiable_unwrap(phase: torch.Tensor) -> torch.Tensor:
        diff = torch.diff(phase, prepend=phase[..., :1])
        diff_wrapped = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.cumsum(diff_wrapped, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = torch.angle(x)
        if self.unwrap:
            if x.dim() == 2:
                phase = self.differentiable_unwrap(phase.T).T
            else:
                phase = self.differentiable_unwrap(phase)
        return phase / (2*torch.pi *self.h)

class Weight(nn.Module):
    r"""
    Applies element-wise weighting to the input tensor.

    Parameters
    ----------
    a : torch.Tensor
        Weight vector of shape (N_sub,).
    """

    def __init__(self, a: torch.Tensor):
        super().__init__()
        self.a = a

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            return self.a * X
        elif X.ndim == 2:
            return self.a[:, None] * X
        else:
            raise ValueError(f'Unsupported input shape for Weight: {X.shape}')