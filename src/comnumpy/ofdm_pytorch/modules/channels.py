import torch
import torch.nn as nn
class AWGNChannel(nn.Module):
    r"""
    Additive White Gaussian Noise (AWGN) channel.

    Signal Model
    ------------
    y[n] = x[n] + b[n]

    where b[n] ~ CN(0, sigma^2) and sigma^2 = P_s * 10^(-OSNR_dB/10) * os

    Parameters
    ----------
    OSNR_dB : float
        Optical Signal to Noise Ratio in dB. Default is 20.
    os : int
        Oversampling factor. Default is 1.
    is_complex : bool
        Whether the signal is complex. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is None.
    """

    def __init__(self, OSNR_dB: float = 20.0, os: int = 1, is_complex: bool = True, seed: int = None):
        super().__init__()
        self.OSNR_dB = OSNR_dB
        self.os = os
        self.is_complex = is_complex
        self._seed = seed  # salvam seed-ul initial

        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

    def reset_seed(self):
        """Reset generator to initial seed — same noise at every call."""
        if self._seed is not None:
            self.generator.manual_seed(self._seed)

    def get_sigma2(self, signal_power: torch.Tensor) -> torch.Tensor:
        return signal_power * (10 ** (-self.OSNR_dB / 10)) * self.os

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        signal_power = torch.mean(torch.abs(x) ** 2)
        sigma2 = self.get_sigma2(signal_power)

        if self.is_complex:
            scale = torch.sqrt(sigma2 / 2)
            noise_real = torch.normal(mean=0.0, std=scale.item(),
                                      size=(N,), generator=self.generator)
            noise_imag = torch.normal(mean=0.0, std=scale.item(),
                                      size=(N,), generator=self.generator)
            noise = torch.complex(noise_real, noise_imag)
        else:
            scale = torch.sqrt(sigma2)
            noise = torch.normal(mean=0.0, std=scale.item(),
                                 size=(N,), generator=self.generator)

        return x + noise

class LaserPhaseNoise(nn.Module):
    r"""
    Laser Phase Noise channel.

    Applies phase noise to the input signal based on the laser linewidth,
    modeled as a Wiener process.

    Signal Model
    ------------
    y[n] = x[n] * exp(+j * theta[n])  # transmitter
    y[n] = x[n] * exp(-j * theta[n])  # receiver

    Parameters
    ----------
    linewidth : float
        Laser linewidth [Hz]. Default is 100_000.
    fs : float
        Sampling frequency [Hz]. Default is 1e9.
    seed : int, optional
        Seed for reproducibility. Default is None.
    mode : str
        'transmitter' applies exp(+j*theta),
        'receiver' applies exp(-j*theta). Default is 'transmitter'.
    """

    def __init__(self, linewidth: float = 100_000, fs: float = 1e9, 
                 seed: int = None, mode: str = 'transmitter'):
        super().__init__()
        self.linewidth = linewidth
        self.fs = fs
        self._seed = seed
        self.mode = mode

        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

    def reset_seed(self):
        if self._seed is not None:
            self.generator.manual_seed(self._seed)

    @property
    def sigma2(self) -> float:
        return 2 * torch.pi * self.linewidth / self.fs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nb_samples = x.shape[0]
        dtheta = torch.sqrt(torch.tensor(self.sigma2)) * torch.randn(
            nb_samples, generator=self.generator)
        theta = torch.cumsum(dtheta, dim=0)

        if self.mode == 'transmitter':
            return x * torch.exp(+1j * theta)
        elif self.mode == 'receiver':
            return x * torch.exp(-1j * theta)