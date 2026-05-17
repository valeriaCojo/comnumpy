import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from scipy import signal
from comnumpy.core.generics import Processor
from .utils import compute_sigma2


@dataclass
class AWGN(Processor):
    r"""
    Additive White Gaussian Noise (AWGN) channel.

    This class models an AWGN channel, which adds complex Gaussian noise to a signal.
    It is characterized by a noise power specified by sigma squared (sigma2).

    Signal Model
    ------------

    .. math::

       y_u[n] = x_u[n] + b_u[n]

    where:

    * :math:`b[n]\sim \mathcal{N}(0, \sigma^2)` is a Gaussian additive noise.

    For complex signals, a circular Gaussian noise is applied to the signal.
    The value of :math:`\sigma^2` is computed with respect to the method specified as input.

    Attributes
    ----------
    value : float, optional
        The value associated with the given method. Default is 1.
    unit : Literal["sigma2", "snr", "snr_dB", "snr_dBm"], optional
        The unit to compute the noise power. Default is ``"sigma2"``.
    sigma2s : float, optional
        Signal power. Default is 1.
    sigma2s_method : Literal["fixed", "measured"], optional
        Method used to obtain the signal power. If ``"measured"``, the signal
        power is estimated from the input signal. Default is ``"fixed"``.
    seed : int, optional
        The seed for the noise generator. Default is None.
    name : str, optional
        Name of the channel instance. Default is ``"awgn"``.
    """
    value: float = 1.
    unit: Literal["sigma2", "snr", "snr_dB", "snr_dBm"] = "sigma2"
    sigma2s: float = 1.
    sigma2s_method: Literal["fixed", "measured"] = "fixed"
    seed: int = None
    name: str = 'awgn'

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def get_sigma2s(self, x):
        # extract signal power
        match self.sigma2s_method:
            case "measured": 
                sigma2s = np.sum(np.abs(x)**2) / np.prod(x.shape)
            case "fixed":
                sigma2s = self.sigma2s
            case _:
                raise ValueError(f"Unknown sigma2s_method='{self.sigma2s_method}'. Expected one of: 'fixed', 'measured'.")

        return sigma2s

    def noise_rvs(self, x):
        is_complex = np.iscomplexobj(x)

        # compute sigma2s
        sigma2s = self.get_sigma2s(x)
        sigma2n = compute_sigma2(self.value, self.unit, sigma2s)
        shape = x.shape
        if is_complex:
            scale = np.sqrt(sigma2n / 2)
            b_r = self.rng.normal(scale=scale, size=shape)
            b_i = self.rng.normal(scale=scale, size=shape)
            b = b_r + 1j * b_i
        else:
            scale = np.sqrt(sigma2n)
            b = self.rng.normal(scale=scale, size=shape)

        self._b = b
        self.sigma2 = sigma2n

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.noise_rvs(x)
        y = x + self._b
        return y

@dataclass
class AWGNChannel(Processor):
    r"""
    Additive White Gaussian Noise (AWGN) channel.

    Signal Model
    ------------

    .. math::

        y[n] = x[n] + b[n]

    where:

    * :math:`b[n] \sim \mathcal{CN}(0, \sigma^2)` este zgomotul complex gaussian,
    * :math:`\sigma^2 = P_s \cdot 10^{-\text{OSNR}_{dB}/10} \cdot N_{os}`

    Attributes
    ----------
    OSNR_dB : float
        Optical Signal to Noise Ratio [dB]. Default is 20.
    os : int
        Oversampling factor pentru normare. Default is 1.
    is_complex : bool
        Daca semnalul este complex. Default is True.
    seed : int, optional
        Seed pentru reproductibilitate. Default is None.
    name : str
        Numele procesorului. Default is 'awgn_channel'.
    """
    OSNR_dB: float = 20.0
    os: int = 1
    is_complex: bool = True
    seed: int = None
    name: str = 'awgn_channel'

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def get_sigma2(self, signal_power):
        return signal_power * (10 ** (-self.OSNR_dB / 10)) * self.os

    def forward(self, x: np.ndarray) -> np.ndarray:
        N = len(x)
        signal_power = np.mean(np.abs(x) ** 2)
        sigma2 = self.get_sigma2(signal_power)

        if self.is_complex:
            scale = np.sqrt(sigma2 / 2)
            noise = (self.rng.normal(scale=scale, size=N) +
                     1j * self.rng.normal(scale=scale, size=N))
        else:
            scale = np.sqrt(sigma2)
            noise = self.rng.normal(scale=scale, size=N)

        return x + noise



@dataclass
class FIRChannel(Processor):
    r"""
    Finite Impulse Response (FIR) channel with given impulse response.

    Signal Model
    ------------

    The output signal :math:`y[n]` is computed as the convolution of the input signal :math:`x[n]` with the impulse response :math:`h[l]`:

    .. math::

       y[n] = \sum_{l=0}^{L-1} h[l] x[n-l]

    where:

    - :math:`h[l]` is the impulse response of the channel.
    - :math:`L` is the length of the impulse response.

    Attributes
    ----------
    h : np.ndarray
        The impulse response of the FIR channel. Should be a 1-dimensional numpy array.
    mode : Literal["full", "same", "valid"], optional
        Convolution mode passed to ``scipy.signal.convolve``. Default is ``"full"``.
    is_mimo : bool, optional
        Whether the channel supports MIMO input. Default is False.
    name : str, optional
        The name of the channel instance. Default is ``"fir"``.
    """
    h: np.array
    mode: Literal["full", "same", "valid"] = "full"
    is_mimo: bool = False
    name: str = "fir"

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = signal.convolve(x, self.h, mode=self.mode)
        return y


@dataclass
class LaserPhaseNoise(Processor):
    r"""
    Laser Phase Noise channel.

    Applies phase noise to the input signal based on the laser linewidth,
    modeled as a Wiener process.

    Signal Model
    ------------

    .. math::

        y[n] = x[n] \cdot e^{j\theta[n]}

    where:

    * :math:`\theta[n] = \sum_{k=0}^{n} d\theta[k]` is the accumulated phase noise,
    * :math:`d\theta[n] \sim \mathcal{N}(0, \sigma^2_\theta)`,
    * :math:`\sigma^2_\theta = 2\pi \Delta f / f_s`,
    * :math:`\Delta f` is the laser linewidth [Hz],
    * :math:`f_s` is the sampling frequency [Hz].

    Attributes
    ----------
    linewidth : float
        Laser linewidth [Hz]. Default is 100_000.
    fs : float
        Sampling frequency [Hz]. Default is 1e9.
    seed : int, optional
        Seed for reproducibility. Default is None.
    name : str
        Name of the processor. Default is 'laser_phase_noise'.

    Example
    -------
    >>> lpn = LaserPhaseNoise(linewidth=100_000, fs=1e9)
    >>> y = lpn(x)
    """
    linewidth: float = 100_000
    fs: float = 1e9
    seed: int = None
    name: str = 'laser_phase_noise'

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    @property
    def sigma2(self):
        return 2 * np.pi * self.linewidth / self.fs

    def forward(self, x: np.ndarray) -> np.ndarray:
        nb_samples = len(x)
        dtheta = np.sqrt(self.sigma2) * self.rng.standard_normal(nb_samples)
        theta = np.cumsum(dtheta)
        return x * np.exp(1j * theta)

@dataclass
class LaserPhaseNoiseDual(Processor):
    r"""
    Laser Phase Noise channel.

    Applies phase noise to the input signal based on the laser linewidth,
    modeled as a Wiener process.

    Signal Model
    ------------

    .. math::

        y[n] = x[n] \cdot e^{j\theta[n]}

    where:

    * :math:`\theta[n] = \sum_{k=0}^{n} d\theta[k]` is the accumulated phase noise,
    * :math:`d\theta[n] \sim \mathcal{N}(0, \sigma^2_\theta)`,
    * :math:`\sigma^2_\theta = 2\pi \Delta f / f_s`,
    * :math:`\Delta f` is the laser linewidth [Hz],
    * :math:`f_s` is the sampling frequency [Hz].

    Attributes
    ----------
    linewidth : float
        Laser linewidth [Hz]. Default is 100_000.
    fs : float
        Sampling frequency [Hz]. Default is 1e9.
    seed : int, optional
        Seed for reproducibility. Default is None.
    mode : str
        Operational mode: 'transmitter' applies e^{+j*theta},
        'receiver' applies e^{-j*theta}. Default is 'transmitter'.
    name : str
        Name of the processor. Default is 'laser_phase_noise'.

    Example
    -------
    >>> lpn_tx = LaserPhaseNoise(linewidth=100_000, fs=1e9, mode='transmitter')
    >>> lpn_rx = LaserPhaseNoise(linewidth=100_000, fs=1e9, mode='receiver')
    >>> y = lpn_tx(x)
    >>> z = lpn_rx(y)
    """
    linewidth: float = 100_000
    fs: float = 1e9
    seed: int = None
    mode: str = 'transmitter'
    name: str = 'laser_phase_noise'

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    @property
    def sigma2(self):
        return 2 * np.pi * self.linewidth / self.fs

    def forward(self, x: np.ndarray) -> np.ndarray:
        nb_samples = len(x)
        dtheta = np.sqrt(self.sigma2) * self.rng.standard_normal(nb_samples)
        theta = np.cumsum(dtheta)

        if self.mode == 'transmitter':
            return x * np.exp(+1j * theta)
        elif self.mode == 'receiver':
            return x * np.exp(-1j * theta)