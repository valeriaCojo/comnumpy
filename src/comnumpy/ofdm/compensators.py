import numpy as np
from typing import Literal
from dataclasses import dataclass, field
from scipy.fft import fft, fftshift
from comnumpy.core import Processor, Sequential
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial, WeightAmplifier
from .processors import CarrierAllocator, IFFTProcessor, CyclicPrefixer
from .utils import get_standard_carrier_allocation


@dataclass
class FrequencyDomainEqualizer(WeightAmplifier):
    r"""
    A frequency domain equalizer that applies weights to compensate for channel effects in the frequency domain.

    This class extends the `WeightAmplifier` to operate in the frequency domain, using the Fast Fourier Transform (FFT) to compute weights that equalize the input signal. The equalizer can optionally shift the zero-frequency component to the center of the spectrum.

    Signal Model
    ------------

    Given a channel impulse response :math:`h[l]`, the Frequency Domain Equalizer applies the following weight amplifier

    .. math::

        y[n] = \left(\frac{1}{H[n]}\right) \cdot x[n]

    * :math:`H[n]` corresponds to the inverse of the :math:`n`-th bin of the channel's Discrete Fourier Transform (DFT).

    Attributes
    ----------
    h : np.ndarray, optional
        The impulse response of the channel to be equalized. Default is None.
    shift : bool, optional
        If True, applies a frequency shift to center the zero-frequency component. Default is True.
    axis : int, optional
        The axis along which to compute the FFT and apply the weights. Default is -2.
    name : str
        Name of the frequency domain equalizer instance.

    Example
    -------
    >>> equalizer = FrequencyDomainEqualizer(h=np.array([1, 0.5, 0.2]))
    >>> X = np.random.randn(4, 3, 2)  # Example input tensor
    >>> Y = equalizer(X)
    """
    h : np.ndarray = None
    axis: int = 0
    shift: bool = False
    norm: Literal["ortho", "backward", "forward"] = "ortho"
    weight: np.ndarray = field(init=False, default=None)
    name : str = "frequency domain equalizer"

    def __post_init__(self):
        if self.h is None:
            raise ValueError("The impulse response 'h' must be provided.")

    def prepare(self, X):
        """
        Compute the amplifier weight from the channel impulse response
        """
        N_sc = X.shape[self.axis]
        Hw = fft(self.h, n=N_sc,  axis=self.axis)
        weight = 1./Hw 
        if self.shift:
            weight = fftshift(weight)
        self.weight = weight


@dataclass
class PhaseModulator(Processor):
    r"""
    Phase Modulator for Constant Envelope OFDM (CE-OFDM).

    Signal Model
    ------------
    This processor applies a nonlinear phase modulation to the input signal in order
    to generate a constant envelope signal.

    The modulation is defined as:

    .. math::

        y[n] = e^{j 2 \pi h x[n]}

    where:

    * :math:`x[n]` is the input real-valued OFDM time signal,
    * :math:`h` is the modulation index,
    * :math:`y[n]` is the phase-modulated complex signal.

    This operation maps the amplitude information of :math:`x[n]` into the **phase**
    of the complex exponential. As a result, the transmitted signal has:

    .. math::

        |y[n]| = 1

    meaning the signal has a **constant envelope**.

    This property is useful in communication systems where nonlinear power amplifiers
    are used, since constant envelope signals are more robust to nonlinear distortions.

    Notes
    -----
    In CE-OFDM systems, the input signal :math:`x[n]` is typically obtained from an OFDM
    modulator with **Hermitian symmetry**, ensuring that the time-domain signal is real.

    Example
    -------
    >>> mod = PhaseModulator(h=0.5)
    >>> x = np.array([0.1, 0.2, -0.1])
    >>> y = mod(x)

    """
    h: float = 1.0
    name:str = "phase_modulator"

    def forward(self, X: np.ndarray) -> np.ndarray:

        Y = np.exp(2j * np.pi * self.h * X)

        return Y

@dataclass
class PhaseDemodulator(Processor):
    r"""
    Phase Demodulator for Constant Envelope OFDM (CE-OFDM).

    Signal Model
    ------------
    This processor performs phase demodulation in order to recover the
    original OFDM time-domain signal from the constant envelope signal.

    The received CE-OFDM signal has the form:

    .. math::

        y[n] = e^{j 2 \pi h x[n]}

    where:

    * :math:`x[n]` is the original real OFDM time-domain signal
    * :math:`h` is the modulation index
    * :math:`y[n]` is the transmitted constant-envelope signal

    The demodulation consists in extracting the phase of the received signal:

    .. math::

        z[n] = \frac{\angle y[n]}{2 \pi h}

    where:

    * :math:`\angle y[n]` represents the phase of the complex signal
    * :math:`z[n]` is the recovered estimate of the original signal :math:`x[n]`

    Principle
    ---------
    The phase modulator maps the real-valued OFDM signal into the phase
    of a complex exponential. The demodulator simply retrieves this phase
    and rescales it by the factor :math:`2\pi h`.

    Ideally:

    .. math::

        z[n] \approx x[n]

    Notes
    -----
    In practical systems the recovered signal can contain distortions due to:

    * channel noise
    * phase wrapping effects
    * nonlinearities

    Example
    -------
    >>> demod = PhaseDemodulator(h=0.5)
    >>> y = np.exp(2j*np.pi*0.5*np.array([0.1, 0.2]))
    >>> z = demod(y)

    """

    h: float = 1.0
    name: str = "phase_demodulator"

    def forward(self, X: np.ndarray) -> np.ndarray:

        phase = np.angle(X)

        Z = phase / (2 * np.pi * self.h)

        return Z

class SubcarrierWeight(Processor):

    def __init__(self, gamma=0.02):
        self.gamma = gamma
        self.W = None

    def forward(self, X):

        N = X.shape[0]

        if self.W is None:

            k = np.arange(N)
            k = k - N//2
            k = np.abs(k)

            W = 1 / np.sqrt(1 + self.gamma * k**2)

            # normalize average power
            W = W / np.sqrt(np.mean(W**2))

            self.W = W

        return X * self.W[:, None]

class SubcarrierWeight(Processor):

    def __init__(self, sigma2, Nsc, os, h, OSNR):

        self.sigma2 = sigma2
        self.Nsc = Nsc
        self.os = os
        self.h = h
        self.OSNR = OSNR
        self.W = None


    def forward(self, X):

        N = X.shape[0]

        if self.W is None:

            # subcarrier index (distance from carrier)
            k = np.arange(1, N+1)

            term1 = (self.sigma2 * (self.Nsc*self.os)**2) / (2*np.pi**2 * k**2)
            term2 = 1/(2*self.OSNR)

            W = (1/(2*np.pi*self.h)) * np.sqrt(term1 + term2)

            # normalize average power
            W = W / np.sqrt(np.mean(W**2))

            self.W = W

        return X * self.W[:, None]
class SubcarrierUnweight(Processor):

    def __init__(self, weight_block):
        self.weight_block = weight_block

    def forward(self, X):

        W = self.weight_block.W
        return X / W[:, None]
    

