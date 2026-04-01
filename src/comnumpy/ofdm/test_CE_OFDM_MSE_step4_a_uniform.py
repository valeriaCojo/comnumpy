import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src")

from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from comnumpy.core.channels import AWGN
from comnumpy.core.utils import get_alphabet
from comnumpy.ofdm.chains import PhaseNoise
from comnumpy.ofdm.compensators import PhaseModulator, PhaseDemodulator, Weight,Unweight
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor

M = 16
os = 2
sigma_awgn2 = 5e-3
sigma_phase2 = 5e-4
h = 0.2
A = 1

L = 63
N = 2*(L+1)*os
l = np.arange(1, L+1)

n_runs = 1

alphabet = get_alphabet("QAM", M)

vect = np.zeros(N)
vect[1:L+1] = 1
vect[N:N-L-1:-1] = -1

gamma_l = sigma_phase2 / (np.sin(np.pi*l/N)**2) + sigma_awgn2/(A**2)

a_uniform = np.ones(L)
a_uniform *= np.sqrt((N/2) / np.sum(a_uniform**2))
print("Power:", (2/N)*np.sum(a_uniform**2))

mse_th = gamma_l / (8*(np.pi*h*a_uniform)**2)

transmitter = Sequential([
    SymbolGenerator(M),
    Recorder(name = 'data_tx'),
    SymbolMapper(alphabet),
    Serial2Parallel(L),
    Recorder(name='tx_symbols'),
    Weight(a=a_uniform),
    CarrierAllocator(vect, hermitian_sym=True),
    IFFTProcessor(),
    Recorder(name='after'),
    Parallel2Serial(),
    PhaseModulator(h=h),
])

receiver = Sequential([
    PhaseDemodulator(h=h, unwrap=True),
    Serial2Parallel(N),
    FFTProcessor(),
    CarrierExtractor(vect),
    Weight(a=1/a_uniform),
    Recorder(name='rx_symbols'),
    Parallel2Serial(),
    SymbolDemapper(alphabet),
    Recorder(name = 'data_rx'),
])

channel = Sequential([
    PhaseNoise(sigma2=sigma_phase2),
    AWGN(value=sigma_awgn2, unit='sigma2'),
])

chain = Sequential([transmitter, channel, receiver])

chain(L * n_runs)

S_tx = transmitter['tx_symbols'].get_data()
S_rx = receiver['rx_symbols'].get_data()

mse_sim = np.mean(np.abs(S_rx - S_tx)**2, axis=1)

print("Mean |S_tx|:", np.mean(np.abs(S_tx)))
print("Mean |S_rx|:", np.mean(np.abs(S_rx)))

plt.figure(figsize=(8,5))

plt.plot(l, mse_th, 'r--o', label='Theoretical')
plt.plot(l, mse_sim, 'b-o', label='Simulation')

plt.xlabel("Subcarrier index l")
plt.ylabel("MSE")
plt.title("Uniform weights: MSE")
plt.grid(True)
plt.legend()
plt.show()