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
from comnumpy.ofdm.compensators import PhaseModulator, PhaseDemodulator, Weight, Unweight
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
from comnumpy.ofdm.utils import get_standard_carrier_allocation

M = 4
os = 2
sigma_awgn2 = 1e-3
sigma_phase2 = 1e-6
h = 0.1
A = 1

alphabet = get_alphabet("QAM", M)
type_of_carrier = get_standard_carrier_allocation("NoPilot_128", os=os, shift=True, hermitian_sym=True)

carrier = len(type_of_carrier)
carrier_data = np.sum(type_of_carrier == 1)
Nfft = carrier

data_idx = np.where(type_of_carrier == 1)[0]
dc = Nfft // 2

l = np.abs(data_idx - dc)
l[l == 0] = 1

a_l = np.ones_like(l)
a_l = a_l * np.sqrt((Nfft / 2) / np.sum(a_l**2))

b_l = sigma_phase2 / (np.sin(np.pi * l / Nfft)**2) + sigma_awgn2 / (A**2)

weight_block = Weight(a=a_l)
unweight_block = Unweight(a=a_l)

transmitter = Sequential([
    SymbolGenerator(M),
    SymbolMapper(alphabet),
    Recorder(name='data_evm'),
    Serial2Parallel(carrier_data),
    Recorder(name='tx_symbols'),
    weight_block,
    Recorder(name='after_weight'),
    CarrierAllocator(type_of_carrier, hermitian_sym=True),
    IFFTProcessor(),
    Parallel2Serial(),
    PhaseModulator(h=h),
    Recorder(name='tx_phase'),
])

channel = Sequential([
    PhaseNoise(sigma2=sigma_phase2),
    AWGN(value=sigma_awgn2, unit='sigma2'),
])

receiver = Sequential([
    PhaseDemodulator(h=h),
    Recorder(name='rx_phase'),
    Serial2Parallel(carrier),
    FFTProcessor(),
    CarrierExtractor(type_of_carrier),
    unweight_block,
    Recorder(name='rx_symbols'),
    Parallel2Serial(),
    Recorder(name='constellation'),
    SymbolDemapper(alphabet),
])

chain = Sequential([transmitter, channel, receiver])

N_OFDM_symbols = 1000
N = int(carrier_data * N_OFDM_symbols)

n_runs = 1000
mse_accum = np.zeros(carrier_data)

for _ in range(n_runs):
    y = chain(N)

    S_tx = transmitter['tx_symbols'].get_data()
    S_rx = receiver['rx_symbols'].get_data()

    S_tx = S_tx.reshape(carrier_data, N_OFDM_symbols, order='F')
    S_rx = S_rx.reshape(carrier_data, N_OFDM_symbols, order='F')

    mse = np.mean(np.abs(S_rx - S_tx)**2, axis=1)
    mse_accum += mse

mse_sim = mse_accum / n_runs

mse_th = (1 / (8 * (np.pi * h * a_l)**2)) * b_l

sort_idx = np.argsort(l)

l_sorted = l[sort_idx]
mse_th_sorted = mse_th[sort_idx]
mse_sim_sorted = mse_sim[sort_idx]

plt.figure(figsize=(8,5))
plt.plot(l_sorted[:20], mse_th_sorted[:20], 'r--o', label='Theoretical')
plt.plot(l_sorted[:20], mse_sim_sorted[:20], 'b-o', label='Simulation')
plt.xlabel("Distance from DC (l)")
plt.ylabel("MSE")
plt.title("MSE Comparison")
plt.grid(True)
plt.legend()

constellation_rx = receiver["constellation"].get_data()

plt.figure()
plt.plot(np.real(constellation_rx), np.imag(constellation_rx), ".")
plt.xlabel("Real")
plt.ylabel("Imag")
plt.title("Received Constellation")


plt.show()