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
from comnumpy.ofdm.compensators import PhaseModulator, PhaseDemodulator, Weight
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor

M = 16
os = 2
sigma_awgn2 = 5e-3
sigma_phase2 = 5e-4
h = 0.1
A = 1

L = 63
N = 2*(L+1)*os
N_OFDM_symbols = 1000
l = np.arange(1, L+1)

n_runs = 1000

alphabet = get_alphabet("QAM", M)

vect = np.zeros(N)
vect[1:L+1] = 1
vect[N:N-L-1:-1] = -1


gamma_l = sigma_phase2 / (np.sin(np.pi*l/N)**2) + sigma_awgn2/(A**2)

alpha_var = (2/N) * np.sum(gamma_l)
a_var = np.sqrt(gamma_l / alpha_var)

print("Power:", (2/N)*np.sum(a_var**2))

mse_th = gamma_l / (8*(np.pi*h*a_var)**2)

J_var_th = (L * np.sum(gamma_l)) / (4*N*(np.pi*h)**2)

print("Sum(mse_th):", np.sum(mse_th))
print("J_var_th:", J_var_th)


transmitter = Sequential([
    SymbolGenerator(M),
    SymbolMapper(alphabet),
    Serial2Parallel(L),
    Recorder(name='tx_symbols'),
    Weight(a=a_var),
    CarrierAllocator(vect, hermitian_sym=True),
    IFFTProcessor(),
    Parallel2Serial(),
    PhaseModulator(h=h),
])

receiver = Sequential([
    PhaseDemodulator(h=h),
    Serial2Parallel(N),
    FFTProcessor(),
    CarrierExtractor(vect),
    Weight(a=1/a_var),
    Recorder(name='rx_symbols'),
    Parallel2Serial(),
    SymbolDemapper(alphabet),
])

channel = Sequential([
    PhaseNoise(sigma2=sigma_phase2),
    AWGN(value=sigma_awgn2, unit='sigma2'),
])

chain = Sequential([transmitter, channel, receiver])


mse_accum = np.zeros(L)

for _ in range(n_runs):
    chain(L)

    S_tx = transmitter['tx_symbols'].get_data()
    S_rx = receiver['rx_symbols'].get_data()

    mse = np.mean(np.abs(S_rx - S_tx)**2, axis=1)
    mse_accum += mse

mse_sim = mse_accum / n_runs

total_mse_sim = np.sum(mse_sim)
var_mse_sim = np.var(mse_sim)

print("mean MSE:", np.mean(mse_sim))
print("std:", np.std(mse_sim))

print("\n RESULTS: VAR-MSE\n")

print("Per-subcarrier MSE (simulation):")
print(mse_sim)

print("\n Total MSE (simulation):")
print(total_mse_sim)

print("\n Total MSE (theoretical):")
print(J_var_th)

print("\n MSE variance across subcarriers:")
print(var_mse_sim)

plt.figure(figsize=(8,5))
plt.plot(l[:20], mse_th[:20], 'r--o', label='Theoretical')
plt.plot(l[:20], mse_sim[:20], 'b-o', label='Simulation')
plt.xlabel("Subcarrier index l")
plt.ylabel("MSE")
plt.title("Var-MSE weights")
plt.grid(True)
plt.legend()
plt.show()