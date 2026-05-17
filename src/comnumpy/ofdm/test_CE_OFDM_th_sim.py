import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd 
sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src")

from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from comnumpy.core.channels import AWGN, LaserPhaseNoise, AWGNChannel
from comnumpy.core.utils import get_alphabet
from comnumpy.optical.devices import Laser
from comnumpy.ofdm.chains import PhaseNoise
from comnumpy.ofdm.compensators import PhaseModulator, PhaseDemodulator, Weight
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
from comnumpy.core.metrics import compute_evm, compute_ser, compute_ber

M = 16
os = 2
h = 0.2
A = 1
sigma_awgn2 = 5e-3
sigma_phase2 = 5e-4

L = 63
N = 2*(L+1)*os
l = np.arange(1, L+1)

fs = 1e9
linewidth = sigma_phase2 * fs / (2 * np.pi)
OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))
print(f"linewidth: {linewidth:.2f} Hz")
print(f"OSNR_dB: {OSNR_dB:.2f} dB")

# linewidth = 10_000
laser_pn = LaserPhaseNoise(linewidth=linewidth, fs = fs)

n_runs = 1000
alphabet = get_alphabet("QAM", M)

vect = np.zeros(N)
vect[1:L+1] = 1
vect[N:N-L-1:-1] = -1

gamma_l = sigma_phase2 / (np.sin(np.pi*l/N)**2) + sigma_awgn2/(A**2)

# uniform
a_uniform = np.ones(L)
a_uniform *= np.sqrt((N/2)/np.sum(a_uniform**2))
print("weight uniform:", a_uniform)
print("Power:", (2/N)*np.sum(a_uniform**2))


def run_chain(h):

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
        PhaseDemodulator(h=h, unwrap=False),
        Recorder(name='y1[n]'),
        Serial2Parallel(N),
        FFTProcessor(),
        CarrierExtractor(vect),
        Weight(a=1/a_uniform),
        Recorder(name='rx_symbols'),
        Parallel2Serial(),
        Recorder(name = 'constellation'),
        SymbolDemapper(alphabet),
        Recorder(name = 'data_rx'),
    ])

    channel = Sequential([
        # PhaseNoise(sigma2=sigma_phase2),
        laser_pn,
        # AWGN(value=sigma_awgn2, unit='sigma2'),
        AWGNChannel(OSNR_dB=OSNR_dB, os=os),
    ])

    chain = Sequential([transmitter, channel, receiver])

    chain(L*n_runs) 
    
    S_tx = transmitter['tx_symbols'].get_data()
    S_rx = receiver['rx_symbols'].get_data()

    mse = np.mean(np.abs(S_rx - S_tx)**2, axis=1)
    mse_tot = np.sum(mse)

    data_tx = transmitter['data_tx'].get_data()
    data_rx = receiver['data_rx'].get_data()
    data_const = receiver['constellation'].get_data()
    ber = compute_ber(data_tx, data_rx, width=int(np.log2(M)))
    ser = compute_ser(data_tx, data_rx)
    evm = compute_evm(data_tx, data_rx)
    print("data_tx:", data_tx.shape)
    
    return mse, mse_tot, ber, ser, evm, data_const

mse_u_1, mse_u_tot_1, ber_u_1, ser_u_1, evm_u_1, data_const_1 = run_chain(h)
# mse_u_2, mse_u_tot_2, ber_u_2, ser_u_2, evm_u_2, data_const_2 = run_chain(h=0.2)

mse_u_th_1 = gamma_l / (8*(np.pi*h*a_uniform)**2)
mse_u_th_2 = gamma_l / (8*(np.pi*h*a_uniform)**2)

plt.figure(figsize=(8,5))

# UNIFORM
plt.plot(l, mse_u_th_1, 'r-', label='Uniform TH')
plt.plot(l, mse_u_1, 'r:o', label='Uniform SIM')
# plt.plot(l, mse_u_th_2, 'r-', label='Uniform TH')
# plt.plot(l, mse_u_2, 'r:o', label='Uniform SIM')

plt.xlabel("Subcarrier index")
plt.ylabel("MSE")
plt.title("Theory vs Simulation")
plt.grid(True)
plt.legend()

plt.xlabel("real")
plt.ylabel("imag")

plt.show()