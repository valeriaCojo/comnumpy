import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd 
sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src")

from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from comnumpy.core.channels import AWGN, LaserPhaseNoise, AWGNChannel, LaserPhaseNoiseDual
from comnumpy.core.utils import get_alphabet
from comnumpy.optical.devices import Laser
from comnumpy.ofdm.chains import PhaseNoise
from comnumpy.ofdm.compensators import PhaseModulator, PhaseDemodulator, Weight
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
from comnumpy.core.metrics import compute_evm, compute_ser, compute_ber

M = 16
os = 2
h = 0.1
A = 1
sigma_awgn2 = 5e-3
sigma_phase2 = 5e-4
linewidth = 100_000
OSNR_dB = 20
L = 63
N = 2*(L+1)*os
l = np.arange(1, L+1)

fs = 1e9

# derivare linewidth si OSNR
# linewidth = sigma_phase2 * fs / (2 * np.pi)
# OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))

sigma_phase2 = 2 * np.pi * linewidth / fs
sigma_awgn2 = A**2 * (10 ** (-OSNR_dB / 10)) * os

print(f"linewidth: {linewidth:.2f} Hz")
print(f"OSNR_dB: {OSNR_dB:.2f} dB")
print(f"sigma_phase2: {sigma_phase2:.2e}")
print(f"sigma_awgn2: {sigma_awgn2:.2e}")

n_runs = 1000
alphabet = get_alphabet("QAM", M)
vect = np.zeros(N)
vect[1:L+1] = 1
vect[N:N-L-1:-1] = -1

gamma_l = sigma_phase2 / (np.sin(np.pi*l/N)**2) + sigma_awgn2/(A**2)

# uniform
a_uniform = np.ones(L)
a_uniform *= np.sqrt((N/2)/np.sum(a_uniform**2))

# sum-MSE
alpha_sum = (2/N)*np.sum(np.sqrt(gamma_l))
a_sum = np.sqrt(np.sqrt(gamma_l)/alpha_sum)
print("Power:", (2/N)*np.sum(a_sum**2))

# var-MSE
alpha_var = (2/N)*np.sum(gamma_l)
a_var = np.sqrt(gamma_l/alpha_var)
print("Power:", (2/N)*np.sum(a_var**2))

h_list = np.arange(0.05, 1.1, 0.05)
ber_u_list, ber_s_list, ber_v_list = [], [], []
ser_u_list, ser_s_list, ser_v_list = [], [], []

def run_chain(a, h_val):

    transmitter = Sequential([
        SymbolGenerator(M),
        Recorder(name = 'data_tx'),
        SymbolMapper(alphabet),
        Recorder(name='data_evm'),
        Serial2Parallel(L),
        Recorder(name='tx_symbols'),
        Weight(a=a),
        CarrierAllocator(vect, hermitian_sym=True),
        IFFTProcessor(),
        Recorder(name='after'),
        Parallel2Serial(),
        PhaseModulator(h=h_val),
        LaserPhaseNoiseDual(linewidth=linewidth, fs = fs, mode='transmitter'),
    ])

    receiver = Sequential([
        LaserPhaseNoiseDual(linewidth=linewidth, fs = fs, mode='receiver'),
        PhaseDemodulator(h=h_val, unwrap=True),
        Serial2Parallel(N),
        FFTProcessor(),
        CarrierExtractor(vect),
        Weight(a=1/a),
        Recorder(name='rx_symbols'),
        Parallel2Serial(),
        Recorder(name='data_evm'),
        SymbolDemapper(alphabet),
        Recorder(name = 'data_rx'),
    ])

    channel = Sequential([
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
    data_tx_evm = transmitter['data_evm'].get_data()
    data_rx_evm = receiver['data_evm'].get_data()

    ber = compute_ber(data_tx, data_rx, width=int(np.log2(M)))
    ser = compute_ser(data_tx, data_rx)
    evm = compute_evm(data_tx_evm, data_rx_evm)
    return mse, mse_tot, ber, ser, evm

for h_val in h_list:
    mse_u, mse_u_tot, ber_u, ser_u, evm_u = run_chain(a_uniform, h_val)
    mse_s, mse_s_tot, ber_s, ser_s, evm_s = run_chain(a_sum, h_val)
    mse_v, mse_v_tot, ber_v, ser_v, evm_v = run_chain(a_var, h_val)

    # Print rezultate
    print(f"\nh = {h_val:.2f}")
    print(f"  {'':10} {'Uniform':>12} {'Sum-MSE':>12} {'Var-MSE':>12}")
    print(f"  {'BER':10} {ber_u:>12.4e} {ber_s:>12.4e} {ber_v:>12.4e}")
    print(f"  {'SER':10} {ser_u:>12.4e} {ser_s:>12.4e} {ser_v:>12.4e}")

    # BER
    ber_u_list.append(ber_u)
    ber_s_list.append(ber_s)
    ber_v_list.append(ber_v)

    # SER
    ser_u_list.append(ser_u)
    ser_s_list.append(ser_s)
    ser_v_list.append(ser_v)

plt.figure()
plt.semilogy(h_list, ser_u_list, 'r-o', label='Uniform')
plt.semilogy(h_list, ser_s_list, 'g-o', label='Sum-MSE')
plt.semilogy(h_list, ser_v_list, 'b-o', label='Var-MSE')

plt.xlabel("h")
plt.ylabel("SER")
plt.title("SER vs h")
plt.grid(True)
plt.legend()

plt.figure()

plt.semilogy(h_list, ber_u_list, 'r-o', label='Uniform')
plt.semilogy(h_list, ber_s_list, 'g-o', label='Sum-MSE')
plt.semilogy(h_list, ber_v_list, 'b-o', label='Var-MSE')

plt.xlabel("h")
plt.ylabel("BER")
plt.title("BER vs h")
plt.grid(True)
plt.legend()

plt.show()
