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
from comnumpy.ofdm.compensators import PhaseModulator, PhaseDemodulator, Weight
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
from comnumpy.core.metrics import compute_evm, compute_ser, compute_ber

M = 16
A = 1
sigma_awgn2 = 5e-3
sigma_phase2 = 0
L = 63
n_runs = 1000
l = np.arange(1, L+1)
fs = 1e9
linewidth = sigma_phase2 * fs / (2 * np.pi)
alphabet = get_alphabet("QAM", M)

h_list = np.arange(0.05, 0.55, 0.05)
os_list = [2, 4, 8]
colors_os = {2: 'r', 4: 'g', 8: 'b'}

print(f"linewidth: {linewidth:.2f} Hz")

results = {os: {'uniform': {'th': [], 'sim': [], 'ber': []},
                'sum':     {'th': [], 'sim': [], 'ber': []},
                'var':     {'th': [], 'sim': [], 'ber': []}} for os in os_list}

def run_chain(a, h_val, N, vect, laser_pn, OSNR_dB, os):
    transmitter = Sequential([
        SymbolGenerator(M),
        Recorder(name='data_tx'),
        SymbolMapper(alphabet),
        Serial2Parallel(L),
        Recorder(name='tx_symbols'),
        Weight(a=a),
        CarrierAllocator(vect, hermitian_sym=True),
        IFFTProcessor(),
        Parallel2Serial(),
        PhaseModulator(h=h_val),
    ])

    receiver = Sequential([
        PhaseDemodulator(h=h_val, unwrap=True),
        Serial2Parallel(N),
        FFTProcessor(),
        CarrierExtractor(vect),
        Weight(a=1/a),
        Recorder(name='rx_symbols'),
        Parallel2Serial(),
        SymbolDemapper(alphabet),
        Recorder(name='data_rx'),
    ])

    channel = Sequential([
        laser_pn,
        AWGNChannel(OSNR_dB=OSNR_dB, os=os),
    ])

    chain = Sequential([transmitter, channel, receiver])
    chain(L*n_runs)

    S_tx = transmitter['tx_symbols'].get_data()
    S_rx = receiver['rx_symbols'].get_data()
    mse_tot = np.sum(np.mean(np.abs(S_rx - S_tx)**2, axis=1))

    data_tx = transmitter['data_tx'].get_data()
    data_rx = receiver['data_rx'].get_data()
    ber = compute_ber(data_tx, data_rx, width=int(np.log2(M)))

    return mse_tot, ber

for os in os_list:
    N = 2*(L+1)*os
    OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))
    laser_pn = LaserPhaseNoise(linewidth=linewidth, fs=fs)

    vect = np.zeros(N)
    vect[1:L+1] = 1
    vect[N:N-L-1:-1] = -1

    gamma_l = sigma_phase2 / (np.sin(np.pi*l/N)**2) + sigma_awgn2/(A**2)

    # ponderi
    a_uniform = np.ones(L)
    a_uniform *= np.sqrt((N/2)/np.sum(a_uniform**2))

    alpha_sum = (2/N)*np.sum(np.sqrt(gamma_l))
    a_sum = np.sqrt(np.sqrt(gamma_l)/alpha_sum)

    alpha_var = (2/N)*np.sum(gamma_l)
    a_var = np.sqrt(gamma_l/alpha_var)

    for h_val in h_list:

        mse_u_th = np.sum(gamma_l / (8*(np.pi*h_val*a_uniform)**2))
        mse_s_th = (np.sum(np.sqrt(gamma_l))**2) / (4*N*(np.pi*h_val)**2)
        mse_v_th = (L*np.sum(gamma_l)) / (4*N*(np.pi*h_val)**2)

        mse_u_sim, ber_u = run_chain(a_uniform, h_val, N, vect, laser_pn, OSNR_dB, os)
        mse_s_sim, ber_s = run_chain(a_sum,     h_val, N, vect, laser_pn, OSNR_dB, os)
        mse_v_sim, ber_v = run_chain(a_var,      h_val, N, vect, laser_pn, OSNR_dB, os)

        results[os]['uniform']['th'].append(mse_u_th)
        results[os]['uniform']['sim'].append(mse_u_sim)
        results[os]['uniform']['ber'].append(ber_u)

        results[os]['sum']['th'].append(mse_s_th)
        results[os]['sum']['sim'].append(mse_s_sim)
        results[os]['sum']['ber'].append(ber_s)

        results[os]['var']['th'].append(mse_v_th)
        results[os]['var']['sim'].append(mse_v_sim)
        results[os]['var']['ber'].append(ber_v)

        print(f"os={os}, h={h_val:.2f} | BER: U={ber_u:.2e} S={ber_s:.2e} V={ber_v:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
strategies = ['uniform', 'sum', 'var']
titles = ['Uniform', 'Sum-MSE', 'Var-MSE']

for ax, strat, title in zip(axes, strategies, titles):
    for os in os_list:
        c = colors_os[os]
        ax.semilogy(h_list, results[os][strat]['th'],
                    color=c, ls='-', label=f'TH os={os}')
        ax.semilogy(h_list, results[os][strat]['sim'],
                    color=c, ls='--', marker='o', markersize=4, label=f'SIM os={os}')
    ax.set_title(title)
    ax.set_xlabel("h")
    ax.set_ylabel("Total MSE")
    ax.grid(True)
    ax.legend(fontsize=8)

plt.suptitle(f'Total MSE vs h | sigma_phase2={sigma_phase2:.1e} | sigma_awgn2={sigma_awgn2:.1e}')
plt.tight_layout()

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

for ax, strat, title in zip(axes2, strategies, titles):
    for os in os_list:
        c = colors_os[os]
        ax.semilogy(h_list, results[os][strat]['ber'],
                    color=c, ls='-', marker='o', markersize=4, label=f'os={os}')
    ax.set_title(title)
    ax.set_xlabel("h")
    ax.set_ylabel("BER")
    ax.grid(True)
    ax.legend(fontsize=8)

plt.suptitle(f'BER vs h | sigma_phase2={sigma_phase2:.1e} | sigma_awgn2={sigma_awgn2:.1e}')
plt.tight_layout()

plt.show()