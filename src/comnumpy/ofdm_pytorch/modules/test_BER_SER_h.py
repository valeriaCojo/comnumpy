# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src\comnumpy\ofdm_pytorch")
# from comnumpy.core.utils import get_alphabet
# from modules.generics import Sequential, Recorder
# from modules.generators import SymbolGenerator
# from modules.mappers import SymbolMapper, SymbolDemapper
# from modules.processors import Serial2Parallel, Parallel2Serial, CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
# from modules.channels import AWGNChannel, LaserPhaseNoise
# from modules.compensators import PhaseModulator, PhaseDemodulator, Weight
# from modules.metrics import compute_ber, compute_ser

# M = 16
# os = 2
# A = 1
# linewidth = 100_000
# OSNR_dB = 20
# L = 63
# N = 2*(L+1)*os
# l_np = np.arange(1, L+1)
# fs = 1e9
# sigma_phase2 = 2 * np.pi * linewidth / fs
# sigma_awgn2 = A**2 * (10 ** (-OSNR_dB / 10)) * os
# n_runs = 1000
# n_iterations = 10000

# print(f"linewidth: {linewidth:.2f} Hz")
# print(f"OSNR_dB: {OSNR_dB:.2f} dB")
# print(f"sigma_phase2: {sigma_phase2:.2e}")
# print(f"sigma_awgn2: {sigma_awgn2:.2e}")

# alphabet_np = get_alphabet("QAM", M)
# alphabet = torch.tensor(alphabet_np, dtype=torch.complex64)

# vect = torch.zeros(N)
# vect[1:L+1] = 1
# indices = torch.arange(N-1, N-L-1, -1)
# vect[indices] = -1

# gamma_l = sigma_phase2 / (np.sin(np.pi*l_np/N)**2) + sigma_awgn2/(A**2)
# alpha_var = (2/N) * np.sum(gamma_l)
# a_var_th = np.sqrt(gamma_l / alpha_var)

# h_list = np.arange(0.05, 0.3, 0.01)
# # learning_rates = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
# learning_rates = [2e-2, 1e-2, 5e-3]

# results = {lr: {} for lr in learning_rates}

# for lr in learning_rates:
#     print(f"\nRunning lr={lr}")

#     for h_val in h_list:
#         print(f"  h={h_val:.2f}")

#         # a = nn.Parameter(torch.Tensor(a_var_th))
#         a = nn.Parameter(torch.ones(L))
#         optimizer = torch.optim.Adam([a], lr=lr)
#         awgn = AWGNChannel(OSNR_dB=OSNR_dB, os=os, seed=42)

#         for i in range(n_iterations):
#             optimizer.zero_grad()
#             # awgn.reset_seed() — eliminat

#             a_norm = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

#             transmitter = Sequential([
#                 SymbolGenerator(M),
#                 Recorder(name='data_tx'),
#                 SymbolMapper(alphabet),
#                 Serial2Parallel(L),
#                 Recorder(name='tx_symbols'),
#                 Weight(a=a_norm),
#                 CarrierAllocator(vect, hermitian_sym=True),
#                 IFFTProcessor(),
#                 Parallel2Serial(),
#                 PhaseModulator(h=h_val),
#                 LaserPhaseNoise(linewidth=linewidth, fs=fs, mode='transmitter'),
#             ])

#             receiver = Sequential([
#                 LaserPhaseNoise(linewidth=linewidth, fs=fs, mode='receiver'),
#                 PhaseDemodulator(h=h_val, unwrap=True),
#                 Serial2Parallel(N),
#                 FFTProcessor(),
#                 CarrierExtractor(vect),
#                 Weight(a=1/a_norm),
#                 Recorder(name='rx_symbols'),
#                 Parallel2Serial(),
#                 SymbolDemapper(alphabet),
#                 Recorder(name='data_rx'),
#             ])

#             channel = Sequential([awgn])
#             chain = Sequential([transmitter, channel, receiver])
#             chain(L * n_runs)

#             S_tx = transmitter['tx_symbols'].get_data()
#             S_rx = receiver['rx_symbols'].get_data()

#             mse_per_subcarrier = torch.mean(torch.abs(S_rx - S_tx)**2, dim=1)
#             loss = torch.var(mse_per_subcarrier)
#             loss.backward()
#             optimizer.step()

#             if i % 500 == 0:
#                 print(f"    iter {i}, loss: {loss.item():.6f}")

#         # BER si SER — rulare separata dupa convergenta
#         with torch.no_grad():
#             a_norm_final = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

#             transmitter_eval = Sequential([
#                 SymbolGenerator(M),
#                 Recorder(name='data_tx'),
#                 SymbolMapper(alphabet),
#                 Serial2Parallel(L),
#                 Recorder(name='tx_symbols'),
#                 Weight(a=a_norm_final),
#                 CarrierAllocator(vect, hermitian_sym=True),
#                 IFFTProcessor(),
#                 Parallel2Serial(),
#                 PhaseModulator(h=h_val),
#                 LaserPhaseNoise(linewidth=linewidth, fs=fs, mode='transmitter'),
#             ])

#             receiver_eval = Sequential([
#                 LaserPhaseNoise(linewidth=linewidth, fs=fs, mode='receiver'),
#                 PhaseDemodulator(h=h_val, unwrap=True),
#                 Serial2Parallel(N),
#                 FFTProcessor(),
#                 CarrierExtractor(vect),
#                 Weight(a=1/a_norm_final),
#                 Recorder(name='rx_symbols'),
#                 Parallel2Serial(),
#                 SymbolDemapper(alphabet),
#                 Recorder(name='data_rx'),
#             ])

#             channel_eval = Sequential([AWGNChannel(OSNR_dB=OSNR_dB, os=os)])
#             chain_eval = Sequential([transmitter_eval, channel_eval, receiver_eval])
#             chain_eval(L * n_runs)

#             data_tx = transmitter_eval['data_tx'].get_data()
#             data_rx = receiver_eval['data_rx'].get_data()
#             ber = compute_ber(data_tx, data_rx, width=int(np.log2(M)))
#             ser = compute_ser(data_tx, data_rx)

#         results[lr][h_val] = {'ber': ber, 'ser': ser}
#         print(f"    BER: {ber:.4e}, SER: {ser:.4e}")

# # --- Grafic BER vs h ---
# plt.figure(figsize=(10, 5))
# for lr in learning_rates:
#     ber_list = [results[lr][h]['ber'] for h in h_list]
#     plt.semilogy(h_list, ber_list, label=f'lr={lr}')
# plt.xlabel("Phase Modulation Index h")
# plt.ylabel("BER")
# plt.title("BER vs h")
# plt.ylim([1e-4, 1e0])
# plt.legend()
# plt.grid(True, which='both')

# # --- Grafic SER vs h ---
# plt.figure(figsize=(10, 5))
# for lr in learning_rates:
#     ser_list = [results[lr][h]['ser'] for h in h_list]
#     plt.semilogy(h_list, ser_list, label=f'lr={lr}')
# plt.xlabel("Phase Modulation Index h")
# plt.ylabel("SER")
# plt.title("SER vs h")
# plt.ylim([1e-4, 1e0])
# plt.legend()
# plt.grid(True, which='both')

# plt.show()

import matplotlib.pyplot as plt
import numpy as np

h = [0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29]

ber = [2.2302e-1,1.8260e-1,1.5117e-1,1.2305e-1,1.0293e-1,8.5071e-2,6.8944e-2,5.5901e-2,4.4306e-2,3.5675e-2,2.4960e-2,2.1270e-2,1.6972e-2,1.0702e-2,2.8373e-3,1.1111e-3,1.2659e-3,1.3730e-3,1.6190e-3,2.4643e-3,1.7421e-3,2.6349e-3,4.1468e-3,4.4881e-3,4.6667e-3]

ser = [6.5254e-1,5.7749e-1,5.0414e-1,4.2929e-1,3.6981e-1,3.1105e-1,2.5752e-1,2.1200e-1,1.6990e-1,1.3827e-1,9.7762e-2,8.3619e-2,6.7127e-2,4.2460e-2,1.1270e-2,4.4444e-3,4.6667e-3,4.6508e-3,5.4444e-3,7.6984e-3,5.4762e-3,7.6825e-3,1.1984e-2,1.2635e-2,1.3587e-2]

# --- Grafic BER vs h ---
plt.figure(figsize=(10, 5))
plt.semilogy(h, ber, 'b-o', markersize=4, label='BER')
plt.xlabel("Phase Modulation Index h")
plt.ylabel("BER")
plt.title("BER vs h (lr=0.01)")
plt.ylim([1e-4, 1e0])
plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
plt.grid(True, which='both')
plt.legend()

# --- Grafic SER vs h ---
plt.figure(figsize=(10, 5))
plt.semilogy(h, ser, 'r-o', markersize=4, label='SER')
plt.xlabel("Phase Modulation Index h")
plt.ylabel("SER")
plt.title("SER vs h (lr=0.01)")
plt.ylim([1e-4, 1e0])
plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
plt.grid(True, which='both')
plt.legend()

# --- Grafic BER si SER pe acelasi grafic ---
plt.figure(figsize=(10, 5))
plt.semilogy(h, ber, 'b-o', markersize=4, label='BER')
plt.semilogy(h, ser, 'r-o', markersize=4, label='SER')
plt.xlabel("Phase Modulation Index h")
plt.ylabel("BER / SER")
plt.title("BER and SER vs h (lr=0.01)")
plt.ylim([1e-4, 1e0])
plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
plt.grid(True, which='both')
plt.legend()

plt.show()