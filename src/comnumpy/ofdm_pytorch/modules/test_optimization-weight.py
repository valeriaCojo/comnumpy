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

# M = 16
# os = 2
# h = 0.1
# A = 1
# sigma_awgn2 = 5e-3
# sigma_phase2 = 5e-4
# L = 63
# N = 2*(L+1)*os
# l_np = np.arange(1, L+1)
# fs = 1e9
# linewidth = sigma_phase2 * fs / (2 * np.pi)
# OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))
# n_runs = 1000
# print('linewidth[Hz]:', linewidth)
# print('OSNR[dB]:', OSNR_dB)
# alphabet_np = get_alphabet("QAM", M)
# alphabet = torch.tensor(alphabet_np, dtype=torch.complex64)

# vect = torch.zeros(N)
# vect[1:L+1] = 1
# indices = torch.arange(N-1, N-L-1, -1)
# vect[indices] = -1

# gamma_l = sigma_phase2 / (np.sin(np.pi*l_np/N)**2) + sigma_awgn2/(A**2)

# # 1) a
# a = nn.Parameter(torch.ones(L))
# print("a shape:", a.shape)
# print("a requires_grad:", a.requires_grad)

# # canal cu seed fix
# laser_pn = LaserPhaseNoise(linewidth=linewidth, fs=fs, seed=42)
# awgn = AWGNChannel(OSNR_dB=OSNR_dB, os=os, seed=42)

# optimizer = torch.optim.Adam([a], lr=2e-2)
# # optimizer = torch.optim.SGD([a], lr=1e-2, momentum=0.9)
# n_iterations = 1000
# loss_total = []

# for i in range(n_iterations):
#     optimizer.zero_grad()

#     laser_pn.reset_seed()
#     awgn.reset_seed()

#     # normalizam a — diferentiabil, fara no_grad
#     a_norm = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

#     transmitter = Sequential([
#         SymbolGenerator(M),
#         Recorder(name='data_tx'),
#         SymbolMapper(alphabet),
#         Serial2Parallel(L),
#         Recorder(name='tx_symbols'),
#         Weight(a=a_norm),
#         CarrierAllocator(vect, hermitian_sym=True),
#         IFFTProcessor(),
#         Parallel2Serial(),
#         PhaseModulator(h=h),
#     ])

#     receiver = Sequential([
#         PhaseDemodulator(h=h, unwrap=True),
#         Serial2Parallel(N),
#         FFTProcessor(),
#         CarrierExtractor(vect),
#         Weight(a=1/a_norm),
#         Recorder(name='rx_symbols'),
#         Parallel2Serial(),
#         Recorder(name = 'constellation'),
#         SymbolDemapper(alphabet),
#         Recorder(name = 'data_rx'),
#     ])

#     channel = Sequential([laser_pn, awgn])
#     chain = Sequential([transmitter, channel, receiver])
#     chain(L * n_runs)

#     #2) loss 
#     S_tx = transmitter['tx_symbols'].get_data()  
#     S_rx = receiver['rx_symbols'].get_data()     
#     loss = torch.mean(torch.abs(S_rx - S_tx)**2)

#     # data_tx = transmitter['data_tx'].get_data()
#     # rx_symbols = receiver['rx_symbols'].get_data()
#     # loss = soft_classification_loss(rx_symbols, data_tx, alphabet, T=0.01)

#     # 3) backpropagation + update a
#     loss.backward()
#     optimizer.step()

#     loss_total.append(loss.item())

#     if i % 100 == 0:
#         print(f"Iter {i}, Loss: {loss.item():.6f}")

# plt.figure(figsize=(8, 4))
# plt.plot(loss_total)
# plt.xlabel("Iteration")
# plt.ylabel("Loss (MSE)")
# plt.title("Convergence curve")
# plt.grid(True)

# # 4) a_final 
# with torch.no_grad():
#     a_final = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))
# print("a learned:", a_final)

# # a_sum th
# alpha_sum = (2/N) * np.sum(np.sqrt(gamma_l))
# a_sum_th = torch.tensor(np.sqrt(np.sqrt(gamma_l) / alpha_sum), dtype=torch.float32)
# print("a_sum:", a_sum_th)

# print("a_learned[:5]:", a_final.detach().numpy()[:5])
# print("a_sum_th[:5]: ", a_sum_th.numpy()[:5])
# print("=?", torch.allclose(a_final, a_sum_th, atol=0.1))

# plt.figure(figsize=(8, 4))
# plt.plot(l_np, a_final.detach().numpy(), 'b-o', label='Learned a')
# plt.plot(l_np, a_sum_th.numpy(), 'r--', label='Sum-MSE optimal (theoretical)')
# plt.xlabel("Subcarrier index")
# plt.ylabel("Weight value")
# plt.title("Learned weights vs theoretical sum-MSE weights")
# plt.legend()
# plt.grid(True)
# plt.show()


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

# M = 16
# os = 2
# h = 0.1
# A = 1
# sigma_awgn2 = 5e-3
# sigma_phase2 = 5e-4
# L = 63
# N = 2*(L+1)*os
# l_np = np.arange(1, L+1)
# fs = 1e9
# linewidth = sigma_phase2 * fs / (2 * np.pi)
# OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))
# n_runs = 1000
# n_iterations = 1000
# print('linewidth[Hz]:', linewidth)
# print('OSNR[dB]:', OSNR_dB)

# alphabet_np = get_alphabet("QAM", M)
# alphabet = torch.tensor(alphabet_np, dtype=torch.complex64)

# vect = torch.zeros(N)
# vect[1:L+1] = 1
# indices = torch.arange(N-1, N-L-1, -1)
# vect[indices] = -1

# gamma_l = sigma_phase2 / (np.sin(np.pi*l_np/N)**2) + sigma_awgn2/(A**2)

# alpha_sum = (2/N) * np.sum(np.sqrt(gamma_l))
# a_sum_th = torch.tensor(np.sqrt(np.sqrt(gamma_l) / alpha_sum), dtype=torch.float32)

# learning_rates = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 1e-3]
# results = {}

# for lr in learning_rates:
#     print(f"\nRunning lr={lr}")

#     a = nn.Parameter(torch.ones(L))
#     optimizer = torch.optim.Adam([a], lr=lr)
#     loss_total = []

#     laser_pn = LaserPhaseNoise(linewidth=linewidth, fs=fs, seed=42)
#     awgn = AWGNChannel(OSNR_dB=OSNR_dB, os=os, seed=42)

#     for i in range(n_iterations):
#         optimizer.zero_grad()
#         laser_pn.reset_seed()
#         awgn.reset_seed()

#         a_norm = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

#         transmitter = Sequential([
#             SymbolGenerator(M),
#             Recorder(name='data_tx'),
#             SymbolMapper(alphabet),
#             Serial2Parallel(L),
#             Recorder(name='tx_symbols'),
#             Weight(a=a_norm),
#             CarrierAllocator(vect, hermitian_sym=True),
#             IFFTProcessor(),
#             Parallel2Serial(),
#             PhaseModulator(h=h),
#         ])

#         receiver = Sequential([
#             PhaseDemodulator(h=h, unwrap=True),
#             Serial2Parallel(N),
#             FFTProcessor(),
#             CarrierExtractor(vect),
#             Weight(a=1/a_norm),
#             Recorder(name='rx_symbols'),
#             Parallel2Serial(),
#             Recorder(name='constellation'),
#             SymbolDemapper(alphabet),
#             Recorder(name='data_rx'),
#         ])

#         channel = Sequential([laser_pn, awgn])
#         chain = Sequential([transmitter, channel, receiver])
#         chain(L * n_runs)

#         S_tx = transmitter['tx_symbols'].get_data()
#         S_rx = receiver['rx_symbols'].get_data()
#         loss = torch.mean(torch.abs(S_rx - S_tx)**2)

#         loss.backward()
#         optimizer.step()
#         loss_total.append(loss.item())

#         if i % 200 == 0:
#             print(f"  Iter {i}, Loss: {loss.item():.6f}")

#     with torch.no_grad():
#         a_final = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

#     results[lr] = {
#         'loss': loss_total,
#         'a_final': a_final.detach().numpy()
#     }

# # grafic convergenta pentru toate lr-urile
# plt.figure(figsize=(10, 5))
# for lr in learning_rates:
#     plt.plot(results[lr]['loss'], label=f'lr={lr}')
# plt.xlabel("Iteration")
# plt.ylabel("Loss (MSE)")
# plt.title("Convergence curve — comparison of learning rates")
# plt.legend()
# plt.grid(True)

# # grafic ponderi invatate pentru fiecare lr
# fig, axes = plt.subplots(2, 3, figsize=(15, 8))
# axes = axes.flatten()

# for idx, lr in enumerate(learning_rates):
#     axes[idx].plot(l_np, results[lr]['a_final'], 'b-o', markersize=3, label=f'Learned a (lr={lr})')
#     axes[idx].plot(l_np, a_sum_th.numpy(), 'r--', label='Sum-MSE optimal (theoretical)')
#     axes[idx].set_xlabel("Subcarrier index")
#     axes[idx].set_ylabel("Weight value")
#     axes[idx].set_title(f"lr = {lr}")
#     axes[idx].legend(fontsize=8)
#     axes[idx].grid(True)

# plt.suptitle("Learned weights vs theoretical sum-MSE weights — different learning rates")
# plt.tight_layout()
# plt.show()





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

# M = 16
# os = 2
# h = 0.1
# A = 1
# sigma_awgn2 = 5e-3
# sigma_phase2 = 5e-4
# L = 63
# N = 2*(L+1)*os
# l_np = np.arange(1, L+1)
# fs = 1e9
# linewidth = sigma_phase2 * fs / (2 * np.pi)
# OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))
# n_runs = 1000

# T = 0.01

# alphabet_np = get_alphabet("QAM", M)
# alphabet = torch.tensor(alphabet_np, dtype=torch.complex64)

# vect = torch.zeros(N)
# vect[1:L+1] = 1
# indices = torch.arange(N-1, N-L-1, -1)
# vect[indices] = -1

# gamma_l = sigma_phase2 / (np.sin(np.pi*l_np/N)**2) + sigma_awgn2/(A**2)

# def soft_classification_loss(S_rx, S_tx, alphabet, T=T):
#     S_rx_flat    = S_rx.reshape(-1)
#     S_tx_flat    = S_tx.reshape(-1)

#     S_rx_col     = S_rx_flat.unsqueeze(1)
#     alphabet_row = alphabet.unsqueeze(0)

#     dists_sq     = torch.abs(S_rx_col - alphabet_row) ** 2
#     log_probs    = -dists_sq / T
#     log_probs    = log_probs - log_probs.logsumexp(dim=1, keepdim=True)
#     probs        = log_probs.exp()
#     true_dists   = torch.abs(S_tx_flat.unsqueeze(1) - alphabet_row) ** 2
#     true_indices = true_dists.argmin(dim=1)
#     p_correct    = probs[torch.arange(len(S_tx_flat)), true_indices]
#     return 1.0 - p_correct.mean()

# # 1) a
# a = nn.Parameter(torch.rand(L))
# print("a shape:", a.shape)
# print("a requires_grad:", a.requires_grad)

# # canal cu seed fix
# laser_pn = LaserPhaseNoise(linewidth=linewidth, fs=fs, seed=42)
# awgn = AWGNChannel(OSNR_dB=OSNR_dB, os=os, seed=42)

# optimizer = torch.optim.Adam([a], lr=1e-2)
# n_iterations = 1000
# loss_total = []

# for i in range(n_iterations):
#     optimizer.zero_grad()

#     laser_pn.reset_seed()
#     awgn.reset_seed()

#     # normalizam a — diferentiabil, fara no_grad
#     a_norm = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

#     transmitter = Sequential([
#         SymbolGenerator(M),
#         Recorder(name='data_tx'),
#         SymbolMapper(alphabet),
#         Serial2Parallel(L),
#         Recorder(name='tx_symbols'),
#         Weight(a=a_norm),
#         CarrierAllocator(vect, hermitian_sym=True),
#         IFFTProcessor(),
#         Parallel2Serial(),
#         PhaseModulator(h=h),
#     ])

#     receiver = Sequential([
#         PhaseDemodulator(h=h, unwrap=True),
#         Serial2Parallel(N),
#         FFTProcessor(),
#         CarrierExtractor(vect),
#         Weight(a=1/a_norm),
#         Recorder(name='rx_symbols'),
#         Parallel2Serial(),
#         Recorder(name='constellation'),
#         SymbolDemapper(alphabet),
#         Recorder(name='data_rx'),
#     ])

#     channel = Sequential([laser_pn, awgn])
#     chain = Sequential([transmitter, channel, receiver])
#     chain(L * n_runs)

#     # 2) loss
#     S_tx = transmitter['tx_symbols'].get_data()
#     S_rx = receiver['rx_symbols'].get_data()
#     loss = soft_classification_loss(S_rx, S_tx, alphabet, T)

#     # 3) backpropagation + update a
#     loss.backward()
#     optimizer.step()

#     loss_total.append(loss.item())

#     if i % 100 == 0:
#         print(f"Iter {i}, Loss: {loss.item():.6f}")

# plt.figure(figsize=(8, 4))
# plt.plot(loss_total)
# plt.xlabel("Iteration")
# plt.ylabel("Loss (Soft-SER)")
# plt.title("Convergence curve — Classification loss")
# plt.grid(True)

# # 4) a_final
# with torch.no_grad():
#     a_final = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))
# print("a learned:", a_final)

# # a_sum th
# alpha_sum = (2/N) * np.sum(np.sqrt(gamma_l))
# a_sum_th = torch.tensor(np.sqrt(np.sqrt(gamma_l) / alpha_sum), dtype=torch.float32)
# print("a_sum:", a_sum_th)

# print("a_learned[:5]:", a_final.detach().numpy()[:5])
# print("a_sum_th[:5]: ", a_sum_th.numpy()[:5])
# print("=?", torch.allclose(a_final, a_sum_th, atol=0.1))

# plt.figure(figsize=(8, 4))
# plt.plot(l_np, a_final.detach().numpy(), 'b-o', label='Learned a (classification loss)')
# plt.plot(l_np, a_sum_th.numpy(), 'r--', label='Sum-MSE optimal (theoretical)')
# plt.xlabel("Subcarrier index")
# plt.ylabel("Weight value")
# plt.title("Learned weights (classification loss) vs theoretical sum-MSE weights")
# plt.legend()
# plt.grid(True)
# plt.show()






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
# from modules.mappers import SymbolMapper
# from modules.processors import Serial2Parallel, Parallel2Serial, CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
# from modules.channels import AWGNChannel, LaserPhaseNoise
# from modules.compensators import PhaseModulator, PhaseDemodulator, Weight

# M = 16
# os = 2
# h = 0.1
# A = 1
# sigma_awgn2 = 5e-3
# sigma_phase2 = 5e-4
# L = 63
# N = 2*(L+1)*os
# l_np = np.arange(1, L+1)
# fs = 1e9
# linewidth = sigma_phase2 * fs / (2 * np.pi)
# OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))
# n_runs = 1000

# alphabet_np = get_alphabet("QAM", M)
# alphabet = torch.tensor(alphabet_np, dtype=torch.complex64)

# vect = torch.zeros(N)
# vect[1:L+1] = 1
# indices = torch.arange(N-1, N-L-1, -1)
# vect[indices] = -1

# gamma_l = sigma_phase2 / (np.sin(np.pi*l_np/N)**2) + sigma_awgn2/(A**2)

# # a_var teoretic
# alpha_var = (2/N) * np.sum(gamma_l)
# a_var_th = np.sqrt(gamma_l / alpha_var)

# laser_pn = LaserPhaseNoise(linewidth=linewidth, fs=fs, seed=42)
# awgn = AWGNChannel(OSNR_dB=OSNR_dB, os=os, seed=42)

# # a trainable
# a = nn.Parameter(torch.rand(L))
# print("a shape:", a.shape)
# print("a requires_grad:", a.requires_grad)

# optimizer = torch.optim.Adam([a], lr=1e-2)
# n_iterations = 1000
# loss_total = []

# for i in range(n_iterations):
#     optimizer.zero_grad()
#     laser_pn.reset_seed()
#     awgn.reset_seed()

#     a_norm = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

#     transmitter = Sequential([
#         SymbolGenerator(M),
#         SymbolMapper(alphabet),
#         Serial2Parallel(L),
#         Recorder(name='tx_symbols'),
#         Weight(a=a_norm),
#         CarrierAllocator(vect, hermitian_sym=True),
#         IFFTProcessor(),
#         Parallel2Serial(),
#         PhaseModulator(h=h),
#     ])

#     receiver = Sequential([
#         PhaseDemodulator(h=h, unwrap=True),
#         Serial2Parallel(N),
#         FFTProcessor(),
#         CarrierExtractor(vect),
#         Weight(a=1/a_norm),
#         Recorder(name='rx_symbols'),
#         Parallel2Serial(),
#     ])

#     channel = Sequential([laser_pn, awgn])
#     chain = Sequential([transmitter, channel, receiver])
#     chain(L * n_runs)

#     S_tx = transmitter['tx_symbols'].get_data()
#     S_rx = receiver['rx_symbols'].get_data()
    
#     # loss = Var[MSE_l]
#     mse_per_subcarrier = torch.mean(torch.abs(S_rx - S_tx)**2, dim=1)
#     loss = torch.var(mse_per_subcarrier)

#     loss.backward()
#     optimizer.step()
#     loss_total.append(loss.item())

#     if i % 100 == 0:
#         print(f"Iter {i}, Loss: {loss.item():.6f}")

# # a_final
# with torch.no_grad():
#     a_final = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

# print("\na_learned[:5]:", a_final.detach().numpy()[:5])
# print("a_var_th[:5]: ", a_var_th[:5])
# print("=?", np.allclose(a_final.detach().numpy(), a_var_th, atol=0.1))

# plt.figure(figsize=(8, 4))
# plt.plot(loss_total)
# plt.xlabel("Iteration")
# plt.ylabel("Loss (Var MSE)")
# plt.title("Convergence curve - Var-MSE")
# plt.grid(True)

# plt.figure(figsize=(8, 4))
# plt.plot(l_np, a_final.detach().numpy(), 'b-o', label='Learned a')
# plt.plot(l_np, a_var_th, 'r--', label='Var-MSE optimal (theoretical)')
# plt.xlabel("Subcarrier index")
# plt.ylabel("Weight value")
# plt.title("Learned weights vs theoretical var-MSE weights")
# plt.legend()
# plt.grid(True)
# plt.show()




import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src\comnumpy\ofdm_pytorch")
from comnumpy.core.utils import get_alphabet
from modules.generics import Sequential, Recorder
from modules.generators import SymbolGenerator
from modules.mappers import SymbolMapper
from modules.processors import Serial2Parallel, Parallel2Serial, CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
from modules.channels import AWGNChannel, LaserPhaseNoise
from modules.compensators import PhaseModulator, PhaseDemodulator, Weight

M = 16
os = 2
h = 0.1
A = 1
sigma_awgn2 = 5e-3
sigma_phase2 = 5e-4
L = 63
N = 2*(L+1)*os
l_np = np.arange(1, L+1)
fs = 1e9
linewidth = sigma_phase2 * fs / (2 * np.pi)
OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))
n_runs = 1000
n_iterations = 10000

alphabet_np = get_alphabet("QAM", M)
alphabet = torch.tensor(alphabet_np, dtype=torch.complex64)

vect = torch.zeros(N)
vect[1:L+1] = 1
indices = torch.arange(N-1, N-L-1, -1)
vect[indices] = -1

gamma_l = sigma_phase2 / (np.sin(np.pi*l_np/N)**2) + sigma_awgn2/(A**2)

# a_var teoretic
alpha_var = (2/N) * np.sum(gamma_l)
a_var_th = np.sqrt(gamma_l / alpha_var)

# learning_rates = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 1e-3]
learning_rates = [1e-3]
results = {}

for lr in learning_rates:
    print(f"\nRunning lr={lr}")

    # a = nn.Parameter(torch.rand(L))
    a = nn.Parameter(torch.Tensor(a_var_th))
    # a = nn.Parameter(torch.ones(L))

    optimizer = torch.optim.Adam([a], lr=lr)
    loss_total = []

    laser_pn = LaserPhaseNoise(linewidth=linewidth, fs=fs, seed=42)
    awgn = AWGNChannel(OSNR_dB=OSNR_dB, os=os, seed=42)

    for i in range(n_iterations):
        optimizer.zero_grad()
        laser_pn.reset_seed()
        awgn.reset_seed()

        a_norm = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))
        
        transmitter = Sequential([
            SymbolGenerator(M),
            SymbolMapper(alphabet),
            Serial2Parallel(L),
            Recorder(name='tx_symbols'),
            Weight(a=a_norm),
            CarrierAllocator(vect, hermitian_sym=True),
            IFFTProcessor(),
            Parallel2Serial(),
            PhaseModulator(h=h),
            LaserPhaseNoise(linewidth=linewidth, fs=fs, mode = 'transmitter'),
        ])

        receiver = Sequential([
            LaserPhaseNoise(linewidth=linewidth, fs=fs, mode = 'receiver'),
            PhaseDemodulator(h=h, unwrap=True),
            Serial2Parallel(N),
            FFTProcessor(),
            CarrierExtractor(vect),
            Weight(a=1/a_norm),
            Recorder(name='rx_symbols'),
            Parallel2Serial(),
        ])

        channel = Sequential([awgn])
        chain = Sequential([transmitter, channel, receiver])
        chain(L * n_runs)

        S_tx = transmitter['tx_symbols'].get_data()
        S_rx = receiver['rx_symbols'].get_data()
        
        # loss = Var[MSE_l]
        mse_per_subcarrier = torch.mean(torch.abs(S_rx - S_tx)**2, dim=1)
        # print(mse_per_subcarrier.shape)
        loss = torch.var(mse_per_subcarrier)

        loss.backward()
        optimizer.step()
        loss_total.append(loss.item())

        if i % 50 == 0:
            print(f"  Iter {i}, Loss: {loss.item():.6f}")

    with torch.no_grad():
        a_final = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

    results[lr] = {
        'loss': loss_total,
        'a_final': a_final.detach().numpy()
    }

    print(f"  a_learned[:5]: {a_final.detach().numpy()[:5]}")
    print(f"  a_var_th[:5]:  {a_var_th[:5]}")
    print(f"  =? {np.allclose(a_final.detach().numpy(), a_var_th, atol=0.1)}")

# grafic convergenta pentru toate lr-urile
plt.figure(figsize=(10, 5))
for lr in learning_rates:
    plt.plot(results[lr]['loss'], label=f'lr={lr}')
plt.xlabel("Iteration")
plt.ylabel("Loss (Var-MSE)")
plt.title("Convergence curve — Var-MSE — comparison of learning rates")
plt.legend()
plt.grid(True)

# ─grafic ponderi invatate pentru fiecare lr
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, lr in enumerate(learning_rates):
    axes[idx].plot(l_np, results[lr]['a_final'], 'b-o', markersize=3, label=f'Learned a (lr={lr})')
    axes[idx].plot(l_np, a_var_th, 'r--', label='Var-MSE optimal (theoretical)')
    axes[idx].set_xlabel("Subcarrier index")
    axes[idx].set_ylabel("Weight value")
    axes[idx].set_title(f"lr = {lr}")
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True)

plt.suptitle("Learned weights vs theoretical var-MSE weights — different learning rates")
plt.tight_layout()
plt.show()




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
# from modules.mappers import SymbolMapper
# from modules.processors import Serial2Parallel, Parallel2Serial, CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
# from modules.channels import AWGNChannel, LaserPhaseNoise
# from modules.compensators import PhaseModulator, PhaseDemodulator, Weight

# LOSS_MODE = 'classification'
# T = 0.01   

# M = 16
# os = 2
# h = 0.1
# A = 1
# sigma_awgn2 = 5e-3
# sigma_phase2 = 5e-4
# L = 63
# N = 2*(L+1)*os
# l_np = np.arange(1, L+1)
# fs = 1e9
# linewidth = sigma_phase2 * fs / (2 * np.pi)
# OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))
# n_runs = 2000

# alphabet_np = get_alphabet("QAM", M)
# alphabet = torch.tensor(alphabet_np, dtype=torch.complex64)

# vect = torch.zeros(N)
# vect[1:L+1] = 1
# indices = torch.arange(N-1, N-L-1, -1)
# vect[indices] = -1

# gamma_l = sigma_phase2 / (np.sin(np.pi*l_np/N)**2) + sigma_awgn2/(A**2)

# # a_var teoretic
# alpha_var = (2/N) * np.sum(gamma_l)
# a_var_th = np.sqrt(gamma_l / alpha_var)

# def soft_classification_loss(S_rx, S_tx, alphabet, T):
#     S_rx_flat    = S_rx.reshape(-1)
#     S_tx_flat    = S_tx.reshape(-1)
#     S_rx_col     = S_rx_flat.unsqueeze(1)
#     alphabet_row = alphabet.unsqueeze(0)
#     dists_sq     = torch.abs(S_rx_col - alphabet_row) ** 2
#     log_probs    = -dists_sq / T
#     log_probs    = log_probs - log_probs.logsumexp(dim=1, keepdim=True)
#     probs        = log_probs.exp()
#     true_dists   = torch.abs(S_tx_flat.unsqueeze(1) - alphabet_row) ** 2
#     true_indices = true_dists.argmin(dim=1)
#     p_correct    = probs[torch.arange(len(S_tx_flat)), true_indices]
#     return 1.0 - p_correct.mean()

# # a trainable
# a = nn.Parameter(torch.ones(L))
# print("a shape:", a.shape)
# print("a requires_grad:", a.requires_grad)

# laser_pn = LaserPhaseNoise(linewidth=linewidth, fs=fs, seed=42)
# awgn = AWGNChannel(OSNR_dB=OSNR_dB, os=os, seed=42)

# optimizer = torch.optim.Adam([a], lr=1e-2)
# n_iterations = 1000
# loss_total = []

# for i in range(n_iterations):
#     optimizer.zero_grad()
#     laser_pn.reset_seed()
#     awgn.reset_seed()

#     a_norm = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

#     transmitter = Sequential([
#         SymbolGenerator(M),
#         SymbolMapper(alphabet),
#         Serial2Parallel(L),
#         Recorder(name='tx_symbols'),
#         Weight(a=a_norm),
#         CarrierAllocator(vect, hermitian_sym=True),
#         IFFTProcessor(),
#         Parallel2Serial(),
#         PhaseModulator(h=h),
#     ])

#     receiver = Sequential([
#         PhaseDemodulator(h=h, unwrap=True),
#         Serial2Parallel(N),
#         FFTProcessor(),
#         CarrierExtractor(vect),
#         Weight(a=1/a_norm),
#         Recorder(name='rx_symbols'),
#         Parallel2Serial(),
#     ])

#     channel = Sequential([laser_pn, awgn])
#     chain = Sequential([transmitter, channel, receiver])
#     chain(L * n_runs)

#     S_tx = transmitter['tx_symbols'].get_data()
#     S_rx = receiver['rx_symbols'].get_data()
    
#     if LOSS_MODE == 'var_mse':
#         mse_per_subcarrier = torch.mean(torch.abs(S_rx - S_tx)**2, dim=1)
#         loss = torch.var(mse_per_subcarrier)

#     elif LOSS_MODE == 'classification':
#         loss = soft_classification_loss(S_rx, S_tx, alphabet, T)

#     else:
#         raise ValueError(f"LOSS_MODE necunoscut: {LOSS_MODE!r}")

#     loss.backward()
#     optimizer.step()
#     loss_total.append(loss.item())

#     if i % 200 == 0:
#         print(f"Iter {i}, Loss: {loss.item():.6f}")

# # a_final
# with torch.no_grad():
#     a_final = a * torch.sqrt(torch.tensor(N/2) / torch.sum(a**2))

# print("\na_learned[:5]:", a_final.detach().numpy()[:5])
# print("a_var_th[:5]: ", a_var_th[:5])
# print("=?", np.allclose(a_final.detach().numpy(), a_var_th, atol=0.1))

# plt.figure(figsize=(8, 4))
# plt.plot(loss_total)
# plt.xlabel("Iteration")
# plt.ylabel(f"Loss ({LOSS_MODE})")
# plt.title(f"Convergence curve — {LOSS_MODE}")
# plt.grid(True)

# plt.figure(figsize=(8, 4))
# plt.plot(l_np, a_final.detach().numpy(), 'b-o', label=f'Learned a ({LOSS_MODE})')
# plt.plot(l_np, a_var_th, 'r--', label='Var-MSE optimal (theoretical)')
# plt.xlabel("Subcarrier index")
# plt.ylabel("Weight value")
# plt.title(f"Learned weights ({LOSS_MODE}) vs theoretical var-MSE weights")
# plt.legend()
# plt.grid(True)
# plt.show()