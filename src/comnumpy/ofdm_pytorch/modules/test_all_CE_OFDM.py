import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src\comnumpy\ofdm_pytorch")
# sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src")
from comnumpy.core.utils import get_alphabet
from modules.generics import Sequential, Recorder
from modules.generators import SymbolGenerator
from modules.mappers import SymbolMapper, SymbolDemapper
from modules.processors import Serial2Parallel, Parallel2Serial, CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
from modules.channels import AWGNChannel, LaserPhaseNoise
from modules.compensators import PhaseModulator, PhaseDemodulator, Weight

M = 16
os = 2
h = 0.1
A = 1
# sigma_awgn2 = 5e-3
# sigma_phase2 = 5e-4
sigma_awgn2 = 5e-3
sigma_phase2 = 5e-4
L = 63
N = 2*(L+1)*os
l_np = np.arange(1, L+1)
fs = 1e9
linewidth = sigma_phase2 * fs / (2 * np.pi)
OSNR_dB = -10 * np.log10(sigma_awgn2 / (1 * os))
print(f"linewidth: {linewidth:.2f} Hz")
print(f"OSNR_dB: {OSNR_dB:.2f} dB")
n_runs = 1000

alphabet_np = get_alphabet("QAM", M)
alphabet = torch.tensor(alphabet_np, dtype=torch.complex64)

vect = torch.zeros(N)
vect[1:L+1] = 1
indices = torch.arange(N-1, N-L-1, -1)
vect[indices] = -1

gamma_l = sigma_phase2 / (np.sin(np.pi*l_np/N)**2) + sigma_awgn2/(A**2)

# Uniform
a_uniform = torch.ones(L)
a_uniform *= torch.sqrt(torch.tensor((N/2)/torch.sum(a_uniform**2).item()))

# Sum-MSE
alpha_sum = (2/N)*np.sum(np.sqrt(gamma_l))
a_sum = torch.tensor(np.sqrt(np.sqrt(gamma_l)/alpha_sum), dtype=torch.float32)

# Var-MSE
alpha_var = (2/N)*np.sum(gamma_l)
a_var = torch.tensor(np.sqrt(gamma_l/alpha_var), dtype=torch.float32)


def run_chain(a):

    transmitter = Sequential([
        SymbolGenerator(M),
        SymbolMapper(alphabet),
        Serial2Parallel(L),
        Recorder(name='tx_symbols'),
        Weight(a=a),
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
        Weight(a=1/a),
        Recorder(name='rx_symbols'),
        Parallel2Serial(),
        Recorder(name = 'constellation'),
        SymbolDemapper(alphabet),
        Recorder(name = 'data_rx'),
    ])

    channel = Sequential([
        # laser_pn,
        AWGNChannel(OSNR_dB=OSNR_dB, os=os),
    ])

    chain = Sequential([transmitter, channel, receiver])

    with torch.no_grad():
        chain(L * n_runs)

    S_tx = transmitter['tx_symbols'].get_data()
    S_rx = receiver['rx_symbols'].get_data()

    mse = torch.mean(torch.abs(S_rx - S_tx)**2, dim=1)
    mse_tot = torch.sum(mse).item()
    return mse.numpy(), mse_tot

mse_u, mse_u_tot = run_chain(a_uniform)
mse_s, mse_s_tot = run_chain(a_sum)
mse_v, mse_v_tot = run_chain(a_var)

print(f"Total MSE - Uniform: {mse_u_tot:.4f}")
print(f"Total MSE - Sum-MSE: {mse_s_tot:.4f}")
print(f"Total MSE - Var-MSE: {mse_v_tot:.4f}")
print("Order (should be sum <= var ? uniform):")
print(mse_s_tot,mse_v_tot,mse_u_tot)
# Theoretical
mse_u_th = gamma_l / (8*(np.pi*h*a_uniform.numpy())**2)
mse_s_th = gamma_l / (8*(np.pi*h*a_sum.numpy())**2)
mse_v_th = gamma_l / (8*(np.pi*h*a_var.numpy())**2)

plt.figure(figsize=(8, 5))
plt.plot(l_np, mse_u_th, 'r-', label='Uniform TH')
plt.plot(l_np, mse_u, 'r:o', label='Uniform SIM')
plt.plot(l_np, mse_s_th, 'g-', label='Sum TH')
plt.plot(l_np, mse_s, 'g:o', label='Sum SIM')
plt.plot(l_np, mse_v_th, 'b-', label='Var TH')
plt.plot(l_np, mse_v, 'b:o', label='Var SIM')
plt.xlabel("Subcarrier index")
plt.ylabel("MSE")
plt.title("Theory vs Simulation (PyTorch)")
plt.grid(True)
plt.legend()
plt.show()