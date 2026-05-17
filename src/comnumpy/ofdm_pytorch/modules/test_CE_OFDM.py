import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src\comnumpy\ofdm_pytorch")
from comnumpy.core.utils import get_alphabet
from modules.generics import Sequential, Recorder
from modules.generators import SymbolGenerator
from modules.mappers import SymbolMapper, SymbolDemapper
from modules.processors import Serial2Parallel, Parallel2Serial, CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
from modules.channels import AWGNChannel, LaserPhaseNoise
from modules.compensators import PhaseModulator, PhaseDemodulator, Weight
from modules.metrics import compute_ber

M = 16
os = 2
h = 0.1
A = 1
sigma_awgn2 = 2e-2
sigma_phase2 = 6.89e-4

L = 63
N = 2*(L+1)*os
l = np.arange(1, L+1)

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


gamma_l = sigma_phase2 / (torch.sin(torch.pi * torch.arange(1, L+1) / N)**2) + sigma_awgn2/(A**2)
a_uniform = torch.ones(L)
a_uniform *= torch.sqrt(torch.tensor((N/2)/torch.sum(a_uniform**2).item()))

laser_pn = LaserPhaseNoise(linewidth=linewidth, fs=fs)

transmitter = Sequential([
    SymbolGenerator(M),
    Recorder(name='data_tx'),
    SymbolMapper(alphabet),
    Serial2Parallel(L), 
    Recorder(name='tx_symbols'),
    Weight(a=a_uniform),
    CarrierAllocator(vect, hermitian_sym=True),
    IFFTProcessor(),
    Parallel2Serial(),
    PhaseModulator(h=h),
])

receiver = Sequential([
    PhaseDemodulator(h=h, unwrap=True),
    Serial2Parallel(N),
    FFTProcessor(),
    CarrierExtractor(vect),
    Weight(a=1/a_uniform),
    Parallel2Serial(),
    Recorder(name='constellation'),
    SymbolDemapper(alphabet),
    Recorder(name='data_rx'),
])

channel = Sequential([
    laser_pn,
    AWGNChannel(OSNR_dB=OSNR_dB, os=os),
])

chain = Sequential([transmitter, channel, receiver])

with torch.no_grad():
    chain(L * n_runs)


data_tx = transmitter['data_tx'].get_data()
data_rx = receiver['data_rx'].get_data()
constellation = receiver['constellation'].get_data()

#BER
ber = compute_ber(data_tx, data_rx, width=int(np.log2(M)))
print(f"BER = {ber:.3e}")


plt.figure()
plt.plot(constellation.real.numpy(), constellation.imag.numpy(), ".")
plt.xlabel("Re")
plt.ylabel("Im")
plt.title(f"Constellation (h={h}, Uniform weights)")
plt.axis('equal')
plt.grid()
plt.show()