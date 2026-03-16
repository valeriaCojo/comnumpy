import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src")
from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.channels import AWGN, FIRChannel
from comnumpy.core.compensators import LinearEqualizer
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ser
from comnumpy.ofdm.chains import OFDMTransmitter, OFDMReceiver

M = 16
N_h = 5
N = 1280
sigma2 = 0.01 # or sigma2 = 0.01
alphabet = get_alphabet("QAM", M)

#generate a random selective channel
h = 0.1*(np.random.randn(N_h) + 1j*np.random.randn(N_h))
h[0] = 1

# create a simple single carrier chain and simulate
simple_chain = Sequential([
        SymbolGenerator(M),
        Recorder(name="data_tx"),
        SymbolMapper(alphabet),
        FIRChannel(h),
        AWGN(value=sigma2),
        Recorder(name="data_rx"),
        LinearEqualizer(h, method="mmse"),
        Recorder(name="data_rx_eq"),
        SymbolDemapper(alphabet)
    ])

start_time = time.time()
s_rx = simple_chain(N)
stop_time = time.time()

# extract signals, compute ser and elapsed time
s_tx = simple_chain["data_tx"].get_data()
ser = compute_ser(s_tx, s_rx)
elapsed_time = stop_time - start_time
print(f"SER: {ser}")
print(f"elapsed time: {elapsed_time} s")

# plot signal and save
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
for indice, processor_name in enumerate(["data_rx", "data_rx_eq"]):
    data_rx = simple_chain[processor_name].get_data()
    axes[indice].plot(np.real(data_rx), np.imag(data_rx), ".")
    axes[indice].set_title(f"Received signal ({processor_name})")
    axes[indice].set_aspect("equal", adjustable="box")
    axes[indice].set_xlim([-2, 2])
    axes[indice].set_ylim([-2, 2])

plt.show()