import numpy as np
import sys
import matplotlib.pyplot as plt

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
L = 63
os = 2
N = 2*(L+1)*os

sigma2_awgn = 5e-3
sigma2_noise = 5e-4
n_runs = 1000

alphabet = get_alphabet("QAM", M)

vect = np.zeros(N)
vect[1:L+1] = 1
vect[N:N-L-1:-1] = -1

l = np.arange(1, L+1)

def run_simulation(h):

    gamma_l = sigma2_noise / (np.sin(np.pi*l/N)**2) + sigma2_awgn

    a_uniform = np.ones(L)
    a_uniform *= np.sqrt((N/2) / np.sum(a_uniform**2))

    mse_th = gamma_l / (8*(np.pi*h*a_uniform)**2)

    transmitter = Sequential([
        SymbolGenerator(M),
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
        PhaseDemodulator(h=h, unwrap=False),
        Serial2Parallel(N),
        FFTProcessor(),
        CarrierExtractor(vect),
        Weight(a=1/a_uniform),
        Recorder(name='rx_symbols'),
        Parallel2Serial(),
    ])

    channel = Sequential([
        PhaseNoise(sigma2=sigma2_noise),
        AWGN(value=sigma2_awgn, unit='sigma2')
    ])

    chain = Sequential([transmitter, channel, receiver])

    mse_acc = np.zeros(L)

    for _ in range(n_runs):
        chain(L)

        S_tx = transmitter['tx_symbols'].get_data()
        S_rx = receiver['rx_symbols'].get_data()

        mse_acc += np.mean(np.abs(S_rx - S_tx)**2, axis=1)

    mse_sim = mse_acc / n_runs

    return mse_th, mse_sim


mse_th_01, mse_sim_01 = run_simulation(0.1)
mse_th_02, mse_sim_02 = run_simulation(0.2)

plt.figure(figsize=(8,5))

plt.plot(l, mse_sim_01, 'bo-', label='Simulation (h=0.1)')
plt.plot(l, mse_th_01, 'r--', label='Theoretical (h=0.1)')

plt.plot(l, mse_sim_02, 'ro-', label='Simulation (h=0.2)')
plt.plot(l, mse_th_02, 'r--', label='Theoretical (h=0.2)')

plt.xlabel("Index")
plt.xlim([0, 20])
plt.ylabel("Mean Square Error (MSE)")
plt.title("MSE vs subcarrier index")
plt.grid(True)
plt.legend()

plt.show()