import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src")
from comnumpy.core import Sequential, Recorder
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
from comnumpy.core.channels import AWGN, FIRChannel
from comnumpy.core.compensators import LinearEqualizer
from comnumpy.core.utils import get_alphabet
from comnumpy.core.metrics import compute_ser
from comnumpy.ofdm.chains import PhaseNoise, OFDMTransmitter, OFDMReceiver
from comnumpy.ofdm.compensators import PhaseModulator, PhaseDemodulator, SubcarrierWeight, SubcarrierUnweight
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor, CyclicPrefixer, CyclicPrefixRemover
from comnumpy.ofdm.utils import get_standard_carrier_allocation
from comnumpy.core.visualizers import TimeScope, SpectrumScope
from comnumpy.core.metrics import compute_evm, compute_ser
M = 4
os=4
sigma_awgn2 = 0.01# or sigma2 = 0.01
sigma_phase2 = 1e-6
alphabet = get_alphabet("QAM", M)
type_of_carrier= get_standard_carrier_allocation("NoPilot_128", os=os, shift=True, hermitian_sym=True)
# print("data carriers:", np.sum(type_of_carrier == 1))
# print("conjugate carriers:", np.sum(type_of_carrier == -1))
# print("null carriers:", np.sum(type_of_carrier == 0))
carrier = len(type_of_carrier)
carrier_data = np.sum(type_of_carrier==1)
N_cp = carrier//8
n_samples = N_cp+carrier
N_OFDM_symbols = 1000
N = int(carrier_data*N_OFDM_symbols)
weight_block=SubcarrierWeight(sigma2=sigma_awgn2, Nsc=carrier_data, os=os, h=0.1, OSNR=20)
transmitter = Sequential([
        SymbolGenerator(M),
        Recorder(name='data_ser'),
        SymbolMapper(alphabet),
        Recorder(name='data_evm'),
        Serial2Parallel(carrier_data),
        Recorder(name='tx_symbols'),
        # weight_block,
        CarrierAllocator(type_of_carrier, hermitian_sym=True),
        Recorder(name='carrier_allocator'),
        IFFTProcessor(),
        Recorder(name='signal_after_IFFT'),
        # CyclicPrefixer(N_cp=N_cp),
        Parallel2Serial(),
        Recorder(name='h_signal'),
        PhaseModulator(h=0.1),
        Recorder(name='tx_signal'),
        # TimeScope(num=1, title='real', name='time domain'),
])

channel = Sequential([
        PhaseNoise(sigma2=sigma_phase2),
        AWGN(value=sigma_awgn2, unit='sigma2'),
        # PhaseNoise(sigma2=1e-3)
])

receiver = Sequential([
        PhaseDemodulator(h=0.1),
        Serial2Parallel(carrier, method='zero-padding'),
        # CyclicPrefixRemover(N_cp),
        FFTProcessor(),
        # SubcarrierUnweight(weight_block),
        CarrierExtractor(type_of_carrier),
        # SubcarrierUnweight(weight_block),
        Recorder(name='rx_symbols'),
        Parallel2Serial(),
        Recorder(name='data_evm'),
        SymbolDemapper(alphabet),
        Recorder(name='data_ser')
])

chain = Sequential([transmitter, channel, receiver])

y = chain(N)

tx = transmitter["signal_after_IFFT"].get_data()
print(np.max(np.abs(tx)))

tx = transmitter["tx_signal"].get_data()
print(f"Max imaginary amplitude: {np.max(np.abs(np.imag(tx))):.1e}")
phase_signal_after_modulator = transmitter['tx_signal'].get_data()
print("min amplitude:", np.min(np.abs(phase_signal_after_modulator)))
print("max amplitude:", np.max(np.abs(phase_signal_after_modulator)))
#grafic partea reala si imaginara--> pe cerc 
plt.figure()
plt.plot(np.real(phase_signal_after_modulator[:2000]), np.imag(phase_signal_after_modulator[:2000]), ".")
plt.title("CE-OFDM complex trajectory")
plt.xlabel("Real")
plt.ylabel("Imag")
plt.axis("equal")

phase = np.angle(phase_signal_after_modulator)
#plotarea fazei
plt.figure()
plt.plot(phase[:2000])
plt.title("Phase of CE-OFDM signal after modulator")
plt.xlabel("Sample")
plt.ylabel("Phase (rad)")

tx = transmitter["h_signal"].get_data()
#partea reala si imaginara dupa hermitian+P2S
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.real(tx[:1000]))
plt.ylim({-2,2})
plt.title("Hermitian OFDM Real")
plt.subplot(2,1,2)
plt.plot(np.imag(tx[:1000]))
plt.ylim({-2,2})
plt.title("Hermitian OFDM Imag")
plt.tight_layout()
print("Imaginary part after h_simetry:", np.max(np.abs(np.imag(tx))))

########## EVM
data_tx_evm = transmitter['data_evm'].get_data()
data_rx_evm = receiver['data_evm'].get_data()
evm = compute_evm(data_tx_evm, data_rx_evm)
print("evm(%) = {}".format(evm*100))

########## SER
data_tx = transmitter["data_ser"].get_data()
data_rx = receiver["data_ser"].get_data()
ser = compute_ser(data_tx, data_rx)
print("ser = {}".format(ser))


######### MSE teoretic

h = 0.2
A = 1.0

Nc = carrier

# numărul de subpurtătoare date
L = carrier_data

# a_l = 1 fara weighting
a_l = np.ones(L)
l = np.arange(L)
l_formula = l + 1

mse_th = (1 / (8 * (np.pi * h * a_l)**2)) * (
    sigma_phase2 / (np.sin(np.pi * l_formula / Nc)**2) +
    sigma_awgn2 / (A**2)
)

plt.figure(figsize=(8,5))
plt.plot(l, mse_th, color='red', marker='o', linewidth=2, markersize=4)   # curba
plt.xlabel("Index")
plt.ylabel("MSE")
plt.title("Theoretical MSE") 
plt.xlim(0, L-20)
plt.grid(True)



#### MSE_2

S_tx = transmitter['data_evm'].get_data()
S_rx = receiver['data_evm'].get_data()
print(S_rx.shape)
print(S_tx.shape)

mse_sim = np.abs(S_rx - S_tx)**2

l = np.arange(1, carrier_data+1)

plt.figure(figsize=(8,5))
plt.plot(l, mse_sim)
plt.xlabel("Index l")
plt.ylabel("MSE")
plt.title("Simulation MSE")
plt.grid(True)


#
freq = transmitter["carrier_allocator"].get_data()
N = freq.shape[0]
left = freq[1:N//2,0]
right = np.conjugate(freq[-1:N//2:-1,0])

print("Hermitian error:", np.max(np.abs(left-right)))


constellation = receiver["data_evm"].get_data()
#constelatia la receptie
plt.figure()
plt.plot(np.real(constellation), np.imag(constellation), ".")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("OFDM symbols")
plt.show()  

