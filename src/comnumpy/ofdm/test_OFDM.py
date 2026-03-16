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
from comnumpy.ofdm.chains import OFDMTransmitter, OFDMReceiver
from comnumpy.ofdm.compensators import FrequencyDomainEqualizer
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor, CyclicPrefixer, CyclicPrefixRemover
from comnumpy.ofdm.utils import get_standard_carrier_allocation
from comnumpy.core.visualizers import TimeScope, SpectrumScope
from comnumpy.core.metrics import compute_evm, compute_ser
M = 16
os=4
sigma2 = 0.01 # or sigma2 = 0.01
alphabet = get_alphabet("QAM", M)
type_of_carrier= get_standard_carrier_allocation("NoPilot_128", os=os, shift=True)
carrier = len(type_of_carrier)
carrier_data = np.sum(type_of_carrier==1)
print(carrier_data)
N_cp = carrier//8
n_samples = N_cp+carrier
N_OFDM_symbols = 20
N = int(carrier_data*N_OFDM_symbols)

transmitter = Sequential([
        SymbolGenerator(M),
        Recorder(name='data_ser'),
        SymbolMapper(alphabet),
        Recorder(name="mapper"),
        Recorder(name='data_evm'),
        Serial2Parallel(carrier_data),
        CarrierAllocator(type_of_carrier),
        IFFTProcessor(),
        CyclicPrefixer(N_cp=N_cp),
        Parallel2Serial(),
        Recorder(name='tx_signal'),
        # Recorder(name='data_after_OFDM_Transmitter'),
        TimeScope(num=1, title='full', name='time domain_OFDM_complex_signal'),
])

channel = Sequential([
        AWGN(value=sigma2, unit='sigma2'),
])

receiver = Sequential([
        Serial2Parallel(N_cp+carrier, method='zero-padding'),
        CyclicPrefixRemover(N_cp),
        FFTProcessor(),
        CarrierExtractor(type_of_carrier),
        Parallel2Serial(),
        Recorder(name='data_evm'),
        SymbolDemapper(alphabet),
        Recorder(name='data_ser')
])

chain = Sequential([transmitter, channel, receiver])

y = chain(N)

tx = transmitter["tx_signal"].get_data()
print("Max imaginary amplitude:", np.max(np.abs(np.imag(tx))))
print("Max imaginary amplitude:", np.max(np.abs(np.real(tx))))
plt.figure()

plt.subplot(2,1,1)
plt.plot(np.real(tx[:1000]))
plt.title("OFDM Real part")

plt.subplot(2,1,2)
plt.plot(np.imag(tx[:1000]))
plt.title("OFDM Imag part")

plt.tight_layout()

data_tx_evm = transmitter['data_evm'].get_data()
data_rx_evm = receiver['data_evm'].get_data()
evm = compute_evm(data_tx_evm, data_rx_evm)
print("evm(%) = {}".format(evm*100))


data_tx = transmitter["data_ser"].get_data()
data_rx = receiver["data_ser"].get_data()
ser = compute_ser(data_tx, data_rx)
print("ser = {}".format(ser))


constellation = receiver["data_evm"].get_data()
plt.figure()
plt.plot(np.real(constellation), np.imag(constellation), ".")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("OFDM symbols")


plt.show()  

