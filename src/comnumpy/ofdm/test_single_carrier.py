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
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor, CyclicPrefixer, CyclicPrefixRemover, HermitianPrefixer
from comnumpy.ofdm.utils import get_standard_carrier_allocation
from comnumpy.core.visualizers import TimeScope, SpectrumScope
from comnumpy.core.metrics import compute_evm, compute_ser
M = 4
N = 1000
os=4
alphabet = get_alphabet("QAM", M)
sigma2 = 0.015



transmitter = Sequential([
        SymbolGenerator(M),
        Recorder(name='data_tx'),
        SymbolMapper(alphabet),
        Recorder(name='data_qam'),
        # Recorder(name='data_after_OFDM_Transmitter'),
        # TimeScope(title='Transmitter OFDM', name='time domain'),
])

channel = Sequential([
        AWGN(value=sigma2, unit='sigma2'),
])

receiver = Sequential([
        Recorder(name='data_qam_rx'),
        SymbolDemapper(alphabet),
        Recorder(name='data_rx')
])


chain = Sequential([transmitter, channel, receiver])

y = chain(N)

data_tx = transmitter['data_qam'].get_data()
data_rx = receiver['data_qam_rx'].get_data()
evm = compute_evm(data_tx, data_rx)
print("evm(%) = {}".format(evm*100))


data_tx = transmitter["data_tx"].get_data()
data_rx = receiver["data_rx"].get_data()
ser = compute_ser(data_tx, data_rx)
print("ser = {}".format(ser))


constellation = receiver["data_qam_rx"].get_data()
plt.figure()
plt.plot(np.real(constellation), np.imag(constellation), ".")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("OFDM symbols")


plt.show()  

