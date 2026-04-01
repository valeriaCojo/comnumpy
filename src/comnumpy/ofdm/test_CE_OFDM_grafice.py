import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd 
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
from comnumpy.core.metrics import compute_evm, compute_ser, compute_ber

M = 16
os = 2
sigma_awgn2 = 5e-3
sigma_phase2 = 5e-4
A = 1
h_list = np.arange(0.05, 0.5, 0.05)

L = 63
N = 2*(L+1)*os
n_runs = 1000

alphabet = get_alphabet("QAM", M)

vect = np.zeros(N)
vect[1:L+1] = 1
vect[N:N-L-1:-1] = -1

ber_list = []

for h_val in h_list:

    transmitter = Sequential([
        SymbolGenerator(M),
        Recorder(name='data_tx'),
        SymbolMapper(alphabet),
        Serial2Parallel(L),
        CarrierAllocator(vect, hermitian_sym=True),
        IFFTProcessor(),
        Parallel2Serial(),
        PhaseModulator(h=h_val),
        Recorder(name='y')
    ])

    receiver = Sequential([
        PhaseDemodulator(h=h_val, unwrap=False),
        Serial2Parallel(N),
        FFTProcessor(),
        CarrierExtractor(vect),
        Parallel2Serial(),
        SymbolDemapper(alphabet),
        Recorder(name='data_rx')
    ])
    channel = Sequential([
            PhaseNoise(sigma2=sigma_phase2),
            AWGN(value=sigma_awgn2, unit='sigma2'),
        ])

    chain = Sequential([transmitter, channel, receiver])
    chain(L * n_runs)

    y = transmitter['y'].get_data()
    data_tx = transmitter['data_tx'].get_data()
    data_rx = receiver['data_rx'].get_data()

    ber = compute_ber(data_tx, data_rx, width=int(np.log2(M)))
    ber_list.append(ber)

    print(f"h = {h_val:.2f}, BER = {ber:.3e}")

    plt.figure()

    plt.scatter(np.real(y), np.imag(y), s=1)

    plt.xlabel("Re{y[n]}")
    plt.ylabel("Im{y[n]}")
    plt.title(f"Signal y[n] (h = {h_val:.2f}, no unwrap)")

    plt.axis('equal')
    plt.grid()

plt.figure()

plt.semilogy(h_list, ber_list, 'r-o')

plt.xlabel("h (modulation index)")
plt.ylabel("BER")
plt.title("BER vs h (no unwrap, no channel)")

plt.grid()
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import pandas as pd 
# sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src")

# from comnumpy.core import Sequential, Recorder
# from comnumpy.core.generators import SymbolGenerator
# from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
# from comnumpy.core.processors import Serial2Parallel, Parallel2Serial
# from comnumpy.core.channels import AWGN
# from comnumpy.core.utils import get_alphabet
# from comnumpy.ofdm.chains import PhaseNoise
# from comnumpy.ofdm.compensators import PhaseModulator, PhaseDemodulator, Weight
# from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
# from comnumpy.core.metrics import compute_evm, compute_ser, compute_ber


# M = 16
# os = 2
# sigma_awgn2 = 5e-3
# sigma_phase2 = 5e-4
# h_list = np.arange(0.05, 0.5, 0.05)

# L = 63
# N = 2*(L+1)*os
# n_runs = 1000

# alphabet = get_alphabet("QAM", M)

# vect = np.zeros(N)
# vect[1:L+1] = 1
# vect[N:N-L-1:-1] = -1

# ber_list = []

# ber_no_channel = []
# ber_channel_no_unwrap = []
# ber_channel_with_unwrap = []
# ber_no_channel_with_unwrap = []
# const1_list = []
# const2_list = []
# const3_list = []
# const4_list = []

# for h_val in h_list:

#     transmitter_1 = Sequential([
#         SymbolGenerator(M),
#         Recorder(name='data_tx'),
#         SymbolMapper(alphabet),
#         Serial2Parallel(L),
#         CarrierAllocator(vect, hermitian_sym=True),
#         IFFTProcessor(),
#         Parallel2Serial(),
#         PhaseModulator(h=h_val),
#     ])

#     receiver_1 = Sequential([
#         PhaseDemodulator(h=h_val, unwrap=False),
#         Serial2Parallel(N),
#         FFTProcessor(),
#         CarrierExtractor(vect),
#         Parallel2Serial(),
#         Recorder(name = 'constellation'),
#         SymbolDemapper(alphabet),
#         Recorder(name='data_rx')
#     ])

#     chain_1 = Sequential([transmitter_1, receiver_1])
#     chain_1(L * n_runs)

#     data_tx_1 = transmitter_1['data_tx'].get_data()
#     data_rx_1 = receiver_1['data_rx'].get_data()

#     ber1 = compute_ber(data_tx_1, data_rx_1, width=int(np.log2(M)))
#     ber_no_channel.append(ber1)
#     const1 = receiver_1['constellation'].get_data()
#     const1_list.append(const1)

#     transmitter_2 = Sequential([
#         SymbolGenerator(M),
#         Recorder(name='data_tx'),
#         SymbolMapper(alphabet),
#         Serial2Parallel(L),
#         CarrierAllocator(vect, hermitian_sym=True),
#         IFFTProcessor(),
#         Parallel2Serial(),
#         PhaseModulator(h=h_val),
#     ])

#     receiver_2 = Sequential([
#         PhaseDemodulator(h=h_val, unwrap=False),
#         Serial2Parallel(N),
#         FFTProcessor(),
#         CarrierExtractor(vect),
#         Parallel2Serial(),
#         Recorder(name = 'constellation'),
#         SymbolDemapper(alphabet),
#         Recorder(name='data_rx')
#     ])

#     channel = Sequential([
#         # PhaseNoise(sigma2=sigma_phase2),
#         AWGN(value=sigma_awgn2, unit='sigma2'),
#     ])

#     chain_2 = Sequential([transmitter_2, channel, receiver_2])
#     chain_2(L * n_runs)

#     data_tx_2 = transmitter_2['data_tx'].get_data()
#     data_rx_2 = receiver_2['data_rx'].get_data()

#     ber2 = compute_ber(data_tx_2, data_rx_2, width=int(np.log2(M)))
#     ber_channel_no_unwrap.append(ber2)

#     const2 = receiver_2['constellation'].get_data()
#     const2_list.append(const2)

#     transmitter_3 = Sequential([
#         SymbolGenerator(M),
#         Recorder(name='data_tx'),
#         SymbolMapper(alphabet),
#         Serial2Parallel(L),
#         CarrierAllocator(vect, hermitian_sym=True),
#         IFFTProcessor(),
#         Parallel2Serial(),
#         PhaseModulator(h=h_val),
#     ])

#     receiver_3 = Sequential([
#         PhaseDemodulator(h=h_val, unwrap=True),
#         Serial2Parallel(N),
#         FFTProcessor(),
#         CarrierExtractor(vect),
#         Parallel2Serial(),
#         Recorder(name = 'constellation'),
#         SymbolDemapper(alphabet),
#         Recorder(name='data_rx')
#     ])

#     chain_3 = Sequential([transmitter_3, channel, receiver_3])
#     chain_3(L * n_runs)

#     data_tx_3 = transmitter_3['data_tx'].get_data()
#     data_rx_3 = receiver_3['data_rx'].get_data()

#     ber3 = compute_ber(data_tx_3, data_rx_3, width=int(np.log2(M)))
#     ber_channel_with_unwrap.append(ber3)

#     const3 = receiver_3['constellation'].get_data()
#     const3_list.append(const3)
    
#     transmitter_4 = Sequential([
#         SymbolGenerator(M),
#         Recorder(name='data_tx'),
#         SymbolMapper(alphabet),
#         Serial2Parallel(L),
#         CarrierAllocator(vect, hermitian_sym=True),
#         IFFTProcessor(),
#         Parallel2Serial(),
#         PhaseModulator(h=h_val),
#     ])

#     receiver_4 = Sequential([
#         PhaseDemodulator(h=h_val, unwrap=True),
#         Serial2Parallel(N),
#         FFTProcessor(),
#         CarrierExtractor(vect),
#         Parallel2Serial(),
#         Recorder(name='constellation'),
#         SymbolDemapper(alphabet),
#         Recorder(name='data_rx')
#     ])

#     chain_4 = Sequential([transmitter_4, receiver_4])
#     chain_4(L * n_runs)

#     data_tx_4 = transmitter_4['data_tx'].get_data()
#     data_rx_4 = receiver_4['data_rx'].get_data()

#     ber4 = compute_ber(data_tx_4, data_rx_4, width=int(np.log2(M)))
#     ber_no_channel_with_unwrap.append(ber4)

#     const4 = receiver_4['constellation'].get_data()
#     const4_list.append(const4)
#     print(f"h={h_val:.2f} | no ch={ber1:.2e} | no unwrap={ber2:.2e} | unwrap={ber3:.2e} | no ch +unwrap={ber4:.2e}")

# plt.figure()

# plt.semilogy(h_list, ber_no_channel, 'b-o', label='No channel, no unwrap')
# plt.semilogy(h_list, ber_no_channel_with_unwrap, 'k-o', label='No channel, unwrap')
# plt.semilogy(h_list, ber_channel_no_unwrap, 'r-o', label='Channel, no unwrap')
# plt.semilogy(h_list, ber_channel_with_unwrap, 'g-o', label='Channel, with unwrap')


# plt.xlabel("h (modulation index)")
# plt.ylabel("BER")
# plt.title("BER vs h")
# plt.legend()
# plt.grid()

# indices = [0, 1, 2, 3, 4]

# for idx in indices:

#     h_val = h_list[idx]

#     c1 = const1_list[idx]
#     c2 = const2_list[idx]
#     c3 = const3_list[idx]
#     c4 = const4_list[idx]

#     plt.figure(figsize=(16,4))

#     plt.subplot(1,4,1)
#     plt.scatter(np.real(c1), np.imag(c1), s=2)
#     plt.title("No channel")
#     plt.xlabel("Real")
#     plt.ylabel("Imag")
#     plt.grid()

#     plt.subplot(1,4,2)
#     plt.scatter(np.real(c4), np.imag(c4), s=2)
#     plt.title("No channel + unwrap")
#     plt.xlabel("Real")
#     plt.ylabel("Imag")
#     plt.grid()

#     plt.subplot(1,4,3)
#     plt.scatter(np.real(c2), np.imag(c2), s=2)
#     plt.title("Channel, no unwrap")
#     plt.xlabel("Real")
#     plt.ylabel("Imag")
#     plt.grid()

#     plt.subplot(1,4,4)
#     plt.scatter(np.real(c3), np.imag(c3), s=2)
#     plt.title("Channel, with unwrap")
#     plt.xlabel("Real")
#     plt.ylabel("Imag")
#     plt.grid()
    
#     plt.suptitle(f"Constellation comparison (h = {h_val:.2f})")
#     plt.tight_layout()

# plt.show()