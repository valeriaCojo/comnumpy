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
h = 0.2
A = 1

L = 63
N = 2*(L+1)*os
l = np.arange(1, L+1)

n_runs = 1000

alphabet = get_alphabet("QAM", M)

vect = np.zeros(N)
vect[1:L+1] = 1
vect[N:N-L-1:-1] = -1

gamma_l = sigma_phase2 / (np.sin(np.pi*l/N)**2) + sigma_awgn2/(A**2)
# uniform
a_uniform = np.ones(L)
# a_uniform = np.random.randn(len(l))
a_uniform *= np.sqrt((N/2)/np.sum(a_uniform**2))

# sum-MSE
alpha_sum = (2/N)*np.sum(np.sqrt(gamma_l))
a_sum = np.sqrt(np.sqrt(gamma_l)/alpha_sum)
print("Power:", (2/N)*np.sum(a_sum**2))

# var-MSE
alpha_var = (2/N)*np.sum(gamma_l)
a_var = np.sqrt(gamma_l/alpha_var)
print("Power:", (2/N)*np.sum(a_var**2))

def run_chain(a):

    transmitter = Sequential([
        SymbolGenerator(M),
        Recorder(name = 'data_tx'),
        SymbolMapper(alphabet),
        Recorder(name='data_evm'),
        Serial2Parallel(L),
        Recorder(name='tx_symbols'),
        Weight(a=a),
        CarrierAllocator(vect, hermitian_sym=True),
        IFFTProcessor(),
        Recorder(name='after'),
        Parallel2Serial(),
        PhaseModulator(h=h),
    ])

    receiver = Sequential([
        PhaseDemodulator(h=h, unwrap=True),
        Serial2Parallel(N),
        FFTProcessor(),
        CarrierExtractor(vect),
        Weight(a=1/a),
        Recorder(name='rx_symbols'),
        Parallel2Serial(),
        Recorder(name='data_evm'),
        SymbolDemapper(alphabet),
        Recorder(name = 'data_rx'),
    ])

    channel = Sequential([
        PhaseNoise(sigma2=sigma_phase2),
        AWGN(value=sigma_awgn2, unit='sigma2'),
    ])

    chain = Sequential([transmitter, channel, receiver])

    chain(L*n_runs) 
    
    S_tx = transmitter['tx_symbols'].get_data()
    S_rx = receiver['rx_symbols'].get_data()

    mse = np.mean(np.abs(S_rx - S_tx)**2, axis=1)
    mse_tot = np.sum(mse)

    data_tx = transmitter['data_tx'].get_data()
    data_rx = receiver['data_rx'].get_data()
    data_tx_evm = transmitter['data_evm'].get_data()
    data_rx_evm = receiver['data_evm'].get_data()

    ber = compute_ber(data_tx, data_rx, width=int(np.log2(M)))
    ser = compute_ser(data_tx, data_rx)
    evm = compute_evm(data_tx_evm, data_rx_evm)
    # print("data_tx:", data_tx.shape)
    return mse, mse_tot, ber, ser, evm

mse_u, mse_u_tot, ber_u, ser_u, evm_u = run_chain(a_uniform)
mse_s, mse_s_tot, ber_s, ser_s, evm_s = run_chain(a_sum)
mse_v, mse_v_tot, ber_v, ser_v, evm_v = run_chain(a_var)



print(f"Uniform: BER={ber_u:.4e}, SER={ser_u:.4e}, EVM={evm_u:.4e}")
print(f"Sum-MSE: BER={ber_s:.4e}, SER={ser_s:.4e}, EVM={evm_s:.4e}")
print(f"Var-MSE: BER={ber_v:.4e}, SER={ser_v:.4e}, EVM={evm_v:.4e}")

plt.show()


