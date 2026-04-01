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
from comnumpy.optical.devices import Laser
from comnumpy.ofdm.chains import PhaseNoise
from comnumpy.ofdm.compensators import PhaseModulator, PhaseDemodulator, Weight
from comnumpy.ofdm.processors import CarrierAllocator, CarrierExtractor, FFTProcessor, IFFTProcessor
from comnumpy.core.metrics import compute_evm, compute_ser, compute_ber

M = 16
os = 2
h = 0.3
A = 1
sigma_awgn2 = 5e-3
sigma_phase2 = 5e-4

# OSNR_dB = 20
# OSNR_abs = 10**(OSNR_dB/10)
# sigma_awgn2 = A**2 / (2 * OSNR_abs * os)

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
print("weight uniform:", a_uniform)
print("Power:", (2/N)*np.sum(a_uniform**2))

# sum-MSE
alpha_sum = (2/N)*np.sum(np.sqrt(gamma_l))
a_sum = np.sqrt(np.sqrt(gamma_l)/alpha_sum)
print("Power:", (2/N)*np.sum(a_sum**2))
print("weight sum:",a_sum)
# var-MSE
alpha_var = (2/N)*np.sum(gamma_l)
a_var = np.sqrt(gamma_l/alpha_var)
print("weight var:",a_var)
print("Power:", (2/N)*np.sum(a_var**2))

def run_chain(a):

    transmitter = Sequential([
        SymbolGenerator(M),
        Recorder(name = 'data_tx'),
        SymbolMapper(alphabet),
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
        Recorder(name='y1[n]'),
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
        PhaseNoise(sigma2=sigma_phase2),
        # Laser(linewidth=100_000, theta0=None, fs=1e9),
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
    data_const = receiver['constellation'].get_data()
    ber = compute_ber(data_tx, data_rx, width=int(np.log2(M)))
    ser = compute_ser(data_tx, data_rx)
    evm = compute_evm(data_tx, data_rx)
    print("data_tx:", data_tx.shape)

    # plt.figure()
    # plt.plot(np.real(data_const), np.imag(data_const), ".")

    # signal_after_demodulator=receiver['y1[n]'].get_data()
    # plt.figure()
    # plt.plot(signal_after_demodulator)
    
    return mse, mse_tot, ber, ser, evm, data_const

mse_u, mse_u_tot, ber_u, ser_u, evm_u, data_const = run_chain(a_uniform)
mse_s, mse_s_tot, ber_s, ser_s, evm_s, data_const = run_chain(a_sum)
mse_v, mse_v_tot, ber_v, ser_v, evm_v, data_const = run_chain(a_var)

print("BER:", ber_s,ber_u, ber_v)
mse_u_th = gamma_l / (8*(np.pi*h*a_uniform)**2)
mse_s_th = gamma_l / (8*(np.pi*h*a_sum)**2)
mse_v_th = gamma_l / (8*(np.pi*h*a_var)**2)

J_uniform = np.sum(gamma_l / (8*(np.pi*h*a_uniform)**2))
J_sum = (np.sum(np.sqrt(gamma_l))**2) / (4*N*(np.pi*h)**2)
J_var = (L*np.sum(gamma_l)) / (4*N*(np.pi*h)**2)

table = pd.DataFrame({
    "Strategy": ["Uniform", "Var-MSE", "Sum-MSE"],
    "Total MSE (th)": [J_uniform, J_sum, J_var],
})

print("Error SUM:", abs(J_sum - mse_u_tot))
print("Error VAR:", abs(J_var - mse_s_tot))
print("Error UNIFORM:", abs(J_uniform - mse_v_tot))

# print(table)
print("UNIFORM")
print("Total MSE (th):", J_uniform)
print("Total MSE (sim):", mse_u_tot)

print("\nSUM-MSE")
print("Total MSE (th):", J_sum)
print("Total MSE (sim):", mse_s_tot)

print("\nVAR-MSE")
print("Total MSE (th):", J_var)
print("Total MSE (sim):", mse_v_tot)

print("Order (should be sum <= var <= uniform):")
print(mse_s_tot,mse_v_tot,mse_u_tot)

plt.figure(figsize=(8,5))

# UNIFORM
plt.plot(l, mse_u_th, 'r-', label='Uniform TH')
plt.plot(l, mse_u, 'r:o', label='Uniform SIM')

# SUM
plt.plot(l, mse_s_th, 'g-', label='Sum TH')
plt.plot(l, mse_s, 'g:o', label='Sum SIM')

# VAR
plt.plot(l, mse_v_th, 'b-', label='Var TH')
plt.plot(l, mse_v, 'b:o', label='Var SIM')

plt.xlabel("Subcarrier index")
plt.ylabel("MSE")
plt.title("Theory vs Simulation")
plt.grid(True)
plt.legend()

plt.xlabel("real")
plt.ylabel("imag")

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
# h = 0.1
# A = 1

# L = 63
# N = 2*(L+1)*os
# l = np.arange(1, L+1)

# n_runs = 1000

# alphabet = get_alphabet("QAM", M)

# vect = np.zeros(N)
# vect[1:L+1] = 1
# vect[N:N-L-1:-1] = -1

# gamma_l = sigma_phase2 / (np.sin(np.pi*l/N)**2) + sigma_awgn2/(A**2)
# # uniform
# a_uniform = np.ones(L)
# # a_uniform = np.random.randn(len(l))
# a_uniform *= np.sqrt((N/2)/np.sum(a_uniform**2))
# print("Power:", (2/N)*np.sum(a_uniform**2))
# # print(a_uniform)
# # sum-MSE
# alpha_sum = (2/N)*np.sum(np.sqrt(gamma_l))
# a_sum = np.sqrt(np.sqrt(gamma_l)/alpha_sum)
# print("Power:", (2/N)*np.sum(a_sum**2))

# # var-MSE
# alpha_var = (2/N)*np.sum(gamma_l)
# a_var = np.sqrt(gamma_l/alpha_var)
# print("Power:", (2/N)*np.sum(a_var**2))

# def run_chain(a):

#     transmitter = Sequential([
#         SymbolGenerator(M),
#         Recorder(name = 'data_tx'),
#         SymbolMapper(alphabet),
#         Recorder(name='data_evm'),
#         Serial2Parallel(L),
#         Recorder(name='tx_symbols'),
#         Weight(a=a),
#         CarrierAllocator(vect, hermitian_sym=True),
#         IFFTProcessor(),
#         Recorder(name='after'),
#         Parallel2Serial(),
#         PhaseModulator(h=h),
#     ])

#     receiver = Sequential([
#         PhaseDemodulator(h=h, unwrap=True),
#         Serial2Parallel(N),
#         FFTProcessor(),
#         CarrierExtractor(vect),
#         Weight(a=1/a),
#         Recorder(name='rx_symbols'),
#         Parallel2Serial(),
#         Recorder(name='data_evm'),
#         SymbolDemapper(alphabet),
#         Recorder(name = 'data_rx'),
#     ])

#     channel = Sequential([
#         PhaseNoise(sigma2=sigma_phase2),
#         AWGN(value=sigma_awgn2, unit='sigma2'),
#     ])

#     chain = Sequential([transmitter, channel, receiver])

#     chain(L*n_runs) 
    
#     S_tx = transmitter['tx_symbols'].get_data()
#     S_rx = receiver['rx_symbols'].get_data()

#     mse = np.mean(np.abs(S_rx - S_tx)**2, axis=1)
#     mse_tot = np.sum(mse)

#     data_tx = transmitter['data_tx'].get_data()
#     data_rx = receiver['data_rx'].get_data()
#     data_tx_evm = transmitter['data_evm'].get_data()
#     data_rx_evm = receiver['data_evm'].get_data()

#     ber = compute_ber(data_tx, data_rx, width=int(np.log2(M)))
#     ser = compute_ser(data_tx, data_rx)
#     evm = compute_evm(data_tx_evm, data_rx_evm)
#     # print("data_tx:", data_tx.shape)
#     return mse, mse_tot, ber, ser, evm

# mse_u, mse_u_tot, ber_u, ser_u, evm_u = run_chain(a_uniform)
# mse_s, mse_s_tot, ber_s, ser_s, evm_s = run_chain(a_sum)
# mse_v, mse_v_tot, ber_v, ser_v, evm_v = run_chain(a_var)

# mse_u_th = gamma_l / (8*(np.pi*h*a_uniform)**2)
# mse_s_th = gamma_l / (8*(np.pi*h*a_sum)**2)
# mse_v_th = gamma_l / (8*(np.pi*h*a_var)**2)

# J_uniform = np.sum(gamma_l / (8*(np.pi*h*a_uniform)**2))
# J_sum = (np.sum(np.sqrt(gamma_l))**2) / (4*N*(np.pi*h)**2)
# J_var = (L*np.sum(gamma_l)) / (4*N*(np.pi*h)**2)

# table = pd.DataFrame({
#     "Strategy": ["Uniform", "Var-MSE", "Sum-MSE"],
#     "Total MSE (th)": [J_uniform, J_sum, J_var],
# })

# print("Error SUM:", abs(J_sum - mse_u_tot))
# print("Error VAR:", abs(J_var - mse_s_tot))
# print("Error UNIFORM:", abs(J_uniform - mse_v_tot))

# # print(table)
# print("UNIFORM")
# print("Total MSE (th):", J_uniform)
# print("Total MSE (sim):", mse_u_tot)

# print("\nSUM-MSE")
# print("Total MSE (th):", J_sum)
# print("Total MSE (sim):", mse_s_tot)

# print("\nVAR-MSE")
# print("Total MSE (th):", J_var)
# print("Total MSE (sim):", mse_v_tot)

# print("Order (should be sum <= var <= uniform):")
# print(mse_s_tot,mse_v_tot,mse_u_tot)

# plt.figure(figsize=(8,5))

# # UNIFORM
# plt.plot(l, mse_u_th, 'r-', label='Uniform TH')
# plt.plot(l, mse_u, 'r:o', label='Uniform SIM')

# # SUM
# plt.plot(l, mse_s_th, 'g-', label='Sum TH')
# plt.plot(l, mse_s, 'g:o', label='Sum SIM')

# # VAR
# plt.plot(l, mse_v_th, 'b-', label='Var TH')
# plt.plot(l, mse_v, 'b:o', label='Var SIM')

# plt.xlabel("Subcarrier index")
# plt.ylabel("MSE")
# plt.title("Theory vs Simulation")
# plt.grid(True)
# plt.legend()

# print(f"Uniform: BER={ber_u:.4e}, SER={ser_u:.4e}, EVM={evm_u:.4e}")
# print(f"Sum-MSE: BER={ber_s:.4e}, SER={ser_s:.4e}, EVM={evm_s:.4e}")
# print(f"Var-MSE: BER={ber_v:.4e}, SER={ser_v:.4e}, EVM={evm_v:.4e}")

# plt.show()
