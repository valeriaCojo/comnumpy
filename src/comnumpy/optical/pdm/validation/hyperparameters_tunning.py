import numpy as np
import matplotlib.pyplot as plt
import optuna
import pandas
import sys
import pandas as pd
from matplotlib.ticker import NullFormatter, LogFormatterMathtext
from time import time

import sys
sys.path.insert(0, "src\\")

from comnumpy.core import Sequential
from comnumpy.core.channels import AWGN
from comnumpy.core.generators import SymbolGenerator
from comnumpy.core.mappers import SymbolMapper, SymbolDemapper
from comnumpy.core.utils import get_alphabet, hard_projector
from comnumpy.optical.pdm.sinks import IQ_Scope, IQ_Scope_PostProcessing
from comnumpy.core.filters import SRRCFilter
from comnumpy.core.processors import Upsampler, Downsampler
from comnumpy.core.monitors import Recorder
from comnumpy.core.metrics import compute_ser, compute_ber, compute_evm

from comnumpy.optical.pdm.generics import PDMWrapper, ChannelWrapper_, ChannelWrapper__, ChannelWrapper
from comnumpy.optical.pdm.channels import SOP, SOP_, PDL_, PMD_, PMD, PDL__, SOP__, PMD__
from comnumpy.optical.pdm.compensators import MCMA_to_DD_Czegledi, DD_Czegledi
from comnumpy.optical.pdm.utils import *

# parameters
type, M = "QAM", 16
N = 1_200_000
alphabet = get_alphabet(type, M, type='bin')

Ts = 1 / 28e9
seg = 20 # number of fiber segments

# SRRC filter
roll_off = 0.35
srrc_taps = 30

# Oversampling 
oversampling = 2
Fs = (1/Ts) * oversampling

### Impairments parameters ###
# SOP drift
pol_linewidth = 28e1

Fiber_length = 1000 # Km
D_pmd = 0.1e-12 # ps/sqrt(km), for aggresive cases, choose 0.1
t_dgd_k, rot_angle_k = build_pmd_segments(Fiber_length, D_pmd, seg)
pmd_params = [t_dgd_k, rot_angle_k]
pdl_params = 0.15 # segment-wise [1e-3]


# Convergence #
conv = 500_000
tx = Sequential( [
            Upsampler(oversampling),
            SRRCFilter(roll_off, oversampling, srrc_taps, method='fft')
] )


channel = Sequential( [
            PDL_(pdl_params),
            PMD_(t_dgd_k, Fs),
            SOP_(T_symb=Ts, linewidth=pol_linewidth, segments=seg),

] )

rx = Sequential( [
            Downsampler(oversampling),
] )

channel_params = [ pol_linewidth, pmd_params, pdl_params ]

SNR = 20

def run_chain(mu):
    chain = Sequential([
                SymbolGenerator(M),
                Recorder(name='data_tx'),
                PDMWrapper(DifferentialEncoding(M)),
                PDMWrapper(tx, 'tx'),
                ChannelWrapper_(seq_obj=channel, L=seg, params=channel_params, debug=True),
                PDMWrapper( AWGN(value=SNR, unit='snr_dB'), name='noise'),
                PDMWrapper( SRRCFilter(roll_off, oversampling, srrc_taps, method='fft') ),
                MCMA_(alphabet, 7, mu, oversampling),
                PDMWrapper(rx,'rx'),
                PDMWrapper(DifferentialDecoding(M)),
                Recorder(name='data_rx'),
                ])
    return chain

def objective(trial):
    start = time()
    mu = trial.suggest_float('mu', 1e-4, 1e-2)
    chain = run_chain(mu)
    y = chain(N)[:,conv:]
    stop = time()
    print('Ellapsed time:', stop-start)
    data_tx = chain['data_tx'].get_data()

    data_tx1 = np.reshape(data_tx, (2,-1))[0,conv:]
    data_tx2 = np.reshape(data_tx, (2,-1))[1,conv:]

    ser1  = compute_ser(data_tx1, y[0,:])
    ser2  = compute_ser(data_tx2, y[1,:])
    mean_ser = np.mean( np.array([ser1,ser2]) )
    return mean_ser

study = optuna.create_study(study_name='S3_MCMA_28e1', storage="sqlite:///S3_MCMA_28e1.db", load_if_exists=True)
study.optimize(objective, n_trials=20)
# fig = optuna.visualization.plot_optimization_history(study)
study.best_params
df_optuna = study.trials_dataframe()
df = pandas.DataFrame({'mu':df_optuna['params_mu'], 'SER':df_optuna['value']})
df.to_csv('opt_S3_MCMA_28e1.csv', index=False)
# fig.show()