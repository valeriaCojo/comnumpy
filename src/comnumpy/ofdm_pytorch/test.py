import numpy as np
import torch
import sys
sys.path.insert(0, r"C:\Users\Valeria\Desktop\Master_OFDM\comnumpy\src")

from modules.generators import SymbolGenerator
from modules.mappers import SymbolMapper
from comnumpy.core.utils import get_alphabet
from modules.processors import Serial2Parallel, Parallel2Serial
from modules.channels import AWGNChannel

from modules.channels import LaserPhaseNoise
from modules.compensators import Weight

