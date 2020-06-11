# Preamble
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import time 
import os
import sys
import argparse

sys.path.append('/home/rwright/CHUMM/') # may need to specify
sys.path.append('/Users/ruby/Documents/GitHub/CHUMM/') # may need to specify
from STFTools import *
from AccretionTools import *
from ParticleTools import *
from VRPythonTools import *
from GenPythonTools import *
from multiprocessing import Process, cpu_count


############ 0. ARGUMENT PROCESSING ############
# Parse the arguments for accretion calculation
if True:
    parser=argparse.ArgumentParser()
    parser.add_argument('-path',type=str, default=None,
                        help='Folder to analyse')
    parser.add_argument('-mcut',type=float, default=10,
                        help='Mass cut: only consider haloes above this mass')

path=parser.parse_args().path
mcut=10**parser.parse_args().mcut

add_recycling_data_serial(path=path,mcut=mcut)