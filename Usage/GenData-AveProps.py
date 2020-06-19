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

path=parser.parse_args().path
run_name=os.getcwd().split('/')[-1] #Grab simulation name from folder name
base_halo_data=open_pickle(f'B4_HaloData_{run_name}.dat')#*

gen_averaged_accretion_data(base_halo_data=base_halo_data,path=path)