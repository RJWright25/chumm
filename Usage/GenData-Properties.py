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
    parser.add_argument('-numproc',type=int, default=1,
                        help='Number of processes to use')
    parser.add_argument('-mcut',type=float, default=10,
                        help='Mass cut: only consider haloes above this mass')
    parser.add_argument('-fullhalo',type=int, default=0,
                        help='Flag: write data for all halo particles (not just accreted) ')

path=parser.parse_args().path
mcut=10**parser.parse_args().mcut
fullhalo=bool(parser.parse_args().fullhalo)   
numproc=int(parser.parse_args().numproc)

accfiles_all=list_dir(path)
accfiles_valid=[accfile for accfile in accfiles_all if ('All' not in accfile and 'recyc' not in accfile and 'ave' not in accfile)]
accfiles_n=len(list_dir(accfiles_valid))
indices=gen_mp_indices(list(range(accfiles_n)),n=numproc)

processes=[]
for iproc in range(numproc):
    print(f"Starting process {iproc}, has files {indices[iproc]['indices']}")
    kwargs={'path':path,'mcut':mcut,'fullhalo':fullhalo,'fileidx':indices[iproc]['indices']}
    p=Process(target=add_particle_data_serial,kwargs=kwargs)
    processes.append(p)
    p.start()

for p in processes:
    p.join()
