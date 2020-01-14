#   ______   __    __  __    __  __       __  __       __ 
#  /      \ /  |  /  |/  |  /  |/  \     /  |/  \     /  |
# /$$$$$$  |$$ |  $$ |$$ |  $$ |$$  \   /$$ |$$  \   /$$ |
# $$ |  $$/ $$ |__$$ |$$ |  $$ |$$$  \ /$$$ |$$$  \ /$$$ |
# $$ |      $$    $$ |$$ |  $$ |$$$$  /$$$$ |$$$$  /$$$$ |
# $$ |   __ $$$$$$$$ |$$ |  $$ |$$ $$ $$/$$ |$$ $$ $$/$$ |
# $$ \__/  |$$ |  $$ |$$ \__$$ |$$ |$$$/ $$ |$$ |$$$/ $$ |
# $$    $$/ $$ |  $$ |$$    $$/ $$ | $/  $$ |$$ | $/  $$ |
#  $$$$$$/  $$/   $$/  $$$$$$/  $$/      $$/ $$/      $$/

#    _____          _         __             _    _       _                        _    _ __  __       _       _   _                      __   __  __               
#   / ____|        | |       / _|           | |  | |     | |                      | |  | |  \/  |     | |     | | (_)                    / _| |  \/  |              
#  | |     ___   __| | ___  | |_ ___  _ __  | |__| | __ _| | ___     __ _  ___ ___| |  | | \  / |_   _| | __ _| |_ _  ___  _ __     ___ | |_  | \  / | __ _ ___ ___ 
#  | |    / _ \ / _` |/ _ \ |  _/ _ \| '__| |  __  |/ _` | |/ _ \   / _` |/ __/ __| |  | | |\/| | | | | |/ _` | __| |/ _ \| '_ \   / _ \|  _| | |\/| |/ _` / __/ __|
#  | |___| (_) | (_| |  __/ | || (_) | |    | |  | | (_| | | (_) | | (_| | (_| (__| |__| | |  | | |_| | | (_| | |_| | (_) | | | | | (_) | |   | |  | | (_| \__ \__ \
#   \_____\___/ \__,_|\___| |_| \___/|_|    |_|  |_|\__,_|_|\___/   \__,_|\___\___|\____/|_|  |_|\__,_|_|\__,_|\__|_|\___/|_| |_|  \___/|_|   |_|  |_|\__,_|___/___/
                                                                                                                                                                  
                                                                                                                                                                  
# GenData-PartData.py - Script to generate integrated particle histories for a given simulation.
# Author: RUBY WRIGHT 

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
from GenPythonTools import *
from VRPythonTools import *
from STFTools import *
from ParticleTools import *
from AccretionTools import *
from multiprocessing import Process, cpu_count

############ 0. ARGUMENT PROCESSING ############
parser=argparse.ArgumentParser()
parser.add_argument('-np', type=int, default=1,
                    help='number of processes to use')
parser.add_argument('-gen_bph', type=int, default=1,
                    help='generate base particle histories')
parser.add_argument('-sum_bph', type=int, default=1,
                    help='sum base particle histories')
n_processes = parser.parse_args().np
gen_bph=parser.parse_args().gen_bph
sum_bph=parser.parse_args().sum_bph

# Load base halo data
run_name=os.getcwd().split('/')[-1] #Grab simulation name from folder name
base_halo_data=open_pickle(f'B1_HaloData_{run_name}.dat')#*

############ 1. SAVE PARTICLE STATES FOR EACH SNAPSHOT ############
# This is run in parallel, each snap individually.
# Here, we run the gen_particle_history_serial tool for each snapshot,
# which saves the host structure of each particle, sorts their IDs, 
# and saves their index in particle data for future reference. 

if gen_bph:
    snaps_for_history=[snap for snap in range(len(base_halo_data)) if base_halo_data[snap]['Part_FilePath']] #Only generate histories for non-padded snaps
    snaps_mp_lists=gen_mp_indices(snaps_for_history,n=n_processes)
    kwargs=[{'snaps':snaps_mp_lists[iprocess]['indices']} for iprocess in range(n_processes)]

    print(f"Distributing snaps for particle states amongst {n_processes} cores")
    processes=[]
    if __name__ == '__main__':
        for iprocess in range(n_processes):
            print(f'Starting process {iprocess}')
            p=Process(target=gen_particle_history_serial, args=(base_halo_data,),kwargs=kwargs[iprocess])
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

############ 2. INTEGRATE PARTICLE HISTORIES FOR EACH SNAPSHOT ############
# This is run in serial, each snap sequentially.
# Here, we run the postprocess_particle_history_serial tool over the 
# result from above, which for each particle sequentially adds to a flag 
# indicating the sum of how many snaps a given particle has been part of 
# structure in the past. 

if sum_bph:
    postprocess_particle_history_serial(base_halo_data,path='part_histories')
