####### GENERATE ACCRETION DATA #######
#(for a given set of parameters)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py
import time 
import os
import sys
import argparse

sys.path.append('/home/rwright/CHUMM/')

from STFTools import *
from AccretionTools import *
from VRPythonTools import *
from GenPythonTools import *
from multiprocessing import Process, cpu_count

########## 0. PREAMBLE ##########

parser=argparse.ArgumentParser()

parser.add_argument('-snap', type=int,
                    help='snap to calculate accretion for')
parser.add_argument('-pre', type=int,default=1,
                    help='accretion snapshot gap')
parser.add_argument('-post', type=int,default=1,
                    help='fidelity checking gap')
parser.add_argument('-hil_lo', type=int,default=-1,
                    help='halo index list lower limit (for testing: -1=all, not test)')
parser.add_argument('-hil_hi', type=int,default=-1,
                    help='halo index list upper limit (for testing, -1=all, not test)')
parser.add_argument('-gen_ad', type=int,default=1,
                    help='Flag: generate accretion data (and sum)')
parser.add_argument('-gen_pd', type=int,default=1,
                    help='Flag: add particle data to existing acc_data')
parser.add_argument('-np', type=int, default=1,
                    help='number of processes to use')

# Parse the arguments
snap = parser.parse_args().snap
pre_depth=parser.parse_args().pre
post_depth=parser.parse_args().post
halo_index_list_lo=parser.parse_args().hil_lo
halo_index_list_hi=parser.parse_args().hil_hi
gen_ad=bool(parser.parse_args().gen_ad)
gen_pd=bool(parser.parse_args().gen_pd)
n_processes = parser.parse_args().np

print('Arguments parsed:')
print(f'snap: {snap}, pre_depth: {pre_depth}, post_depth: {post_depth}, hil_lo: {halo_index_list_lo}, hil_hi {halo_index_list_hi}')
print(f'gen_ad: {gen_ad}, gen_pd: {gen_pd}, n_processes: {n_processes}')

# Use the directory name to get the runs
run_name=os.getcwd().split('/')[-1]
if 'EAGLE' in run_name:
    partdata_filetype='EAGLE'
else:
    partdata_filetype='GADGET'

#Load in halo data
base_halo_data=open_pickle(f'B1_HaloData_{run_name}.dat')

#Process arguments: if we're parsed a halo index list range that isn't -1, then use testing mode
if halo_index_list_lo==-1:
    test=False
    num_halos=base_halo_data[snap]["Count"]
    halo_index_lists=gen_mp_indices(indices=list(range(num_halos)),n=n_processes,test=test)
else:
    test=True
    halo_index_list=list(range(halo_index_list_lo,halo_index_list_hi))
    halo_index_lists=gen_mp_indices(indices=halo_index_list,n=n_processes,test=test)

# Determine output directory for this calculation
if test:
    calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}_test/'
else:
    calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}/'

output_dir=calc_dir+f'snap_{str(snap).zfill(3)}/'

########## 1. GENERATE ACCRETION DATA ##########
if gen_ad:
    t1_acc=time.time()

    # Clear the output directory 
    os.system(f'rm -rf {output_dir}*')

    processes=[]
    kwargs=[{'snap':snap,'halo_index_list':halo_index_lists[iprocess],'pre_depth':pre_depth,'post_depth':post_depth} for iprocess in range(n_processes)]

    if __name__ == '__main__':
        for iprocess in range(len(kwargs)):
            print(f'Starting process {iprocess}')
            p=Process(target=gen_accretion_data_fof_serial, args=(base_halo_data,),kwargs=kwargs[iprocess])
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

    t2_acc=time.time()

########## 2. ADD PARTICLE DATA ##########
if gen_pd:
    t1_part=time.time()

    processes=[]
    accdata_files=os.listdir(output_dir)
    accdata_paths=[output_dir+accdata_file for accdata_file in accdata_files if 'summed' not in accdata_file]
    kwargs=[{'accdata_path':accdata_path,
    'datasets':['ParticleIDs','AExpMaximumTemperature','Coordinates','Density','InternalEnergy','MaximumTemperature','StarFormationRate','Temperature','Velocity']} for accdata_path in accdata_paths]

    if __name__ == '__main__':
        for iprocess in range(len(kwargs)):
            print(f'Starting process {iprocess}')
            p=Process(target=add_gas_particle_data, args=(base_halo_data,),kwargs=kwargs[iprocess])
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

    t2_part=time.time()

########## 3. SUM ACCRETION DATA ##########
if gen_ad:
    t1_sum=time.time()

    postprocess_acc_data_serial(output_dir)

    t2_sum=time.time()

########## 4. PRINT PERFORMANCE ##########
print()
print('******************************************************')
print()

if gen_ad:
    print(f'Generated accretion data for snap {snap} in {t2_acc-t1_acc} sec')
if gen_pd:
    print(f'Added particle data to accretion data for snap {snap} in {t2_part-t1_part} sec')
if gen_ad:
    print(f'Summed accretion data for snap {snap} in {t2_sum-t1_sum} sec')

print()
print('******************************************************')
print()