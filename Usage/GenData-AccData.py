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
                                                                                                                                                                  
                                                                                                                                                                  
# GenData-AccData.py - Miscellaneous python tools for use in the rest of the package. 
# Author: RUBY WRIGHT 

# Halo data must first have been generated as per GenData-HaloData.py.
# Particle histories must also have been generated as per GenData-PartData.py. 

# PREAMBLE
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import time 
import os
import sys
import argparse

sys.path.append('/home/rwright/CHUMM/') # may need to specify
from STFTools import *
from AccretionTools import *
from VRPythonTools import *
from GenPythonTools import *
from multiprocessing import Process, cpu_count

############ 0. ARGUMENT PROCESSING ############
# Parse the arguments for accretion calculation
if True:
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
    parser.add_argument('-add_pd', type=int,default=1,
                        help='Flag: add particle data to existing acc_data')
    parser.add_argument('-sum_ad', type=int,default=1,
                        help='Flag: sum the generated accretion data')
    parser.add_argument('-np', type=int, default=1,
                        help='number of processes to use')
    snap = parser.parse_args().snap
    pre_depth=parser.parse_args().pre
    post_depth=parser.parse_args().post
    halo_index_list_lo=parser.parse_args().hil_lo
    halo_index_list_hi=parser.parse_args().hil_hi
    gen_ad=bool(parser.parse_args().gen_ad)
    add_pd=bool(parser.parse_args().add_pd)
    sum_ad=bool(parser.parse_args().sum_ad)
    n_processes = parser.parse_args().np
    print('Arguments parsed:')
    print(f'snap: {snap}, pre_depth: {pre_depth}, post_depth: {post_depth}, hil_lo: {halo_index_list_lo}, hil_hi {halo_index_list_hi}')
    print(f'gen_ad: {gen_ad}, add_pd: {add_pd}, sum_ad: {sum_ad}, n_processes: {n_processes}')

    # Use the directory name to get the run name
    run_name=os.getcwd().split('/')[-1]

    # Load in halo data
    base_halo_data=open_pickle(f'B1_HaloData_{run_name}.dat')

    # Process arguments: if we're parsed a halo index list range that isn't -1, then use testing mode
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

############ 1. GENERATE ACCRETION DATA ############
# This is run in parallel, splitting halos based on gen_mp_indices.
# Here, we generate files containing data about the particles which
# have accreted or been ejected from a halo, at a time and manner
# specified by the parsed arguments. 

if gen_ad:
    t1_acc=time.time()

    # Clear the output directory 
    os.system(f'rm -rf {output_dir}*')

    # Multiprocessing arguments
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


############ 2. SUM ACCRETION DATA ############
# This is run in serial, based on the files generated above. 
# Here, we sum the accretion data to create a database of 
# accretion rates (of various types) for all halos in the simulation.

if sum_ad:
    t1_sum=time.time()
    #recalc dir in case we want to use 1 process
    calc_list=os.listdir(f'acc_data')
    for icalc_dir in calc_list:
        if f'pre{str(pre_depth).zfill(2)}_post{str(post_depth)}' in icalc_dir:
            calc_dir_forsum=icalc_dir
            break
    if not calc_dir_forsum.endswith('/')
        calc_dir_forsum=calc_dir_forsum+'/'
    output_dir_forsum='acc_data/'+calc_dir_forsum+f'snap_{str(snap).zfill(3)}'
    postprocess_acc_data_serial(base_halo_data,output_dir_forsum)

    t2_sum=time.time()


############ 3. ADD PARTICLE DATA ############
# This is run in parallel, based on the files generated above. 
# Here, we add desired gas particle data to the accretion file.

if add_pd:
    t1_part=time.time()

    # Multiprocessing arguments
    processes=[]
    accdata_files=os.listdir(output_dir)
    accdata_paths=[output_dir+accdata_file for accdata_file in accdata_files if 'summed' not in accdata_file]
    kwargs=[{'accdata_path':accdata_path,
    'datasets':['ParticleIDs','AExpMaximumTemperature','Coordinates','Density','InternalEnergy','MaximumTemperature','StarFormationRate','Temperature','Velocity']} for accdata_path in accdata_paths]#* specify this

    if __name__ == '__main__':
        for iprocess in range(len(kwargs)):
            print(f'Starting process {iprocess}')
            p=Process(target=add_gas_particle_data, args=(base_halo_data,),kwargs=kwargs[iprocess])
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

    t2_part=time.time()



############ 4. PRINT PERFORMANCE ############
# Print performance of above.

print()
print('******************************************************')
print()

if gen_ad:
    print(f'Generated accretion data for snap {snap} in {t2_acc-t1_acc} sec')
if sum_ad:
    print(f'Summed accretion data for snap {snap} in {t2_sum-t1_sum} sec')
if add_pd:
    print(f'Added particle data to accretion data for snap {snap} in {t2_part-t1_part} sec')
    
print()
print('******************************************************')
print()