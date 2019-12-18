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

sys.path.append('/Users/ruby/Documents/GitHub/CHUMM/')
sys.path.append('/home/rwright/CHUMM/')
from STFTools import *
from AccretionTools import *
from VRPythonTools import *
from GenPythonTools import *
from multiprocessing import Process, cpu_count

############ 0. ARGUMENT PROCESSING ############
# Parse the arguments for accretion calculation
if True:
    parser=argparse.ArgumentParser()
    parser.add_argument('-detailed',type=int, default=1,
                        help='Flag: generate detailed (rather than FOF) accretion data')    
    parser.add_argument('-compression',type=int, default=1,
                        help='Flag: compress the resulting hdf5 datasets')
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
                        help='Flag: generate accretion data')
    parser.add_argument('-sum_ad', type=int,default=1,
                        help='Flag: sum the generated accretion data')
    parser.add_argument('-np', type=int, default=1,
                        help='number of processes to use')
    
    detailed=bool(parser.parse_args().detailed)
    compression=bool(parser.parse_args().compression)
    snap = parser.parse_args().snap
    pre_depth=parser.parse_args().pre
    post_depth=parser.parse_args().post
    halo_index_list_lo=parser.parse_args().hil_lo
    halo_index_list_hi=parser.parse_args().hil_hi
    gen_ad=bool(parser.parse_args().gen_ad)
    sum_ad=bool(parser.parse_args().sum_ad)
    n_processes = parser.parse_args().np
    
    print()
    print('**********************************************************************************************************************')
    print('Arguments parsed:')
    print(f'Generate accretion data: {gen_ad}, sum accretion data: {sum_ad} (at snap {snap})')
    print(f'Detailed accretion data: {detailed} (with n_processes: {n_processes}, compress: {compression}, pre_depth: {pre_depth}, post_depth: {post_depth}, hil_lo: {halo_index_list_lo}, hil_hi {halo_index_list_hi})')
    print('**********************************************************************************************************************')
    print()

    # Use the directory name to get the run name
    run_name=os.getcwd().split('/')[-1]

    # Load in halo data
    base_halo_data=open_pickle(f'B1_HaloData_{run_name}.dat')

    # Process arguments: if we're parsed a halo index list range that isn't -1, then use testing mode
    if halo_index_list_lo==-1:
        test=False
        halo_index_list_massordered=np.argsort(base_halo_data[snap]["Mass_200crit"])[::-1]
        halo_index_lists=gen_mp_indices(indices=halo_index_list_massordered,n=n_processes,test=test)
    else:
        test=True
        halo_index_list=list(range(halo_index_list_lo,halo_index_list_hi))
        halo_index_lists=gen_mp_indices(indices=halo_index_list,n=n_processes,test=test)

    # Determine output directory for this calculation
    if detailed:
        if test:
            calc_dir=f'acc_data/detailed_pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}_test/'
        else:
            calc_dir=f'acc_data/detailed_pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}/'
    else:
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
    kwargs=[{'snap':snap,'halo_index_list':halo_index_lists[iprocess],'pre_depth':pre_depth,'post_depth':post_depth,'compression':compression} for iprocess in range(n_processes)]

    if __name__ == '__main__':
        for iprocess in range(len(kwargs)):
            print(f'Starting process {iprocess}')
            if detailed:
                p=Process(target=gen_accretion_data_detailed_serial, args=(base_halo_data,),kwargs=kwargs[iprocess])
            else:
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

    snap_calcdir=calc_dir+f'snap_{str(snap).zfill(3)}/'
    postprocess_accretion_data(base_halo_data,path=snap_calcdir)

    t2_sum=time.time()


############ 4. PRINT PERFORMANCE ############
# Print performance of above.

print()
print('******************************************************')
print()

if gen_ad:
    print(f'Generated accretion data for snap {snap} in {t2_acc-t1_acc} sec')
# if sum_ad:
#     print(f'Summed accretion data for snap {snap} in {t2_sum-t1_sum} sec')

    
print()
print('******************************************************')
print()