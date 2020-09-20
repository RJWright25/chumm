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

# Author: RUBY WRIGHT 
                                                                                                                                                                                                                                                                                               
# GenData-AccData.py - Generation script for accretion data (any algorithm). 

# Preamble
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py
import time 
import os
import sys
import argparse

sys.path.append('/Users/ruby/Documents/GitHub/CHUMM/')
sys.path.append('/home/rwright/Software/CHUMM/')
from STFTools import *
from AccretionTools import *
from VRPythonTools import *
from GenPythonTools import *
from multiprocessing import Process, cpu_count

############ 0. ARGUMENT PROCESSING ############
# Parse the arguments for accretion calculation
if True:
    parser=argparse.ArgumentParser()
    parser.add_argument('-algorithm',type=int, default=1,
                        help='Flag: 0: standard, -1: fof, 1: r200')
    parser.add_argument('-partdata',type=int, default=1,
                        help='Flag: write particle data for each halo')
    parser.add_argument('-r200_facs_in',type=str, default="",
                        help='list: which factors of r200 to calculate SO accretion to',)
    parser.add_argument('-r200_facs_out',type=str, default="",
                        help='list: which factors of r200 to calculate SO outflow from')
    parser.add_argument('-vmax_facs_in',type=str, default="",
                        help='list: which factors of vmax to cut accretion at')
    parser.add_argument('-vmax_facs_out',type=str, default="",
                        help='list: which factors of vmax to cut outflow at')                 
    parser.add_argument('-snap', type=int,
                        help='snap to calculate accretion/outflow for')
    parser.add_argument('-pre', type=int,default=1,
                        help='accretion snapshot gap')
    parser.add_argument('-post', type=int,default=1,
                        help='fidelity checking gap')
    parser.add_argument('-hil_lo', type=int,default=-1,
                        help='halo index list lower limit (for testing: -1=all, not test)')
    parser.add_argument('-hil_hi', type=int,default=-1,
                        help='halo index list upper limit (for testing, -1=all, not test)')
    parser.add_argument('-hil_cap', type=int,default=1,
                        help='halo index list limit')
    parser.add_argument('-gen_ad', type=int,default=1,
                        help='Flag: generate accretion data')  
    parser.add_argument('-col_ad', type=int,default=1,
                        help='Flag: collate accretion data')  
    parser.add_argument('-np_calc', type=int, default=1,
                        help='number of processes for accretion calc')
    
    algorithm=parser.parse_args().algorithm
    partdata=bool(parser.parse_args().partdata)    

    if not algorithm==-1:
        try:
            r200_facs_in=[float(fac) for fac in parser.parse_args().r200_facs_in.split(',')]
        except:
            r200_facs_in=[]
        try:
            r200_facs_out=[float(fac) for fac in parser.parse_args().r200_facs_out.split(',')]    
        except:
            r200_facs_out=[]
    else:
        r200_facs_in=None
        r200_facs_out=None
        
    try:
        vmax_facs_in=[float(fac) for fac in parser.parse_args().vmax_facs_in.split(',')]
    except:
        vmax_facs_in=[]
    try:
        vmax_facs_out=[float(fac) for fac in parser.parse_args().vmax_facs_out.split(',')]
    except:
        vmax_facs_out=[]

    snap = parser.parse_args().snap
    pre_depth=parser.parse_args().pre
    post_depth=parser.parse_args().post
    halo_index_list_lo=parser.parse_args().hil_lo
    halo_index_list_hi=parser.parse_args().hil_hi
    halo_index_list_cap=parser.parse_args().hil_cap
    gen_ad=bool(parser.parse_args().gen_ad)
    col_ad=bool(parser.parse_args().col_ad)
    n_processes = parser.parse_args().np_calc
    
    print()
    print('**********************************************************************************************************************')
    print('Arguments parsed:')
    print(f'Generate accretion data: {gen_ad} (at snap {snap})')
    if algorithm==-1:
        print(f'Algorithm: FOF only')
    elif algorithm==0:
        print(f'Algorithm: EAGLE FOF and SO')
    else:
        print(f'Algorithm: R200 only')

    print(f'with n_processes: {n_processes}, write particle data: {partdata}, pre_depth: {pre_depth}, post_depth: {post_depth}, hil_lo: {halo_index_list_lo}, hil_hi {halo_index_list_hi})')
    print(f'with r200_facs_in = {r200_facs_in}')
    print(f'with r200_facs_out = {r200_facs_out}')
    print(f'with vmax_facs_in = {vmax_facs_in}')
    print(f'with vmax_facs_out = {vmax_facs_out}')
    print('**********************************************************************************************************************')
    print()

    # Use the directory name to get the run name
    run_name=os.getcwd().split('/')[-1]

    # Load in halo data
    base_halo_data=open_pickle(f'B1_HaloData_{run_name}.dat')

    # Process arguments: if we're parsed a halo index list range that isn't -1, then use testing mode
    if halo_index_list_lo==-1:
        test=False
        halo_index_list_massordered=np.argsort(base_halo_data[snap]["Mass_200crit"])
        halo_index_lists=gen_mp_indices(indices=halo_index_list_massordered,n=n_processes,test=test)
    else:
        test=True
        halo_index_list=list(range(halo_index_list_lo,halo_index_list_hi))
        halo_index_lists=gen_mp_indices(indices=halo_index_list,n=n_processes,test=test)
    
    # Determine output directory for this calculation
    if algorithm==-1:
        if test:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}_FOFonly_test/'
        else:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}_FOFonly/'
    elif algorithm==0:
        if test:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}_test/'
        else:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}/'
    else:
        if test:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}_R200only_test/'
        else:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(n_processes).zfill(2)}_R200only/'

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
    if not algorithm==-1:
        kwargs=[{'snap':snap,
                'halo_index_list':halo_index_lists[iprocess],
                'pre_depth':pre_depth,
                'post_depth':post_depth,
                'write_partdata':partdata,
                'r200_facs_in':r200_facs_in,
                'r200_facs_out':r200_facs_out,
                'vmax_facs_in':vmax_facs_in,
                'vmax_facs_out':vmax_facs_out} 
                for iprocess in range(n_processes)]
    else:
        kwargs=[{'snap':snap,
                'halo_index_list':halo_index_lists[iprocess],
                'pre_depth':pre_depth,
                'post_depth':post_depth,
                'write_partdata':partdata,
                'vmax_facs_in':vmax_facs_in,
                'vmax_facs_out':vmax_facs_out} 
                for iprocess in range(n_processes)]

    if __name__ == '__main__':
        for iprocess in range(len(kwargs)):
            print(f'Starting process {iprocess}')
            if algorithm==0:
                p=Process(target=gen_accretion_data_eagle, args=(base_halo_data,),kwargs=kwargs[iprocess])
            elif algorithm==-1:
                p=Process(target=gen_accretion_data_fof, args=(base_halo_data,),kwargs=kwargs[iprocess])
            else:
                p=Process(target=gen_accretion_data_r200, args=(base_halo_data,),kwargs=kwargs[iprocess])
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

    t2_acc=time.time()

    print()
    print('***************************************************')
    print(f'Generated accretion data in {t2_acc-t1_acc:.2f} sec')
    print('***************************************************')
    print()

############ 2. COLLATE ACCRETION DATA ############
# This is run in serial, collates the ffiles generated above. 

if col_ad:
    t1_col=time.time() 
    output_dir=calc_dir+f'snap_{str(snap).zfill(3)}/'
    postprocess_accretion_data_serial(base_halo_data,output_dir)        
    t2_col=time.time()

