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
                                                                                                                                                                  
                                                                                                                                                                  
# AccretionTools.py - Python routines to read the outputs of STF algorithms (presently VELOCIraptor and TreeFrog) and simulation data and generate halo accretion data.                     
# Author: RUBY WRIGHT 

# PREAMBLE
import os
import numpy as np
import h5py
import astropy.units as u
import read_eagle
import time

from GenPythonTools import *

#GET PARTICLE INDICES

def get_particle_indices(base_halo_data,IDs_sorted,indices_sorted,IDs_taken,types_taken=None,snap_taken=None,snap_desired=None):

    """
    get_particle_indices : function
	----------

    Given a list of particle IDs, find their index and type in particle data at the desired snap.

	Parameters
	----------
    base_halo_data : list of dict
        Base halo data from gen_base_halo_data.

    IDs_sorted : dict of lists
        Lists of sorted particle IDs from particle histories at the desired snap. 

    indices_sorted : dict of lists
        Lists of sorted particle indices from particle histories at the desired snap. 

    IDs_taken : list of int
        The IDs to search for at the desired snap. 

    types_taken : list of int
        The corresponding types of the IDs above (if available). 

    snap_taken : int
        The snap at which the IDs were taken.

    snap_desired : int
        The snap at which to find the indices.

    Returns
	----------
    Tuple of parttypes_desired, historyindices_desired, partindices_desired

    indices : list of int
        The indices in particle data of the IDs at snap_desired. 

    types : list of int
        The corresponding types for indices above. 

    """
    #Number of particle IDs parsed
    npart=len(IDs_taken)

    #Indicating whether we are searching for future or past particles
    search_after=snap_desired>snap_taken #flag as to whether index is desired after the ID was taken
    search_now=snap_desired==snap_taken #flag as to whether index is desired at the snap the ID was taken

    #if can't find particle, give it the index -1
    indices_sorted['-1']=[-1]

    #Particle types from dictionary of particle histories
    parttype_keys=list(IDs_sorted.keys())
    parttypes=sorted([int(parttype_key) for parttype_key in parttype_keys])
    
    #For each particle type, determine which lists will need to be searched
    search_types={}
    if len(parttypes)>2:
        if search_now:#if searching current snap, particles will always be same type
            for itype in parttypes:
                search_types[str(itype)]=[itype]
            search_types[str(-1)]=parttypes
        else:# if past or future
            search_types[str(1)]=[1]#dm particle will always be dm
            search_types[str(-1)]=parttypes#if we don't have type, again have to search them all
            if search_after:# if searching for particles after IDs were taken 
                search_types[str(0)]=[0,4,5]#gas particles in future could be gas, star or BH
                search_types[str(4)]=[4,5]#star particles in future could be star or BH
                search_types[str(5)]=[5]#BH particles in future could be only BH
            else:# if searching for particles before IDs were taken 
                search_types[str(0)]=[0]#gas particles in past can only be gas
                search_types[str(4)]=[4,0]#star particles in past can only be star or gas
                search_types[str(5)]=[4,0,5]#BH particles in past can be gas, star or BH
    else:
        search_types={'0':[0],'1':[1],'-1':[0,1]}
    
    #if not given types, set all to -1
    try:
        len(types_taken)
    except:
        types_taken=np.array(np.zeros(npart)-1,dtype=np.int8)

    #Initialising outputs
    historyindices_atsnap=np.zeros(npart)-1
    partindices_atsnap=np.zeros(npart)-1
    parttypes_atsnap=np.zeros(npart)-1
    
    #Iterate through particles and type, history index and partdata index at the desited snap
    ipart=0
    npart_sorted={str(itype):len(IDs_sorted[f'{itype}']) for itype in parttypes}

    for ipart,ipart_id,ipart_type in zip(list(range(npart)),IDs_taken,types_taken):
        out_type=-1
        if ipart%500==0:
            # print(f'{ipart/npart*100:.2f}% done typing')
            pass
        #find new type
        search_in=search_types[str(int(ipart_type))]
        if len(search_in)==1:
            out_type=search_in[0]
        else:
            isearch=0
            for itype in search_in:
                test_index=bisect_left(a=IDs_sorted[f'{itype}'],x=ipart_id,hi=npart_sorted[str(itype)])
                if IDs_sorted[f'{itype}'][test_index]==ipart_id:
                    out_type=itype
                    break
                else:
                    continue
                isearch=isearch+1
        
        if out_type==-1:
            print(f'Warning: couldnt find particle {ipart_id}.')
            print(f'When taken (snap {snap_taken}), the particle was of type {ipart_type} but (at snap {snap_desired}) could not be found in {search_in} lists')

        parttypes_atsnap[ipart]=out_type

    for itype in parttypes:
        itype_mask=np.where(parttypes_atsnap==itype)
        itype_indices=binary_search(items=np.array(IDs_taken)[itype_mask],sorted_list=IDs_sorted[f'{itype}'],check_entries=False)
        historyindices_atsnap[itype_mask]=itype_indices
    
    #Convert types and indices to integet
    parttypes_atsnap=parttypes_atsnap.astype(int)
    historyindices_atsnap=historyindices_atsnap.astype(int)

    #Use the parttypes and history indices to find the particle data indices
    partindices_atsnap=np.array([indices_sorted[str(ipart_type)][ipart_historyindex] for ipart_type,ipart_historyindex in zip(parttypes_atsnap,historyindices_atsnap)],dtype=int)

    #Return types, history indices, and particle data indices
    return parttypes_atsnap,historyindices_atsnap,partindices_atsnap


