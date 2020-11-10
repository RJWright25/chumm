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
                                                                                                                                                                  
                                                                                                                                                                  
# ParticleTools.py - Python routines to process particle data pertinent to halo accretion.      
# Author: RUBY WRIGHT 

# PREAMBLE
import os
import sys
import time
import numpy as np
import h5py
import astropy.units as u

sys.path.append('/home/rwright/Software/read_eagle/build/lib/python3.7/site-packages/')
import read_eagle

from GenPythonTools import *
from VRPythonTools import *
from STFTools import *

########################### CREATE PARTICLE HISTORIES ###########################

def gen_particle_history_serial(base_halo_data,snaps=None):

    """

    gen_particle_history_serial : function
	----------

    Generate and save particle history data from VELOCIraptor property and particle files.

	Parameters
	----------
    base_halo_data : list of dictionaries
        The halo data list of dictionaries previously generated (by gen_base_halo_data). Should contain information
        re: the type of particle file be reading. 

    snaps : list of ints
        The list of absolute snaps (corresponding to index in base_halo_data) for which we will add 
        particles in halos or subhalos (and save accordingly). The running lists will build on the previous snap. 

	Returns
	----------
    None.

    Saves to file:
    PartHistory_xxx-outname.hdf5 : hdf5 file with datasets

        '/PartTypeX/ParticleIDs' - SORTED particle IDs from simulation.
        '/PartTypeX/ParticleIndex' - Corresponding indices of particles. 
        '/PartTypeX/HostStructure' - Host structure (from STF) of particles. (-1: no host structure)
    
	"""

    # If not given snaps, do for all snaps in base_halo_data (can deal with padded snaps)
    if snaps==None:
        snaps=list(range(len(base_halo_data)))

    # Find which snaps are valid/not padded and which aren't 
    try:
        valid_snaps=[len(base_halo_data[snap].keys())>3 for snap in snaps] #which indices of snaps are valid
        valid_snaps=np.compress(valid_snaps,snaps)
        run_outname=base_halo_data[valid_snaps[0]]['outname']

    except:
        print("Couldn't validate snaps")
        return []

    # Standard names of particle types
    PartNames=['Gas','DM','','','Star','BH']

    # Which simulation type do we have?
    if base_halo_data[valid_snaps[0]]['Part_FileType']=='EAGLE':
        PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
        SimType='EAGLE'
    else:
        PartTypes=[0,1] #Gas, DM
        SimType='OtherHydro'

    # If the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    if not os.path.isdir("part_histories"):
        os.mkdir("part_histories")

    # Iterate through snapshots and flip switches as required
    isnap=0
    for snap in valid_snaps:
        # Initialise output file (will be truncated if already exists)
        outfile_name="part_histories/PartHistory_"+str(snap).zfill(3)+"_"+run_outname+".hdf5"
        if os.path.exists(outfile_name):
            os.remove(outfile_name)
        outfile=h5py.File(outfile_name,'w')

        # Load the EAGLE data for this snapshot
        if SimType=='EAGLE':
            t1=time.time()
            EAGLE_boxsize=base_halo_data[snap]['SimulationInfo']['BoxSize_Comoving']
            EAGLE_Snap=read_eagle.EagleSnapshot(base_halo_data[snap]['Part_FilePath'])
            EAGLE_Snap.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
            t2=time.time()
            print(f"Loaded and sliced EAGLE data from snapshot {snap} in {t2-t1:.2f} sec")

        # Load the halo particle lists for this snapshot for each particle type
        t1=time.time()
        snap_fof_particle_data=get_FOF_particle_lists(base_halo_data,snap)#don't need to add subhalo particles as we have each subhalo separately
        
        # if not type(snap_fof_particle_data)==dict:
        #     print(f'Skipping histories for snap {snap} - could not retrieve FOF particle lists')
        #     continue
        
        # Count halos and particles in each
        n_halos=len(list(snap_fof_particle_data["Particle_IDs"].keys()))
        n_part_ihalo=[len(snap_fof_particle_data["Particle_IDs"][str(ihalo)]) for ihalo in range(n_halos)]
        n_part_tot=np.sum(n_part_ihalo)
        
        # Store IDs, Types, and assign host IDs
        structure_Particles={}
        structure_Particles['ParticleIDs']=np.concatenate([snap_fof_particle_data['Particle_IDs'][str(ihalo)] for ihalo in range(n_halos)])
        structure_Particles['ParticleTypes']=np.concatenate([snap_fof_particle_data['Particle_Types'][str(ihalo)] for ihalo in range(n_halos)]); del snap_fof_particle_data #remove the unnecessary fof data to save memory
        structure_Particles['HostStructureID']=np.concatenate([np.ones(n_part_ihalo[ihalo],dtype='int64')*haloid for ihalo,haloid in enumerate(base_halo_data[snap]['ID'])])

        t2=time.time()
        print(f"Loaded, concatenated and sorted halo particle lists for snap {snap} in {t2-t1:.2f} sec")

        # Map IDs to indices from particle data, and initialise array
        Particle_History_Flags=dict()
        for itype in PartTypes:
            ###############################################
            ##### Step 1: PARTICLE DATA - SORTING IDs #####
            ###############################################

            # Load new snap data
            if SimType=='EAGLE': 
                try:
                    Particle_IDs_Unsorted_itype=EAGLE_Snap.read_dataset(itype,"ParticleIDs")
                except:
                    print(f'No {PartNames[itype]} PartType{itype} particles found at snap {snap} - skipping to next particle type')
                    continue
            else:
                h5py_Snap=h5py.File(base_halo_data[snap]['Part_FilePath'])
                Particle_IDs_Unsorted_itype=h5py_Snap['PartType'+str(itype)+'/ParticleIDs']
            N_Particles_itype=len(Particle_IDs_Unsorted_itype)
            print(f'There are n = {N_Particles_itype} PartType{itype} particles loaded for snap {snap}')

            # Sort IDs and initialise hdf5 file with mapped IDs
            print(f"Mapping IDs to indices for PartType{itype} particles at snap {snap} ...")
            itype_IDs_argsort=np.argsort(Particle_IDs_Unsorted_itype)
            itype_IDs_sorted=Particle_IDs_Unsorted_itype[(itype_IDs_argsort,)];del Particle_IDs_Unsorted_itype
            
            # Dump sorted IDs and particle argsort to hdf5
            outfile.create_dataset(f'/PartType{itype}/ParticleIDs',dtype=np.int64,compression='gzip',data=itype_IDs_sorted)
            outfile.create_dataset(f'/PartType{itype}/ParticleIndex',dtype=np.int32,compression='gzip',data=itype_IDs_argsort);del itype_IDs_argsort
            outfile[f'/PartType{itype}/ParticleIDs'].attrs.create('npart',data=N_Particles_itype,dtype=np.int64)

            ###############################################
            ##### Step 2: FOF DATA - ADDING HOSTS IDs #####
            ###############################################
            t1_fof=time.time()     
            # Initialise hosts to -1            
            itype_hostIDs=np.ones(N_Particles_itype,dtype='int64')-np.int64(2)
            # Find which structure particles are this type
            itype_structure_mask=np.where(structure_Particles["ParticleTypes"]==itype)
            # Find the index of the structure particles in the sorted particle IDs list of this type 
            itype_structure_partindex=binary_search(sorted_list=itype_IDs_sorted,items=structure_Particles["ParticleIDs"][itype_structure_mask]); del itype_IDs_sorted
            # Add host ID for structure particles
            itype_hostIDs[(itype_structure_partindex,)]=structure_Particles['HostStructureID'][itype_structure_mask]
            
            t2_fof=time.time()     
            print(f"Added host halos in {t2_fof-t1_fof:.2f} sec for PartType{itype} particles")

            # Dump structure IDs and particle argsort to hdf5
            outfile.create_dataset(f'/PartType{itype}/HostStructure',dtype=np.int64,compression='gzip',data=itype_hostIDs)
        
        outfile.close()
        isnap+=1#go to next snap

    return None #Don't return anything just save the data 

########################### POSTPROCESS PARTICLE HISTORIES ###########################

def postprocess_particle_history_serial(base_halo_data,path='part_histories'):

    """

    postprocess_particle_history_serial : function
	----------

    Process the existing particle histories to generate a flag for each particle at each snap as to whether it had been processed at any time UP TO this snap. 

	Parameters
	----------
    path : str
        The location of the existing particle histories. 


    Returns
    ----------
    None.

    Saves to file:
    PartHistory_xxx-outname.hdf5 : hdf5 file with EXTRA dataset:

        /PartTypeX/Processed_L1 #no_snaps this particle has been in a halo 

    """

    # Find all particle history files in the provided directory
    ordered_parthistory_files=sorted(os.listdir(path))
    isnap0_skipped=False
    # Iterate through each particle history file
    for isnap,history_filename in enumerate(ordered_parthistory_files):
        # Initialise input file
        infile_file=h5py.File(path+'/'+history_filename,'r+')
        # Find the actual snap
        snap_abs=int(history_filename.split('_')[1])

        # Find which particles are included
        PartTypes_keys=list(infile_file.keys())
        PartTypes=[PartType_keys.split('PartType')[-1] for PartType_keys in PartTypes_keys]
        PartTypes_n={str(itype):infile_file[f'/PartType{itype}/ParticleIDs'].attrs['npart'] for itype in PartTypes}

        # If there is no parttypes then skip this snap
        if len(PartTypes_keys)==0:
            print(f'Skipping snap {snap_abs}')
            isnap0_skipped=True
            continue
        print(PartTypes_keys)

        # If this is the first history snap, initialise the previous processing data structure (and sorted IDs)
        if not (isnap==0 or isnap0_skipped):
            iprev_itype_processing_level=isnap_itype_processing_level
            iprev_itype_sorted_IDs=isnap_itype_sorted_IDs
        else:
            iprev_itype_processing_level={str(itype):np.zeros(PartTypes_n[str(itype)]) for itype in PartTypes}

        ###############################################
        ##### Step 1: Transfer old processing level ###
        ###############################################
        if not (isnap==0 or isnap0_skipped):
            iprev_itype_processing_count=[np.sum(iprev_itype_processing_level[str(itype)]>0) for itype in PartTypes]
            iprev_all_processed_count=int(np.sum(iprev_itype_processing_count))
            iprev_all_processed_IDs=np.zeros(iprev_all_processed_count,dtype=np.int64)
            iprev_all_processed_Types=np.zeros(iprev_all_processed_count,dtype=np.int8)
            iprev_all_processed_L1=np.zeros(iprev_all_processed_count,dtype=np.int8)

            # Iterate through each particle type and add to prev processed list
            iprev_all_processed_index_start=0
            for iitype,itype in enumerate(PartTypes):
                # Count number of previously processed particles of this type and assign indices
                iprev_iitype_processing_count=iprev_itype_processing_count[iitype]
                iprev_all_processed_index_end=iprev_all_processed_index_start+iprev_iitype_processing_count

                # Get mask of previous particles which were processed
                iprev_itype_processed_mask=np.where(isnap_itype_processing_level[str(itype)]>0)

                # Save the IDs, types and level
                print('Processed indices start/end itype: ',iprev_all_processed_index_start,iprev_all_processed_index_end)
                iprev_all_processed_IDs[iprev_all_processed_index_start:iprev_all_processed_index_end]=iprev_itype_sorted_IDs[str(itype)][iprev_itype_processed_mask]
                iprev_all_processed_Types[iprev_all_processed_index_start:iprev_all_processed_index_end]=np.zeros(iprev_iitype_processing_count,dtype=np.int8)+np.int8(itype)
                iprev_all_processed_L1[iprev_all_processed_index_start:iprev_all_processed_index_end]=iprev_itype_processing_level[str(itype)][iprev_itype_processed_mask]
                

                # Progress starting index
                iprev_all_processed_index_start=iprev_all_processed_index_end

            # Delete the full previous data
            del iprev_itype_sorted_IDs
            del iprev_itype_processing_level

            # Load in current sorted IDs and new processing after we can get rid of the old ones
            isnap_itype_sorted_IDs={str(itype):infile_file[f'/PartType{itype}/ParticleIDs'].value for itype in PartTypes}
            isnap_itype_processing_level={str(itype):np.zeros(PartTypes_n[str(itype)]) for itype in PartTypes}

            # Find indices, types of old IDs in new data
            print('Finding current indices and types of previously processed particles ...')
            iprev_processed_parttypes_atsnap,iprev_processed_historyindices_atsnap,x=get_particle_indices(base_halo_data=base_halo_data,
                                                                                                        IDs_taken=iprev_all_processed_IDs,
                                                                                                        IDs_sorted=isnap_itype_sorted_IDs,
                                                                                                        types_taken=iprev_all_processed_Types,
                                                                                                        snap_taken=snap_abs-1,
                                                                                                        snap_desired=snap_abs,
                                                                                                        return_partindices=False)

            # Iterate through each of the processed particles and add to the new array
            ipart_processed=0
            for iprev_processed_type_now,iprev_processed_historyindex_now,iprev_processed_L1 in zip(iprev_processed_parttypes_atsnap,iprev_processed_historyindices_atsnap,iprev_all_processed_L1):
                ipart_processed+=1
                if iprev_processed_type_now>=0:
                    isnap_itype_processing_level[str(iprev_processed_type_now)][iprev_processed_historyindex_now]=iprev_processed_L1
                if ipart_processed%10**5==0:
                    print(f'{ipart_processed/iprev_all_processed_count*100:.1f}% done transferring old processing data')

        else:
            isnap_itype_sorted_IDs={str(itype):infile_file[f'/PartType{itype}/ParticleIDs'].value for itype in PartTypes}
            isnap_itype_processing_level={str(itype):np.zeros(PartTypes_n[str(itype)]) for itype in PartTypes}

        ###############################################
        ##### Step 2: Add new processing level ###
        ###############################################

        # Iterate through each particle type
        print(f'Adding to histories for snap {snap_abs} ...')
        for itype in PartTypes:
            isnap_itype_hoststructures=infile_file[f'/PartType{itype}/HostStructure'].value
            isnap_itype_processed_mask=np.where(isnap_itype_hoststructures>0);del isnap_itype_hoststructures
            print(f'n = {len(isnap_itype_processed_mask[0])} PartType{itype} particles were in structure at snap {snap_abs}')
            isnap_itype_processing_level[str(itype)][isnap_itype_processed_mask]+=1
            try:
                infile_file[f'PartType{itype}'].create_dataset('Processed_L1',data=isnap_itype_processing_level[str(itype)],dtype=np.int8)
            except:
                infile_file[f'PartType{itype}/Processed_L1'][:]=isnap_itype_processing_level[str(itype)]
        
        isnap0_skipped=False
        print(f'Finished adding to histories for snap {snap_abs}')

        infile_file.close()

########################### GET PARTICLE INDICES ###########################

def get_particle_indices(base_halo_data,IDs_taken,IDs_sorted,indices_sorted={},types_taken=None,snap_taken=None,snap_desired=None,return_partindices=True,verbose=False):

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
    
    return_partindices : bool
        Return list of indices in particle data from particle index sorted matching.
    
    verbose : bool
        Print any problems/warnings. 

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
    if len(parttypes)==4:
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
    elif len(parttypes)==3:
        if search_now:#if searching current snap, particles will always be same type
            for itype in parttypes:
                search_types[str(itype)]=[itype]
            search_types[str(-1)]=parttypes
        else:# if past or future
            search_types[str(1)]=[1]#dm particle will always be dm
            search_types[str(-1)]=parttypes#if we don't have type, again have to search them all
            if search_after:# if searching for particles after IDs were taken 
                search_types[str(0)]=[0,4]#gas particles in future could be gas, star
                search_types[str(4)]=[4]#star particles in future could be star 
            else:# if searching for particles before IDs were taken 
                search_types[str(0)]=[0]#gas particles in past can only be gas
                search_types[str(4)]=[4,0]#star particles in past can only be star or gas
                search_types[str(5)]=[4,0]#BH particles in past can be gas, star 
    elif len(parttypes)==2:
        if search_after:# if searching for particles after IDs were taken 
                search_types[str(0)]=[0,4]#gas particles in future could be gas, star 
                search_types[str(1)]=[1]#gas particles in future could be gas, star or BH
                search_types[str(4)]=[4,5]#star particles in future could be star or BH
        else:# if searching for particles before IDs were taken 
                search_types[str(0)]=[0]#gas particles in past can only be gas
                search_types[str(1)]=[0]#gas particles in past can only be gas
                search_types[str(4)]=[4,0]#star particles in past can only be star or gas

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
        if ipart%100000==0 and verbose:
            print(f'{ipart/npart*100:.2f}% done typing')
            pass
        #find new type
        search_in=search_types[str(int(ipart_type))]
        if len(search_in)==1:
            out_type=search_in[0]
        else:
            isearch=0
            for itype in search_in:
                test_index=bisect_left(a=IDs_sorted[f'{itype}'],x=ipart_id,hi=npart_sorted[str(itype)])
                if not test_index>=npart_sorted[str(itype)]:
                    if IDs_sorted[f'{itype}'][test_index]==ipart_id:
                        out_type=itype
                        break
                    else:
                        continue
                else:
                    continue
                isearch=isearch+1
        
        if out_type==-1 and verbose:
            print(f'Warning: couldnt find particle {ipart_id}.')
            print(f'When taken (snap {snap_taken}), the particle was of type {int(ipart_type)} but (at snap {snap_desired}) could not be found in {search_in} lists')

        parttypes_atsnap[ipart]=out_type

    for itype in parttypes:
        if verbose:
            print(f'Getting history indices for itype {itype}')
        itype_mask=np.where(parttypes_atsnap==itype)
        itype_indices=binary_search(items=np.array(IDs_taken)[itype_mask],sorted_list=IDs_sorted[f'{itype}'],check_entries=False,verbose=True)
        historyindices_atsnap[itype_mask]=itype_indices
    
    #Convert types and indices to integet
    parttypes_atsnap=parttypes_atsnap.astype(int)
    historyindices_atsnap=historyindices_atsnap.astype(int)

    #Use the parttypes and history indices to find the particle data indices
    if return_partindices:
        if verbose:
            print(f'Getting particle indices ...')
            partindices_atsnap=np.zeros(npart,dtype=np.int32)
            for iipart,(ipart_type,ipart_historyindex) in enumerate(zip(parttypes_atsnap,historyindices_atsnap)):
                partindices_atsnap[iipart]=indices_sorted[str(ipart_type)][ipart_historyindex]
                if iipart%10000==0:
                    print(f'{iipart/npart*100:.1f} % done getting particle indices')
        else:
            partindices_atsnap=np.array([indices_sorted[str(ipart_type)][ipart_historyindex] for ipart_type,ipart_historyindex in zip(parttypes_atsnap,historyindices_atsnap)],dtype=np.int32)
    else:
        partindices_atsnap=None
    
    #Return types, history indices, and particle data indices
    return parttypes_atsnap,historyindices_atsnap,partindices_atsnap


