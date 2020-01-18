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
import numpy as np
import h5py
import astropy.units as u
import read_eagle
import time

from GenPythonTools import *
from VRPythonTools import *
from STFTools import *

########################### CREATE PARTICLE HISTORIES ###########################

def gen_particle_history_serial(base_halo_data,snaps=None):

    """

    gen_particle_history_serial : function
	----------

    Generate and save particle history data from velociraptor property and particle files.

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

        '/PartTypeX/PartID' - SORTED particle IDs from simulation.
        '/PartTypeX/PartIndex' - Corresponding indices of particles. 
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

        # Load the Halo particle lists for this snapshot for each particle type
        t1=time.time()
        snap_fof_particle_data=get_FOF_particle_lists(base_halo_data,snap)#don't need to add subhalo particles as we have each subhalo separately
        
        if not type(snap_fof_particle_data)==dict:
            print(f'Skipping histories for snap {snap} - could not retrieve FOF particle lists')
            continue

        n_halos=len(list(snap_fof_particle_data["Particle_IDs"].keys()))
        n_part_ihalo=[len(snap_fof_particle_data["Particle_IDs"][str(ihalo)]) for ihalo in range(n_halos)]
        n_part_tot=np.sum(n_part_ihalo)
        
        ipart_IDs=np.concatenate([snap_fof_particle_data['Particle_IDs'][str(ihalo)] for ihalo in range(n_halos)])
        ipart_Types=np.concatenate([snap_fof_particle_data['Particle_Types'][str(ihalo)] for ihalo in range(n_halos)])
        ipart_hostIDs=np.concatenate([np.ones(n_part_ihalo[ihalo],dtype='int64')*haloid for ihalo,haloid in enumerate(base_halo_data[snap]['ID'])])
        
        structure_Particles={'ParticleIDs':ipart_IDs,'ParticleTypes':ipart_Types,'HostStructureID':ipart_hostIDs}
        structure_Particles_bytype={str(itype):{} for itype in PartTypes}
        
        for itype in PartTypes:
            itype_mask=np.where(structure_Particles["ParticleTypes"]==itype)
            structure_Particles_bytype[str(itype)]["ParticleIDs"]=structure_Particles["ParticleIDs"][itype_mask]
            structure_Particles_bytype[str(itype)]["HostStructureID"]=structure_Particles["HostStructureID"][itype_mask]
            
        t2=time.time()
        print(f"Loaded, concatenated and sorted halo particle lists for snap {snap} in {t2-t1:.2f} sec")
        if '376' in run_outname:
            print(f"There are {n_part_tot} particles in all FOFs ({np.nansum(n_part_tot)/(2*376**3)*100:.2f}%)")
        else:
            print(f"There are {n_part_tot} particles in all FOFs ({np.nansum(n_part_tot)/(2*752**3)*100:.2f}%)")

        # Map IDs to indices from particle data, and initialise array
        Particle_History_Flags=dict()
        
        for itype in PartTypes:
            t1=time.time()
            # Load new snap data
            if SimType=='EAGLE': 
                try:
                    Particle_IDs_Unsorted_itype=EAGLE_Snap.read_dataset(itype,"ParticleIDs")
                    print(f'{PartNames[itype]} IDs loaded')
                    print(f'There are n = {len(Particle_IDs_Unsorted_itype)} PartType{itype} particles loaded for snap {snap}')
                except:
                    print(f'No {PartNames[itype]} IDs found')
                    Particle_IDs_Unsorted_itype=[]

                N_Particles_itype=len(Particle_IDs_Unsorted_itype)
            else:
                h5py_Snap=h5py.File(base_halo_data[snap]['Part_FilePath'])
                Particle_IDs_Unsorted_itype=h5py_Snap['PartType'+str(itype)+'/ParticleIDs']
                N_Particles_itype=len(Particle_IDs_Unsorted_itype)

            # Initialise flag data structure with mapped IDs
            print(f"Mapping IDs to indices for all {PartNames[itype]} particles at snap {snap} ...")
            Particle_History_Flags[str(itype)]={"ParticleIDs_Sorted":np.sort(Particle_IDs_Unsorted_itype),
                                                "ParticleIndex_Original":np.argsort(Particle_IDs_Unsorted_itype),
                                                "HostStructureID":np.ones(N_Particles_itype,dtype='int64')-np.int64(2)}
            t2=time.time()
            print(f"Mapped IDs to indices for all {PartNames[itype]} particles at snap {snap} in {t2-t1:.2f} sec")
            
            # Flip switches of new particles
            ipart_switch=0
            all_Structure_IDs_itype=structure_Particles_bytype[str(itype)]["ParticleIDs"]
            all_Structure_HostStructureID_itype=np.int64(structure_Particles_bytype[str(itype)]["HostStructureID"])
            all_Structure_IDs_itype_partindex=binary_search(sorted_list=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"],items=all_Structure_IDs_itype)
            
            print("Adding host indices ...")
            Particle_History_Flags[str(itype)]["HostStructureID"][(all_Structure_IDs_itype_partindex,)]=all_Structure_HostStructureID_itype
            print(f"Added host halos in {t2-t1:.2f} sec for {PartNames[itype]} particles")

        # Done with processing, now save to hdf5 file with PartType groups
        print(f'Dumping data to file')
        for itype in PartTypes:
            outfile.create_dataset(f'/PartType{itype}/ParticleIDs',dtype=np.int64,compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"])
            outfile.create_dataset(f'/PartType{itype}/ParticleIndex',dtype=np.int32,compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIndex_Original"])
            outfile.create_dataset(f'/PartType{itype}/HostStructure',dtype=np.int64,compression='gzip',data=Particle_History_Flags[str(itype)]["HostStructureID"])
        outfile.close()
        t2=time.time()
        print(f'Dumped snap {snap} data to file in {t2-t1} sec')
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

    ordered_parthistory_files=sorted(os.listdir(path))

    for isnap,history_filename in enumerate(ordered_parthistory_files):
        
        infile_file=h5py.File(path+'/'+history_filename,'r+')
        snap_abs=int(history_filename.split('_')[1])
        PartTypes_keys=list(infile_file.keys())
        PartTypes=[PartType_keys.split('PartType')[-1] for PartType_keys in PartTypes_keys]

        print(f'Loading in existing histories data for snap {snap_abs}')
        Part_Histories_IDs={str(parttype):infile_file["PartType"+str(parttype)+'/ParticleIDs'].value for parttype in PartTypes}
        Part_Histories_Indices={str(parttype):infile_file["PartType"+str(parttype)+'/ParticleIndex'].value for parttype in PartTypes}

        ##### DARK MATTER
        print(f'Processing DM data for snap {snap_abs}...')
        t1=time.time()
        try:
            current_hosts_DM=infile_file["PartType1/HostStructure"].value##ordered by ID
            
        except:
            print(f'Couldnt retrieve DM data for isnap {isnap}')
            continue
        if isnap==0:#initialise our arrays
            n_part_DM=len(current_hosts_DM)
            DM_flags=np.array(np.zeros(n_part_DM),dtype=np.int8)

        indices_in_structure=np.where(current_hosts_DM>0)
        DM_flags[indices_in_structure]=DM_flags[indices_in_structure]+1
        try:
            infile_file["PartType1"].create_dataset("Processed_L1",data=DM_flags,compression='gzip',dtype=np.uint8)
        except:
            infile_file["PartType1"]['Processed_L1'][:]=DM_flags
        t2=time.time()
        print(f'Finished with DM for snap {snap_abs} in {t2-t1:.1f} sec')

        ##### GAS
        print(f'Processing gas data for snap {snap_abs}...')
        t1=time.time()
        if isnap==0:#initialise our arrays
            print('Initialising gas processing data (first snap)')
            current_IDs_gas=Part_Histories_IDs[str(0)]
            current_indices_gas=Part_Histories_Indices[str(0)]
            current_hosts_gas=infile_file["PartType0/HostStructure"].value ##ordered by ID
            n_part_gas_now=len(current_IDs_gas)
            n_part_gas_prev=n_part_gas_now
            gas_flags_L1=np.array(np.zeros(n_part_gas_now),dtype=np.int8)
            
        else:
            print('Loading previous gas processing data')
            prev_IDs_gas=current_IDs_gas
            prev_hosts_gas=current_hosts_gas
            current_IDs_gas=Part_Histories_IDs[str(0)]
            current_indices_gas=Part_Histories_Indices[str(0)]
            current_hosts_gas=infile_file["PartType0/HostStructure"].value
            n_part_gas_now=len(current_IDs_gas)
            n_part_gas_prev=len(prev_IDs_gas)
        
        delta_particles=n_part_gas_prev-n_part_gas_now
        
        t1=time.time()
        if delta_particles<1:
            print(f"Gas particle count changed by {delta_particles} - first processing sum")
            indices_in_structure=np.where(current_hosts_gas>0)
            gas_flags_L1[indices_in_structure]=gas_flags_L1[indices_in_structure]+1

        else:
            print(f"Gas particle count changed by {delta_particles} - carrying over old information")
            gas_flags_L1_old=gas_flags_L1
            gas_flags_L1=np.array(np.zeros(n_part_gas_now),dtype=np.int8)

            print('Finding old processed particles ...')
            processed_old_indices=np.where(gas_flags_L1_old>0)
            processed_old_IDs=prev_IDs_gas[processed_old_indices]
            processed_old_flag=gas_flags_L1_old[processed_old_indices]

            


            #find these IDs at this snap
            parttypes_atsnap,historyindices_atsnap,partindices_atsnap=get_particle_indices(base_halo_data,IDs_sorted=Part_Histories_IDs,
                                                                                                          indices_sorted=Part_Histories_Indices,
                                                                                                          IDs_taken=processed_old_IDs,
                                                                                                          types_taken=np.zeros(len(processed_old_IDs)),
                                                                                                          snap_taken=snap_abs-1,
                                                                                                          snap_desired=snap_abs)

            # #sanity check
            # prev_IDs_totransferfor=processed_old_IDs
            # new_indices_righttype=historyindices_atsnap[np.where(parttypes_atsnap==0)]
            # new_IDs_totransferto=Part_Histories_IDs[str(0)][(new_indices_righttype,)]
            # print('Comparing old IDs with processing data to the IDs this data will be transferred to')
            # print(np.column_stack((prev_IDs_totransferfor[np.where(parttypes_atsnap==0)],new_IDs_totransferto)))
            # print(f'mismatches: {np.nansum(prev_IDs_totransferfor[np.where(parttypes_atsnap==0)]!=new_IDs_totransferto)}')

            iipart_processed=0
            #transferring old processing data
            for ipart_prevprocessing,ipart_prevhistoryindex,ipart_newhistoryindex,ipart_newtype in zip(gas_flags_L1_old,processed_old_indices[0],historyindices_atsnap,parttypes_atsnap):
                if ipart_newtype==0:#if particle still gas
                    gas_flags_L1[ipart_newhistoryindex]=ipart_prevprocessing
                else:
                    pass
                iipart_processed=iipart_processed+1
        
            processed_new_IDs=Part_Histories_IDs

            print('Finding new processed particles ...')
            #adding new processing data
            processed_new_indices=np.where(current_hosts_gas>0)
            gas_flags_L1[processed_new_indices]=gas_flags_L1[processed_new_indices]+1

        try:
            infile_file["PartType0"].create_dataset("Processed_L1",data=gas_flags_L1,compression='gzip',dtype=np.uint8)
        except:
            infile_file["PartType0"]['Processed_L1'][:]=gas_flags_L1
            
        t2=time.time()
        print(f'Finished with gas for snap {snap_abs} in {t2-t1:.1f} sec')

        infile_file.close()

########################### GET PARTICLE INDICES ###########################

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
                if not test_index>=npart_sorted[str(itype)]:
                    if IDs_sorted[f'{itype}'][test_index]==ipart_id:
                        out_type=itype
                        break
                    else:
                        continue
                else:
                    continue
                isearch=isearch+1
        
        if out_type==-1:
            print(f'Warning: couldnt find particle {ipart_id}.')
            print(f'When taken (snap {snap_taken}), the particle was of type {int(ipart_type)} but (at snap {snap_desired}) could not be found in {search_in} lists')

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


