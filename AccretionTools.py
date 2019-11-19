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
from VRPythonTools import *
from STFTools import *
from pandas import DataFrame as df

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
    PartNames=['gas','DM','','','star','BH']

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
            print(f"Loaded and sliced EAGLE data from snapshot {snap} in {t2-t1} sec")

        # Load the Halo particle lists for this snapshot for each particle type
        t1=time.time()
        snap_Halo_Particle_Lists=get_particle_lists(base_halo_data[snap],include_unbound=True,add_subparts_to_fofs=False)#don't need to add subhalo particles as we have each subhalo separately
        n_halos=len(snap_Halo_Particle_Lists["Particle_IDs"])
        n_halo_particles=[len(snap_Halo_Particle_Lists["Particle_IDs"][ihalo]) for ihalo in range(n_halos)]
        allhalo_Particle_hosts=np.concatenate([np.ones(n_halo_particles[ihalo],dtype='int64')*haloid for ihalo,haloid in enumerate(base_halo_data[snap]['ID'])])
        structure_Particles=df({'ParticleIDs':np.concatenate(snap_Halo_Particle_Lists['Particle_IDs']),'ParticleTypes':np.concatenate(snap_Halo_Particle_Lists['Particle_Types']),"HostStructureID":allhalo_Particle_hosts},dtype=np.int64).sort_values(["ParticleIDs"])
        structure_Particles_bytype={str(itype):np.array(structure_Particles[["ParticleIDs","HostStructureID"]].loc[structure_Particles["ParticleTypes"]==itype]) for itype in PartTypes}
        n_structure_particles=np.sum([len(structure_Particles_bytype[str(itype)][:,0]) for itype in PartTypes])
        t2=time.time()
        print(f"Loaded, concatenated and sorted halo particle lists for snap {snap} in {t2-t1} sec")
        print(f"There are {np.sum(n_structure_particles)} particles in structure")

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
            Particle_History_Flags[str(itype)]={"ParticleIDs_Sorted":np.sort(Particle_IDs_Unsorted_itype),"ParticleIndex_Original":np.argsort(Particle_IDs_Unsorted_itype),"HostStructureID":np.ones(N_Particles_itype,dtype='int64')-np.int64(2)}
            t2=time.time()
            print(f"Mapped IDs to indices for all {PartNames[itype]} particles at snap {snap} in {t2-t1} sec")
            
            # Flip switches of new particles
            print("Adding host indices ...")
            t1=time.time()
            ipart_switch=0
            all_Structure_IDs_itype=structure_Particles_bytype[str(itype)][:,0]
            all_Structure_HostStructureID_itype=structure_Particles_bytype[str(itype)][:,1]
            all_Structure_IDs_itype_partindex=binary_search(sorted_list=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"],items=all_Structure_IDs_itype)
            for ipart_switch, ipart_index in enumerate(all_Structure_IDs_itype_partindex):#for each particle in structure, add its host structure to the array (if not in structure, HostStructure=-1)
                if ipart_switch%100000==0:
                    print(ipart_switch/len(all_Structure_IDs_itype_partindex)*100,f'% done adding host halos for {PartNames[itype]} particles')
                Particle_History_Flags[str(itype)]["HostStructureID"][ipart_index]=np.int64(all_Structure_HostStructureID_itype[ipart_switch])
            t2=time.time()
            print(f"Added host halos in {t2-t1} sec for {PartNames[itype]} particles")

        # Done with processing, now save to hdf5 file with PartType groups
        print(f'Dumping data to file')
        t1=time.time()
        for itype in PartTypes:
            dset_write=outfile.create_dataset(f'/PartType{itype}/ParticleIDs',dtype=np.int64,compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"])
            dset_write=outfile.create_dataset(f'/PartType{itype}/ParticleIndex',dtype=np.int32,compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIndex_Original"])
            dset_write=outfile.create_dataset(f'/PartType{itype}/HostStructure',dtype=np.int64,compression='gzip',data=Particle_History_Flags[str(itype)]["HostStructureID"])
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
    PartHistory_xxx-outname.hdf5 : hdf5 file with datasets

        /PartTypeX/Processed_L1 #no_snaps this particle has been in a halo with substructure
        /PartTypeX/Processed_L2 #no_snaps this particle has been in a halo with NO substructure (<Processed_L1)
        /PartTypeX/HostStructure
        /PartTypeX/ParticleIDs
        /PartTypeX/ParticleIndex

    """

    ordered_parthistory_files=sorted(os.listdir(path))

    for isnap,history_filename in enumerate(ordered_parthistory_files):
        
        infile_file=h5py.File(path+'/'+history_filename,'r+')
        snap_abs=int(history_filename.split('_')[1])

        print(f"Arranging halo data for snap {snap_abs}...")
        t1=time.time()
        halo_l2_IDs=set([base_halo_data[snap_abs]["ID"][ihalo] for ihalo in np.where(base_halo_data[snap_abs]["numSubStruct"]==0)[0]])# no substructure
        t2=time.time()
        print(f'Done in {t2-t1}')

        ##### DARK MATTER
        print(f'Processing DM Data for snap {snap_abs}...')
        t1=time.time()
        current_hosts_DM=infile_file["PartType1/HostStructure"].value##ordered by ID
        if isnap==0:#initialise our arrays
            n_part_DM=len(current_hosts_DM)
            DM_flags_L1=np.array(np.zeros(n_part_DM),dtype=np.int8)
            DM_flags_L2=np.array(np.zeros(n_part_DM),dtype=np.int8)

        indices_in_structure=np.where(current_hosts_DM>0)[0]
        iipart=0
        for ipart in indices_in_structure:
            iipart=iipart+1
            if iipart%100000==0:
                print(np.round(iipart/len(indices_in_structure)*100,2),'% done adding flags for DM particles')
            DM_flags_L1[ipart]=DM_flags_L1[ipart]+1
            host_ID=current_hosts_DM[ipart]
            if host_ID in halo_l2_IDs:
                DM_flags_L2[ipart]=DM_flags_L2[ipart]+1
        
        try:
            infile_file["PartType1"].create_dataset("Processed_L1",data=DM_flags_L1,compression='gzip',dtype=np.uint8)
            infile_file["PartType1"].create_dataset("Processed_L2",data=DM_flags_L2,compression='gzip',dtype=np.uint8)
        except:
            infile_file["PartType1"]['Processed_L1'][:]=DM_flags_L1
            infile_file["PartType1"]['Processed_L2'][:]=DM_flags_L2

        t2=time.time()
        print(f'Finished with DM for snap {snap_abs} in {t2-t1}')

        ##### GAS
        print(f'Processing gas Data for snap {snap_abs}...')
        t1=time.time()
        if isnap==0:#initialise our arrays
            current_IDs_gas=infile_file["PartType0/ParticleIDs"].value
            current_hosts_gas=infile_file["PartType0/HostStructure"].value##ordered by ID
            n_part_gas_now=len(current_IDs_gas)
            n_part_gas_prev=n_part_gas_now
            gas_flags_L1=np.array(np.zeros(n_part_gas_now),dtype=np.int8)
            gas_flags_L2=np.array(np.zeros(n_part_gas_now),dtype=np.int8)
        else:
            prev_IDs_gas=current_IDs_gas
            prev_hosts_gas=current_hosts_gas
            current_IDs_gas=infile_file["PartType0/ParticleIDs"].value
            current_hosts_gas=infile_file["PartType0/HostStructure"].value##ordered by ID
            n_part_gas_now=len(current_IDs_gas)
            n_part_gas_prev=len(prev_IDs_gas)
        
        delta_particles=n_part_gas_prev-n_part_gas_now
        
        if delta_particles<1:
            print("No change in gas particle count since last snap (i.e. first snap)")
            indices_in_structure=np.where(current_hosts_gas>0)[0]
            iipart=0
            for ipart in indices_in_structure:
                iipart=iipart+1
                host_ID=current_hosts_gas[ipart]
                if iipart%100000==0:
                    print(np.round(iipart/len(indices_in_structure)*100,2),'% done adding flags for gas particles')
                gas_flags_L1[ipart]=gas_flags_L1[ipart]+1
                if host_ID in halo_l2_IDs:
                    gas_flags_L2[ipart]=gas_flags_L2[ipart]+1
        else:
            print(f"Gas particle count changed by {delta_particles} - carrying over old information")
            gas_flags_L1_old=gas_flags_L1
            gas_flags_L2_old=gas_flags_L2
            gas_flags_L1=np.array(np.zeros(n_part_gas_now),dtype=np.int8)
            gas_flags_L2=np.array(np.zeros(n_part_gas_now),dtype=np.int8)

            print('Finding old processed particles ...')
            particles_prev_processed_L1=[(prev_IDs_gas[ipart],gas_flags_L1_old[ipart]) for ipart in np.where(gas_flags_L1_old>0)[0]]
            particles_prev_processed_L2=[(prev_IDs_gas[ipart],gas_flags_L2_old[ipart]) for ipart in np.where(gas_flags_L2_old>0)[0]]
            
            ipart_L1=0
            for ipart_prevID, ipart_L1_level in particles_prev_processed_L1:
                ipart_L1=ipart_L1+1
                if ipart_L1%100000==0:  
                    print(f'{np.round(ipart_L1/len(particles_prev_processed_L1)*100,2)}% done with carrying over L1 flags for gas')
                ipart_currentindex=binary_search(items=[ipart_prevID],sorted_list=current_IDs_gas,check_entries=True)[0]
                if ipart_currentindex>-1:#if particle found
                    gas_flags_L1[ipart_currentindex]=ipart_L1_level
                else:
                    pass
            
            ipart_L2=1
            for ipart_prevID, ipart_L2_level in particles_prev_processed_L2:
                ipart_L2=ipart_L2+1
                if ipart_L2%100000==0:
                    print(f'{np.round(ipart_L2/len(particles_prev_processed_L2)*100,2)}% done with carrying over L2 flags')
                ipart_currentindex=binary_search(items=[ipart_prevID],sorted_list=current_IDs_gas,check_entries=True)[0]
                if ipart_currentindex>-1:#if particle found
                    gas_flags_L2[ipart_currentindex]=ipart_L2_level
                else:
                    pass


            print(f"Now adding new flags for gas particles")
            indices_in_structure=np.where(current_hosts_gas>0)[0]
            iipart=0
            for ipart in indices_in_structure:
                iipart=iipart+1
                host_ID=current_hosts_gas[ipart]
                if iipart%10000==0:
                    print(np.round(iipart/len(indices_in_structure)*100,2),'% done adding flags for gas particles')
                gas_flags_L1[ipart]=gas_flags_L1[ipart]+1
                if host_ID in halo_l2_IDs:
                    gas_flags_L2[ipart]=gas_flags_L2[ipart]+1
        print('About to save: L1!=L2?', np.sum(gas_flags_L2!=gas_flags_L1))

        try:
            infile_file["PartType0"].create_dataset("Processed_L1",data=gas_flags_L1,compression='gzip',dtype=np.uint8)
            infile_file["PartType0"].create_dataset("Processed_L2",data=gas_flags_L2,compression='gzip',dtype=np.uint8)
        except:
            infile_file["PartType0"]['Processed_L1'][:]=gas_flags_L1
            infile_file["PartType0"]['Processed_L2'][:]=gas_flags_L2
            
        t2=time.time()
        print(f'Finished with Gas for snap {snap_abs} in {t2-t1}')

        infile_file.close()

########################### GET PARTICLE INDICES ###########################

def get_particle_indices(base_halo_data,SortedIDs,SortedIndices,PartIDs,PartTypes,snap_taken,snap_desired):

    """
    get_particle_indices : function
	----------

    Given a list of particle IDs, find their index and type in particle data at the desired snap.

	Parameters
	----------
    base_halo_data : list of dict
        Base halo data from gen_base_halo_data.

    SortedIDs : dict of lists
        Lists of sorted particle IDs from particle histories. 

    SortedIndices : dict of lists
        Lists of sorted particle indices from particle histories. 

    PartIDs : list of int
        The IDs to search for at the desired snap. 

    PartTypes : list of int
        The corresponding types of the IDs above. 

    snap_taken : int
        The snap at which the IDs were taken.

    snap_desired : int
        The snap at which to find the indices.

    Returns
	----------
    Tuple of Indices, Types.

    Indices : list of int
        The indices in particle data of the IDs at snap_desired. 

    Types : list of int
        The corresponding types for indices above. 


    """
    npart=len(PartIDs)
    search_after=snap_desired>snap_taken #flag as to whether index is desired after the ID was taken
    search_now=snap_desired==snap_taken #flag as to whether index is desired at the snap the ID was taken

    parttype_keys=list(SortedIDs.keys())
    parttypes=[int(parttype_key) for parttype_key in parttype_keys]
    
    search_types={}
    if len(parttypes)>2:
        if search_now:#if searching current snap, particles will always be same type
            for itype in parttypes:
                search_types[itype]=[itype]
        else:# if past or future
            search_types[str(1)]=[1]#dm particle will always be dm
            if search_after:# if searching for particles after IDs were taken 
                search_types[str(0)]=[0,4,5]#gas particles in future could be gas, star or BH
                search_types[str(4)]=[4,5]#star particles in future could be star or BH
                search_types[str(5)]=[5]#BH particles in future could be only BH
            else:# if searching for particles before IDs were taken 
                search_types[str(0)]=[0]#gas particles in past can only be gas
                search_types[str(4)]=[4,0]#star particles in past can only be star or gas
                search_types[str(5)]=[4,0,5]#BH particles in past can be gas, star or BH
    else:
        search_types={'0':[0],'1':[1]}

    historyindices_atsnap=np.zeros(npart)-1
    partindices_atsnap=np.zeros(npart)-1
    parttypes_atsnap=np.zeros(npart)-1

    print(f'Search types:')
    print(search_types)
    ipart=0
    for ipart,ipart_id,ipart_type in zip(list(range(npart)),PartIDs,PartTypes):
        #find new type
        search_in=search_types[str(ipart_type)]
        if len(search_in)==1:
            out_type=search_in[0]
        else:
            for itype in search_in:
                test_index=binary_search(items=[ipart_id],sorted_list=SortedIDs[f'{itype}'],check_entries=True)[0]
                if test_index>=0:
                    out_type=itype
                    break
                else:
                    continue
        parttypes_atsnap[ipart]=out_type

    for itype in parttypes:
        itype_mask=np.where(parttypes_atsnap==itype)
        itype_indices=binary_search(items=np.array(PartIDs)[itype_mask],sorted_list=SortedIDs[f'{itype}'],check_entries=False)
        historyindices_atsnap[itype_mask]=itype_indices
    
    parttypes_atsnap=parttypes_atsnap.astype(int)
    historyindices_atsnap=historyindices_atsnap.astype(int)

    #use the parttypes and history indices to find the particle data indices
    partindices_atsnap=np.array([SortedIndices[str(ipart_type)][ipart_historyindex] for ipart_type,ipart_historyindex in zip(parttypes_atsnap,historyindices_atsnap)],dtype=int)

    return parttypes_atsnap,historyindices_atsnap,partindices_atsnap

########################### GENERATE ACCRETION DATA ###########################

def gen_accretion_data_fof_serial(base_halo_data,snap=None,halo_index_list=None,pre_depth=1,post_depth=1):
    
    """

    gen_accretion_data_fof_serial : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists from VELOCIraptor FOF outputs. 

    ** note: particle histories and base_halo_data must have been created as per gen_particle_history_serial (this file)
             and gen_base_halo_data in STFTools.py

	Parameters
	----------
    base_halo_data : list of dictionaries
        The minimal halo data list of dictionaries previously generated ("B1" is sufficient)

    snap : int
        The index in the base_halo_data for which to calculate accretion rates (should be actual snap index)
        We will retrieve particle data based on the flags at this index
    
    halo_index_list : dict
        "iprocess": int
        "indices: list of int
        List of the halo indices for which to calculate accretion rates. If 'None',
        find for all halos in the base_halo_data dictionary at the desired snapshot. 

    pre_depth : int
        How many snaps to skip back to when comparing particle lists.
        Initial snap for calculation will be snap-pre_depth. 

    pre_depth : int
        How many snaps to skip back to when comparing particle lists.
        Initial snap (s1) for calculation will be s1=snap-pre_depth, and we will check particle histories at s1-1. 

	Returns
	----------
    
    FOF_AccretionData_snap{snap2}_pre{pre_depth}_post{post_depth}_px.hdf5: hdf5 file with datasets
        Header contains attributes:
            "snap1"
            "snap2"
            "snap3"
            "snap1_LookbackTime"
            "snap2_LookbackTime"
            "snap3_LookbackTime"
            "ave_LookbackTime"
            "delta_LookbackTime"
            "snap1_z"
            "snap2_z"
            "snap3_z"
            "ave_z

        There is a group for each halo: ihalo_xxxxxx
        
        Inflow:

            '/Inflow/PartTypeX/ParticleIDs': ParticleID (in particle data for given type) of all accreted particles.
            '/Inflow/PartTypeX/Masses': Mass (in particle data for given type) of all accreted particles.
            '/Inflow/PartTypeX/Fidelity': Whether this particle stayed in the halo at the given fidelity gap. 
            '/Inflow/PartTypeX/PreviousHost': Which structure was this particle host to (-1: not in any fof object, 0 if CGM (subhalos only), >0: ID of previous halo).
            '/Inflow/PartTypeX/Processed_L1': How many snaps has this particle been part of any structure in the past. 
            '/Inflow/PartTypeX/Processed_L2': How many snaps has this particle been part of halos with no substructure in the past. 
            + more for PartType0 if add_gas_particle_data is run. 

        Outflow: 

            '/Outflow/PartTypeX/ParticleIDs': ParticleID (in particle data for given type) of all outflow particles.
            '/Outflow/PartTypeX/Masses': Mass (in particle data for given type) of all outflow particles.
            '/Outflow/PartTypeX/Destination_S2': Where did the particle end up after outflow at snap 2 (-1: not in halo or group, 0: CGM (only subhalos), >1: ID of destination subhalo in same field halo)
            '/Outflow/PartTypeX/Destination_S3': Where did the particle end up after outflow at snap 3 (-1: not in halo or group, 0: CGM (only subhalos), 1: reaccreted, >1: ID of destination subhalo in same field halo)
            + more for PartType0 if add_gas_particle_data is run. 

        Where there will be num_total_halos ihalo datasets. 
    
    
    """
    
    
    # Initialising halo index list
    t1_io=time.time()

    if halo_index_list==None:
        halo_index_list_snap2=list(range(len(base_halo_data[snap]["hostHaloID"])))#use all halos if not handed halo index list
        iprocess="x"
        num_processes=1
        test=True
    else:
        try:
            halo_index_list_snap2=halo_index_list["indices"] #extract index list from input dictionary
            iprocess=str(halo_index_list["iprocess"]).zfill(2) #the process for this index list (this is just used for the output file name)
            print(f'iprocess {iprocess} has {len(halo_index_list_snap2)} halo indices: {halo_index_list_snap2}')
            num_processes=halo_index_list["np"]
            test=halo_index_list["test"]
        except:
            print('Not parsed a valud halo index list. Exiting.')
            return None

    # Create log file and directories
    acc_log_dir=f"job_logs/acc_logs/"
    if not os.path.exists(acc_log_dir):
        os.mkdir(acc_log_dir)
    if test:
        run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{num_processes}_test/"
    else:
        run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{num_processes}/"

    if not os.path.exists(run_log_dir):
        try:
            os.mkdir(run_log_dir)
        except:
            pass

    run_snap_log_dir=run_log_dir+f'snap_{str(snap).zfill(3)}/'

    if not os.path.exists(run_snap_log_dir):
        try:
            os.mkdir(run_snap_log_dir)
        except:
            pass
    if test:
        fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}_test.log"
        print(f'iprocess {iprocess} will save progress to log file: {fname_log}')

    else:
        fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}.log"

    if os.path.exists(fname_log):
        os.remove(fname_log)
    
    with open(fname_log,"a") as progress_file:
        progress_file.write('Initialising and loading in data ...\n')
    progress_file.close()

    # Assigning snap
    if snap==None:
        snap=len(base_halo_data)-1#if not given snap, just use the last one

    # Find previous snap (to compare halo particles) and subsequent snap (to check accretion fidelity)
    snap1=snap-pre_depth
    snap2=snap
    snap3=snap+post_depth

    # Find the indices of halos at snap1 and snap3 (ordered by snap2 halo indices)
    halo_index_list_snap1=[find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=pre_depth) for ihalo in halo_index_list_snap2]
    halo_index_list_snap3=[find_descen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=post_depth) for ihalo in halo_index_list_snap2]

    # Initialising outputs
    if not os.path.exists('acc_data'):#create folder for outputs if doesn't already exist
        os.mkdir('acc_data')
    if test:
        calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_test/'
    else:
        calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}/'

    if not os.path.exists(calc_dir):#create folder for outputs if doesn't already exist
        try:
            os.mkdir(calc_dir)
        except:
            pass
    calc_snap_dir=calc_dir+f'snap_{str(snap2).zfill(3)}/'
    
    if not os.path.exists(calc_snap_dir):#create folder for outputs if doesn't already exist
        try:
            os.mkdir(calc_snap_dir)
        except:
            pass

    run_outname=base_halo_data[snap]['outname']#extract output name (simulation name)
    outfile_name=calc_snap_dir+'FOF_AccretionData_pre'+str(pre_depth).zfill(2)+'_post'+str(post_depth).zfill(2)+'_snap'+str(snap).zfill(3)+'_p'+str(iprocess).zfill(3)+f'_n{str(len(halo_index_list_snap2)).zfill(6)}.hdf5'
    if os.path.exists(outfile_name):#if the accretion file already exists, get rid of it 
        os.remove(outfile_name)

    # Make header for accretion data  based on base halo data 
    output_hdf5=h5py.File(outfile_name,"w")#initialise file object
    header_hdf5=output_hdf5.create_group("Header")
    lt_ave=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']+base_halo_data[snap2]['SimulationInfo']['LookbackTime'])/2
    z_ave=(base_halo_data[snap1]['SimulationInfo']['z']+base_halo_data[snap2]['SimulationInfo']['z'])/2
    dt=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']-base_halo_data[snap2]['SimulationInfo']['LookbackTime'])
    t1=base_halo_data[snap1]['SimulationInfo']['LookbackTime']
    t2=base_halo_data[snap2]['SimulationInfo']['LookbackTime']
    t3=base_halo_data[snap3]['SimulationInfo']['LookbackTime']
    z1=base_halo_data[snap1]['SimulationInfo']['z']
    z2=base_halo_data[snap2]['SimulationInfo']['z']
    z3=base_halo_data[snap3]['SimulationInfo']['z']
    header_hdf5.attrs.create('ave_LookbackTime',data=lt_ave,dtype=np.float16)
    header_hdf5.attrs.create('ave_z',data=z_ave,dtype=np.float16)
    header_hdf5.attrs.create('delta_LookbackTime',data=dt,dtype=np.float16)
    header_hdf5.attrs.create('snap1_LookbackTime',data=t1,dtype=np.float16)
    header_hdf5.attrs.create('snap2_LookbackTime',data=t2,dtype=np.float16)
    header_hdf5.attrs.create('snap3_LookbackTime',data=t3,dtype=np.float16)
    header_hdf5.attrs.create('snap1_z',data=z1,dtype=np.float16)
    header_hdf5.attrs.create('snap2_z',data=z2,dtype=np.float16)
    header_hdf5.attrs.create('snap3_z',data=z3,dtype=np.float16)
    header_hdf5.attrs.create('snap1',data=snap1,dtype=np.int16)
    header_hdf5.attrs.create('snap2',data=snap2,dtype=np.int16)
    header_hdf5.attrs.create('snap3',data=snap3,dtype=np.int16)
    header_hdf5.attrs.create('pre_depth',data=snap2-snap1,dtype=np.int16)
    header_hdf5.attrs.create('post_depth',data=snap3-snap2,dtype=np.int16)
    header_hdf5.attrs.create('outname',data=np.string_(base_halo_data[snap2]['outname']))
    header_hdf5.attrs.create('total_num_halos',data=base_halo_data[snap2]['Count'])

    # Now find which simulation type we're dealing with
    part_filetype=base_halo_data[snap]["Part_FileType"]
    print(f'Particle data type: {part_filetype}')

    # Standard particle type names from simulation
    PartNames=['gas','DM','','','star','BH']
    
    # Assign the particle types we're considering 
    if part_filetype=='EAGLE':
        PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
        constant_mass={str(0):False,str(1):True,str(4):False,str(5):False}
    else:
        PartTypes=[0,1] #Gas, DM
        constant_mass={str(0):True,str(1):True}

    # Read in particle masses
    h_val=base_halo_data[snap2]['SimulationInfo']['h_val']
    if part_filetype=='EAGLE':# if an EAGLE snapshot
        print('Reading in EAGLE snapshot data ...')
        EAGLE_boxsize=base_halo_data[snap1]['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap_1=read_eagle.EagleSnapshot(base_halo_data[snap1]['Part_FilePath'])
        EAGLE_Snap_1.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
        EAGLE_Snap_2=read_eagle.EagleSnapshot(base_halo_data[snap2]['Part_FilePath'])
        EAGLE_Snap_2.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
        Part_Data_Masses_Snap1=dict();Part_Data_IDs_Snap1=dict()
        Part_Data_Masses_Snap2=dict();Part_Data_IDs_Snap2=dict()
        for itype in PartTypes:
            print(f'Loading itype {itype} data ...')
            if not itype==1:#everything except DM
                try:
                    Part_Data_Masses_Snap1[str(itype)]=EAGLE_Snap_1.read_dataset(itype,"Mass")*10**10/h_val #CHECK THIS√
                    Part_Data_Masses_Snap2[str(itype)]=EAGLE_Snap_2.read_dataset(itype,"Mass")*10**10/h_val #CHECK THIS√
                except:
                    print('No particles of this type were found.')
                    Part_Data_Masses_Snap1[str(itype)]=[]
                    Part_Data_Masses_Snap2[str(itype)]=[]
            else:#for DM, find particle data file and save 
                hdf5file=h5py.File(base_halo_data[snap1]['Part_FilePath'])#hdf5 file
                Part_Data_Masses_Snap1[str(itype)]=hdf5file['Header'].attrs['MassTable'][1]*10**10/h_val #CHECK THIS√
                Part_Data_Masses_Snap2[str(itype)]=hdf5file['Header'].attrs['MassTable'][1]*10**10/h_val #CHECK THIS√
        print('Done reading in EAGLE snapshot data')
    else:#assuming constant mass
        Part_Data_Masses_Snap1=dict()
        hdf5file=h5py.File(base_halo_data[snap1]['Part_FilePath'])
        MassTable=hdf5file["Header"].attrs["MassTable"]
        Part_Data_Masses_Snap1[str(1)]=MassTable[1]*10**10/h_val#CHECK THIS
        Part_Data_Masses_Snap1[str(0)]=MassTable[0]*10**10/h_val#CHECK THIS
        Part_Data_Masses_Snap2[str(1)]=MassTable[1]*10**10/h_val#CHECK THIS
        Part_Data_Masses_Snap2[str(0)]=MassTable[0]*10**10/h_val#CHECK THIS

    #Load in particle histories: snap 1
    print(f'Retrieving & organising particle histories for snap = {snap1} ...')
    Part_Histories_File_snap1=h5py.File("part_histories/PartHistory_"+str(snap1).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes}
    Part_Histories_Index_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes}
    Part_Histories_HostStructure_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes}
    Part_Histories_Processed_L1_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/Processed_L1'] for parttype in [0,1]}
    Part_Histories_Processed_L2_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/Processed_L2'] for parttype in [0,1]}
    Part_Histories_npart_snap1={str(parttype):len(Part_Histories_IDs_snap1[str(parttype)]) for parttype in PartTypes}

    #Load in particle histories: snap 2
    print(f'Retrieving & organising particle histories for snap = {snap2} ...')
    Part_Histories_File_snap2=h5py.File("part_histories/PartHistory_"+str(snap2).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes}
    Part_Histories_Index_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes}
    Part_Histories_HostStructure_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes}
    Part_Histories_npart_snap2={str(parttype):len(Part_Histories_IDs_snap2[str(parttype)]) for parttype in PartTypes}

    #Load in particle lists from VR
    print('Retrieving VR halo particle lists ...')
    snap_1_halo_particles=get_particle_lists(base_halo_data[snap1],halo_index_list=halo_index_list_snap1,include_unbound=True,add_subparts_to_fofs=True)
    snap_2_halo_particles=get_particle_lists(base_halo_data[snap2],halo_index_list=halo_index_list_snap2,include_unbound=True,add_subparts_to_fofs=True)
    snap_3_halo_particles=get_particle_lists(base_halo_data[snap3],halo_index_list=halo_index_list_snap3,include_unbound=True,add_subparts_to_fofs=True)

    t2_io=time.time()
    print()
    print('*********************************************************')
    print(f'Done with I/O in {(t2_io-t1_io):.2f} sec - entering main halo loop ...')
    print('*********************************************************')
    with open(fname_log,"a") as progress_file:
        progress_file.write(f'Done with I/O in {(t2_io-t1_io):.2f} sec - entering main halo loop ...\n')
    progress_file.close()

    count=0
    halos_done=0
    num_halos_thisprocess=len(halo_index_list_snap2)
    for iihalo,ihalo_s2 in enumerate(halo_index_list_snap2):# for each halo at snap 2
        with open(fname_log,"a") as progress_file:
            progress_file.write(f' \n')
            progress_file.write(f'Starting with ihalo {ihalo_s2} ...\n')
        progress_file.close()

        t1_halo=time.time()
        t1_preamble=time.time()
        # Create group for this halo in output file
        ihalo_hdf5=output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))
        ihalo_in_hdf5=ihalo_hdf5.create_group('Inflow')
        ihalo_out_hdf5=ihalo_hdf5.create_group('Outflow')

        # Find halo progenitor and descendants 
        ihalo_s1=halo_index_list_snap1[iihalo]#find progenitor
        ihalo_s3=halo_index_list_snap3[iihalo]#find descendant
        try:
            idhalo_s1=base_halo_data[snap1]['ID'][ihalo_s1]
            idhalo_s3=base_halo_data[snap3]['ID'][ihalo_s3]
        except:
            idhalo_s1=np.nan
            idhalo_s3=np.nan

        #Record halo position and velocity
        if ihalo_s1>=0:
            ihalo_hdf5.attrs.create('snap1_com',data=[base_halo_data[snap1]['Xc'][ihalo_s1],base_halo_data[snap1]['Yc'][ihalo_s1],base_halo_data[snap1]['Zc'][ihalo_s1]],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap1_v',data=[base_halo_data[snap1]['VXc'][ihalo_s1],base_halo_data[snap1]['VYc'][ihalo_s1],base_halo_data[snap1]['VZc'][ihalo_s1]],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap1_R200',data=base_halo_data[snap1]['R_200crit'][ihalo_s1],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap1_M200',data=base_halo_data[snap1]['Mass_200crit'][ihalo_s1],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap1_Vmax',data=base_halo_data[snap1]['Vmax'][ihalo_s1],dtype=np.float32)
        if ihalo_s2>=0:
            ihalo_hdf5.attrs.create('snap2_com',data=[base_halo_data[snap2]['Xc'][ihalo_s2],base_halo_data[snap2]['Yc'][ihalo_s2],base_halo_data[snap2]['Zc'][ihalo_s2]],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap2_v',data=[base_halo_data[snap2]['VXc'][ihalo_s2],base_halo_data[snap2]['VYc'][ihalo_s2],base_halo_data[snap2]['VZc'][ihalo_s2]],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap2_R200',data=base_halo_data[snap2]['R_200crit'][ihalo_s2],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap2_M200',data=base_halo_data[snap2]['Mass_200crit'][ihalo_s2],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap2_Vmax',data=base_halo_data[snap2]['Vmax'][ihalo_s2],dtype=np.float32)
        if ihalo_s3>=0:
            ihalo_hdf5.attrs.create('snap3_com',data=[base_halo_data[snap3]['Xc'][ihalo_s3],base_halo_data[snap3]['Yc'][ihalo_s3],base_halo_data[snap3]['Zc'][ihalo_s3]],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap3_v',data=[base_halo_data[snap3]['VXc'][ihalo_s3],base_halo_data[snap3]['VYc'][ihalo_s3],base_halo_data[snap3]['VZc'][ihalo_s3]],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap3_R200',data=base_halo_data[snap3]['R_200crit'][ihalo_s3],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap3_M200',data=base_halo_data[snap2]['Mass_200crit'][ihalo_s3],dtype=np.float32)
            ihalo_hdf5.attrs.create('snap3_Vmax',data=base_halo_data[snap3]['Vmax'][ihalo_s3],dtype=np.float32)
        ihalo_tracked=(ihalo_s1>-1 and ihalo_s3>-1)#track if have both progenitor and descendant
        structuretype=base_halo_data[snap2]["Structuretype"][ihalo_s2]#structure type

        # If we have a subhalo, find its progenitor host group (for CGM accretion)
        if structuretype>10:
            isub=True
            ifield=False
            try:
                current_hostgroupID=base_halo_data[snap2]["hostHaloID"][ihalo_s2]
                current_hostindex=np.where(current_hostgroupID==base_halo_data[snap2]["ID"])[0][0]
                prev_hostindex=find_progen_index(base_halo_data,index2=current_hostindex,snap2=snap2,depth=1) #host index at previous snapshot 
                prev_hostgroupID=base_halo_data[snap1]["ID"][prev_hostindex] #the host halo ID of this subhalo at the previous snapshot
            except:#if can't find progenitor, don't try to compare for CGM accretion
                prev_hostHaloID=np.nan
                print("Couldn't find the progenitor group - not checking for CGM accretion")
        else:
            isub=False
            ifield=True
            prev_hostHaloID=np.nan

        # Print halo data for outputs 
        print()
        print('**********************************************')
        if ifield:
            print('Halo index: ',ihalo_s2,f' - field halo')
        if isub:
            print('Halo index: ',ihalo_s2,f' - sub halo')
            print(f'Host halo at previous snap: {prev_hostgroupID}')
        print(f'Progenitor: {idhalo_s1} | Descendant: {idhalo_s3}')
        print('**********************************************')
        print()
        
        t2_preamble=time.time()

        # If this halo is going to be tracked (and is not a subsubhalo) then we continue
        if ihalo_tracked and structuretype<25:# if we found both the progenitor and the descendent (and it's not a subsubhalo)
            snap1_IDs_temp=snap_1_halo_particles['Particle_IDs'][iihalo]#IDs in the halo at the previous snap
            snap1_Types_temp=snap_1_halo_particles['Particle_Types'][iihalo]#Types of particles in the halo at the previous snap
            snap2_IDs_temp=snap_2_halo_particles['Particle_IDs'][iihalo]#IDs in the halo at the current snap
            snap2_Types_temp=snap_2_halo_particles['Particle_Types'][iihalo]# Types of particles in the halo at the current snap
            snap3_IDs_temp_set=set(snap_3_halo_particles['Particle_IDs'][iihalo])# Set of IDs in the halo at the subsequent snapshot (to compare with)
            
            ############ GRABBING DATA FOR INFLOW PARTICLES (at snap 1) ############
            # Returns mask for s2 of particles which are in s2 but not in s1
            print(f"Finding and indexing new particles to ihalo {ihalo_s2} ...")
            t1_new=time.time()
            new_particle_IDs_mask_snap2=np.isin(snap2_IDs_temp,snap1_IDs_temp,assume_unique=True,invert=True)
            new_particle_IDs_where_snap2=np.where(new_particle_IDs_mask_snap2)
            new_particle_IDs=snap2_IDs_temp[new_particle_IDs_where_snap2]
            new_particle_Types_snap2=snap2_Types_temp[new_particle_IDs_where_snap2]
            new_particle_Types_snap1,new_particle_historyindices_snap1,new_particle_partindices_snap1=get_particle_indices(base_halo_data=base_halo_data,
                                                                    SortedIDs=Part_Histories_IDs_snap1,
                                                                    SortedIndices=Part_Histories_Index_snap1,
                                                                    PartIDs=new_particle_IDs,
                                                                    PartTypes=new_particle_Types_snap2,
                                                                    snap_taken=snap2,
                                                                    snap_desired=snap1)
            new_particle_tranformed=np.logical_not(new_particle_Types_snap1==new_particle_Types_snap2)
            ihalo_nin=np.sum(new_particle_IDs_mask_snap2)
            print(f"n(in) = {ihalo_nin}")
            t2_new=time.time()

            ihalo_snap1_inflow_type=new_particle_Types_snap1
            ihalo_snap1_inflow_transformed=new_particle_tranformed
            ihalo_snap1_inflow_history_L1=np.zeros(ihalo_nin)
            ihalo_snap1_inflow_history_L2=np.zeros(ihalo_nin)
            ihalo_snap1_inflow_structure=np.zeros(ihalo_nin)+np.nan
            ihalo_snap1_inflow_fidelity=np.zeros(ihalo_nin)
            ihalo_snap1_inflow_masses=np.zeros(ihalo_nin)+np.nan

            # Find processing history, previous host, fidelity
            for iipartin,ipartin_ID,ipartin_snap1_type,ipartin_snap1_historyindex,ipartin_snap1_partindex in zip(list(range(ihalo_nin)),new_particle_IDs,new_particle_Types_snap1,new_particle_historyindices_snap1,new_particle_partindices_snap1):
                if ipartin_snap1_type==0 or ipartin_snap1_type==1:#if DM or gas, this has been recorded
                    ihalo_snap1_inflow_history_L1[iipartin]=Part_Histories_Processed_L1_snap1[str(ipartin_snap1_type)][ipartin_snap1_historyindex]
                    ihalo_snap1_inflow_history_L2[iipartin]=Part_Histories_Processed_L2_snap1[str(ipartin_snap1_type)][ipartin_snap1_historyindex]
                else:#assume stars have been processed
                    ihalo_snap1_inflow_history_L1[iipartin]=1
                    ihalo_snap1_inflow_history_L2[iipartin]=1
                ihalo_snap1_inflow_structure[iipartin]=Part_Histories_HostStructure_snap1[str(ipartin_snap1_type)][ipartin_snap1_historyindex]
                ihalo_snap1_inflow_fidelity[iipartin]=(ipartin_ID in snap3_IDs_temp_set)
            
            if isub:#if subhalo, check which particles came from CGM
                ihalo_cgm_inflow_particles_mask=prev_hostgroupID==ihalo_snap1_inflow_structure
                ihalo_cgm_inflow_particles_where=np.where(ihalo_cgm_inflow_particles_mask)
                ihalo_snap1_inflow_structure[ihalo_cgm_inflow_particles_where]=np.zeros(np.sum(ihalo_cgm_inflow_particles_mask))

            # Find mass
            for itype in PartTypes:
                ihalo_itype_snap1_inflow_mask=ihalo_snap1_inflow_type==itype
                ihalo_itype_snap1_inflow_where=np.where(ihalo_itype_snap1_inflow_mask)
                ihalo_itype_snap1_inflow_n=np.sum(ihalo_itype_snap1_inflow_mask)
                ihalo_itype_snap1_inflow_partindices=new_particle_partindices_snap1[ihalo_itype_snap1_inflow_where]
                if constant_mass[str(itype)]:
                    ihalo_itype_snap1_inflow_masses=np.ones(ihalo_itype_snap1_inflow_n)*Part_Data_Masses_Snap1[str(itype)]
                else:
                    ihalo_itype_snap1_inflow_masses=np.array([Part_Data_Masses_Snap1[str(itype)][ihalo_itype_snap1_inflow_partindex] for ihalo_itype_snap1_inflow_partindex in ihalo_itype_snap1_inflow_partindices])
                ihalo_snap1_inflow_masses[ihalo_itype_snap1_inflow_where]=ihalo_itype_snap1_inflow_masses
            
            ############ GRABBING DATA FOR OUTFLOW PARTICLES (at snap 2) ############
            # # Returns mask for s1 of particles which are in s1 but not in s2
            print(f"Finding and indexing particles which left ihalo {ihalo_s2} ...")
            t1_out=time.time()
            out_particle_IDs_mask_snap1=np.isin(snap1_IDs_temp,snap2_IDs_temp,assume_unique=True,invert=True)
            out_particle_IDs_where_snap1=np.where(out_particle_IDs_mask_snap1)
            out_particle_IDs=snap1_IDs_temp[out_particle_IDs_where_snap1]
            out_particle_Types_snap1=snap1_Types_temp[out_particle_IDs_where_snap1]
            out_particle_Types_snap2,out_particle_historyindices_snap2,out_particle_partindices_snap2=get_particle_indices(base_halo_data=base_halo_data,
                                                        SortedIDs=Part_Histories_IDs_snap2,
                                                        SortedIndices=Part_Histories_Index_snap2,
                                                        PartIDs=out_particle_IDs,
                                                        PartTypes=out_particle_Types_snap1,
                                                        snap_taken=snap1,
                                                        snap_desired=snap2)
            out_particle_tranformed=np.logical_not(out_particle_Types_snap1==out_particle_Types_snap2)
            ihalo_nout=np.sum(out_particle_IDs_mask_snap1)
            t2_out=time.time()
            print(f"n(out) = {ihalo_nout}")
            
            with open(fname_log,"a") as progress_file:
                progress_file.write(f'       n(in): total = {ihalo_nin}\n')
                progress_file.write(f'       n(out): total = {ihalo_nout}\n')
            progress_file.close()
            
            ihalo_snap2_outflow_type=out_particle_Types_snap2
            ihalo_snap2_outflow_transformed=out_particle_tranformed
            ihalo_snap2_outflow_destination=np.zeros(ihalo_nout)+np.nan
            ihalo_snap3_outflow_recycled=np.zeros(ihalo_nout)+np.nan
            ihalo_snap2_outflow_masses=np.zeros(ihalo_nout)+np.nan

            # Find processing history, previous host, fidelity
            for iipartout,ipartout_ID,ipartout_snap2_type,ipartout_snap2_historyindex,ipartout_snap2_partindex in zip(list(range(ihalo_nout)),out_particle_IDs,out_particle_Types_snap2,out_particle_historyindices_snap2,out_particle_partindices_snap2):
                ihalo_snap2_outflow_destination[iipartout]=Part_Histories_HostStructure_snap2[str(ipartout_snap2_type)][ipartout_snap2_historyindex]
                ihalo_snap3_outflow_recycled[iipartout]=(iipartout in snap3_IDs_temp_set)
            
            if isub:#if subhalo, check which particles went to CGM current_hostgroupID
                ihalo_cgm_outflow_particles_mask=(current_hostgroupID==ihalo_snap2_outflow_destination)
                ihalo_cgm_outflow_particles_where=np.where(ihalo_cgm_outflow_particles_mask)
                ihalo_snap2_outflow_destination[ihalo_cgm_outflow_particles_where]=np.zeros(np.sum(ihalo_cgm_outflow_particles_mask))
            
            # Find mass
            for itype in PartTypes:
                ihalo_itype_snap2_outflow_mask=ihalo_snap2_outflow_type==itype
                ihalo_itype_snap2_outflow_where=np.where(ihalo_itype_snap2_outflow_mask)
                ihalo_itype_snap2_outflow_n=np.sum(ihalo_itype_snap2_outflow_mask)
                ihalo_itype_snap2_outflow_partindices=out_particle_partindices_snap2[ihalo_itype_snap2_outflow_where]
                if constant_mass[str(itype)]:
                    ihalo_itype_snap2_outflow_masses=np.ones(ihalo_itype_snap2_outflow_n)*Part_Data_Masses_Snap2[str(itype)]
                else:
                    ihalo_itype_snap2_outflow_masses=np.array([Part_Data_Masses_Snap2[str(itype)][ihalo_itype_snap2_outflow_partindex] for ihalo_itype_snap2_outflow_partindex in ihalo_itype_snap2_outflow_partindices])
                ihalo_snap2_outflow_masses[ihalo_itype_snap2_outflow_where]=ihalo_itype_snap2_outflow_masses

            ############ SAVE DATA FOR INLFOW & OUTFLOW PARTICLES ###########
            for iitype, itype in enumerate(PartTypes):

                # Saving INFLOW data for this parttype of the halo to file 
                ihalo_itype_snap1_inflow_mask=ihalo_snap1_inflow_type==itype#type the inflow particles based on snap 1 state
                ihalo_itype_snap1_inflow_where=np.where(ihalo_itype_snap1_inflow_mask)

                ihalo_in_parttype_hdf5=ihalo_in_hdf5.create_group('PartType'+str(itype))
                ihalo_in_parttype_hdf5.create_dataset('ParticleIDs',data=new_particle_IDs[ihalo_itype_snap1_inflow_where],dtype=np.int64)#######
                ihalo_in_parttype_hdf5.create_dataset('Transformed',data=ihalo_snap1_inflow_transformed[ihalo_itype_snap1_inflow_where],dtype=np.uint8)
                ihalo_in_parttype_hdf5.create_dataset('Processed_L1',data=ihalo_snap1_inflow_history_L1[ihalo_itype_snap1_inflow_where],dtype=np.uint8)
                ihalo_in_parttype_hdf5.create_dataset('Processed_L2',data=ihalo_snap1_inflow_history_L2[ihalo_itype_snap1_inflow_where],dtype=np.uint8)
                ihalo_in_parttype_hdf5.create_dataset('PreviousHost',data=ihalo_snap1_inflow_structure[ihalo_itype_snap1_inflow_where],dtype=np.int64)
                ihalo_in_parttype_hdf5.create_dataset('Fidelity',data=ihalo_snap1_inflow_fidelity[ihalo_itype_snap1_inflow_where],dtype=np.uint8)
                ihalo_in_parttype_hdf5.create_dataset('Masses',data=ihalo_snap1_inflow_masses[ihalo_itype_snap1_inflow_where],dtype=np.float64)

                # Saving OUTFLOW data for this parttype of the halo to file 
                ihalo_itype_snap2_outflow_mask=ihalo_snap2_outflow_type==itype#type the inflow particles based on snap 1 state
                ihalo_itype_snap2_outflow_where=np.where(ihalo_itype_snap2_outflow_mask)

                ihalo_out_parttype_hdf5=ihalo_out_hdf5.create_group('PartType'+str(itype))
                ihalo_out_parttype_hdf5.create_dataset('ParticleIDs',data=out_particle_IDs[ihalo_itype_snap2_outflow_where],dtype=np.int64)#######
                ihalo_out_parttype_hdf5.create_dataset('Transformed',data=ihalo_snap2_outflow_transformed[ihalo_itype_snap2_outflow_where],dtype=np.uint8)
                ihalo_out_parttype_hdf5.create_dataset('Destination',data=ihalo_snap2_outflow_destination[ihalo_itype_snap2_outflow_where],dtype=np.uint8)
                ihalo_out_parttype_hdf5.create_dataset('Recycled',data=ihalo_snap3_outflow_recycled[ihalo_itype_snap2_outflow_where],dtype=np.uint8)
                ihalo_out_parttype_hdf5.create_dataset('Masses',data=ihalo_snap2_outflow_masses[ihalo_itype_snap2_outflow_where],dtype=np.float64)

        else:#if halo not tracked, return np.nan for fidelity, ids, prevhost
            for itype in PartTypes:
                # print(f'Saving {PartNames[itype]} data for ihalo {ihalo_s2} (not tracked) to hdf5 ...')
                ihalo_in_parttype_hdf5=ihalo_in_hdf5.create_group('PartType'+str(itype))
                ihalo_in_parttype_hdf5.create_dataset('ParticleIDs',data=np.nan,dtype=np.float16)
                ihalo_in_parttype_hdf5.create_dataset('Masses',data=np.nan,dtype=np.float16)
                ihalo_in_parttype_hdf5.create_dataset('Fidelity',data=np.nan,dtype=np.float16)
                ihalo_in_parttype_hdf5.create_dataset('PreviousHost',data=np.nan,dtype=np.float16)
                ihalo_in_parttype_hdf5.create_dataset('Processed_L1',data=np.nan,dtype=np.float16)
                ihalo_in_parttype_hdf5.create_dataset('Processed_L2',data=np.nan,dtype=np.float16)
                ihalo_out_parttype_hdf5.create_dataset('Transformed',data=np.nan,dtype=np.float16)

                # Saving OUTFLOW data for this parttype of the halo to file 
                ihalo_out_parttype_hdf5=ihalo_out_hdf5.create_group('PartType'+str(itype))
                ihalo_out_parttype_hdf5.create_dataset('ParticleIDs',data=np.nan,dtype=np.float16)
                ihalo_out_parttype_hdf5.create_dataset('Masses',data=np.nan,dtype=np.float16)
                ihalo_out_parttype_hdf5.create_dataset('Destination',data=np.nan,dtype=np.float16)
                ihalo_out_parttype_hdf5.create_dataset('Recycled',data=np.nan,dtype=np.float16)
                halo_out_parttype_hdf5.create_dataset('Transformed',data=np.nan,dtype=np.float16)
                
        t2_halo=time.time()

        with open(fname_log,"a") as progress_file:
            progress_file.write(f"Done with ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
            progress_file.write(f"[took {t2_halo-t1_halo} sec]\n")
            progress_file.write(f" \n")
        progress_file.close()

        print()

    #Close the output file, finish up
    output_hdf5.close()

########################### POSTPROCESS/SUM ACCRETION DATA ###########################

def postprocess_acc_data_serial(base_halo_data,path):
    """

    postprocess_acc_data_serial : function
	----------

    Collate and post process all the accretion data in the provided directory (which must only contain the required data).

	Parameters
	----------
    base_halo_data : the halo data list of dictionaries for this run
    path : string indicating the directory in which the accretion data is stored (nominally acc_data/)

	Returns
	----------
    
    Combined_AccData.hdf5: hdf5 file with datasets:
        summed outputs
        ---------------
        In group '/Inflow':
        '/PartTypeX/All_TotalDeltaN': Total number of particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_TotalDeltaM': Total mass of particles of type X new to the halo  (length: num_total_halos)
        '/PartTypeX/All_CosmologicalDeltaN': Total number of cosmological origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_CosmologicalDeltaM': Total mass of cosmological origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_CGMDeltaN': Total number of CGM origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_CGMDeltaM': Total mass of CGM origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_ClumpyDeltaN': Total number of clumpy origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_ClumpyDeltaM': Total mass of clumpy origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_PrimordialDeltaN': Total number of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_PrimordialDeltaM': Total mass of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_ProcessedCosmologicalDeltaN': Total number of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/All_ProcessedCosmologicalDeltaM': Total mass of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)

        '/PartTypeX/Stable_TotalDeltaN': Total number of particles of type X new (and LOYAL) to the halo (length: num_total_halos)
        '/PartTypeX/Stable_TotalDeltaM': Total mass of particles of type X new (and LOYAL) to the halo (length: num_total_halos)
        '/PartTypeX/Stable_CosmologicalDeltaN': Total number of cosmological origin particles of type X new (and LOYAL) to the halo (length: num_total_halos)
        '/PartTypeX/Stable_CosmologicalDeltaM': Total mass of cosmological origin particles of type X new (and LOYAL) to the halo (length: num_total_halos)
        '/PartTypeX/Stable_CGMDeltaN': Total number of CGM origin particles of type X new (and LOYAL) to the halo (length: num_total_halos)
        '/PartTypeX/Stable_CGMDeltaM': Total mass of CGM origin particles of type X new (and LOYAL) to the halo (length: num_total_halos)
        '/PartTypeX/Stable_ClumpyDeltaN': Total number of clumpy origin particles of type X new (and LOYAL) to the halo (length: num_total_halos)
        '/PartTypeX/Stable_ClumpyDeltaM': Total mass of clumpy origin particles of type X new (and LOYAL) to the halo (length: num_total_halos)
        '/PartTypeX/Stable_PrimordialDeltaN': Total number of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/Stable_PrimordialDeltaM': Total mass of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/Stable_ProcessedCosmologicalDeltaN': Total number of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)
        '/PartTypeX/Stable_ProcessedCosmologicalDeltaM': Total mass of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)


        '/Header' contains attributes: 
        't1'
        't2'
        'dt'
        'z_ave'
        'lt_ave'
        etc
    
    """
    t1=time.time()
    print(f'Summing accretion data from path: {path}')
    if not path.endswith('/'):
        path=path+'/'

    # List the contents of the provided directory
    acc_data_filelist=os.listdir(path)
    acc_data_filelist=sorted(acc_data_filelist)
    acc_data_filelist_trunc=[filename for filename in acc_data_filelist if (('px' not in filename) and ('FOF' in filename) and ('DS' not in filename) and ('summed' not in filename))]
    print('Summing accretion data from the following files:')
    print(np.array(acc_data_filelist_trunc))
    acc_data_filelist=acc_data_filelist_trunc
    acc_data_outfile_name=acc_data_filelist[0].split('_p0')[0]+'_summed.hdf5'

    if os.path.exists(path+acc_data_outfile_name):
        print("Deleting existing combined data first")
        os.remove(path+acc_data_outfile_name)

    print(f'Output file name: {acc_data_outfile_name}')
    
    # Initialise output file
    collated_output_file=h5py.File(path+acc_data_outfile_name,'w')
    
    # Open existing files in list structure
    acc_data_hdf5files=[h5py.File(path+acc_data_file,'r') for acc_data_file in acc_data_filelist]
    acc_data_snap=acc_data_hdf5files[0]['Header'].attrs['snap2']
    total_num_halos=0
    for ifile in acc_data_hdf5files:
        groups=list(ifile.keys())
        for group in groups:
            if 'ihalo' in group:
                total_num_halos=total_num_halos+1
    if total_num_halos<1000:
        print(f'Using array size {3*10**5}')
        total_num_halos=3*10**5
    else:
        total_num_halos=base_halo_data[acc_data_snap]['Count']

    print(f'Collating data for {total_num_halos} halos')
    
    # Copy over header information from first file
    acc_data_hdf5files_header=acc_data_hdf5files[0]['Header']
    acc_data_hdf5files_header_attrs=list(acc_data_hdf5files_header.attrs)
    collated_output_file_header=collated_output_file.create_group('Header')

    print("Attributes of accretion calculation: ")
    for attribute in acc_data_hdf5files_header_attrs:
        collated_output_file_header.attrs.create(attribute,data=acc_data_hdf5files_header.attrs[attribute])
        print(attribute,collated_output_file_header.attrs[attribute])

    # Add extra header info
    try:
        collated_output_file_header.attrs.create('outname',data=np.string_(base_halo_data[-1]['outname']))
        collated_output_file_header.attrs.create('pre_depth',data=acc_data_hdf5files_header.attrs['snap2']-acc_data_hdf5files_header.attrs['snap1'])
        collated_output_file_header.attrs.create('post_depth',data=acc_data_hdf5files_header.attrs['snap3']-acc_data_hdf5files_header.attrs['snap2'])
        collated_output_file_header.attrs.create('total_num_halos',data=total_num_halos)
    except:
        pass

    # Names of the new outputs
    new_outputs_inflow=[
    "All_TotalDeltaM_In",
    "All_TotalDeltaN_In",
    "All_CosmologicalDeltaN_In",
    'All_CosmologicalDeltaM_In',
    'All_CGMDeltaN_In',
    'All_CGMDeltaM_In',
    'All_ClumpyDeltaN_In',
    'All_ClumpyDeltaM_In',
    'All_PrimordialDeltaN_In',
    'All_PrimordialDeltaM_In',
    'All_ProcessedCosmologicalDeltaN_In',
    'All_ProcessedCosmologicalDeltaM_In',   
    "Stable_TotalDeltaM_In",
    "Stable_TotalDeltaN_In",
    "Stable_CosmologicalDeltaN_In",
    'Stable_CosmologicalDeltaM_In',
    'Stable_CGMDeltaN_In',
    'Stable_CGMDeltaM_In',
    'Stable_ClumpyDeltaN_In',
    'Stable_ClumpyDeltaM_In',
    'Stable_PrimordialDeltaN_In',
    'Stable_PrimordialDeltaM_In',
    'Stable_ProcessedCosmologicalDeltaN_In',
    'Stable_ProcessedCosmologicalDeltaM_In'
    ]

    new_outputs_outflow=[
    "All_TotalDeltaM_Out",
    "All_TotalDeltaN_Out",
    "All_FieldDeltaM_Out",
    "All_FieldDeltaN_Out",
    "All_CGMDeltaM_Out",
    "All_CGMDeltaN_Out",
    "All_OtherHaloDeltaM_Out",
    "All_OtherHaloDeltaN_Out",
    "All_RecycledDeltaN_Out",#at snap 3
    "All_RecycledDeltaM_Out"]#at snap 3


    # Initialise all new outputs
    first_file=acc_data_hdf5files[0]
    first_halo_group=[key for key in list(first_file.keys()) if 'ihalo' in key][0]
    first_halo_inflow_keys=list(first_file[first_halo_group]['Inflow'].keys())
    no_parttypes=len(first_halo_inflow_keys)#grab from first file
    print(f'Grabbing data for part types: {first_halo_inflow_keys}')
    itypes=[0,1,4,5][:no_parttypes]
    summed_acc_data={}
    new_outputs_keys_bytype_in=[f'Inflow/PartType{itype}/'+field for field in new_outputs_inflow for itype in itypes]
    new_outputs_keys_bytype_out=[f'Outflow/PartType{itype}/'+field for field in new_outputs_outflow for itype in itypes]
    for outfield in new_outputs_keys_bytype_out:
        summed_acc_data[outfield]=np.zeros(total_num_halos)+np.nan
    for infield in new_outputs_keys_bytype_in:
        summed_acc_data[infield]=np.zeros(total_num_halos)+np.nan

    iihalo=0
    for ifile,acc_data_filetemp in enumerate(acc_data_hdf5files):
        print(f"Reading from file {ifile+1}/{len(acc_data_hdf5files)}: {acc_data_filetemp}")
        ihalo_group_list_all=list(acc_data_filetemp.keys())
        ihalo_group_list=[ihalo_group for ihalo_group in ihalo_group_list_all if ihalo_group.startswith('ihalo')]
        for ihalo_group in ihalo_group_list:
            iihalo=iihalo+1
            ihalo=int(ihalo_group.split('_')[-1])
            for itype in itypes:
                # Load in the details of particles new to this halo
                try:
                    fidelities=acc_data_filetemp[ihalo_group+f'/Inflow/PartType{itype}/Fidelity'].value
                    masses=acc_data_filetemp[ihalo_group+f'/Inflow/PartType{itype}/Masses'].value
                    prevhosts=acc_data_filetemp[ihalo_group+f'/Inflow/PartType{itype}/PreviousHost'].value
                    if itype==0 or itype==1:
                        processed_l1=acc_data_filetemp[ihalo_group+f'/Inflow/PartType{itype}/Processed_L1'].value
                        processed_l2=acc_data_filetemp[ihalo_group+f'/Inflow/PartType{itype}/Processed_L2'].value
                    else:
                        processed_l1=np.ones(len(masses))#stars/bh will always be processed at some level
                        processed_l2=np.ones(len(masses))#stars/bh will always be processed at lowest level
                except:
                    # print(f'ihalo {ihalo} does not have accretion data for part type = {itype}')
                    continue

                #Load in the details of particles which left this halo
                try:
                    masses_out=acc_data_filetemp[ihalo_group+f'/Outflow/PartType{itype}/Masses'].value
                    destination_s2_out=acc_data_filetemp[ihalo_group+f'/Outflow/PartType{itype}/Destination'].value
                    recycled_out=acc_data_filetemp[ihalo_group+f'/Outflow/PartType{itype}/Recycled'].value
                except:
                    # print(f'ihalo {ihalo} does not have accretion data for part type = {itype}')
                    continue

                if not np.isfinite(np.sum(fidelities)):
                    # print(f'ihalo {ihalo} does not have accretion data for part type = {itype}')
                    continue

                # Define masks based on particle properties
                stable_mask=fidelities>0

                cosmological_mask=prevhosts<0
                cgm_mask=prevhosts==0
                clumpy_mask=prevhosts>0
                primordial_mask=processed_l1==0
                recycled_mask=np.logical_and(np.logical_or(cgm_mask,cosmological_mask),np.logical_not(primordial_mask))

                stable_cosmological_mask=np.logical_and(stable_mask,cosmological_mask)
                stable_cgm_mask=np.logical_and(stable_mask,cgm_mask)
                stable_clumpy_mask=np.logical_and(stable_mask,clumpy_mask)
                stable_primordial_mask=np.logical_and(stable_mask,primordial_mask)
                stable_processedcosmological_mask=np.logical_and(stable_mask,recycled_mask)

                summed_acc_data[f'Inflow/PartType{itype}/All_TotalDeltaN_In'][ihalo]=np.size(masses)
                summed_acc_data[f'Inflow/PartType{itype}/All_TotalDeltaM_In'][ihalo]=np.sum(masses)
                summed_acc_data[f'Inflow/PartType{itype}/All_CosmologicalDeltaN_In'][ihalo]=np.size(np.compress(cosmological_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/All_CosmologicalDeltaM_In'][ihalo]=np.sum(np.compress(cosmological_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/All_CGMDeltaN_In'][ihalo]=np.size(np.compress(cgm_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/All_CGMDeltaM_In'][ihalo]=np.sum(np.compress(cgm_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/All_ClumpyDeltaN_In'][ihalo]=np.size(np.compress(clumpy_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/All_ClumpyDeltaM_In'][ihalo]=np.sum(np.compress(clumpy_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/All_PrimordialDeltaN_In'][ihalo]=np.size(np.compress(primordial_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/All_PrimordialDeltaM_In'][ihalo]=np.sum(np.compress(primordial_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/All_ProcessedCosmologicalDeltaN_In'][ihalo]=np.size(np.compress(recycled_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/All_ProcessedCosmologicalDeltaM_In'][ihalo]=np.sum(np.compress(recycled_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_TotalDeltaN_In'][ihalo]=np.size(np.compress(stable_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_TotalDeltaM_In'][ihalo]=np.sum(np.compress(stable_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_CosmologicalDeltaN_In'][ihalo]=np.size(np.compress(stable_cosmological_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_CosmologicalDeltaM_In'][ihalo]=np.sum(np.compress(stable_cosmological_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_CGMDeltaN_In'][ihalo]=np.size(np.compress(stable_cgm_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_CGMDeltaM_In'][ihalo]=np.sum(np.compress(stable_cgm_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_ClumpyDeltaN_In'][ihalo]=np.size(np.compress(stable_clumpy_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_ClumpyDeltaM_In'][ihalo]=np.sum(np.compress(stable_clumpy_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_PrimordialDeltaN_In'][ihalo]=np.size(np.compress(stable_primordial_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_PrimordialDeltaM_In'][ihalo]=np.sum(np.compress(stable_primordial_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_ProcessedCosmologicalDeltaN_In'][ihalo]=np.size(np.compress(stable_processedcosmological_mask,masses))
                summed_acc_data[f'Inflow/PartType{itype}/Stable_ProcessedCosmologicalDeltaM_In'][ihalo]=np.sum(np.compress(stable_processedcosmological_mask,masses))

                outfield_mask=(destination_s2_out==-1)
                outhalo_mask=(destination_s2_out>1)
                outcgm_mask=destination_s2_out==0
                reaccreted_mask=recycled_out==1

                summed_acc_data[f'Outflow/PartType{itype}/All_TotalDeltaN_Out'][ihalo]=np.size(masses_out)
                summed_acc_data[f'Outflow/PartType{itype}/All_TotalDeltaM_Out'][ihalo]=np.sum(masses_out)
                summed_acc_data[f'Outflow/PartType{itype}/All_FieldDeltaN_Out'][ihalo]=np.size(np.compress(outfield_mask,masses_out))
                summed_acc_data[f'Outflow/PartType{itype}/All_FieldDeltaM_Out'][ihalo]=np.sum(np.compress(outfield_mask,masses_out))
                summed_acc_data[f'Outflow/PartType{itype}/All_CGMDeltaN_Out'][ihalo]=np.size(np.compress(outcgm_mask,masses_out))
                summed_acc_data[f'Outflow/PartType{itype}/All_CGMDeltaM_Out'][ihalo]=np.sum(np.compress(outcgm_mask,masses_out))
                summed_acc_data[f'Outflow/PartType{itype}/All_OtherHaloDeltaN_Out'][ihalo]=np.size(np.compress(outhalo_mask,masses_out))
                summed_acc_data[f'Outflow/PartType{itype}/All_OtherHaloDeltaM_Out'][ihalo]=np.sum(np.compress(outhalo_mask,masses_out))
                summed_acc_data[f'Outflow/PartType{itype}/All_RecycledDeltaN_Out'][ihalo]=np.size(np.compress(reaccreted_mask,masses_out))
                summed_acc_data[f'Outflow/PartType{itype}/All_RecycledDeltaM_Out'][ihalo]=np.sum(np.compress(reaccreted_mask,masses_out))


    collated_output_file_inflow=collated_output_file.create_group('Inflow')
    collated_output_file_outflow=collated_output_file.create_group('Outflow')

    for itype in itypes:
        collated_output_file_inflow_itype=collated_output_file_inflow.create_group(f'PartType{itype}')
        collated_output_file_outflow_itype=collated_output_file_outflow.create_group(f'PartType{itype}')
        for new_field in new_outputs_inflow:
            collated_output_file_inflow_itype.create_dataset(name=new_field,data=summed_acc_data[f'Inflow/PartType{itype}/'+new_field],dtype=np.float32)
        for new_field in new_outputs_outflow:
            collated_output_file_outflow_itype.create_dataset(name=new_field,data=summed_acc_data[f'Outflow/PartType{itype}/'+new_field],dtype=np.float32)
    collated_output_file.close()
    t2=time.time()
    print(f'Finished collating files in {t2-t1} sec')
    return None

########################### ADD PARTICLE DATA TO ACC DATA ###########################

def add_particle_acc_data(base_halo_data,accdata_path,datasets=None):
    """

    add_particle_acc_data : function 
	----------

    Add EAGLE particle data to the particles in accretion files. 

	Parameters
	----------
    base_halo_data: dict
        The base halo data dictionary (encodes particle data filepath, snap, particle histories).

    accdata_path : str
        The file path to the base hdf5 accretion data file. 

    datasets: list 
        Dictionary for each parttype of the desired datasets.

    Returns
	----------
        Requested gas datasets for snap 1 and snap 2, saved to file at accdata_path. 

    """
    print('Starting with I/O for adding particle data ...')
    t1_io=time.time()

    partdata_filetype=base_halo_data[-1]['Part_FileType']
    partdata_outname=base_halo_data[-1]['outname']

    #Determine particle types to extract from data type
    if 'EAGLE' in partdata_filetype:
        parttypes=[0,1,4,5]
    else:
        parttypes=[0,1]

    #Load in the accretion file and header details
    acc_file=h5py.File(accdata_path,'r+')
    snap3=acc_file['Header'].attrs['snap3']
    snap2=acc_file['Header'].attrs['snap2']
    snap1=acc_file['Header'].attrs['snap1']
    pre_depth=snap2-snap1
    post_depth=snap3-snap2
    ihalo_groups=sorted(list(acc_file.keys()))
    ihalo_groups_trunc=[ihalo_group for ihalo_group in ihalo_groups if 'ihalo_' in ihalo_group]
    ihalo_count=len(ihalo_groups_trunc)
    
    #Get the calculation data
    acc_filename=accdata_path.split('/')[-1]
    acc_directory_split=accdata_path.split('/')[:-1]
    for idir in acc_directory_split:#grab the directory
        if 'pre' in idir:
            calc_dir=idir
        else:
            pass
    if 'test' in calc_dir:
        test=True
    else:
        test=False
    num_processes=int((calc_dir.split('np')[-1]).split('_')[0])
    iprocess=int(acc_filename.split('_p')[-1][:3])
    
    #Initialise log file
    acc_log_dir=f"job_logs/acc_logs/"
    if not os.path.exists(acc_log_dir):
        os.mkdir(acc_log_dir)
    if test:
        run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_test/"
    else:
        run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}/"
    if not os.path.exists(run_log_dir):
        try:
            os.mkdir(run_log_dir)
        except:
            pass
    run_snap_log_dir=run_log_dir+f'snap_{str(snap2).zfill(3)}/'
    if not os.path.exists(run_snap_log_dir):
        try:
            os.mkdir(run_snap_log_dir)
        except:
            pass    
    if test:
        fname_log=run_snap_log_dir+f'partdata_p{str(iprocess).zfill(2)}_n{str(ihalo_count).zfill(6)}_test.log'
    else:
        fname_log=run_snap_log_dir+f'partdata_p{str(iprocess).zfill(2)}_n{str(ihalo_count).zfill(6)}.log'
    if os.path.exists(fname_log):
        os.remove(fname_log)

    #Write progress to log file
    with open(fname_log,"w") as progress_file:
        progress_file.write('Loading in data ...')
    progress_file.close()

    #START WITH I/O
    #Load particle histories
    parthist_file_snap2=h5py.File(f'part_histories/PartHistory_{str(snap2).zfill(3)}_{partdata_outname}.hdf5','r')
    parthist_file_snap1=h5py.File(f'part_histories/PartHistory_{str(snap1).zfill(3)}_{partdata_outname}.hdf5','r')
    parthist_IDs_snap1={str(itype):parthist_file_snap1[f'PartType{itype}']['ParticleIDs'].value for itype in parttypes}
    parthist_indices_snap1={str(itype):parthist_file_snap1[f'PartType{itype}']['ParticleIndex'].value for itype in parttypes}
    parthist_IDs_snap2={str(itype):parthist_file_snap2[f'PartType{itype}']['ParticleIDs'].value for itype in parttypes}
    parthist_indices_snap2={str(itype):parthist_file_snap2[f'PartType{itype}']['ParticleIndex'].value for itype in parttypes}
   
    #Load particle data (depending on simulation type)
    if 'EAGLE' in partdata_filetype:
        #Default datasets (will be added for snap1 and snap2)
        if datasets==None:
            datasets={}
            gas_datasets=['ParticleIDs',
                        'AExpMaximumTemperature',
                        'Coordinates',
                        'Density',
                        'InternalEnergy',
                        'MaximumTemperature',
                        'StarFormationRate',
                        'Temperature',
                        'Velocity']
            dm_datasets=['ParticleIDs',
                        'Coordinates',
                        'Velocity']
            star_datasets=['ParticleIDs',
                        'AExpMaximumTemperature',
                        'Coordinates',
                        'BirthDensity',
                        'InternalEnergy',
                        'MaximumTemperature',
                        'StellarFormationTime',
                        'Velocity']
            bh_datasets=['ParticleIDs',
                        'Coordinates',
                        'Velocity']
            datasets={'0':gas_datasets,'1':dm_datasets,'4':star_datasets,'5':bh_datasets}
        
        #record parttypes as those parsed
        parttypes=[int(key) for key in list(datasets.keys())]

        print('Reading in EAGLE snapshot data ...')
        EAGLE_boxsize=base_halo_data[snap1]['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap_1=read_eagle.EagleSnapshot(base_halo_data[snap1]['Part_FilePath'])
        EAGLE_Snap_1.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
        EAGLE_Snap_2=read_eagle.EagleSnapshot(base_halo_data[snap2]['Part_FilePath'])
        EAGLE_Snap_2.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
        
        particle_datasets_snap1={str(itype):{dataset:[] for dataset in datasets[str(itype)]} for itype in parttypes}
        particle_datasets_snap2={str(itype):{dataset:[] for dataset in datasets[str(itype)]} for itype in parttypes}
        for itype in parttypes:
            for dataset in datasets[str(itype)]:
                particle_datasets_snap1[str(itype)][dataset]=EAGLE_Snap_1.read_dataset(itype,dataset)
                particle_datasets_snap2[str(itype)][dataset]=EAGLE_Snap_2.read_dataset(itype,dataset)

    else:#non-eagle file -- GADGET OR SWIFT (don't have read routine)
        partdata_filetype='GADGET'
        if datasets==None:
            datasets={}
            gas_datasets=['ParticleIDs',
                        'Coordinates',
                        'Density',
                        'InternalEnergy',
                        'Velocity']
            dm_datasets=['ParticleIDs',
                        'Coordinates',
                        'Velocity']
            datasets={'0':gas_datasets,'1':dm_datasets}
        
        #record parttypes as those parsed
        parttypes=[int(key) for key in list(datasets.keys())]

        PartFile_Snap_1=h5py.File(base_halo_data[snap1]['Part_FilePath'],'r')
        PartFile_Snap_2=h5py.File(base_halo_data[snap2]['Part_FilePath'],'r')

        particle_datasets_snap1={{dataset:[] for dataset in datasets[str(itype)]} for itype in parttypes}
        particle_datasets_snap2={{dataset:[] for dataset in datasets[str(itype)]} for itype in parttypes}
        for itype in parttypes:
            for dataset in datasets[str(itype)]:
                particle_datasets_snap1[str(itype)][dataset]=EAGLE_Snap_1.read_dataset(itype,dataset)
                particle_datasets_snap2[str(itype)][dataset]=EAGLE_Snap_2.read_dataset(itype,dataset)
    
    t2_io=time.time()

    #Save the shape and type of each dataset for each particle
    dataset_shapes={str(itype):{dataset:[] for dataset in datasets[str(itype)]} for itype in parttypes}
    dataset_types={str(itype):{dataset:[] for dataset in datasets[str(itype)]} for itype in parttypes}
    for itype in parttypes:
        for dataset in datasets[str(itype)]:
            dataset_shapes[str(itype)][dataset]=np.size(particle_datasets_snap2[str(itype)][dataset][0])
            dataset_types[str(itype)][dataset]=np.float32
        dataset_types[str(itype)]['ParticleIDs']=np.int64
    print(f'Finished with I/O for adding particle data in {t2_io-t1_io:.2f} sec')
    
    #Write progress to log file
    with open(fname_log,"a") as progress_file:
        progress_file.write(" \n")
        progress_file.write(f'Finished I/O in {t2_io-t1_io}! Entering main halo loop... \n')
    progress_file.close()
    
    #Enter main halo loop
    for iihalo,ihalo_group in enumerate(ihalo_groups_trunc):
        t1_halo=time.time()
        #Write progress to log file
        print(f'Processing {ihalo_group}')
        with open(fname_log,"a") as progress_file:
            progress_file.write(" \n")
            progress_file.write(f'Processing halo {ihalo_group} ({iihalo+1} out of {ihalo_count}) \n')
        progress_file.close()

        #Load the particle IDs for inflow and outflow for this halo 
        IDs_in_snap1={str(itype):acc_file[ihalo_group]['Inflow'][f'PartType{itype}']['ParticleIDs'].value for itype in parttypes}
        IDs_out_snap2={str(itype):acc_file[ihalo_group]['Outflow'][f'PartType{itype}']['ParticleIDs'].value for itype in parttypes}
        
        #Initialise datasets
        ihalo_datasets_inflow={str(itype):{} for itype in parttypes}
        ihalo_datasets_outflow={str(itype):{} for itype in parttypes}

        #Check if halo is valid
        if np.size(IDs_in_snap1['0'])==1 and type(IDs_in_snap1['0'])==np.float16:#if an invalid halo, save nan datasets
            print(f'Not processing {ihalo_group}')
            for itype in parttypes:
                for dataset in datasets[str(itype)]:
                    ihalo_datasets_inflow[str(itype)][f'snap1_{dataset}']=np.nan
                    ihalo_datasets_outflow[str(itype)][f'snap1_{dataset}']=np.nan
                    ihalo_datasets_inflow[str(itype)][f'snap2_{dataset}']=np.nan
                    ihalo_datasets_outflow[str(itype)][f'snap2_{dataset}']=np.nan

        #If we have a valid halo, proceed to extract datasets for particles
        else:            
            for itype in parttypes:
                #initialise empty halo datasets
                for dataset in datasets[str(itype)]:
                    ihalo_datasets_inflow[str(itype)][f'snap2_{dataset}']=np.zeros((len(IDs_in_snap1[str(itype)]),dataset_shapes[str(itype)][dataset]))
                    ihalo_datasets_inflow[str(itype)][f'snap1_{dataset}']=np.zeros((len(IDs_in_snap1[str(itype)]),dataset_shapes[str(itype)][dataset]))
                    ihalo_datasets_outflow[str(itype)][f'snap2_{dataset}']=np.zeros((len(IDs_out_snap2[str(itype)]),dataset_shapes[str(itype)][dataset]))
                    ihalo_datasets_outflow[str(itype)][f'snap1_{dataset}']=np.zeros((len(IDs_out_snap2[str(itype)]),dataset_shapes[str(itype)][dataset]))  
            
            #Iterate through each parttype to find particle data
            for itype in parttypes:
                ihalo_itype_npart_in=len(IDs_in_snap1[str(itype)])
                ihalo_itype_npart_out=len(IDs_out_snap2[str(itype)])
                
                #Find index and type data for INFLOW particles at snap 1 and snap 2
                # snap 1 - types were taken here
                ihalo_itype_inflow_data_snap1=get_particle_indices(base_halo_data,
                                                    SortedIDs=parthist_IDs_snap1,
                                                    SortedIndices=parthist_indices_snap1,
                                                    PartIDs=IDs_in_snap1[str(itype)],
                                                    PartTypes=np.array(np.ones(ihalo_itype_npart_in)*itype,dtype=int),
                                                    snap_taken=snap1,
                                                    snap_desired=snap1)
                #snap 2 - types were taken at snap 1 so transformation may have occurred
                ihalo_itype_inflow_data_snap2=get_particle_indices(base_halo_data,
                                                    SortedIDs=parthist_IDs_snap2,
                                                    SortedIndices=parthist_indices_snap2,
                                                    PartIDs=IDs_in_snap1[str(itype)],
                                                    PartTypes=np.array(np.ones(ihalo_itype_npart_in)*itype,dtype=int),
                                                    snap_taken=snap1,
                                                    snap_desired=snap2)
                #Find index and type data for OUTFLOW particles at snap 1 and snap 2
                # snap 1 - types were taken at snap 2 so transformation may have occurred
                ihalo_itype_outflow_data_snap1=get_particle_indices(base_halo_data,
                                                    SortedIDs=parthist_IDs_snap1,
                                                    SortedIndices=parthist_indices_snap1,
                                                    PartIDs=IDs_out_snap2[str(itype)],
                                                    PartTypes=np.array(np.ones(ihalo_itype_npart_out)*itype,dtype=int),
                                                    snap_taken=snap2,
                                                    snap_desired=snap1)
                # snap 2 - types were taken here
                ihalo_itype_outflow_data_snap2=get_particle_indices(base_halo_data,
                                                    SortedIDs=parthist_IDs_snap2,
                                                    SortedIndices=parthist_indices_snap2,
                                                    PartIDs=IDs_out_snap2[str(itype)],
                                                    PartTypes=np.array(np.ones(ihalo_itype_npart_out)*itype,dtype=int),
                                                    snap_taken=snap2,
                                                    snap_desired=snap2)
                
                #the above tuples have (type | historyindex | particleindex)

                #Now iterate through each dataset and each particle (inflow and outflow) and find its data
                for dataset in datasets[str(itype)]:
                    dataset_size=dataset_shapes[str(itype)][dataset];dataset_type=dataset_types[str(itype)][dataset]
                    
                    #inflow particles
                    for iipart_inflow in range(ihalo_itype_npart_in):
                        ipart_inflow_snap1_type=ihalo_itype_inflow_data_snap1[0][iipart_inflow]
                        ipart_inflow_snap1_partdataindex=ihalo_itype_inflow_data_snap1[2][iipart_inflow]
                        ipart_inflow_snap2_type=ihalo_itype_inflow_data_snap2[0][iipart_inflow]#maybe transformed
                        ipart_inflow_snap2_partdataindex=ihalo_itype_inflow_data_snap2[2][iipart_inflow]#maybe transformed
                        
                        #non-transformed
                        ihalo_datasets_inflow[str(itype)][f'snap1_{dataset}'][iipart_inflow]=particle_datasets_snap1[str(ipart_inflow_snap1_type)][ipart_inflow_snap1_partdataindex]
                        #transformed
                        try:
                            ihalo_datasets_inflow[str(itype)][f'snap2_{dataset}'][iipart_inflow]=particle_datasets_snap2[str(ipart_inflow_snap2_type)][ipart_inflow_snap2_partdataindex]
                        except:
                            nan_output=[np.nan]*dataset_shapes[str(itype)][dataset]
                            if np.size(nan_output)==1:
                                nan_output=nan_output[0]
                            ihalo_datasets_inflow[str(itype)][f'snap2_{dataset}'][iipart_inflow]=nan_output
                    
                    #outflow particles
                    for iipart_outflow in range(ihalo_itype_npart_in):
                        ipart_outflow_snap1_type=ihalo_itype_outflow_data_snap1[0][outflow]#maybe transformed
                        ipart_outflow_snap1_partdataindex=ihalo_itype_outflow_data_snap1[2][outflow]#maybe transformed
                        ipart_outflow_snap2_type=ihalo_itype_outflow_data_snap2[0][outflow]
                        ipart_outflow_snap2_partdataindex=ihalo_itype_outfloww_data_snap2[2][outflow]
                        
                        #non-transformed
                        ihalo_datasets_outflow[str(itype)][f'snap2_{dataset}'][iipart_outflow]=particle_datasets_snap2[str(ipart_outflow_snap2_type)][ipart_outflow_snap2_partdataindex]
                        #transformed
                        try:
                            ihalo_datasets_outflow[str(itype)][f'snap1_{dataset}'][iipart_outflow]=particle_datasets_snap1[str(ipart_outflow_snap1_type)][ipart_inflow_snap1_partdataindex]
                        except:
                            nan_output=[np.nan]*dataset_shapes[str(itype)][dataset]
                            if np.size(nan_output)==1:
                                nan_output=nan_output[0]
                            ihalo_datasets_outflow[str(itype)][f'snap1_{dataset}'][iipart_outflow]=nan_output
               
        h_val=base_halo_data[-1]['SimulationInfo']['h_val']
        scalefactor_snap1=base_halo_data[snap1]['SimulationInfo']['ScaleFactor']
        scalefactor_snap2=base_halo_data[snap2]['SimulationInfo']['ScaleFactor']

        for dataset in datasets[str(itype)]:
            try:
                del acc_file[ihalo_group]['Inflow'][f'PartType{itype}'][f'snap2_{dataset}']
                del acc_file[ihalo_group]['Inflow'][f'PartType{itype}'][f'snap1_{dataset}']
                del acc_file[ihalo_group]['Outflow'][f'PartType{itype}'][f'snap2_{dataset}']
                del acc_file[ihalo_group]['Outflow'][f'PartType{itype}'][f'snap1_{dataset}']
                print(f'Overwriting data for {ihalo}, dataset {dataset}')

                acc_file[ihalo_group]['Inflow'][f'PartType{itype}'].create_dataset(f'snap2_{dataset}',data=ihalo_datasets_inflow[str(itype)][f'snap2_{dataset}'],dtype=dataset_types[str(itype)][dataset])
                acc_file[ihalo_group]['Inflow'][f'PartType{itype}'].create_dataset(f'snap1_{dataset}',data=ihalo_datasets_inflow[str(itype)][f'snap1_{dataset}'],dtype=dataset_types[str(itype)][dataset])
                acc_file[ihalo_group]['Outflow'][f'PartType{itype}'].create_dataset(f'snap2_{dataset}',data=ihalo_datasets_outflow[str(itype)][f'snap2_{dataset}'],dtype=dataset_types[str(itype)][dataset])
                acc_file[ihalo_group]['Outflow'][f'PartType{itype}'].create_dataset(f'snap1_{dataset}',data=ihalo_datasets_outflow[str(itype)][f'snap1_{dataset}'],dtype=dataset_types[str(itype)][dataset])
            except:

                acc_file[ihalo_group]['Inflow'][f'PartType{itype}'].create_dataset(f'snap2_{dataset}',data=ihalo_datasets_inflow[str(itype)][f'snap2_{dataset}'],dtype=dataset_types[str(itype)][dataset])
                acc_file[ihalo_group]['Inflow'][f'PartType{itype}'].create_dataset(f'snap1_{dataset}',data=ihalo_datasets_inflow[str(itype)][f'snap1_{dataset}'],dtype=dataset_types[str(itype)][dataset])
                acc_file[ihalo_group]['Outflow'][f'PartType{itype}'].create_dataset(f'snap2_{dataset}',data=ihalo_datasets_outflow[str(itype)][f'snap2_{dataset}'],dtype=dataset_types[str(itype)][dataset])
                acc_file[ihalo_group]['Outflow'][f'PartType{itype}'].create_dataset(f'snap1_{dataset}',data=ihalo_datasets_outflow[str(itype)][f'snap1_{dataset}'],dtype=dataset_types[str(itype)][dataset])
            
        t2_halo=time.time()

        with open(fname_log,"a") as progress_file:
            progress_file.write(" \n")
            progress_file.write(f'Done processing halo {ihalo_group} ({iihalo+1} out of {ihalo_count}) - took {t2_halo-t1_halo} \n')
            progress_file.write(f'Progress: {(iihalo+1)/ihalo_count*100:.1f}%\n')
        progress_file.close()

    acc_file.close()

########################### READ ALL ACC DATA ###########################

def get_particle_acc_data(directory,halo_index_list=None):

    print('Indexing halos ...')
    t1=time.time()
    if not directory.endswith('/'):
        directory=directory+'/'
    accdata_filelist=os.listdir(directory)
    accdata_filelist_trunc=sorted([directory+accfile for accfile in accdata_filelist if (('summed' not in accfile) and ('px' not in accfile) and ('DS' not in accfile))])
    accdata_files=[h5py.File(accdata_filename,'r') for accdata_filename in accdata_filelist_trunc]
    accdata_halo_lists=[sorted(list(accdata_file.keys()))[1:] for accdata_file in accdata_files]
    accdata_halo_lists_flattened=flatten(accdata_halo_lists)

    if halo_index_list==None:
        halo_index_list=list(range(len(accdata_halo_lists_flattened)-1))
    
    if type(halo_index_list)==int:
        halo_index_list=[halo_index_list]
    else:
        halo_index_list=list(halo_index_list)
    
    desired_num_halos=len(halo_index_list)
    ihalo_files=np.ones(desired_num_halos)+np.nan
    
    for iihalo,ihalo in enumerate(halo_index_list):
        for ifile,ihalo_list in enumerate(accdata_halo_lists):
            if f'ihalo_'+str(ihalo).zfill(6) in ihalo_list:
                ihalo_files[iihalo]=ifile
                break
            else:
                pass
    t2=time.time()
    print(f'Done indexing halos in {t2-t1:.1f} sec')


    if 'EAGLE' in directory:
        parttypes=[0,1,4]
    else:
        parttypes=[0,1]

    partfields_in={}
    partfields_out={}
    for itype in parttypes:
        ihalo_group0=list(accdata_files[0].keys())[-1]
        fields_in_itype=list(accdata_files[0][ihalo_group0]['Inflow'][f'PartType{itype}'].keys())
        fields_out_itype=list(accdata_files[0][ihalo_group0]['Outflow'][f'PartType{itype}'].keys())
        partfields_in[str(itype)]=fields_in_itype
        partfields_out[str(itype)]=fields_out_itype
    particle_acc_data_in={f"PartType{itype}":{field: [[] for i in range(desired_num_halos)] for field in partfields_in[str(itype)]} for itype in parttypes}
    particle_acc_data_out={f"PartType{itype}":{field: [[] for i in range(desired_num_halos)] for field in partfields_out[str(itype)]} for itype in parttypes}
    particle_acc_files=[]    

    print('Now retrieving halo data from file ...')
    t1=time.time()
    for iihalo,ihalo in enumerate(halo_index_list):
        ihalo_name='ihalo_'+str(ihalo).zfill(6)
        ifile=ihalo_files[iihalo]
        if iihalo%1000==0:
            print(f'{iihalo/desired_num_halos*100:.1f}% of halo data loaded')
        for itype in parttypes:
            for field in partfields_in[str(itype)]:
                ihalo_itype_ifield=accdata_files[int(ihalo_files[iihalo])][ihalo_name+f'/Inflow/PartType{itype}/'+field].value
                particle_acc_data_in[f'PartType{itype}'][field][iihalo]=ihalo_itype_ifield
            for field in partfields_out[str(itype)]:
                ihalo_itype_ifield=accdata_files[int(ihalo_files[iihalo])][ihalo_name+f'/Outflow/PartType{itype}/'+field].value
                particle_acc_data_out[f'PartType{itype}'][field][iihalo]=ihalo_itype_ifield

    particle_acc_data={"Inflow":particle_acc_data_in,"Outflow":particle_acc_data_out}
    t2=time.time()
    print(f'Done in {t2-t1}')

    return particle_acc_data
 
########################### READ SUMMED ACC DATA ###########################

def get_summed_acc_data(base_halo_data,accdata_path):

    """

    read_acc_rate_file : function
	----------

    Read the accretion data from a certain file. 

	Parameters
	----------
    path : string 
        Indicates the file in which the accretion data is stored (nominally acc_data/AccretionData_snap{snap2}_pre{pre_depth}_post{post_depth}.hdf5)

	Returns
	----------
    
    accretion_data : dict
        With different fields for inflow and outflow. 

    Each dictionary entry will be of length n_halos, and each of these entries will be a dictionary

    """
    # Define output fields

    # Load collated file
    hdf5file=h5py.File(accdata_path,'r')

    # Load in metadata
    acc_metadata=dict()
    hdf5header_attrs=list(hdf5file['/Header'].attrs)
    for attribute in hdf5header_attrs:
        acc_metadata[attribute]=hdf5file['/Header'].attrs[attribute]

    if 'pre_depth' not in hdf5header_attrs:
        acc_metadata['pre_depth']=int(accdata_path.split('pre')[-1][:2])
    if 'post_depth' not in hdf5header_attrs:
        acc_metadata['post_depth']=int(accdata_path.split('post')[-1][:2])
    if 'outname' not in hdf5header_attrs:
        acc_metadata['outname']=str(base_halo_data[-1]['outname'])

    if not type(acc_metadata['outname'])==str:
        acc_metadata['outname']=acc_metadata['outname'].decode('utf-8')

    # Initialise output data
    total_num_halos=acc_metadata['total_num_halos']
    group_list=list(hdf5file.keys())
    part_group_list=['PartType'+str(itype) for itype in [0,1,4]]
    acc_data_inflow={part_group:{} for part_group in part_group_list}
    acc_data_outflow={part_group:{} for part_group in part_group_list}
    
    acc_fields_inflow=[
    "All_TotalDeltaM_In",
    "All_TotalDeltaN_In",
    "All_CosmologicalDeltaN_In",
    'All_CosmologicalDeltaM_In',
    'All_CGMDeltaN_In',
    'All_CGMDeltaM_In',
    'All_ClumpyDeltaN_In',
    'All_ClumpyDeltaM_In',
    'All_PrimordialDeltaN_In',
    'All_PrimordialDeltaM_In',
    'All_ProcessedCosmologicalDeltaN_In',
    'All_ProcessedCosmologicalDeltaM_In',   
    "Stable_TotalDeltaM_In",
    "Stable_TotalDeltaN_In",
    "Stable_CosmologicalDeltaN_In",
    'Stable_CosmologicalDeltaM_In',
    'Stable_CGMDeltaN_In',
    'Stable_CGMDeltaM_In',
    'Stable_ClumpyDeltaN_In',
    'Stable_ClumpyDeltaM_In',
    'Stable_PrimordialDeltaN_In',
    'Stable_PrimordialDeltaM_In',
    'Stable_ProcessedCosmologicalDeltaN_In',
    'Stable_ProcessedCosmologicalDeltaM_In'
    ]

    acc_fields_outflow=[
    "All_TotalDeltaM_Out",
    "All_TotalDeltaN_Out",
    "All_FieldDeltaM_Out",
    "All_FieldDeltaN_Out",
    "All_CGMDeltaM_Out",
    "All_CGMDeltaN_Out",
    "All_OtherHaloDeltaM_Out",
    "All_OtherHaloDeltaN_Out",
    "All_RecycledDeltaN_Out",#at snap 3
    "All_RecycledDeltaM_Out"]#at snap 3

    for part_group_name in part_group_list:
        for dataset in acc_fields_inflow:
            try:
                acc_data_inflow[part_group_name][dataset]=hdf5file['Inflow/'+part_group_name+'/'+dataset].value
            except:
                # print(f'Couldnt retrieve {part_group_name}/{dataset}')
                pass

    for part_group_name in part_group_list:
        for dataset in acc_fields_outflow:
            try:
                acc_data_outflow[part_group_name][dataset]=hdf5file['Outflow/'+part_group_name+'/'+dataset].value    
            except:
                # print(f'Couldnt retrieve {part_group_name}/{dataset}')
                pass
    acc_data={'Inflow':acc_data_inflow,'Outflow':acc_data_outflow}
    
    return acc_metadata, acc_data

