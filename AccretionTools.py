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
from ParticleTools import *
from pandas import DataFrame as df

# ########################### CREATE PARTICLE HISTORIES ###########################

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
        snap_fof_particle_data=get_FOF_particle_lists(base_halo_data,snap)#don't need to add subhalo particles as we have each subhalo separately
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
        print(f"Loaded, concatenated and sorted halo particle lists for snap {snap} in {t2-t1} sec")
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
            print(f"Mapped IDs to indices for all {PartNames[itype]} particles at snap {snap} in {t2-t1} sec")
            
            # Flip switches of new particles
            ipart_switch=0
            all_Structure_IDs_itype=structure_Particles_bytype[str(itype)]["ParticleIDs"]
            all_Structure_HostStructureID_itype=np.int64(structure_Particles_bytype[str(itype)]["HostStructureID"])
            all_Structure_IDs_itype_partindex=binary_search(sorted_list=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"],items=all_Structure_IDs_itype)
            
            print("Adding host indices ...")
            Particle_History_Flags[str(itype)]["HostStructureID"][(all_Structure_IDs_itype_partindex,)]=all_Structure_HostStructureID_itype
            print(f"Added host halos in {t2-t1} sec for {PartNames[itype]} particles")

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
    PartHistory_xxx-outname.hdf5 : hdf5 file with datasets

        /PartTypeX/Processed_L1 #no_snaps this particle has been in a halo 
        /PartTypeX/HostStructure
        /PartTypeX/ParticleIDs
        /PartTypeX/ParticleIndex

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
        current_hosts_DM=infile_file["PartType1/HostStructure"].value##ordered by ID
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
    
########################### GENERATE DETAILED ACCRETION DATA ###########################

def gen_accretion_data_detailed_serial(base_halo_data,snap=None,halo_index_list=None,pre_depth=1,post_depth=1,outflow=False,write_partdata=False):
    
    """

    gen_accretion_data_detailed_serial : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists from VELOCIraptor FOF outputs. 

    ** note: particle histories, base_halo_data and halo particle data must have been generated as per gen_particle_history_serial (this file),
             gen_base_halo_data in STFTools.py and dump_structure_particle_data in STFTools.py

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
        Initial snap (s1) for calculation will be s1=snap-pre_depth, and we will check particle histories at s1.

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
            "ave_LookbackTime" (snap 1 -> snap 2)
            "delta_LookbackTime" (snap 1 -> snap 2)
            "snap1_z"
            "snap2_z"
            "snap3_z"
            "ave_z (snap 1 -> snap 2)

        There is a group for each halo: ihalo_xxxxxx
            - we iterate through each SO region and find the corresponding halo. 
            - return nan datasets if cannot find match for halo.

        Each halo group with attributes:
        "snapx_com"
        "snapx_cminpot"
        "snapx_cmbp"
        "snapx_vmax"
        "snapx_v"
        "snapx_M_200crit"
        "snapx_R_200mean"
        "snapx_R_200crit"
        
        Inflow:
            For each particle type /PartTypeX/:
                Each of the following datasets will have n_in particles - the n_in initially selected as accretion "candidates" from being in the SO particle list at snap 2. 
                The particles are categorised by their type at SNAP 1 (i.e. prior to entering the halo)

                'ParticleIDs': ParticleID (in particle data for given type) of all accreted particles.
                'Masses': Mass of all accretion candidates at snap 1. 
                
                'snap1_Processed': How many snaps has this particle been part of any structure in the past.
                'snap1_FOF': Is the candidate particle included in the FOF at snap 1? 
                'snap1_Bound': Is the candidate particle bound (in the FOF list) at snap 1?
                'snap1_Coordinates': The absolute coordinates of the particle at snap 1. 
                'snap1_Velocity': The absolute instantaneous velocity of the particle at snap 1. 
                'snap1_r_xx': The relative coordinate of the particle at snap 1 relative to halo center from xx. 
                'snap1_rabs_xx': The radius of the particle at snap 1 relative to halo center from xx. 
                'snap1_vrad': The instantaneous radial velocity of the particle at snap 1 relative to halo (using com). 
                'snap1_vtan': The instantaneous tangential velocity of the particle at snap 1 relative to halo (using com). 
                'snap2_FOF': Is the candidate particle included in the FOF at snap 2? 
                'snap2_Bound': Is the candidate particle bound (in the FOF list) at snap 2?
                'snap2_Coordinates': The absolute coordinates of the particle at snap 2. 
                'snap2_Velocity': The absolute instantaneous velocity of the particle at snap 2. 
                'snap2_r_xx': The relative coordinate of the particle at snap 2 relative to halo center from xx. 
                'snap2_rabs_xx': The radius of the particle at snap 2 relative to halo center from xx. 
                'snap2_vrad': The instantaneous radial velocity of the particle at snap 2 relative to halo (using com). 
                'snap2_vtan': The instantaneous tangential velocity of the particle at snap 2 relative to halo (using com). 
                'snap3_FOF': Is the candidate particle included in the FOF at snap 3? 
                'snap3_Bound': Is the candidate particle bound (in the FOF list) at snap 3?
                'ave_vrad_xx': Average radial velocity from snap1_r -> snap2_r, where r is taken relative to halo center from xx. 
                
                Where xx can be from com, cminpot, or cmbp. 

        Outflow: 
            For each particle type /PartTypeX/:
                Each of the following datasets will have n_out particles - the n initially selected by being in the particle list at snap 1.
                The particles are categorised by their type at SNAP 2 (i.e. after entering the halo)

                'ParticleIDs': ParticleID (in particle data for given type) of all accreted particles.
                'Masses': Mass of all accretion candidates at snap 1. 

                'snap1_Processed': How many snaps has this particle been part of any structure in the past.
                'snap1_FOF': Is the candidate particle included in the FOF at snap 1? 
                'snap1_Bound': Is the candidate particle bound (in the FOF list) at snap 1?
                'snap1_Coordinates': The absolute coordinates of the particle at snap 1. 
                'snap1_Velocity': The absolute instantaneous velocity of the particle at snap 1. 
                'snap1_r_xx': The relative coordinate of the particle at snap 1 relative to halo center from xx. 
                'snap1_rabs_xx': The radius of the particle at snap 1 relative to halo center from xx. 
                'snap1_vrad': The instantaneous radial velocity of the particle at snap 1 relative to halo (using com). 
                'snap1_vtan': The instantaneous tangential velocity of the particle at snap 1 relative to halo (using com). 
                'snap2_FOF': Is the candidate particle included in the FOF at snap 2? 
                'snap2_Bound': Is the candidate particle bound (in the FOF list) at snap 2?
                'snap2_Coordinates': The absolute coordinates of the particle at snap 2. 
                'snap2_Velocity': The absolute instantaneous velocity of the particle at snap 2. 
                'snap2_r_xx': The relative coordinate of the particle at snap 2 relative to halo center from xx. 
                'snap2_rabs_xx': The radius of the particle at snap 2 relative to halo center from xx. 
                'snap2_vrad': The instantaneous radial velocity of the particle at snap 2 relative to halo (using com). 
                'snap2_vtan': The instantaneous tangential velocity of the particle at snap 2 relative to halo (using com). 
                'snap3_FOF': Is the candidate particle included in the FOF at snap 3? 
                'snap3_Bound': Is the candidate particle bound (in the FOF list) at snap 3?
                'ave_vrad_xx': Average radial velocity from snap1_r -> snap2_r, where r is taken relative to halo center from xx. 

                Where xx can be from com, cminpot, or cmbp. 

        Where there will be num_total_halos ihalo datasets. 
    
    """
    
    t1_init=time.time()

    ##### Processing inputs #####

    # Processing the snap inputs
    snap1=snap-pre_depth
    snap2=snap
    snap3=snap+post_depth
    snaps=[snap1,snap2,snap3]
    
    # Processing the desired halo index list
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
    
    # Find the indices of halos at snap1 and snap3 (ordered by snap2 halo indices)
    halo_index_list_snap1=[find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=pre_depth) for ihalo in halo_index_list_snap2]
    halo_index_list_snap3=[find_descen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=post_depth) for ihalo in halo_index_list_snap2]

    # Parameters for calculation
    r200_facs={'Inflow':[0.125,0.25,0.375,0.5,0.75,1,1.5,2],'Outflow':[1]} # factors of r200 to calculate SO accretion/outflow to 
    vmax_facs={'Inflow':[-1,0,0.125,0.25,0.375,0.5,0.75,1],'Outflow':[0.125]} # factors of ave_vmax to cut accretion/outflow for 

    halo_defnames={}
    halo_defnames["Inflow"]=np.concatenate([['FOF-haloscale','FOF-subhaloscale'],['SO-r200_fac'+str(ir200_fac+1) for ir200_fac in range(len(r200_facs["Inflow"]))]])
    halo_defnames["Outflow"]=np.concatenate([['FOF-haloscale','FOF-subhaloscale'],['SO-r200_fac'+str(ir200_fac+1) for ir200_fac in range(len(r200_facs["Outflow"]))]])
    
    ihalo_cube_rfac=1.25
    vel_conversion=978.462 #Mpc/Gyr to km/s
    use='cminpot'
    compression='gzip'

    # Create log file and directories, initialising outputs
    if True:
        #Logs
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

        # Assigning snap
        if snap==None:
            snap=len(base_halo_data)-1#if not given snap, just use the last one

    # Create output file with metadata attributes
    run_outname=base_halo_data[snap]['outname']#extract output name (simulation name)
    outfile_name=calc_snap_dir+'AccretionData_pre'+str(pre_depth).zfill(2)+'_post'+str(post_depth).zfill(2)+'_snap'+str(snap).zfill(3)+'_p'+str(iprocess).zfill(3)+'.hdf5'
    
    if not os.path.exists(outfile_name):#if the accretion file doesn't exists, initialise with header
        print(f'Initialising output file at {outfile_name}...')
        output_hdf5=h5py.File(outfile_name,"w")#initialise file object
        # Make header for accretion data  based on base halo data 
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
    else:
        print(f'Opening existing output file at {outfile_name} ...')
        output_hdf5=h5py.File(outfile_name,"r+")#initialise file object

    # Now find which simulation type we're dealing with
    part_filetype=base_halo_data[snap]["Part_FileType"]
    print(f'Particle data type: {part_filetype}')

    # Standard particle type names from simulation
    PartNames=['gas','DM','','','star','BH']

    # Assign the particle types we're considering 
    if part_filetype=='EAGLE':
        PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
        Mass_DM=base_halo_data[snap2]['SimulationInfo']['Mass_DM_Physical']
        Mass_Gas=base_halo_data[snap2]['SimulationInfo']['Mass_Gas_Physical']

    ##### Loading in Data #####
    #Load in FOF particle lists: snap 1, snap 2, snap 3
    FOF_Part_Data={}
    FOF_Part_Data[str(snap1)]=get_FOF_particle_lists(base_halo_data,snap1,halo_index_list=halo_index_list_snap1)
    FOF_Part_Data[str(snap2)]=get_FOF_particle_lists(base_halo_data,snap2,halo_index_list=halo_index_list_snap2)
    FOF_Part_Data[str(snap3)]=get_FOF_particle_lists(base_halo_data,snap3,halo_index_list=halo_index_list_snap3)
    FOF_Part_Data_fields=list(FOF_Part_Data[str(snap1)].keys()) #fields from FOF data

    #Particle data filepath
    hval=base_halo_data[snap1]['SimulationInfo']['h_val'];scalefactors={}
    scalefactors={str(snap):base_halo_data[snap]['SimulationInfo']['ScaleFactor'] for snap in snaps}
    Part_Data_FilePaths={str(snap):base_halo_data[snap]['Part_FilePath'] for snap in snaps}
    Part_Data_fields=['Coordinates','Velocity','Mass','ParticleIDs'] #the fields to be read initially from EAGLE cubes
    Part_Data_comtophys={str(snap):{'Coordinates':scalefactors[str(snap)]/hval, #conversion factors for EAGLE cubes
                                    'Velocity':scalefactors[str(snap)]/hval,
                                    'Mass':10.0**10/hval,
                                    'ParticleIDs':1} for snap in snaps}
    
    #Load in particle histories: snap 1
    print(f'Retrieving & organising particle histories for snap = {snap1} ...')
    Part_Histories_File_snap1=h5py.File("part_histories/PartHistory_"+str(snap1).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIDs'].value for parttype in PartTypes}
    Part_Histories_Index_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIndex'].value for parttype in PartTypes}
    Part_Histories_npart_snap1={str(parttype):len(Part_Histories_IDs_snap1[str(parttype)]) for parttype in PartTypes}
    Part_Histories_HostStructure_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/HostStructure'].value for parttype in PartTypes}
    Part_Histories_Processed_L1_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/Processed_L1'].value for parttype in [0,1]}
    Part_Histories_Processed_L1_snap1[str(4)]=np.ones(Part_Histories_npart_snap1[str(4)]);Part_Histories_Processed_L1_snap1[str(5)]=np.ones(Part_Histories_npart_snap1[str(5)])

    #Load in particle histories: snap 2
    print(f'Retrieving & organising particle histories for snap = {snap2} ...')
    Part_Histories_File_snap2=h5py.File("part_histories/PartHistory_"+str(snap2).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIDs'].value for parttype in PartTypes}
    Part_Histories_Index_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIndex'].value for parttype in PartTypes}
    Part_Histories_HostStructure_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/HostStructure'].value for parttype in PartTypes}
    Part_Histories_npart_snap2={str(parttype):len(Part_Histories_IDs_snap2[str(parttype)]) for parttype in PartTypes}
    
    print()
    t2_init=time.time()
    print('*********************************************************')
    print(f'Done initialising in {(t2_init-t1_init):.2f} sec - entering main halo loop ...')
    print('*********************************************************')

    with open(fname_log,"a") as progress_file:
        progress_file.write(f'Done initialising in {(t2_init-t1_init):.2f} sec - entering main halo loop ...\n')
    progress_file.close()

    ##### Initialising outputs #####
    #Particle
    if write_partdata:
        #hdf5 group
        particle_output_hdf5=output_hdf5.create_group('Particle')
        
        #output dtypes
        output_fields_dtype={}
        output_fields_float16=['ave_vrad_com',"r_com","rabs_com","vrad_com","vtan_com","Mass"]
        for field in output_fields_float16:
            output_fields_dtype[field]=np.float16

        output_fields_int64=["ParticleIDs","Structure"]
        for field in output_fields_int64:
            output_fields_dtype[field]=np.int64
        
        output_fields_int8=["Processed","Particle_InFOF","Particle_Bound","Particle_InHost"]
        for field in output_fields_int8:
            output_fields_dtype[field]=np.int8 

    #Integrated
    num_halos_thisprocess=len(halo_index_list_snap2)
    #hdf5 group
    integrated_output_hdf5=output_hdf5.create_group('Integrated')
    integrated_output_hdf5.create_dataset('ihalo_list',data=halo_index_list_snap2)

    #inflow: all, from field, from CGM, or from other structure (merger)
    #outflow: all, to field, to CGM, or to other structure (transfer)
    output_processedgroups=['Total','Unprocessed','Processed']

    output_datasets={'Inflow':['Gross','Field','Transfer'],
                     'Outflow':['Gross']}
       
    #for inflow create groups
    if outflow:
        output_groups=['Inflow','Outflow']
    else:
        output_groups=['Inflow']

    for output_group in output_groups:
        integrated_output_hdf5.create_group(output_group)
        for itype in PartTypes:
            itype_key=f'PartType{itype}'
            integrated_output_hdf5[output_group].create_group(itype_key)
            for halo_defname in halo_defnames[output_group]:
                integrated_output_hdf5[output_group][itype_key].create_group(halo_defname)
                if 'FOF' not in halo_defname or output_group=='Outflow':
                    datasets=['Gross']
                    processedgroups=['Total']
                else:
                    datasets=output_datasets[output_group]
                    processedgroups=output_processedgroups

                #both calculations - each Vmax cut
                for ivmax_fac, vmax_fac in enumerate(vmax_facs[output_group]):
                    ivmax_key=f'vmax_fac{ivmax_fac+1}'
                    integrated_output_hdf5[output_group][itype_key][halo_defname].create_group(ivmax_key);integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
                    for processedgroup in processedgroups:
                        integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key].create_group(processedgroup)
                        for dataset in datasets:
                            if output_group=='Inflow':
                                suffix='In'
                            else:
                                suffix='Out'

                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'All_'+dataset+f'_DeltaM_{suffix}',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'All_'+dataset+f'_DeltaN_{suffix}',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'Stable_'+dataset+f'_DeltaM_{suffix}',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.int16)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'Stable_'+dataset+f'_DeltaN_{suffix}',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.int16)

    ####################################################################################################################################################################################
    ####################################################################################################################################################################################
    ########################################################################### MAIN HALO LOOP #########################################################################################
    ####################################################################################################################################################################################
    ####################################################################################################################################################################################

    for iihalo,ihalo_s2 in enumerate(halo_index_list_snap2):# for each halo (index at snap 2)

        # If needed, create group for this halo in output file
        if write_partdata:
            try:
                ihalo_hdf5=particle_output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))
                ihalo_hdf5.create_group('Metadata')
                if write_partdata:
                    ihalo_hdf5.create_group('Inflow');ihalo_hdf5.create_group('Outflow')
                    for itype in PartTypes:
                        ihalo_hdf5['Inflow'].create_group(f'PartType{itype}')
                        ihalo_hdf5['Outflow'].create_group(f'PartType{itype}')
            except:
                if write_partdata:
                    ihalo_hdf5=output_hdf5['ihalo_'+str(ihalo_s2).zfill(6)]
                    ihalo_hdf5_inkeys=list(ihalo_hdf5['Inflow'].keys());ihalo_hdf5_outkeys=list(ihalo_hdf5['Outflow'].keys());ihalo_hdf5_mdkeys=list(ihalo_hdf5['Metadata'].keys())
                    for itype in PartTypes:
                        for ihalo_hdf5_inkey in ihalo_hdf5_inkeys: del ihalo_hdf5['Inflow'][f'PartType{itype}'][ihalo_hdf5_inkey]
                        for ihalo_hdf5_outkey in ihalo_hdf5_outkeys: del ihalo_hdf5['Outflow'][f'PartType{itype}'][ihalo_hdf5_inkey]
                        for ihalo_hdf5_mdkey in ihalo_hdf5_inkeys: del ihalo_hdf5['Metadata'][f'PartType{itype}'][ihalo_hdf5_mdkey]
        
        try:     # This catches any exceptions for a given halo and prevents the code from crashing 
            # try:
            ########################################################################################################################################
            ###################################################### ihalo PRE-PROCESSING ############################################################
            ########################################################################################################################################
            t1_halo=time.time()
            t1_haloinit=time.time()
            # Find halo progenitor and descendants
            ihalo_indices={str(snap1):halo_index_list_snap1[iihalo],str(snap2):ihalo_s2,str(snap3):halo_index_list_snap3[iihalo]}
            
            # Record halo properties 
            ihalo_tracked=(ihalo_indices[str(snap1)]>-1 and ihalo_indices[str(snap3)]>-1)#track if have both progenitor and descendant
            ihalo_structuretype=base_halo_data[snap2]["Structuretype"][ihalo_indices[str(snap2)]]#structure type
            ihalo_numsubstruct=base_halo_data[snap2]["numSubStruct"][ihalo_indices[str(snap2)]]
            ihalo_hostHaloID=base_halo_data[snap2]["hostHaloID"][ihalo_indices[str(snap2)]]
            ihalo_sublevel=int(np.floor((ihalo_structuretype-0.01)/10))

            # Print progress to terminal and output file
            print();print('**********************************************')
            print('Halo index: ',ihalo_s2,f' - {ihalo_numsubstruct} substructures')
            print(f'Progenitor: {ihalo_indices[str(snap1)]} | Descendant: {ihalo_indices[str(snap3)]}')
            print('**********************************************');print()
            with open(fname_log,"a") as progress_file:
                progress_file.write(f' \n')
                progress_file.write(f'Starting with ihalo {ihalo_s2} ... \n')
            progress_file.close()
            
            # This catches any halos for which we can't find a progenitor/descendant 
            if ihalo_tracked:
                ### GRAB HALO METADATA ###
                ihalo_metadata={}
                for isnap,snap in enumerate(snaps):
                    ihalo_isnap=ihalo_indices[str(snap)]
                    if ihalo_isnap>=0:
                        ihalo_metadata[f'snap{isnap+1}_com']=np.array([base_halo_data[snap]['Xc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Yc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zc'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_cminpot']=np.array([base_halo_data[snap]['Xcminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Ycminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zcminpot'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_vcom']=np.array([base_halo_data[snap]['VXc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VYc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VZc'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_R_200crit']=base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_R_200mean']=base_halo_data[snap]['R_200mean'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_Mass_200crit']=base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*10**10
                        ihalo_metadata[f'snap{isnap+1}_vmax']=base_halo_data[snap]['Vmax'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_vesc_crit']=np.sqrt(2*base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*base_halo_data[snap]['SimulationInfo']['Gravity']/base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]])
                
                # Average some quantities
                ihalo_metadata['sublevel']=ihalo_sublevel
                ihalo_metadata['ave_R_200crit']=0.5*base_halo_data[snap1]['R_200crit'][ihalo_indices[str(snap1)]]+0.5*base_halo_data[snap2]['R_200crit'][ihalo_indices[str(snap2)]]
                ihalo_metadata['ave_vmax']=0.5*base_halo_data[snap1]['Vmax'][ihalo_indices[str(snap1)]]+0.5*base_halo_data[snap2]['Vmax'][ihalo_indices[str(snap2)]]

                # Write halo metadata to file (if desired)
                if write_partdata:
                    for ihalo_mdkey in list(ihalo_metadata.keys()): 
                        size=np.size(ihalo_metadata[ihalo_mdkey])
                        if size>1:
                            ihalo_hdf5['Metadata'].create_dataset(ihalo_mdkey,data=ihalo_metadata[ihalo_mdkey],dtype=np.float32,shape=(1,size))
                        else:
                            ihalo_hdf5['Metadata'].create_dataset(ihalo_mdkey,data=ihalo_metadata[ihalo_mdkey],dtype=np.float32)

                t2_haloinit=time.time()

                ### GET HALO DATA FROM VELOCIRAPTOR AND EAGLE ###
                # Grab the FOF particle data 
                t1_retrieve=time.time()

                ihalo_fof_particles={}
                for snap in snaps:
                    ihalo_fof_particles[str(snap)]={field:FOF_Part_Data[str(snap)][field][str(ihalo_indices[str(snap)])] for field in FOF_Part_Data_fields}
                    ihalo_fof_particles[str(snap)]['SortedIndices']=np.argsort(ihalo_fof_particles[str(snap)]['Particle_IDs'])
                    ihalo_fof_particles[str(snap)]['SortedIDs']=ihalo_fof_particles[str(snap)]['Particle_IDs'][(ihalo_fof_particles[str(snap)]['SortedIndices'],)]
                    ihalo_fof_particles[str(snap)]['ParticleIDs_set']=set(ihalo_fof_particles[str(snap)]['Particle_IDs'])

                # Grab/slice EAGLE datacubes
                print(f'Retrieving datacubes for ihalo {ihalo_s2} ...')
                #cube parameters
                ihalo_com_physical={str(snap):np.array(ihalo_metadata[f'snap{isnap+1}_{use}']) for isnap,snap in enumerate(snaps)}
                ihalo_com_comoving={str(snap):np.array(ihalo_metadata[f'snap{isnap+1}_{use}'])/Part_Data_comtophys[str(snap)]['Coordinates'] for isnap,snap in enumerate(snaps)}
                ihalo_vcom_physical={str(snap):np.array(ihalo_metadata[f'snap{isnap+1}_vcom']) for isnap,snap in enumerate(snaps)}
                ihalo_cuberadius_physical={str(snap):ihalo_metadata[f'snap{isnap+1}_R_200mean']*ihalo_cube_rfac for snap in snaps}
                ihalo_cuberadius_comoving={str(snap):ihalo_metadata[f'snap{isnap+1}_R_200mean']/Part_Data_comtophys[str(snap)]['Coordinates']*ihalo_cube_rfac for isnap,snap in enumerate(snaps)}
                
                #cube outputs
                ihalo_cube_particles={str(snap):{field:[] for field in Part_Data_fields} for snap in snaps}
                ihalo_cube_npart={str(snap):{} for snap in snaps}

                #get cube for each snap
                for snap in snaps:
                    ihalo_EAGLE_snap=read_eagle.EagleSnapshot(Part_Data_FilePaths[str(snap)])
                    ihalo_EAGLE_snap.select_region(xmin=ihalo_com_comoving[str(snap)][0][0]-ihalo_cuberadius_comoving[str(snap)],xmax=ihalo_com_comoving[str(snap)][0][0]+ihalo_cuberadius_comoving[str(snap)],
                                                ymin=ihalo_com_comoving[str(snap)][0][1]-ihalo_cuberadius_comoving[str(snap)],ymax=ihalo_com_comoving[str(snap)][0][1]+ihalo_cuberadius_comoving[str(snap)],
                                                zmin=ihalo_com_comoving[str(snap)][0][2]-ihalo_cuberadius_comoving[str(snap)],zmax=ihalo_com_comoving[str(snap)][0][2]+ihalo_cuberadius_comoving[str(snap)])
                    ihalo_EAGLE_types=[]
                    #get data for each parttype and add to running ihalo_cube_particles
                    for itype in PartTypes:       
                        for ifield,field in enumerate(Part_Data_fields):
                            if not (field=='Mass' and itype==1):#if dataset is not DM mass, read and convert 
                                data=ihalo_EAGLE_snap.read_dataset(itype,field)*Part_Data_comtophys[str(snap)][field];ihalo_cube_npart[str(snap)][str(itype)]=len(data)     
                            else:#if dataset is DM mass, fill flat array with constant
                                data=np.ones(ihalo_cube_npart[str(snap)]['1'])*Mass_DM
                            ihalo_cube_particles[str(snap)][field].extend(data)
                        ihalo_EAGLE_types.extend((np.ones(ihalo_cube_npart[str(snap)][str(itype)])*itype).astype(int))#record particle types
                    ihalo_cube_particles[str(snap)]['ParticleTypes']=np.array(ihalo_EAGLE_types)
                    #convert to np.arrays
                    for field in Part_Data_fields:
                        ihalo_cube_particles[str(snap)][field]=np.array(ihalo_cube_particles[str(snap)][field])
                    #sort the cube particles by ID
                    ihalo_cube_particles[str(snap)]['SortedIndices']=np.argsort(ihalo_cube_particles[str(snap)]['ParticleIDs'])
                    ihalo_cube_particles[str(snap)]['SortedIDs']=ihalo_cube_particles[str(snap)]['ParticleIDs'][(ihalo_cube_particles[str(snap)]['SortedIndices'],)]

                t2_retrieve=time.time()
                print(f'Finished retrieving data from EAGLE and FOF for ihalo {ihalo_s2} in {t2_retrieve-t1_retrieve:.2f} sec')
                
                ########################################################################################################################################
                ############################################################ ihalo INFLOW ##############################################################
                ########################################################################################################################################

                ###### SELECT INFLOW CANDIDATES AS THOSE WITHIN R200crit OR the FOF envelope at snap 2 ######
                #############################################################################################
                t1_inflow_candidates=time.time()

                #find the mean r200 from snap 1 / snap 2
                ihalo_ave_R_200crit_physical=(ihalo_metadata['snap1_R_200crit']+ihalo_metadata['snap2_R_200crit'])/2
                #find radius of each cube particle from halo center
                ihalo_cube_r_snap2=np.sqrt(np.sum(np.square(ihalo_cube_particles[str(snap2)]['Coordinates']-ihalo_com_physical[str(snap2)]),axis=1))
                #find which particles are with in the mean r200
                ihalo_cube_rcut_snap2=np.where(ihalo_cube_r_snap2<ihalo_ave_R_200crit_physical)
                #get the particle data of the particles within r200
                ihalo_cube_inflow_candidate_data_snap2={field:ihalo_cube_particles[str(snap2)][field] for field in Part_Data_fields}
                #get the particle data of the particles in the FOF
                ihalo_fof_inflow_candidate_data_snap2={field:ihalo_fof_particles[str(snap2)][field] for field in FOF_Part_Data_fields}
                #concatenate the IDs of the particles within r200 and the FOF
                ihalo_combined_inflow_candidate_IDs=np.concatenate([ihalo_fof_inflow_candidate_data_snap2['Particle_IDs'],ihalo_cube_inflow_candidate_data_snap2['ParticleIDs']])
                #remove duplicates and convert to np.array with long ints
                ihalo_combined_inflow_candidate_IDs_unique=np.array(np.unique(ihalo_combined_inflow_candidate_IDs),dtype=np.int64)
                #count inflow candidates
                ihalo_combined_inflow_candidate_count=len(ihalo_combined_inflow_candidate_IDs_unique)
                t2_inflow_candidates=time.time()

                ############################## GRAB DATA FOR INFLOW CANDIDATES ##############################
                #############################################################################################
                ihalo_combined_inflow_candidate_data={}

                # 1. OUTPUTS FROM DATACUBE: Coordinates, Velocity, Mass, Type 
                t1_cubeoutputs=time.time()
                ihalo_combined_inflow_candidate_cubeindices={}

                print(f'Inflow candidates for ihalo {ihalo_s2}: n = {ihalo_combined_inflow_candidate_count}')
                for isnap,snap in enumerate(snaps):

                    #find the indices of the IDs in the (sorted) datacube for this halo (will return nan if not in the cube) - outputs sorted cube index
                    ihalo_combined_inflow_candidate_IDindices_temp=binary_search(ihalo_combined_inflow_candidate_IDs_unique,sorted_list=ihalo_cube_particles[str(snap)]['SortedIDs'],check_entries=True)
                    #use the indices from the sorted IDs above to extract the cube indices (will return nan if not in the cube) - outputs raw cube index
                    ihalo_combined_inflow_candidate_cubeindices[str(snap)]=mask_wnans(array=ihalo_cube_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_inflow_candidate_IDindices_temp)
                    
                    #for each snap, grab detailed particle data
                    for field in ['Coordinates','Velocity','Mass','ParticleIDs','ParticleTypes']:
                        ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_cube_particles[str(snap)][field],indices=ihalo_combined_inflow_candidate_cubeindices[str(snap)])
                        
                    #derive other cubdata outputs
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com']=ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Coordinates']-ihalo_com_physical[str(snap)]
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_rabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com']),axis=1))
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_runit_com']=np.divide(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com'],np.column_stack([ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_rabs_com']]*3))
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_v_com']=ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Velocity']-ihalo_vcom_physical[str(snap)]
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_v_com']),axis=1))
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vrad_com']=np.sum(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_runit_com']*ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_v_com'],axis=1)
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vtan_com']=np.sqrt(np.square(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vabs_com'])-np.square(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vrad_com']))

                #include average radial velocity
                ihalo_combined_inflow_candidate_data[f'ave_vrad_com']=(ihalo_combined_inflow_candidate_data[f'snap2_rabs_com']-ihalo_combined_inflow_candidate_data[f'snap1_rabs_com'])/dt*vel_conversion
                t2_cubeoutputs=time.time()

                # 2. OUTPUTS FROM FOF Data: InFOF, Bound
                t1_fofoutputs=time.time()
                ihalo_combined_inflow_candidate_fofindices={}
                for isnap,snap in enumerate(snaps):
                    #find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
                    ihalo_combined_inflow_candidate_IDindices_temp=binary_search(ihalo_combined_inflow_candidate_IDs_unique,sorted_list=ihalo_fof_particles[str(snap)]['SortedIDs'],check_entries=True)
                    #use the indices from the sorted IDs above to extract the fof indices (will return nan if not in the fof) - outputs index
                    ihalo_combined_inflow_candidate_fofindices[str(snap)]=mask_wnans(array=ihalo_fof_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_inflow_candidate_IDindices_temp)
                    
                    #use the fof indices to extract particle data, record which particles couldn't be found
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']=np.isfinite(ihalo_combined_inflow_candidate_IDindices_temp)
                    ihalo_combined_inflow_candidate_fofdata_notinfofmask=np.where(np.logical_not(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']))
                    ihalo_combined_inflow_candidate_fofdata_notinfofmask_count=len(ihalo_combined_inflow_candidate_fofdata_notinfofmask[0])
                    for field in ['Particle_Bound','Particle_InHost']:
                        ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_fof_particles[str(snap)][field],indices=ihalo_combined_inflow_candidate_fofindices[str(snap)])
                        ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_{field}'][ihalo_combined_inflow_candidate_fofdata_notinfofmask]=np.zeros(ihalo_combined_inflow_candidate_fofdata_notinfofmask_count)
                t2_fofoutputs=time.time()

                # 3. OUTPUTS FROM HISTORIES: Processed, Structure (by particle type) -- just for snap 1
                t1_histoutputs=time.time()
                ihalo_combined_inflow_candidate_partindices={}
                ihalo_combined_inflow_candidate_data['snap1_Structure']=np.zeros(ihalo_combined_inflow_candidate_count)
                ihalo_combined_inflow_candidate_data['snap1_Processed']=np.zeros(ihalo_combined_inflow_candidate_count)
                
                for itype in PartTypes:
                    ihalo_combined_inflow_candidate_typemask_snap1=np.where(ihalo_combined_inflow_candidate_data['snap1_ParticleTypes']==itype)
                    ihalo_combined_inflow_candidate_IDs_unique_itype=ihalo_combined_inflow_candidate_IDs_unique[ihalo_combined_inflow_candidate_typemask_snap1]
                    #find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
                    ihalo_combined_inflow_candidate_IDindices_temp=binary_search(ihalo_combined_inflow_candidate_IDs_unique_itype,sorted_list=Part_Histories_IDs_snap1[str(itype)],check_entries=False)
                    #use the indices from the sorted IDs above to extract the partdata indices (will return nan if not in the fof) - outputs index
                    ihalo_combined_inflow_candidate_partindices[str(itype)]=Part_Histories_Index_snap1[str(itype)][(ihalo_combined_inflow_candidate_IDindices_temp,)]
                    #extract host structure and processing
                    ihalo_combined_inflow_candidate_data['snap1_Structure'][ihalo_combined_inflow_candidate_typemask_snap1]=Part_Histories_HostStructure_snap1[str(itype)][ihalo_combined_inflow_candidate_partindices[str(itype)]]
                    ihalo_combined_inflow_candidate_data['snap1_Processed'][ihalo_combined_inflow_candidate_typemask_snap1]=Part_Histories_Processed_L1_snap1[str(itype)][ihalo_combined_inflow_candidate_partindices[str(itype)]]

                t2_histoutputs=time.time()
                
                
                ############################## SAVE DATA FOR INFLOW CANDIDATES ##############################
                #############################################################################################
                
                # Performance measures
                
                t_particle=[]
                t_integrated=[]

                # Iterate through particle types
                for itype in PartTypes:
                    # Mask for particle types - note these are taken at snap 1 (before "entering" halo)
                    itype_key=f'PartType{itype}'
                    ihalo_itype_mask=np.where(ihalo_combined_inflow_candidate_data["snap1_ParticleTypes"]==itype)

                    ### PARTICLE OUTPUTS ###
                    ########################
                    t1_particle=time.time()
                    if write_partdata:
                        ihalo_hdf5['Inflow'][itype_key].create_dataset('ParticleIDs',data=ihalo_combined_inflow_candidate_IDs_unique[ihalo_itype_mask],dtype=output_fields_dtype["ParticleIDs"],compression=compression)
                        ihalo_hdf5['Inflow'][itype_key].create_dataset('Mass',data=ihalo_combined_inflow_candidate_data['snap1_Mass'][ihalo_itype_mask],dtype=output_fields_dtype["Mass"],compression=compression)
                        ihalo_hdf5['Inflow'][itype_key].create_dataset('ave_vrad_com',data=ihalo_combined_inflow_candidate_data['ave_vrad_com'][ihalo_itype_mask],dtype=output_fields_dtype["ave_vrad_com"],compression=compression)

                        #Rest of fields: snap 1
                        ihalo_snap1_inflow_outputs=["Structure","Processed","r_com","rabs_com","vrad_com","vtan_com","Particle_InFOF","Particle_Bound","Particle_InHost"]
                        for ihalo_snap1_inflow_output in ihalo_snap1_inflow_outputs:
                            ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap1_{ihalo_snap1_inflow_output}',data=ihalo_combined_inflow_candidate_data[f'snap1_{ihalo_snap1_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap1_inflow_output],compression=compression)
                        
                        #Rest of fields: snap 2
                        ihalo_snap2_inflow_outputs=["r_com","rabs_com","vrad_com","vtan_com","Particle_InFOF","Particle_Bound","Particle_InHost"]
                        for ihalo_snap2_inflow_output in ihalo_snap2_inflow_outputs:
                            ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap2_{ihalo_snap2_inflow_output}',data=ihalo_combined_inflow_candidate_data[f'snap2_{ihalo_snap2_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap2_inflow_output],compression=compression)
                        
                        #Rest of fields: snap 3
                        ihalo_snap3_inflow_outputs=["Particle_InFOF","Particle_Bound","Particle_InHost",'rabs_com']
                        for ihalo_snap3_inflow_output in ihalo_snap3_inflow_outputs:
                            ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap3_{ihalo_snap3_inflow_output}',data=ihalo_combined_inflow_candidate_data[f'snap3_{ihalo_snap3_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap3_inflow_output],compression=compression)

                    t2_particle=time.time()
                    t_particle.append(t2_particle-t1_particle)
                    
                    ### INTEGRATED OUTPUTS ###
                    ##########################
                    t1_integrated=time.time()
                    ihalo_itype_inflow_masses=ihalo_combined_inflow_candidate_data['snap1_Mass'][ihalo_itype_mask]

                    # Masks for halo inflow definitions
                    halo_itype_inflow_definition={'FOF-haloscale':np.logical_and(ihalo_combined_inflow_candidate_data["snap2_Particle_InFOF"][ihalo_itype_mask],np.logical_not(ihalo_combined_inflow_candidate_data["snap1_Particle_InFOF"][ihalo_itype_mask])),
                                                  'FOF-subhaloscale':np.logical_and(ihalo_combined_inflow_candidate_data["snap2_Particle_InHost"][ihalo_itype_mask],np.logical_not(ihalo_combined_inflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask]))}
                    for ir200_fac, r200_fac in enumerate(r200_facs["Inflow"]):
                        ir200_key=f'SO-r200_fac{ir200_fac+1}'
                        halo_itype_inflow_definition[ir200_key]=np.logical_and(ihalo_combined_inflow_candidate_data["snap2_rabs_com"][ihalo_itype_mask]<r200_fac*ihalo_metadata['ave_R_200crit'],ihalo_combined_inflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_metadata['ave_R_200crit'])

                    # Masks for cuts on inflow velocity as per vmax_facs
                    ihalo_itype_inflow_vmax_masks={'vmax_fac'+str(ivmax_fac+1):-ihalo_combined_inflow_candidate_data[f'snap1_vrad_com'][ihalo_itype_mask]>vmax_fac*ihalo_metadata['ave_vmax']  for ivmax_fac,vmax_fac in enumerate(vmax_facs["Inflow"])}
        
                   # Masks for processing history of particles
                    ihalo_itype_inflow_processed_masks={'Unprocessed':ihalo_combined_inflow_candidate_data["snap1_Processed"][ihalo_itype_mask]==0.0,
                                                        'Processed':ihalo_combined_inflow_candidate_data["snap1_Processed"][ihalo_itype_mask]>0.0,
                                                        'Total': np.isfinite(ihalo_combined_inflow_candidate_data["snap1_Processed"][ihalo_itype_mask])}
                    # Masks for the origin of inflow particles
                    ihalo_itype_inflow_origin_masks={'Gross':np.isfinite(ihalo_combined_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]),
                                                     'Field':ihalo_combined_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]==-1,
                                                     'Transfer':ihalo_combined_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]>0}

                    # Masks for stability
                    ihalo_itype_inflow_stability={}
                    ihalo_itype_inflow_stability={'FOF-haloscale':ihalo_combined_inflow_candidate_data["snap3_Particle_InFOF"][ihalo_itype_mask],
                                                  'FOF-subhaloscale':ihalo_combined_inflow_candidate_data["snap3_Particle_InHost"][ihalo_itype_mask]}
                    for ir200_fac, r200_fac in enumerate(r200_facs["Inflow"]):
                        ir200_key=f'SO-r200_fac{ir200_fac+1}'
                        ihalo_itype_inflow_stability[ir200_key]=ihalo_combined_inflow_candidate_data["snap3_rabs_com"][ihalo_itype_mask]<r200_fac*ihalo_metadata['ave_R_200crit']


                    ## ITERATE THROUGH THE ABOVE MASKS
                    for halo_defname in halo_defnames["Inflow"]:
                        idef_mask=halo_itype_inflow_definition[halo_defname]
                        stability_mask=ihalo_itype_inflow_stability[halo_defname]
                        
                        if 'FOF' not in halo_defname:
                            datasets=['Gross']
                            processedgroups=['Total']
                        else:
                            datasets=output_datasets["Inflow"]
                            processedgroups=output_processedgroups

                        for ivmax_fac, vmax_fac in enumerate(vmax_facs["Inflow"]):
                            ivmax_key=f'vmax_fac{ivmax_fac+1}'
                            ivmax_mask=ihalo_itype_inflow_vmax_masks[ivmax_key]

                            for processedgroup in processedgroups:
                                iprocessed_mask=ihalo_itype_inflow_processed_masks[processedgroup]

                                for dataset in datasets:
                                    idset_key=dataset
                                    origin_mask=ihalo_itype_inflow_origin_masks[dataset]
                                    
                                    masks=[idef_mask,ivmax_mask,iprocessed_mask,origin_mask]
                                    masksname=[halo_defname,ivmax_key,processedgroup,dataset]
                                    # if itype==0 and 'SO' in halo_defname:
                                        
                                    #     # print(f'Calculation: {masksname}')
                                    #     # for depth in range(len(masks)):
                                    #     #     print(masksname[:depth+1])
                                    #     #     running_mask=np.logical_and.reduce(masks[:depth+1])
                                    #     #     print(np.sum(running_mask))
                                    #     #     print(len(np.where(running_mask)[0]))

                                    running_mask=np.logical_and.reduce([idef_mask,ivmax_mask,iprocessed_mask,origin_mask])
                                    stable_running_mask=np.logical_and(running_mask,stability_mask)

                                    all_dset_where=np.where(running_mask)
                                    stable_dset_where=np.where(stable_running_mask)

                                    integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaM_In'][iihalo]=np.float32(np.nansum(ihalo_itype_inflow_masses[all_dset_where]))
                                    integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaN_In'][iihalo]=np.float32(np.nansum(running_mask))
                                    integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaM_In'][iihalo]=np.float32(np.nansum(ihalo_itype_inflow_masses[stable_dset_where]))
                                    integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaN_In'][iihalo]=np.float32(np.nansum(stable_running_mask))
                    
                    t2_integrated=time.time()
                    t_integrated.append(t2_integrated-t1_integrated)
                

                ########################################################################################################################################
                ############################################################ ihalo OUTFLOW ##############################################################
                ########################################################################################################################################

                ###### SELECT OUTFLOW CANDIDATES AS THOSE WITHIN R200crit OR the FOF envelope at snap 1 ######   
                #############################################################################################
                if outflow:
                    t1_outflow_candidates=time.time()

                    #find the mean r200 from snap 1 / snap 2
                    ihalo_ave_R_200crit_physical=(ihalo_metadata['snap1_R_200crit']+ihalo_metadata['snap2_R_200crit'])/2
                    #find radius of each cube particle from halo center
                    ihalo_cube_r_snap1=np.sqrt(np.sum(np.square(ihalo_cube_particles[str(snap1)]['Coordinates']-ihalo_com_physical[str(snap1)]),axis=1))
                    #find which particles are with in the mean r200
                    ihalo_cube_rcut_snap1=np.where(ihalo_cube_r_snap1<ihalo_ave_R_200crit_physical)
                    #get the particle data of the particles within r200
                    ihalo_cube_outflow_candidate_data_snap1={field:ihalo_cube_particles[str(snap1)][field] for field in Part_Data_fields}
                    #get the particle data of the particles in the FOF
                    ihalo_fof_outflow_candidate_data_snap1={field:ihalo_fof_particles[str(snap1)][field] for field in FOF_Part_Data_fields}
                    #concatenate the IDs of the particles within r200 and the FOF
                    ihalo_combined_outflow_candidate_IDs=np.concatenate([ihalo_fof_outflow_candidate_data_snap1['Particle_IDs'],ihalo_cube_outflow_candidate_data_snap1['ParticleIDs']])
                    #remove duplicates and convert to np.array with long ints
                    ihalo_combined_outflow_candidate_IDs_unique=np.array(np.unique(ihalo_combined_outflow_candidate_IDs),dtype=np.int64)
                    #count outflow candidates
                    ihalo_combined_outflow_candidate_count=len(ihalo_combined_outflow_candidate_IDs_unique)
                    t2_outflow_candidates=time.time()

                    ############################## GRAB DATA FOR INFLOW CANDIDATES ##############################
                    #############################################################################################
                    ihalo_combined_outflow_candidate_data={}

                    # 1. OUTPUTS FROM DATACUBE: Coordinates, Velocity, Mass, Type 
                    t1_cubeoutputs=time.time()
                    ihalo_combined_outflow_candidate_cubeindices={}

                    print(f'Outflow candidates for ihalo {ihalo_s2}: n = {ihalo_combined_outflow_candidate_count}')
                    for isnap,snap in enumerate(snaps):

                        #find the indices of the IDs in the (sorted) datacube for this halo (will return nan if not in the cube) - outputs sorted cube index
                        ihalo_combined_outflow_candidate_IDindices_temp=binary_search(ihalo_combined_outflow_candidate_IDs_unique,sorted_list=ihalo_cube_particles[str(snap)]['SortedIDs'],check_entries=True)
                        #use the indices from the sorted IDs above to extract the cube indices (will return nan if not in the cube) - outputs raw cube index
                        ihalo_combined_outflow_candidate_cubeindices[str(snap)]=mask_wnans(array=ihalo_cube_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_outflow_candidate_IDindices_temp)
                        
                        #for each snap, grab detailed particle data
                        for field in ['Coordinates','Velocity','Mass','ParticleIDs','ParticleTypes']:
                            ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_cube_particles[str(snap)][field],indices=ihalo_combined_outflow_candidate_cubeindices[str(snap)])
                            
                        #derive other cubedata outputs
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com']=ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Coordinates']-ihalo_com_physical[str(snap)]
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_rabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com']),axis=1))
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_runit_com']=np.divide(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com'],np.column_stack([ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_rabs_com']]*3))
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_v_com']=ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Velocity']-ihalo_vcom_physical[str(snap)]
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_v_com']),axis=1))
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vrad_com']=np.sum(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_runit_com']*ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_v_com'],axis=1)
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vtan_com']=np.sqrt(np.square(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vabs_com'])-np.square(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vrad_com']))

                    #include average radial velocity
                    ihalo_combined_outflow_candidate_data[f'ave_vrad_com']=(ihalo_combined_outflow_candidate_data[f'snap2_rabs_com']-ihalo_combined_outflow_candidate_data[f'snap1_rabs_com'])/dt*vel_conversion
                    t2_cubeoutputs=time.time()

                    # 2. OUTPUTS FROM FOF Data: InFOF, Bound
                    t1_fofoutputs=time.time()
                    ihalo_combined_outflow_candidate_fofindices={}
                    for isnap,snap in enumerate(snaps):
                        #find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
                        ihalo_combined_outflow_candidate_IDindices_temp=binary_search(ihalo_combined_outflow_candidate_IDs_unique,sorted_list=ihalo_fof_particles[str(snap)]['SortedIDs'],check_entries=True)
                        #use the indices from the sorted IDs above to extract the fof indices (will return nan if not in the fof) - outputs index
                        ihalo_combined_outflow_candidate_fofindices[str(snap)]=mask_wnans(array=ihalo_fof_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_outflow_candidate_IDindices_temp)
                        
                        #use the fof indices to extract particle data, record which particles couldn't be found
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']=np.isfinite(ihalo_combined_outflow_candidate_IDindices_temp)
                        ihalo_combined_outflow_candidate_fofdata_notinfofmask=np.where(np.logical_not(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']))
                        ihalo_combined_outflow_candidate_fofdata_notinfofmask_count=len(ihalo_combined_outflow_candidate_fofdata_notinfofmask[0])
                        for field in ['Particle_Bound','Particle_InHost']:
                            ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_fof_particles[str(snap)][field],indices=ihalo_combined_outflow_candidate_fofindices[str(snap)])
                            ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_{field}'][ihalo_combined_outflow_candidate_fofdata_notinfofmask]=np.zeros(ihalo_combined_outflow_candidate_fofdata_notinfofmask_count)
                    
                    t2_fofoutputs=time.time()

                    ############################## SAVE DATA FOR OUTFLOW CANDIDATES ##############################
                    #############################################################################################
                    t1_outflow=time.time()
                    # Iterate through particle types
                    for itype in PartTypes:
                        # Mask for particle types - note these are taken at snap 2 (after "leaving" halo)
                        itype_key=f'PartType{itype}'
                        ihalo_itype_mask=np.where(ihalo_combined_outflow_candidate_data["snap2_ParticleTypes"]==itype)

                        ### PARTICLE OUTPUTS ###
                        ########################
                        t1_particle=time.time()
                        if write_partdata:
                            ihalo_hdf5['Outflow'][itype_key].create_dataset('ParticleIDs',data=ihalo_combined_outflow_candidate_IDs_unique[ihalo_itype_mask],dtype=output_fields_dtype["ParticleIDs"],compression=compression)
                            ihalo_hdf5['Outflow'][itype_key].create_dataset('Mass',data=ihalo_combined_outflow_candidate_data['snap1_Mass'][ihalo_itype_mask],dtype=output_fields_dtype["Mass"],compression=compression)
                            ihalo_hdf5['Outflow'][itype_key].create_dataset('ave_vrad_com',data=ihalo_combined_outflow_candidate_data['ave_vrad_com'][ihalo_itype_mask],dtype=output_fields_dtype["ave_vrad_com"],compression=compression)

                            #Rest of fields: snap 1
                            ihalo_snap1_outflow_outputs=["r_com","rabs_com","vrad_com","vtan_com","Particle_InFOF","Particle_Bound","Particle_InHost"]
                            for ihalo_snap1_outflow_output in ihalo_snap1_outflow_outputs:
                                ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap1_{ihalo_snap1_outflow_output}',data=ihalo_combined_outflow_candidate_data[f'snap1_{ihalo_snap1_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap1_outflow_output],compression=compression)
                            
                            #Rest of fields: snap 2
                            ihalo_snap2_outflow_outputs=["r_com","rabs_com","vrad_com","vtan_com","Particle_InFOF","Particle_Bound","Particle_InHost"]
                            for ihalo_snap2_outflow_output in ihalo_snap2_outflow_outputs:
                                ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap2_{ihalo_snap2_outflow_output}',data=ihalo_combined_outflow_candidate_data[f'snap2_{ihalo_snap2_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap2_outflow_output],compression=compression)
                            
                            #Rest of fields: snap 3
                            ihalo_snap3_outflow_outputs=["Particle_InFOF","Particle_Bound","Particle_InHost",'rabs_com']
                            for ihalo_snap3_outflow_output in ihalo_snap3_outflow_outputs:
                                ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap3_{ihalo_snap3_outflow_output}',data=ihalo_combined_outflow_candidate_data[f'snap3_{ihalo_snap3_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap3_outflow_output],compression=compression)

                        t2_particle=time.time()
                        t_particle.append(t2_particle-t1_particle)
                        
                        ### INTEGRATED OUTPUTS ###
                        ##########################
                        t1_integrated=time.time()
                        ihalo_itype_outflow_masses=ihalo_combined_outflow_candidate_data['snap1_Mass'][ihalo_itype_mask]

                        # Masks for halo outflow definitions
                        halo_itype_outflow_definition={'FOF-haloscale':np.logical_and(ihalo_combined_outflow_candidate_data["snap1_Particle_InFOF"][ihalo_itype_mask],np.logical_not(ihalo_combined_outflow_candidate_data["snap2_Particle_InFOF"][ihalo_itype_mask])),
                                                    'FOF-subhaloscale':np.logical_and(ihalo_combined_outflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask],np.logical_not(ihalo_combined_outflow_candidate_data["snap2_Particle_InHost"][ihalo_itype_mask]))}
                        for ir200_fac, r200_fac in enumerate(r200_facs["Outflow"]):
                            ir200_key=f'SO-r200_fac{ir200_fac+1}'
                            halo_itype_outflow_definition[ir200_key]=np.logical_and(ihalo_combined_outflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]<r200_fac*ihalo_metadata['ave_R_200crit'],ihalo_combined_outflow_candidate_data["snap2_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_metadata['ave_R_200crit'])

                        # Masks for cuts on outflow velocity as per vmax_facs
                        ihalo_itype_outflow_vmax_masks={'vmax_fac'+str(ivmax_fac+1):ihalo_combined_outflow_candidate_data[f'snap1_vrad_com'][ihalo_itype_mask]>vmax_fac*ihalo_metadata['ave_vmax']  for ivmax_fac,vmax_fac in enumerate(vmax_facs["Outflow"])}
            
                        # Masks for processing history of particles
                        ihalo_itype_outflow_processed_masks={'Total': np.ones(len(ihalo_itype_outflow_masses))}
                        
                        # Masks for the origin of outflow particles
                        ihalo_itype_outflow_origin_masks={'Gross':np.ones(len(ihalo_itype_outflow_masses))}

                        # Masks for stability
                        ihalo_itype_outflow_stability={}
                        ihalo_itype_outflow_stability={'FOF-haloscale':np.logical_not(ihalo_combined_outflow_candidate_data["snap3_Particle_InFOF"][ihalo_itype_mask]),
                                                    'FOF-subhaloscale':np.logical_not(ihalo_combined_outflow_candidate_data["snap3_Particle_InHost"][ihalo_itype_mask])}
                        for ir200_fac, r200_fac in enumerate(r200_facs["Outflow"]):
                            ir200_key=f'SO-r200_fac{ir200_fac+1}'
                            ihalo_itype_outflow_stability[ir200_key]=ihalo_combined_outflow_candidate_data["snap3_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_metadata['ave_R_200crit']

                        ## ITERATE THROUGH THE ABOVE MASKS
                        for halo_defname in halo_defnames["Outflow"]:
                            idef_mask=halo_itype_outflow_definition[halo_defname]
                            stability_mask=ihalo_itype_outflow_stability[halo_defname]
                            
                            datasets=['Gross']
                            processedgroups=['Total']

                            for ivmax_fac, vmax_fac in enumerate(vmax_facs["Outflow"]):
                                ivmax_key=f'vmax_fac{ivmax_fac+1}'
                                ivmax_mask=ihalo_itype_outflow_vmax_masks[ivmax_key]

                                for processedgroup in processedgroups:
                                    iprocessed_mask=ihalo_itype_outflow_processed_masks[processedgroup]

                                    for dataset in datasets:
                                        idset_key=dataset
                                        origin_mask=ihalo_itype_outflow_origin_masks[dataset]
                                        
                                        masks=[idef_mask,ivmax_mask,iprocessed_mask,origin_mask]
                                        masksname=[halo_defname,ivmax_key,processedgroup,dataset]
                                        # if itype==0 and 'SO' in halo_defname:
                                        #     print(f'Calculation: {masksname}')
                                        #     for depth in range(len(masks)):
                                        #         print(masksname[:depth+1])
                                        #         running_mask=np.logical_and.reduce(masks[:depth+1])
                                        #         print(np.sum(running_mask))
                                                # print(len(np.where(running_mask)[0]))

                                        running_mask=np.logical_and.reduce([idef_mask,ivmax_mask,iprocessed_mask,origin_mask])
                                        stable_running_mask=np.logical_and(running_mask,stability_mask)

                                        all_dset_where=np.where(running_mask)
                                        stable_dset_where=np.where(stable_running_mask)

                                        integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaM_Out'][iihalo]=np.float32(np.nansum(ihalo_itype_outflow_masses[all_dset_where]))
                                        integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaN_Out'][iihalo]=np.float32(np.nansum(running_mask))
                                        integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaM_Out'][iihalo]=np.float32(np.nansum(ihalo_itype_outflow_masses[stable_dset_where]))
                                        integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaN_Out'][iihalo]=np.float32(np.nansum(stable_running_mask))
    
                    t2_outflow=time.time()

                t2_halo=time.time()

                # print()
                # print(f'–––––––––––––––––––– PERFORMANCE PROFILING (ihalo {ihalo_s2}: total time = {t2_halo-t1_halo:.2f} sec) ––––––––––––––––––––')
                # print(f'Initialisation: {t2_haloinit-t1_haloinit:.2e} sec [{(t2_haloinit-t1_haloinit)/(t2_halo-t1_halo)*100:.2f}%]')
                # print(f'Data retrieval: {t2_retrieve-t1_retrieve:.2e} sec [{(t2_retrieve-t1_retrieve)/(t2_halo-t1_halo)*100:.2f}%]')
                # print(f'Inflow candidates: {t2_inflow_candidates-t1_inflow_candidates:.2e} sec [{(t2_inflow_candidates-t1_inflow_candidates)/(t2_halo-t1_halo)*100:.2f}%] (n = {ihalo_combined_inflow_candidate_count})')
                # print(f'Inflow cube data: {t2_cubeoutputs-t1_cubeoutputs:.2e} sec [{(t2_cubeoutputs-t1_cubeoutputs)/(t2_halo-t1_halo)*100:.2f}%]')
                # print(f'Inflow FOF data: {t2_fofoutputs-t1_fofoutputs:.2e} sec [{(t2_fofoutputs-t1_fofoutputs)/(t2_halo-t1_halo)*100:.2f}%]')
                # print(f'Inflow history data: {t2_histoutputs-t1_histoutputs:.2e} sec [{(t2_histoutputs-t1_histoutputs)/(t2_halo-t1_halo)*100:.2f}%]')
                # print(f'Inflow saving full particle data: {np.sum(t_particle):.2e} sec [{np.sum(t_particle)/(t2_halo-t1_halo)*100:.2f}%]')
                # print(f'Inflow calculating/saving integrated particle data: {np.sum(t_integrated):.2f} sec [{np.sum(t_integrated)/(t2_halo-t1_halo)*100:.2f}%]')
                # print('––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
                        
                with open(fname_log,"a") as progress_file:
                    progress_file.write(f"Done with ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                    progress_file.write(f"[Took {t2_halo-t1_halo:.2f} sec]\n")
                    progress_file.write(f" \n")
                    progress_file.close()

            else:# Couldn't find the halo progenitor/descendant pair
                print(f'Skipping ihalo {ihalo_s2} - couldnt find progenitor/descendant pair')
                with open(fname_log,"a") as progress_file:
                    progress_file.write(f"Skipping ihalo {ihalo_s2} - no head/tail pair ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                    progress_file.write(f" \n")
                progress_file.close()

        except: # Some other error in the main halo loop
            # except:
            print(f'Skipping ihalo {ihalo_s2} - dont have the reason')
            with open(fname_log,"a") as progress_file:
                progress_file.write(f"Skipping ihalo {ihalo_s2} - unknown reason ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                progress_file.write(f" \n")
            progress_file.close()
            continue


    #Finished with output file
    output_hdf5.close()
    return None

