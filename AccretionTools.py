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
    
#     """

#     gen_accretion_data_fof_serial : function
# 	----------

#     Generate and save accretion rates for each particle type by comparing particle lists from VELOCIraptor FOF outputs. 

#     ** note: particle histories and base_halo_data must have been created as per gen_particle_history_serial (this file)
#              and gen_base_halo_data in STFTools.py

# 	Parameters
# 	----------
#     base_halo_data : list of dictionaries
#         The minimal halo data list of dictionaries previously generated ("B1" is sufficient)

#     snap : int
#         The index in the base_halo_data for which to calculate accretion rates (should be actual snap index)
#         We will retrieve particle data based on the flags at this index
    
#     halo_index_list : dict
#         "iprocess": int
#         "indices: list of int
#         List of the halo indices for which to calculate accretion rates. If 'None',
#         find for all halos in the base_halo_data dictionary at the desired snapshot. 

#     pre_depth : int
#         How many snaps to skip back to when comparing particle lists.
#         Initial snap for calculation will be snap-pre_depth. 

#     pre_depth : int
#         How many snaps to skip back to when comparing particle lists.
#         Initial snap (s1) for calculation will be s1=snap-pre_depth, and we will check particle histories at s1-1. 

# 	Returns
# 	----------
    
#     FOF_AccretionData_snap{snap2}_pre{pre_depth}_post{post_depth}_px.hdf5: hdf5 file with datasets
#         Header contains attributes:
#             "snap1"
#             "snap2"
#             "snap3"
#             "snap1_LookbackTime"
#             "snap2_LookbackTime"
#             "snap3_LookbackTime"
#             "ave_LookbackTime"
#             "delta_LookbackTime"
#             "snap1_z"
#             "snap2_z"
#             "snap3_z"
#             "ave_z

#         There is a group for each halo: ihalo_xxxxxx
        
#         Inflow:
#             For each particle type /PartTypeX/:
#                 Each of the following datasets will have n_in particles - the n_in initially selected as accretion "candidates" from being in the particle list at snap 2. 
#                 The particles are categorised by their type at SNAP 1 (i.e. prior to entering the halo)

#                 'ParticleIDs': ParticleID (in particle data for given type) of all accreted particles.
#                 'snap1_FOF':  Is the candidate particle included in the FOF list at snap 1?
#                 'snap1_SO':  Is the candidate particle included in the SO list at snap 1?
#                 'snap1_Processed': How many snaps has this particle been part of any structure in the past.
#                 'snap2_Bound': Is the candidate particle bound (in the FOF list) at snap 2?
#                 'snap2_FOF': Is the candidate particle included in the FOF at snap 2? (if 0, only in SO list)
#                 'snap3_Bound': Is the candidate particle bound (in the FOF list) at snap 3?
#                 'snap3_FOF': Is the candidate particle included in the FOF at snap 3? 
#                 'snap3_SO': Is the candidate particle included in the SO list at snap 3?

#         Outflow: 
#             For each particle type /PartTypeX/:
#                 Each of the following datasets will have n_out particles - the n initially selected as outflow "candidates" from disappearing from the SO particle list of the halo at snap 2 from snap 1. 
#                 The particles are categorised by their type at SNAP 2 (i.e. after entering the halo)

#                 'ParticleIDs': ParticleID (in particle data for given type) of all accreted particles.
#                 'snap1_FOF': Was the candidate particle included in the FOF at snap 1? (if 0, only in SO list)
#                 'snap2_Destination': Is the candidate particle bound (in the FOF list) at snap 2?
#                 'snap2_FOF': Is the candidate particle included in the FOF at snap 2?
#                 'snap3_Bound': Is the candidate particle bound (in the FOF list) at snap 3?
#                 'snap3_FOF': Is the candidate particle included in the FOF at snap 3?
#                 'snap3_SO': Is the candidate particle included in the SO list at snap 3?


#         Where there will be num_total_halos ihalo datasets. 
    
    
#     """
    
    
#     # Initialising halo index list
#     t1_io=time.time()

#     if halo_index_list==None:
#         halo_index_list_snap2=list(range(len(base_halo_data[snap]["hostHaloID"])))#use all halos if not handed halo index list
#         iprocess="x"
#         num_processes=1
#         test=True
#     else:
#         try:
#             halo_index_list_snap2=halo_index_list["indices"] #extract index list from input dictionary
#             iprocess=str(halo_index_list["iprocess"]).zfill(2) #the process for this index list (this is just used for the output file name)
#             print(f'iprocess {iprocess} has {len(halo_index_list_snap2)} halo indices: {halo_index_list_snap2}')
#             num_processes=halo_index_list["np"]
#             test=halo_index_list["test"]
#         except:
#             print('Not parsed a valud halo index list. Exiting.')
#             return None

#     # Create log file and directories
#     acc_log_dir=f"job_logs/acc_logs/"
#     if not os.path.exists(acc_log_dir):
#         os.mkdir(acc_log_dir)
#     if test:
#         run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_test/"
#     else:
#         run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}/"

#     if not os.path.exists(run_log_dir):
#         try:
#             os.mkdir(run_log_dir)
#         except:
#             pass

#     run_snap_log_dir=run_log_dir+f'snap_{str(snap).zfill(3)}/'

#     if not os.path.exists(run_snap_log_dir):
#         try:
#             os.mkdir(run_snap_log_dir)
#         except:
#             pass
#     if test:
#         fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}_test.log"
#         print(f'iprocess {iprocess} will save progress to log file: {fname_log}')

#     else:
#         fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}.log"

#     if os.path.exists(fname_log):
#         os.remove(fname_log)
    
#     with open(fname_log,"a") as progress_file:
#         progress_file.write('Initialising and loading in data ...\n')
#     progress_file.close()

#     # Assigning snap
#     if snap==None:
#         snap=len(base_halo_data)-1#if not given snap, just use the last one

#     # Find previous snap (to compare halo particles) and subsequent snap (to check accretion fidelity)
#     snap1=snap-pre_depth
#     snap2=snap
#     snap3=snap+post_depth

#     # Find the indices of halos at snap1 and snap3 (ordered by snap2 halo indices)
#     halo_index_list_snap1=[find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=pre_depth) for ihalo in halo_index_list_snap2]
#     halo_index_list_snap3=[find_descen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=post_depth) for ihalo in halo_index_list_snap2]

#     # Initialising outputs
#     if not os.path.exists('acc_data'):#create folder for outputs if doesn't already exist
#         os.mkdir('acc_data')
#     if test:
#         calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_test/'
#     else:
#         calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}/'

#     if not os.path.exists(calc_dir):#create folder for outputs if doesn't already exist
#         try:
#             os.mkdir(calc_dir)
#         except:
#             pass
#     calc_snap_dir=calc_dir+f'snap_{str(snap2).zfill(3)}/'
    
#     if not os.path.exists(calc_snap_dir):#create folder for outputs if doesn't already exist
#         try:
#             os.mkdir(calc_snap_dir)
#         except:
#             pass

#     run_outname=base_halo_data[snap]['outname']#extract output name (simulation name)
#     outfile_name=calc_snap_dir+'FOF_AccretionData_pre'+str(pre_depth).zfill(2)+'_post'+str(post_depth).zfill(2)+'_snap'+str(snap).zfill(3)+'_p'+str(iprocess).zfill(3)+f'_n{str(len(halo_index_list_snap2)).zfill(6)}.hdf5'
#     if os.path.exists(outfile_name):#if the accretion file already exists, get rid of it 
#         os.remove(outfile_name)

#     # Make header for accretion data  based on base halo data 
#     output_hdf5=h5py.File(outfile_name,"w")#initialise file object
#     header_hdf5=output_hdf5.create_group("Header")
#     lt_ave=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']+base_halo_data[snap2]['SimulationInfo']['LookbackTime'])/2
#     z_ave=(base_halo_data[snap1]['SimulationInfo']['z']+base_halo_data[snap2]['SimulationInfo']['z'])/2
#     dt=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']-base_halo_data[snap2]['SimulationInfo']['LookbackTime'])
#     t1=base_halo_data[snap1]['SimulationInfo']['LookbackTime']
#     t2=base_halo_data[snap2]['SimulationInfo']['LookbackTime']
#     t3=base_halo_data[snap3]['SimulationInfo']['LookbackTime']
#     z1=base_halo_data[snap1]['SimulationInfo']['z']
#     z2=base_halo_data[snap2]['SimulationInfo']['z']
#     z3=base_halo_data[snap3]['SimulationInfo']['z']
#     header_hdf5.attrs.create('ave_LookbackTime',data=lt_ave,dtype=np.float16)
#     header_hdf5.attrs.create('ave_z',data=z_ave,dtype=np.float16)
#     header_hdf5.attrs.create('delta_LookbackTime',data=dt,dtype=np.float16)
#     header_hdf5.attrs.create('snap1_LookbackTime',data=t1,dtype=np.float16)
#     header_hdf5.attrs.create('snap2_LookbackTime',data=t2,dtype=np.float16)
#     header_hdf5.attrs.create('snap3_LookbackTime',data=t3,dtype=np.float16)
#     header_hdf5.attrs.create('snap1_z',data=z1,dtype=np.float16)
#     header_hdf5.attrs.create('snap2_z',data=z2,dtype=np.float16)
#     header_hdf5.attrs.create('snap3_z',data=z3,dtype=np.float16)
#     header_hdf5.attrs.create('snap1',data=snap1,dtype=np.int16)
#     header_hdf5.attrs.create('snap2',data=snap2,dtype=np.int16)
#     header_hdf5.attrs.create('snap3',data=snap3,dtype=np.int16)
#     header_hdf5.attrs.create('pre_depth',data=snap2-snap1,dtype=np.int16)
#     header_hdf5.attrs.create('post_depth',data=snap3-snap2,dtype=np.int16)
#     header_hdf5.attrs.create('outname',data=np.string_(base_halo_data[snap2]['outname']))
#     header_hdf5.attrs.create('total_num_halos',data=base_halo_data[snap2]['Count'])

#     # Now find which simulation type we're dealing with
#     part_filetype=base_halo_data[snap]["Part_FileType"]
#     print(f'Particle data type: {part_filetype}')

#     # Standard particle type names from simulation
#     PartNames=['gas','DM','','','star','BH']
    
#     # Assign the particle types we're considering 
#     if part_filetype=='EAGLE':
#         PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
#         constant_mass={str(0):False,str(1):True,str(4):False,str(5):False}
#     else:
#         PartTypes=[0,1] #Gas, DM
#         constant_mass={str(0):True,str(1):True}

#     # Read in particle masses
#     h_val=base_halo_data[snap2]['SimulationInfo']['h_val']
#     if part_filetype=='EAGLE':# if an EAGLE snapshot
#         print('Reading in EAGLE snapshot data ...')
#         EAGLE_boxsize=base_halo_data[snap1]['SimulationInfo']['BoxSize_Comoving']
#         EAGLE_Snap_1=read_eagle.EagleSnapshot(base_halo_data[snap1]['Part_FilePath'])
#         EAGLE_Snap_1.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
#         EAGLE_Snap_2=read_eagle.EagleSnapshot(base_halo_data[snap2]['Part_FilePath'])
#         EAGLE_Snap_2.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
#         Part_Data_Masses_Snap1=dict();Part_Data_IDs_Snap1=dict()
#         Part_Data_Masses_Snap2=dict();Part_Data_IDs_Snap2=dict()
#         for itype in PartTypes:
#             print(f'Loading itype {itype} data ...')
#             if not itype==1:#everything except DM
#                 try:
#                     Part_Data_Masses_Snap1[str(itype)]=EAGLE_Snap_1.read_dataset(itype,"Mass")*10**10/h_val #CHECK THIS√
#                     Part_Data_Masses_Snap2[str(itype)]=EAGLE_Snap_2.read_dataset(itype,"Mass")*10**10/h_val #CHECK THIS√
#                 except:
#                     print('No particles of this type were found.')
#                     Part_Data_Masses_Snap1[str(itype)]=[]
#                     Part_Data_Masses_Snap2[str(itype)]=[]
#             else:#for DM, find particle data file and save 
#                 hdf5file=h5py.File(base_halo_data[snap1]['Part_FilePath'])#hdf5 file
#                 Part_Data_Masses_Snap1[str(itype)]=hdf5file['Header'].attrs['MassTable'][1]*10**10/h_val #CHECK THIS√
#                 Part_Data_Masses_Snap2[str(itype)]=hdf5file['Header'].attrs['MassTable'][1]*10**10/h_val #CHECK THIS√
#         print('Done reading in EAGLE snapshot data')
#     else:#assuming constant mass
#         Part_Data_Masses_Snap1=dict()
#         hdf5file=h5py.File(base_halo_data[snap1]['Part_FilePath'])
#         MassTable=hdf5file["Header"].attrs["MassTable"]
#         Part_Data_Masses_Snap1[str(1)]=MassTable[1]*10**10/h_val#CHECK THIS
#         Part_Data_Masses_Snap1[str(0)]=MassTable[0]*10**10/h_val#CHECK THIS
#         Part_Data_Masses_Snap2[str(1)]=MassTable[1]*10**10/h_val#CHECK THIS
#         Part_Data_Masses_Snap2[str(0)]=MassTable[0]*10**10/h_val#CHECK THIS

#     #Load in particle histories: snap 1
#     print(f'Retrieving & organising particle histories for snap = {snap1} ...')
#     Part_Histories_File_snap1=h5py.File("part_histories/PartHistory_"+str(snap1).zfill(3)+"_"+run_outname+".hdf5",'r')
#     Part_Histories_IDs_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIDs'].value for parttype in PartTypes}
#     Part_Histories_Index_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIndex'].value for parttype in PartTypes}
#     Part_Histories_HostStructure_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/HostStructure'].value for parttype in PartTypes}
#     Part_Histories_Processed_L1_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/Processed_L1'].value for parttype in [0,1]}
#     Part_Histories_Processed_L2_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/Processed_L2'].value for parttype in [0,1]}
#     Part_Histories_npart_snap1={str(parttype):len(Part_Histories_IDs_snap1[str(parttype)]) for parttype in PartTypes}

#     #Load in particle histories: snap 2
#     print(f'Retrieving & organising particle histories for snap = {snap2} ...')
#     Part_Histories_File_snap2=h5py.File("part_histories/PartHistory_"+str(snap2).zfill(3)+"_"+run_outname+".hdf5",'r')
#     Part_Histories_IDs_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIDs'].value for parttype in PartTypes}
#     Part_Histories_Index_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIndex'].value for parttype in PartTypes}
#     Part_Histories_HostStructure_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/HostStructure'].value for parttype in PartTypes}
#     Part_Histories_npart_snap2={str(parttype):len(Part_Histories_IDs_snap2[str(parttype)]) for parttype in PartTypes}

#     #Load in particle lists from VR
#     print('Retrieving VR halo particle lists ...')
#     snap_1_halo_particles_nosp=get_FOF_particle_lists(base_halo_data,snap1,halo_index_list=halo_index_list_snap1,add_subparts_to_fofs=False)
#     snap_2_halo_particles_nosp=get_FOF_particle_lists(base_halo_data,snap2,halo_index_list=halo_index_list_snap2,add_subparts_to_fofs=False)
#     snap_3_halo_particles_nosp=get_FOF_particle_lists(base_halo_data,snap3,halo_index_list=halo_index_list_snap3,add_subparts_to_fofs=False)
#     snap_1_halo_particles_wsp=get_FOF_particle_lists(base_halo_data,snap1,halo_index_list=halo_index_list_snap1,add_subparts_to_fofs=True)
#     snap_2_halo_particles_wsp=get_FOF_particle_lists(base_halo_data,snap2,halo_index_list=halo_index_list_snap2,add_subparts_to_fofs=True)
#     snap_3_halo_particles_wsp=get_FOF_particle_lists(base_halo_data,snap3,halo_index_list=halo_index_list_snap3,add_subparts_to_fofs=True)
#     t2_io=time.time()

#     print()
#     print('*********************************************************')
#     print(f'Done with I/O in {(t2_io-t1_io):.2f} sec - entering main halo loop ...')
#     print('*********************************************************')
#     with open(fname_log,"a") as progress_file:
#         progress_file.write(f'Done with I/O in {(t2_io-t1_io):.2f} sec - entering main halo loop ...\n')
#     progress_file.close()

#     count=0
#     halos_done=0
#     num_halos_thisprocess=len(halo_index_list_snap2)
#     for iihalo,ihalo_s2 in enumerate(halo_index_list_snap2):# for each halo at snap 2

#         t1_halo=time.time()
#         t1_preamble=time.time()

#         # Find halo progenitor and descendants 
#         ihalo_s1=halo_index_list_snap1[iihalo]#find progenitor
#         ihalo_s3=halo_index_list_snap3[iihalo]#find descendant
#         try:
#             idhalo_s1=base_halo_data[snap1]['ID'][ihalo_s1]
#             idhalo_s3=base_halo_data[snap3]['ID'][ihalo_s3]
#         except:
#             idhalo_s1=np.nan
#             idhalo_s3=np.nan
        
#         ihalo_tracked=(ihalo_s1>-1 and ihalo_s3>-1)#track if have both progenitor and descendant
#         structuretype=base_halo_data[snap2]["Structuretype"][ihalo_s2]#structure type
#         numsubstruct=base_halo_data[snap2]["numSubStruct"][ihalo_s2]

#         # If we have a subhalo, find its progenitor host group
#         if structuretype>10:
#             isub=True
#             ifield=False
#             try:
#                 current_hostgroupID=base_halo_data[snap2]["hostHaloID"][ihalo_s2]
#                 current_hostindex=np.where(current_hostgroupID==base_halo_data[snap2]["ID"])[0][0]
#                 prev_hostindex=find_progen_index(base_halo_data,index2=current_hostindex,snap2=snap2,depth=1) #host index at previous snapshot 
#                 prev_hostgroupID=base_halo_data[snap1]["ID"][prev_hostindex] #the host halo ID of this subhalo at the previous snapshot
#             except:#if can't find progenitor, don't try to compare for CGM accretion
#                 prev_hostHaloID=np.nan
#                 prev_hostgroupID=np.nan
#         else:
#             isub=False
#             ifield=True
#             prev_hostHaloID=np.nan
#             prev_hostgroupID=np.nan
        
#         # Create group for this halo in output file
#         ihalo_hdf5=output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))
#         if isub:
#             ihalo_types=[3]
#             ihalo_hdf5_gal=ihalo_hdf5.create_group('subhalo')
#             ihalo_hdf5_gal.attrs.create('HaloType',data=3)#satellite
#             ihalo_hdf5_gal.create_group('Inflow');ihalo_hdf5_gal.create_group('Outflow')

#         if ifield:
#             if numsubstruct>0:
#                 ihalo_types=[0,2]
#                 ihalo_hdf5_gal=ihalo_hdf5.create_group('subhalo')
#                 ihalo_hdf5_gal.attrs.create('HaloType',data=2)#central
#                 ihalo_hdf5_gal.create_group('Inflow');ihalo_hdf5_gal.create_group('Outflow')
#                 ihalo_hdf5_group=ihalo_hdf5.create_group('halo')
#                 ihalo_hdf5_group.attrs.create('HaloType',data=0)#group halo
#                 ihalo_hdf5_group.create_group('Inflow');ihalo_hdf5_group.create_group('Outflow')
#             else:
#                 ihalo_types=[1]
#                 ihalo_hdf5_group=ihalo_hdf5.create_group('halo')
#                 ihalo_hdf5_group.attrs.create('halotype',data=1)#field halo
#                 ihalo_hdf5_group.create_group('Inflow');ihalo_hdf5_group.create_group('Outflow')
        
#         with open(fname_log,"a") as progress_file:
#             progress_file.write(f' \n')
#             progress_file.write(f'Starting with ihalo {ihalo_s2}: types {ihalo_types}... \n')
#         progress_file.close()

#         #Record halo position and velocity
#         if ihalo_s1>=0:
#             ihalo_hdf5.attrs.create('snap1_com',data=[base_halo_data[snap1]['Xc'][ihalo_s1],base_halo_data[snap1]['Yc'][ihalo_s1],base_halo_data[snap1]['Zc'][ihalo_s1]],dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap1_v',data=[base_halo_data[snap1]['VXc'][ihalo_s1],base_halo_data[snap1]['VYc'][ihalo_s1],base_halo_data[snap1]['VZc'][ihalo_s1]],dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap1_R200',data=base_halo_data[snap1]['R_200crit'][ihalo_s1],dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap1_M200',data=base_halo_data[snap1]['Mass_200crit'][ihalo_s1]*10**10,dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap1_Vmax',data=base_halo_data[snap1]['Vmax'][ihalo_s1],dtype=np.float32)
#         if ihalo_s2>=0:
#             ihalo_hdf5.attrs.create('snap2_com',data=[base_halo_data[snap2]['Xc'][ihalo_s2],base_halo_data[snap2]['Yc'][ihalo_s2],base_halo_data[snap2]['Zc'][ihalo_s2]],dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap2_v',data=[base_halo_data[snap2]['VXc'][ihalo_s2],base_halo_data[snap2]['VYc'][ihalo_s2],base_halo_data[snap2]['VZc'][ihalo_s2]],dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap2_R200',data=base_halo_data[snap2]['R_200crit'][ihalo_s2],dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap2_M200',data=base_halo_data[snap2]['Mass_200crit'][ihalo_s2]*10**10,dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap2_Vmax',data=base_halo_data[snap2]['Vmax'][ihalo_s2],dtype=np.float32)
#         if ihalo_s3>=0:
#             ihalo_hdf5.attrs.create('snap3_com',data=[base_halo_data[snap3]['Xc'][ihalo_s3],base_halo_data[snap3]['Yc'][ihalo_s3],base_halo_data[snap3]['Zc'][ihalo_s3]],dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap3_v',data=[base_halo_data[snap3]['VXc'][ihalo_s3],base_halo_data[snap3]['VYc'][ihalo_s3],base_halo_data[snap3]['VZc'][ihalo_s3]],dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap3_R200',data=base_halo_data[snap3]['R_200crit'][ihalo_s3],dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap3_M200',data=base_halo_data[snap3]['Mass_200crit'][ihalo_s3]*10**10,dtype=np.float32)
#             ihalo_hdf5.attrs.create('snap3_Vmax',data=base_halo_data[snap3]['Vmax'][ihalo_s3],dtype=np.float32)


#         # Print halo data for outputs 
#         print()
#         print('**********************************************')
#         if ifield:
#             print('Halo index: ',ihalo_s2,f' - field halo')
#         if isub:
#             print('Halo index: ',ihalo_s2,f' - sub halo')
#             print(f'Host halo at previous snap: {prev_hostgroupID}')
#         print(f'Progenitor: {idhalo_s1} | Descendant: {idhalo_s3}')
#         print('**********************************************')
#         print()
        
#         t2_preamble=time.time()

#         # If this halo is going to be tracked (and is not a subsubhalo) then we continue
#         if ihalo_tracked and structuretype<25:# if we found both the progenitor and the descendent (and it's not a subsubhalo)
#             snap1_IDs_temp_wsp=snap_1_halo_particles_wsp['Particle_IDs'][str(ihalo_s1)]#IDs in the halo at the previous snap
#             snap1_Types_temp_wsp=snap_1_halo_particles_wsp['Particle_Types'][str(ihalo_s1)]#Types of particles in the halo at the previous snap
#             snap2_IDs_temp_wsp=snap_2_halo_particles_wsp['Particle_IDs'][str(ihalo_s2)]#IDs in the halo at the current snap
#             snap2_Types_temp_wsp=snap_2_halo_particles_wsp['Particle_Types'][str(ihalo_s2)]# Types of particles in the halo at the current snap
#             snap2_Bound_temp_wsp=snap_2_halo_particles_wsp['Particle_Bound'][str(ihalo_s2)]# Types of particles in the halo at the current snap

#             snap1_IDs_temp_nosp=snap_1_halo_particles_nosp['Particle_IDs'][str(ihalo_s1)]#IDs in the halo at the previous snap
#             snap1_Types_temp_nosp=snap_1_halo_particles_nosp['Particle_Types'][str(ihalo_s1)]#Types of particles in the halo at the previous snap
#             snap2_IDs_temp_nosp=snap_2_halo_particles_nosp['Particle_IDs'][str(ihalo_s2)]#IDs in the halo at the current snap
#             snap2_Types_temp_nosp=snap_2_halo_particles_nosp['Particle_Types'][str(ihalo_s2)]# Types of particles in the halo at the current snap
#             snap2_Bound_temp_nosp=snap_2_halo_particles_nosp['Particle_Bound'][str(ihalo_s2)]# Types of particles in the halo at the current snap
            
#             ############ GRABBING DATA FOR INFLOW PARTICLES (at snap 1) ############
#             # Returns mask for s2 of particles which are in s2 but not in s1
#             iihalo_type=0
#             for ihalo_type in ihalo_types:
#                 iihalo_type=iihalo_type+1
#                 if ihalo_type==0 or ihalo_type==1:
#                     ihalo_key='halo'
#                 else:
#                     ihalo_key='subhalo'

#                 print(f'Processing for ihalo {ihalo_s2} halo type {ihalo_type} ({iihalo_type}/{len(ihalo_types)})')
#                 with open(fname_log,"a") as progress_file:
#                     progress_file.write(f'Processing for ihalo type {ihalo_type} ...\n')
                
#                 print(f"Finding and indexing new particles to ihalo {ihalo_s2} ...")
#                 t1_new=time.time()
#                 if ihalo_type==0:
#                     snap3_IDs_temp_set=set(snap_3_halo_particles_wsp['Particle_IDs'][str(ihalo_s3)])
#                     new_particle_IDs_mask_snap2=np.isin(snap2_IDs_temp_wsp,snap1_IDs_temp_wsp,assume_unique=True,invert=True)
#                     new_particle_IDs_where_snap2=np.where(new_particle_IDs_mask_snap2)
#                     new_particle_IDs=snap2_IDs_temp_wsp[new_particle_IDs_where_snap2]
#                     new_particle_Types_snap2=snap2_Types_temp_wsp[new_particle_IDs_where_snap2]
#                     new_particle_Bound_snap2=snap2_Bound_temp_wsp[new_particle_IDs_where_snap2]
#                 else:
#                     snap3_IDs_temp_set=set(snap_3_halo_particles_nosp['Particle_IDs'][str(ihalo_s3)])
#                     new_particle_IDs_mask_snap2=np.isin(snap2_IDs_temp_nosp,snap1_IDs_temp_nosp,assume_unique=True,invert=True)
#                     new_particle_IDs_where_snap2=np.where(new_particle_IDs_mask_snap2)
#                     new_particle_IDs=snap2_IDs_temp_nosp[new_particle_IDs_where_snap2]
#                     new_particle_Types_snap2=snap2_Types_temp_nosp[new_particle_IDs_where_snap2]
#                     new_particle_Bound_snap2=snap2_Bound_temp_nosp[new_particle_IDs_where_snap2]
                
#                 ihalo_nin=np.nansum(new_particle_IDs_mask_snap2)
#                 print(f"n(in) = {ihalo_nin}")

#                 new_particle_Types_snap1,new_particle_historyindices_snap1,new_particle_partindices_snap1=get_particle_indices(base_halo_data=base_halo_data,
#                                                                         IDs_sorted=Part_Histories_IDs_snap1,
#                                                                         indices_sorted=Part_Histories_Index_snap1,
#                                                                         IDs_taken=new_particle_IDs,
#                                                                         types_taken=new_particle_Types_snap2,
#                                                                         snap_taken=snap2,
#                                                                         snap_desired=snap1)
#                 new_particle_tranformed=np.logical_not(new_particle_Types_snap1==new_particle_Types_snap2)

#                 t2_new=time.time()

#                 ihalo_snap1_inflow_type=new_particle_Types_snap1
#                 ihalo_snap1_inflow_transformed=new_particle_tranformed
#                 ihalo_snap1_inflow_history_L1=np.zeros(ihalo_nin)
#                 ihalo_snap1_inflow_history_L2=np.zeros(ihalo_nin)
#                 ihalo_snap1_inflow_structure=np.zeros(ihalo_nin)+np.nan
#                 ihalo_snap1_inflow_fidelity=np.zeros(ihalo_nin)
#                 ihalo_snap1_inflow_masses=np.zeros(ihalo_nin)
#                 ihalo_snap2_inflow_bound=new_particle_Bound_snap2

#                 # Find processing history, previous host, fidelity
#                 for iipartin,ipartin_ID,ipartin_snap1_type,ipartin_snap1_historyindex,ipartin_snap1_partindex in zip(list(range(ihalo_nin)),new_particle_IDs,new_particle_Types_snap1,new_particle_historyindices_snap1,new_particle_partindices_snap1):
#                     if ipartin_snap1_type>=0:
#                         if ipartin_snap1_type==0 or ipartin_snap1_type==1:#if DM or gas, this has been recorded
#                             ihalo_snap1_inflow_history_L1[iipartin]=Part_Histories_Processed_L1_snap1[str(ipartin_snap1_type)][ipartin_snap1_historyindex]
#                             ihalo_snap1_inflow_history_L2[iipartin]=Part_Histories_Processed_L2_snap1[str(ipartin_snap1_type)][ipartin_snap1_historyindex]
#                         else:#assume stars have been processed
#                             ihalo_snap1_inflow_history_L1[iipartin]=1
#                             ihalo_snap1_inflow_history_L2[iipartin]=1
#                         ihalo_snap1_inflow_structure[iipartin]=Part_Histories_HostStructure_snap1[str(ipartin_snap1_type)][ipartin_snap1_historyindex]

#                     ihalo_snap1_inflow_fidelity[iipartin]=int(ipartin_ID in snap3_IDs_temp_set)

#                 # Find mass
#                 for itype in PartTypes:
#                     ihalo_itype_snap1_inflow_mask=ihalo_snap1_inflow_type==itype
#                     ihalo_itype_snap1_inflow_where=np.where(ihalo_itype_snap1_inflow_mask)
#                     ihalo_itype_snap1_inflow_n=np.nansum(ihalo_itype_snap1_inflow_mask)
#                     ihalo_itype_snap1_inflow_partindices=new_particle_partindices_snap1[ihalo_itype_snap1_inflow_where]
#                     if constant_mass[str(itype)]:
#                         ihalo_itype_snap1_inflow_masses=np.ones(ihalo_itype_snap1_inflow_n)*Part_Data_Masses_Snap1[str(itype)]
#                     else:
#                         ihalo_itype_snap1_inflow_masses=np.array([Part_Data_Masses_Snap1[str(itype)][ihalo_itype_snap1_inflow_partindex] for ihalo_itype_snap1_inflow_partindex in ihalo_itype_snap1_inflow_partindices])
#                     ihalo_snap1_inflow_masses[ihalo_itype_snap1_inflow_where]=ihalo_itype_snap1_inflow_masses
                
#                 ############ GRABBING DATA FOR OUTFLOW PARTICLES (at snap 2) ############
#                 # # Returns mask for s1 of particles which are in s1 but not in s2
#                 print(f"Finding and indexing particles which left ihalo {ihalo_s2} ...")
#                 t1_out=time.time()
#                 if ihalo_type==0:
#                     out_particle_IDs_mask_snap1=np.isin(snap1_IDs_temp_wsp,snap2_IDs_temp_wsp,assume_unique=True,invert=True)
#                     out_particle_IDs_where_snap1=np.where(out_particle_IDs_mask_snap1)
#                     out_particle_IDs=snap1_IDs_temp_wsp[out_particle_IDs_where_snap1]
#                     out_particle_Types_snap1=snap1_Types_temp_wsp[out_particle_IDs_where_snap1]
#                 else:
#                     out_particle_IDs_mask_snap1=np.isin(snap1_IDs_temp_nosp,snap2_IDs_temp_nosp,assume_unique=True,invert=True)
#                     out_particle_IDs_where_snap1=np.where(out_particle_IDs_mask_snap1)
#                     out_particle_IDs=snap1_IDs_temp_nosp[out_particle_IDs_where_snap1]
#                     out_particle_Types_snap1=snap1_Types_temp_nosp[out_particle_IDs_where_snap1]
                
#                 ihalo_nout=np.nansum(out_particle_IDs_mask_snap1)
#                 print(f"n(out) = {ihalo_nout}")

#                 out_particle_Types_snap2,out_particle_historyindices_snap2,out_particle_partindices_snap2=get_particle_indices(base_halo_data=base_halo_data,
#                                                             IDs_sorted=Part_Histories_IDs_snap2,
#                                                             indices_sorted=Part_Histories_Index_snap2,
#                                                             IDs_taken=out_particle_IDs,
#                                                             types_taken=out_particle_Types_snap1,
#                                                             snap_taken=snap1,
#                                                             snap_desired=snap2)

#                 out_particle_tranformed=np.logical_not(np.array(out_particle_Types_snap1)==np.array(out_particle_Types_snap2))
#                 t2_out=time.time()

#                 with open(fname_log,"a") as progress_file:
#                     progress_file.write(f'       n(in): total = {ihalo_nin}\n')
#                     progress_file.write(f'       n(out): total = {ihalo_nout}\n')
#                 progress_file.close()
                
#                 ihalo_snap2_outflow_type=out_particle_Types_snap2
#                 ihalo_snap2_outflow_transformed=out_particle_tranformed
#                 ihalo_snap2_outflow_destination=np.zeros(ihalo_nout)+np.nan
#                 ihalo_snap3_outflow_recycled=np.zeros(ihalo_nout)+np.nan
#                 ihalo_snap2_outflow_masses=np.zeros(ihalo_nout)

#                 # Find processing history, previous host, fidelity
#                 for iipartout,ipartout_ID,ipartout_snap2_type,ipartout_snap2_historyindex,ipartout_snap2_partindex in zip(list(range(ihalo_nout)),out_particle_IDs,out_particle_Types_snap2,out_particle_historyindices_snap2,out_particle_partindices_snap2):
#                     if ipartout_snap2_type>=0:
#                         ihalo_snap2_outflow_destination[iipartout]=Part_Histories_HostStructure_snap2[str(ipartout_snap2_type)][ipartout_snap2_historyindex]
#                     ihalo_snap3_outflow_recycled[iipartout]=int(ipartout_ID in snap3_IDs_temp_set)
                
#                 # Find mass
#                 for itype in PartTypes:
#                     ihalo_itype_snap2_outflow_mask=ihalo_snap2_outflow_type==itype
#                     ihalo_itype_snap2_outflow_where=np.where(ihalo_itype_snap2_outflow_mask)
#                     ihalo_itype_snap2_outflow_n=np.nansum(ihalo_itype_snap2_outflow_mask)
#                     ihalo_itype_snap2_outflow_partindices=out_particle_partindices_snap2[ihalo_itype_snap2_outflow_where]
#                     if constant_mass[str(itype)]:
#                         ihalo_itype_snap2_outflow_masses=np.ones(ihalo_itype_snap2_outflow_n)*Part_Data_Masses_Snap2[str(itype)]
#                     else:
#                         ihalo_itype_snap2_outflow_masses=np.array([Part_Data_Masses_Snap2[str(itype)][ihalo_itype_snap2_outflow_partindex] for ihalo_itype_snap2_outflow_partindex in ihalo_itype_snap2_outflow_partindices])
#                     ihalo_snap2_outflow_masses[ihalo_itype_snap2_outflow_where]=ihalo_itype_snap2_outflow_masses

#                 ############ SAVE DATA FOR INLFOW & OUTFLOW PARTICLES ###########
#                 if ihalo_type ==0:
#                     print(f'Saving accretion data for ihalo {ihalo_s2}: group halo (TRACKED)')
#                 elif ihalo_type==1:
#                     print(f'Saving accretion data for ihalo {ihalo_s2}: field halo (TRACKED)')
#                 elif ihalo_type==2:
#                     print(f'Saving accretion data for ihalo {ihalo_s2}: central halo (TRACKED)')
#                 elif ihalo_type==3:
#                     print(f'Saving accretion data for ihalo {ihalo_s2}: satellite halo (TRACKED)')

#                 for iitype, itype in enumerate(PartTypes):

#                     # Saving INFLOW data for this parttype of the halo to file 
#                     ihalo_itype_snap1_inflow_mask=ihalo_snap1_inflow_type==itype#type the inflow particles based on snap 1 state
#                     ihalo_itype_snap1_inflow_where=np.where(ihalo_itype_snap1_inflow_mask)
                    
#                     ihalo_in_parttype_hdf5=ihalo_hdf5[ihalo_key]['Inflow'].create_group('PartType'+str(itype))
#                     ihalo_in_parttype_hdf5.create_dataset('ParticleIDs',data=np.array(new_particle_IDs[ihalo_itype_snap1_inflow_where]),dtype=np.int64)#######
#                     ihalo_in_parttype_hdf5.create_dataset('Transformed',data=np.array(ihalo_snap1_inflow_transformed[ihalo_itype_snap1_inflow_where]),dtype=np.int64)
#                     ihalo_in_parttype_hdf5.create_dataset('Processed_L1',data=ihalo_snap1_inflow_history_L1[ihalo_itype_snap1_inflow_where],dtype=np.uint8)
#                     ihalo_in_parttype_hdf5.create_dataset('Processed_L2',data=ihalo_snap1_inflow_history_L2[ihalo_itype_snap1_inflow_where],dtype=np.uint8)
#                     ihalo_in_parttype_hdf5.create_dataset('PreviousHost',data=ihalo_snap1_inflow_structure[ihalo_itype_snap1_inflow_where],dtype=np.int64)
#                     ihalo_in_parttype_hdf5.create_dataset('Fidelity',data=ihalo_snap1_inflow_fidelity[ihalo_itype_snap1_inflow_where],dtype=np.uint8)
#                     ihalo_in_parttype_hdf5.create_dataset('Masses',data=ihalo_snap1_inflow_masses[ihalo_itype_snap1_inflow_where],dtype=np.float64)
#                     ihalo_in_parttype_hdf5.create_dataset('Bound',data=ihalo_snap2_inflow_bound[ihalo_itype_snap1_inflow_where],dtype=np.uint8)

#                     # Saving OUTFLOW data for this parttype of the halo to file 
#                     ihalo_itype_snap2_outflow_mask=ihalo_snap2_outflow_type==itype#type the inflow particles based on snap 1 state
#                     ihalo_itype_snap2_outflow_where=np.where(ihalo_itype_snap2_outflow_mask)
                    
#                     ihalo_out_parttype_hdf5=ihalo_hdf5[ihalo_key]['Outflow'].create_group('PartType'+str(itype))
#                     ihalo_out_parttype_hdf5.create_dataset('ParticleIDs',data=out_particle_IDs[ihalo_itype_snap2_outflow_where],dtype=np.int64)#######
#                     ihalo_out_parttype_hdf5.create_dataset('Transformed',data=ihalo_snap2_outflow_transformed[ihalo_itype_snap2_outflow_where],dtype=np.uint8)
#                     ihalo_out_parttype_hdf5.create_dataset('Destination',data=ihalo_snap2_outflow_destination[ihalo_itype_snap2_outflow_where],dtype=np.int64)
#                     ihalo_out_parttype_hdf5.create_dataset('Recycled',data=ihalo_snap3_outflow_recycled[ihalo_itype_snap2_outflow_where],dtype=np.uint8)
#                     ihalo_out_parttype_hdf5.create_dataset('Masses',data=ihalo_snap2_outflow_masses[ihalo_itype_snap2_outflow_where],dtype=np.float64)

#         else:#if halo not tracked, return np.nan for fidelity, ids, prevhost
#             print(f'Saving accretion data for ihalo {ihalo_s2} (NOT TRACKED)')
#             iihalo_type=0
#             for ihalo_type in ihalo_types:
#                 iihalo_type=iihalo_type+1
#                 if ihalo_type==0 or ihalo_type==1:
#                     ihalo_key='halo'
#                 else:
#                     ihalo_key='subhalo'

#                 for itype in PartTypes:    
#                     # Saving INFLOW data for this parttype of the halo to file 
#                     ihalo_in_parttype_hdf5=ihalo_hdf5[ihalo_key]['Inflow'].create_group('PartType'+str(itype))
#                     ihalo_in_parttype_hdf5.create_dataset('ParticleIDs',data=np.nan,dtype=np.float16)#######
#                     ihalo_in_parttype_hdf5.create_dataset('Transformed',data=np.nan,dtype=np.float16)
#                     ihalo_in_parttype_hdf5.create_dataset('Processed_L1',data=np.nan,dtype=np.float16)
#                     ihalo_in_parttype_hdf5.create_dataset('Processed_L2',data=np.nan,dtype=np.float16)
#                     ihalo_in_parttype_hdf5.create_dataset('PreviousHost',data=np.nan,dtype=np.float16)
#                     ihalo_in_parttype_hdf5.create_dataset('Fidelity',data=np.nan,dtype=np.float16)
#                     ihalo_in_parttype_hdf5.create_dataset('Masses',data=np.nan,dtype=np.float16)
#                     ihalo_in_parttype_hdf5.create_dataset('Bound',data=np.nan,dtype=np.float16)
#                     # Saving OUTFLOW data for this parttype of the halo to file 
#                     ihalo_out_parttype_hdf5=ihalo_hdf5[ihalo_key]['Outflow'].create_group('PartType'+str(itype))
#                     ihalo_out_parttype_hdf5.create_dataset('ParticleIDs',data=np.nan,dtype=np.float16)
#                     ihalo_out_parttype_hdf5.create_dataset('Masses',data=np.nan,dtype=np.float16)
#                     ihalo_out_parttype_hdf5.create_dataset('Destination',data=np.nan,dtype=np.float16)
#                     ihalo_out_parttype_hdf5.create_dataset('Recycled',data=np.nan,dtype=np.float16)
#                     ihalo_out_parttype_hdf5.create_dataset('Transformed',data=np.nan,dtype=np.float16)
                
#         t2_halo=time.time()

#         with open(fname_log,"a") as progress_file:
#             progress_file.write(f"Done with ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
#             progress_file.write(f"[took {t2_halo-t1_halo} sec]\n")
#             progress_file.write(f" \n")
#         progress_file.close()

#         print()

#     #Close the output file, finish up
#     output_hdf5.close()

#     """

#     postprocess_acc_data_serial : function
# 	----------

#     Collate and post process all the accretion data in the provided directory (which must only contain the required data).

# 	Parameters
# 	----------
#     base_halo_data : the halo data list of dictionaries for this run
#     path : string indicating the directory in which the accretion data is stored (nominally acc_data/)

# 	Returns
# 	----------
    
#     Combined_AccData.hdf5: hdf5 file with datasets:
#         summed outputs
#         ---------------
#         In group '/(sub)halo/Inflow':
#         '/PartTypeX/All_TotalDeltaN': Total number of particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/All_TotalDeltaM': Total mass of particles of type X new to the halo  (length: num_total_halos)
#         '/PartTypeX/All_CosmologicalDeltaN': Total number of cosmological origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/All_CosmologicalDeltaM': Total mass of cosmological origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/All_PrimordialDeltaN': Total number of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/All_PrimordialDeltaM': Total mass of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/All_ProcessedCosmologicalDeltaN': Total number of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/All_ProcessedCosmologicalDeltaM': Total mass of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)    
#         '/PartTypeX/All_TransferredDeltaN': Total number of clumpy origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/All_TransferredDeltaM': Total mass of clumpy origin particles of type X new to the halo (length: num_total_halos)

#         '/PartTypeX/Stable_TotalDeltaN': Total number of particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/Stable_TotalDeltaM': Total mass of particles of type X new to the halo  (length: num_total_halos)
#         '/PartTypeX/Stable_CosmologicalDeltaN': Total number of cosmological origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/Stable_CosmologicalDeltaM': Total mass of cosmological origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/Stable_PrimordialDeltaN': Total number of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/Stable_PrimordialDeltaM': Total mass of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/Stable_ProcessedCosmologicalDeltaN': Total number of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/Stable_ProcessedCosmologicalDeltaM': Total mass of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)    
#         '/PartTypeX/Stable_TransferredDeltaN': Total number of clumpy origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/Stable_TransferredDeltaM': Total mass of clumpy origin particles of type X new to the halo (length: num_total_halos)

#         '/PartTypeX/BoundStable_TotalDeltaN': Total number of particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/BoundStable_TotalDeltaM': Total mass of particles of type X new to the halo  (length: num_total_halos)
#         '/PartTypeX/BoundStable_CosmologicalDeltaN': Total number of cosmological origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/BoundStable_CosmologicalDeltaM': Total mass of cosmological origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/BoundStable_PrimordialDeltaN': Total number of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/BoundStable_PrimordialDeltaM': Total mass of primordial (i.e. entirely unprocessed) origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/BoundStable_ProcessedCosmologicalDeltaN': Total number of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/BoundStable_ProcessedCosmologicalDeltaM': Total mass of recycled (i.e. processed at l2 but not at this time) origin particles of type X new to the halo (length: num_total_halos)    
#         '/PartTypeX/BoundStable_TransferredDeltaN': Total number of clumpy origin particles of type X new to the halo (length: num_total_halos)
#         '/PartTypeX/BoundStable_TransferredDeltaM': Total mass of clumpy origin particles of type X new to the halo (length: num_total_halos)

#         In group '/(sub)halo/Outflow':

#         '/Header' contains attributes: 
#         't1'
#         't2'
#         'dt'
#         'z_ave'
#         'lt_ave'
#         etc
    
#     """
#     t1=time.time()
#     print(f'Summing accretion data from path: {accdata_dir}')
#     if not accdata_dir.endswith('/'):
#         accdata_dir=accdata_dir+'/'

#     # List the contents of the provided directory
#     acc_data_filelist=os.listdir(accdata_dir)
#     acc_data_filelist=sorted(acc_data_filelist)
#     acc_data_filelist_trunc=[filename for filename in acc_data_filelist if (('px' not in filename) and ('FOF' in filename) and ('DS' not in filename) and ('summed' not in filename))]
    
#     print('Summing accretion data from the following files:')
#     print(np.array(acc_data_filelist_trunc))
#     acc_data_filelist=acc_data_filelist_trunc
#     acc_data_outfile_name=acc_data_filelist[0].split('_p0')[0]+'_summed.hdf5'

#     if os.path.exists(accdata_dir+acc_data_outfile_name):
#         print("Deleting existing combined data first")
#         os.remove(accdata_dir+acc_data_outfile_name)

#     print(f'Output file name: {acc_data_outfile_name}')
    
#     # Initialise output file
#     collated_output_file=h5py.File(accdata_dir+acc_data_outfile_name,'w')
    
#     # Open existing files in list structure
#     acc_data_hdf5files=[h5py.File(accdata_dir+acc_data_file,'r') for acc_data_file in acc_data_filelist]
#     acc_data_snap=acc_data_hdf5files[0]['Header'].attrs['snap2']
#     total_num_halos=0
#     for ifile in acc_data_hdf5files:
#         groups=list(ifile.keys())
#         for group in groups:
#             if 'ihalo' in group:
#                 total_num_halos=total_num_halos+1
#     if total_num_halos<1000:
#         print(f'Using array size {3*10**5}')
#         total_num_halos=3*10**5
#     else:
#         total_num_halos=base_halo_data[acc_data_snap]['Count']

#     print(f'Collating data for {total_num_halos} halos')
    
#     # Copy over header information from first file
#     acc_data_hdf5files_header=acc_data_hdf5files[0]['Header']
#     acc_data_hdf5files_header_attrs=list(acc_data_hdf5files_header.attrs)
#     collated_output_file_header=collated_output_file.create_group('Header')

#     #Print attributes of accretion calculation
#     print("Attributes of accretion calculation: ")
#     for attribute in acc_data_hdf5files_header_attrs:
#         collated_output_file_header.attrs.create(attribute,data=acc_data_hdf5files_header.attrs[attribute])
#         print(attribute,collated_output_file_header.attrs[attribute])

#     # Add extra header info if needed
#     try:
#         collated_output_file_header.attrs.create('outname',data=np.string_(base_halo_data[-1]['outname']))
#         collated_output_file_header.attrs.create('pre_depth',data=acc_data_hdf5files_header.attrs['snap2']-acc_data_hdf5files_header.attrs['snap1'])
#         collated_output_file_header.attrs.create('post_depth',data=acc_data_hdf5files_header.attrs['snap3']-acc_data_hdf5files_header.attrs['snap2'])
#         collated_output_file_header.attrs.create('total_num_halos',data=total_num_halos)
#     except:
#         pass

#     # Names of the new outputs
#     new_outputs_inflow=[
#     "All_TotalDeltaM_In",
#     "All_TotalDeltaN_In",
#     "All_CosmologicalDeltaN_In",
#     'All_CosmologicalDeltaM_In',
#     'All_TransferredDeltaN_In',
#     'All_TransferredDeltaM_In',
#     'All_PrimordialDeltaN_In',
#     'All_PrimordialDeltaM_In',
#     'All_ProcessedCosmologicalDeltaN_In',
#     'All_ProcessedCosmologicalDeltaM_In',   
#     "Stable_TotalDeltaM_In",
#     "Stable_TotalDeltaN_In",
#     "Stable_CosmologicalDeltaN_In",
#     'Stable_CosmologicalDeltaM_In',
#     'Stable_TransferredDeltaN_In',
#     'Stable_TransferredDeltaM_In',
#     'Stable_PrimordialDeltaN_In',
#     'Stable_PrimordialDeltaM_In',
#     'Stable_ProcessedCosmologicalDeltaN_In',
#     'Stable_ProcessedCosmologicalDeltaM_In',
#     "StableBound_TotalDeltaM_In",
#     "StableBound_TotalDeltaN_In",
#     "StableBound_CosmologicalDeltaN_In",
#     'StableBound_CosmologicalDeltaM_In',
#     'StableBound_TransferredDeltaN_In',
#     'StableBound_TransferredDeltaM_In',
#     'StableBound_PrimordialDeltaN_In',
#     'StableBound_PrimordialDeltaM_In',
#     'StableBound_ProcessedCosmologicalDeltaN_In',
#     'StableBound_ProcessedCosmologicalDeltaM_In'
#     ]

#     new_outputs_outflow=[
#     "All_TotalDeltaM_Out",
#     "All_TotalDeltaN_Out",
#     "All_FieldDeltaM_Out",
#     "All_FieldDeltaN_Out",
#     "All_TransferredDeltaM_Out",
#     "All_TransferredDeltaN_Out",
#     "All_RecycledDeltaN_Out",#at snap 3
#     "All_RecycledDeltaM_Out",
#     "Stable_TotalDeltaM_Out",
#     "Stable_TotalDeltaN_Out",
#     "Stable_FieldDeltaM_Out",
#     "Stable_FieldDeltaN_Out",
#     "Stable_TransferredDeltaM_Out",
#     "Stable_TransferredDeltaN_Out"]#at snap 3


#     # Initialise all new outputs
#     first_file=acc_data_hdf5files[0]
#     first_halo_group=[key for key in list(first_file.keys()) if 'ihalo' in key][0]
#     first_halo_parttype_keys=list(first_file[first_halo_group]['halo']['Inflow'].keys())
#     first_halo_inflow_keys=list(first_file[first_halo_group]['halo']['Inflow']['PartType0'].keys())
#     first_halo_outflow_keys=list(first_file[first_halo_group]['halo']['Outflow']['PartType0'].keys())
#     parttypes=[int(parttype_key.split('Type')[-1]) for parttype_key in first_halo_parttype_keys]
#     print(f'Grabbing data for part types: {parttypes}')

#     new_outputs_keys_bytype_in=[f'Inflow/PartType{itype}/'+field for field in new_outputs_inflow for itype in parttypes]
#     new_outputs_keys_bytype_out=[f'Outflow/PartType{itype}/'+field for field in new_outputs_outflow for itype in parttypes]

#     summed_acc_data={}
#     for infield in new_outputs_keys_bytype_in:
#         summed_acc_data['subhalo/'+infield]=np.zeros(total_num_halos)+np.nan
#         summed_acc_data['halo/'+infield]=np.zeros(total_num_halos)+np.nan
        
#     for outfield in new_outputs_keys_bytype_out:
#         summed_acc_data['subhalo/'+outfield]=np.zeros(total_num_halos)+np.nan
#         summed_acc_data['halo/'+outfield]=np.zeros(total_num_halos)+np.nan

#     #Loop through each acc data file
#     iihalo=0
#     for ifile,acc_data_filetemp in enumerate(acc_data_hdf5files):
#         print(f"Reading from file {ifile+1}/{len(acc_data_hdf5files)}: {acc_data_filetemp}")
#         ihalo_group_list_all=list(acc_data_filetemp.keys())
#         ihalo_group_list=[ihalo_group for ihalo_group in ihalo_group_list_all if ihalo_group.startswith('ihalo')]
        
#         #Loop through each halo group in this acc data file
#         for ihalo_group in ihalo_group_list:
#             print(ihalo_group)
#             iihalo=iihalo+1
#             ihalo=int(ihalo_group.split('_')[-1])
#             ihalo_keys=list(acc_data_filetemp[ihalo_group].keys()) #halo and/or subhalo
            
#             #Loop through each calc type for this halo
#             for ihalo_key in ihalo_keys:
#                 #Loop through each particle type group in this ihalo group and calc 
#                 ihalo_itype_inflow_data={}
#                 ihalo_itype_outflow_data={}
#                 for itype in parttypes:
#                     # Load in the details of particles new to this halo
#                     # try:#if can't load, skip itype for this halo
#                     for inflow_key in first_halo_inflow_keys:
#                         ihalo_itype_inflow_data[inflow_key]=acc_data_filetemp[ihalo_group+f'/{ihalo_key}/Inflow/PartType{itype}/{inflow_key}'].value
#                     # except:#skip
#                     #     continue
                    
#                     #Load in the details of particles which left this halo
#                     # try:#if can't load, skip itype for this halo
#                     for outflow_key in first_halo_outflow_keys:
#                         ihalo_itype_outflow_data[outflow_key]=acc_data_filetemp[ihalo_group+f'/{ihalo_key}/Outflow/PartType{itype}/{outflow_key}'].value
#                     # except:#skip
#                     #     continue

#                     if type(ihalo_itype_inflow_data['Fidelity'])==np.float16:
#                         continue
                    
#                     ############################ PROCESS/CALCULATE INFLOW ############################

#                     # Define masks based on particle properties
#                     stable_mask=ihalo_itype_inflow_data['Fidelity']>0
#                     bound_mask=ihalo_itype_inflow_data['Bound']>0
#                     cosmological_mask=ihalo_itype_inflow_data['PreviousHost']<0
#                     transfer_mask=ihalo_itype_inflow_data['PreviousHost']>0
#                     primordial_mask=ihalo_itype_inflow_data['Processed_L1']==0
#                     processed_mask=ihalo_itype_inflow_data['Processed_L1']>0
#                     processed_cosmological_mask=np.logical_and(processed_mask,cosmological_mask)

#                     stable_cosmological_mask=np.logical_and(stable_mask,cosmological_mask)
#                     stable_primordial_mask=np.logical_and(stable_cosmological_mask,primordial_mask)
#                     stable_processedcosmological_mask=np.logical_and(stable_cosmological_mask,processed_mask)
#                     stable_transfer_mask=np.logical_and(stable_mask,transfer_mask)

#                     stable_bound_mask=np.logical_and(stable_mask,bound_mask)
#                     stable_bound_cosmological_mask=np.logical_and(stable_bound_mask,cosmological_mask)
#                     stable_bound_primordial_mask=np.logical_and(stable_bound_mask,primordial_mask)
#                     stable_bound_processedcosmological_mask=np.logical_and(stable_bound_mask,processed_cosmological_mask)
#                     stable_bound_transfer_mask=np.logical_and(stable_bound_mask,transfer_mask)

#                     # Save the inflow data for this itype, halo and file to our running dictionary
#                     masses=ihalo_itype_inflow_data['Masses']

#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_TotalDeltaN_In'][ihalo]=np.size(masses)
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_TotalDeltaM_In'][ihalo]=np.nansum(masses)
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_CosmologicalDeltaN_In'][ihalo]=np.size(np.compress(cosmological_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_CosmologicalDeltaM_In'][ihalo]=np.nansum(np.compress(cosmological_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_TransferredDeltaN_In'][ihalo]=np.size(np.compress(transfer_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_TransferredDeltaM_In'][ihalo]=np.nansum(np.compress(transfer_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_PrimordialDeltaN_In'][ihalo]=np.size(np.compress(primordial_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_PrimordialDeltaM_In'][ihalo]=np.nansum(np.compress(primordial_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_ProcessedCosmologicalDeltaN_In'][ihalo]=np.size(np.compress(processed_cosmological_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/All_ProcessedCosmologicalDeltaM_In'][ihalo]=np.nansum(np.compress(processed_cosmological_mask,masses))

#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_TotalDeltaN_In'][ihalo]=np.size(np.compress(stable_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_TotalDeltaM_In'][ihalo]=np.nansum(np.compress(stable_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_CosmologicalDeltaN_In'][ihalo]=np.size(np.compress(stable_cosmological_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_CosmologicalDeltaM_In'][ihalo]=np.nansum(np.compress(stable_cosmological_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_TransferredDeltaN_In'][ihalo]=np.size(np.compress(stable_transfer_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_TransferredDeltaM_In'][ihalo]=np.nansum(np.compress(stable_transfer_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_PrimordialDeltaN_In'][ihalo]=np.size(np.compress(stable_primordial_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_PrimordialDeltaM_In'][ihalo]=np.nansum(np.compress(stable_primordial_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_ProcessedCosmologicalDeltaN_In'][ihalo]=np.size(np.compress(stable_processedcosmological_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/Stable_ProcessedCosmologicalDeltaM_In'][ihalo]=np.nansum(np.compress(stable_processedcosmological_mask,masses))

#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_TotalDeltaN_In'][ihalo]=np.size(np.compress(stable_bound_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_TotalDeltaM_In'][ihalo]=np.nansum(np.compress(stable_bound_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_CosmologicalDeltaN_In'][ihalo]=np.size(np.compress(stable_bound_cosmological_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_CosmologicalDeltaM_In'][ihalo]=np.nansum(np.compress(stable_bound_cosmological_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_TransferredDeltaN_In'][ihalo]=np.size(np.compress(stable_bound_transfer_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_TransferredDeltaM_In'][ihalo]=np.nansum(np.compress(stable_bound_transfer_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_PrimordialDeltaN_In'][ihalo]=np.size(np.compress(stable_bound_primordial_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_PrimordialDeltaM_In'][ihalo]=np.nansum(np.compress(stable_bound_primordial_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_ProcessedCosmologicalDeltaN_In'][ihalo]=np.size(np.compress(stable_bound_processedcosmological_mask,masses))
#                     summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/StableBound_ProcessedCosmologicalDeltaM_In'][ihalo]=np.nansum(np.compress(stable_bound_processedcosmological_mask,masses))
                    
#                     ############################ PROCESS/CALCULATE OUTFLOW ############################

#                     outfield_mask=ihalo_itype_outflow_data['Destination']==-1
#                     outhalo_mask=ihalo_itype_outflow_data['Destination']>0
#                     reaccreted_mask=ihalo_itype_outflow_data['Recycled']==1

#                     stable_mask=np.logical_not(reaccreted_mask)
#                     stable_outfield_mask=np.logical_and(stable_mask,outfield_mask)
#                     stable_outhalo_mask=np.logical_and(stable_mask,outhalo_mask)

#                     # Save the outflow data for this itype, halo and file to our running dictionary
#                     masses_out=ihalo_itype_outflow_data['Masses']

#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/All_TotalDeltaN_Out'][ihalo]=np.size(masses_out)
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/All_TotalDeltaM_Out'][ihalo]=np.nansum(masses_out)
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/All_FieldDeltaN_Out'][ihalo]=np.size(np.compress(outfield_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/All_FieldDeltaM_Out'][ihalo]=np.nansum(np.compress(outfield_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/All_TransferredDeltaN_Out'][ihalo]=np.size(np.compress(outhalo_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/All_TransferredDeltaM_Out'][ihalo]=np.nansum(np.compress(outhalo_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/All_RecycledDeltaN_Out'][ihalo]=np.size(np.compress(reaccreted_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/All_RecycledDeltaM_Out'][ihalo]=np.nansum(np.compress(reaccreted_mask,masses_out))

#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/Stable_TotalDeltaN_Out'][ihalo]=np.size(np.compress(stable_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/Stable_TotalDeltaM_Out'][ihalo]=np.nansum(np.compress(stable_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/Stable_FieldDeltaN_Out'][ihalo]=np.size(np.compress(stable_outfield_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/Stable_FieldDeltaM_Out'][ihalo]=np.nansum(np.compress(stable_outfield_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/Stable_TransferredDeltaN_Out'][ihalo]=np.size(np.compress(stable_outhalo_mask,masses_out))
#                     summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/Stable_TransferredDeltaM_Out'][ihalo]=np.nansum(np.compress(stable_outhalo_mask,masses_out))

#     # Create groups for output hdf5
#     collated_output_file_halos=collated_output_file.create_group('halo');collated_output_file_subhalos=collated_output_file.create_group('subhalo')
#     collated_output_file_halos_inflow=collated_output_file_halos.create_group('Inflow');collated_output_file_halos_outflow=collated_output_file_halos.create_group('Outflow')
#     collated_output_file_subhalos_inflow=collated_output_file_subhalos.create_group('Inflow');collated_output_file_subhalos_outflow=collated_output_file_subhalos.create_group('Outflow')

#     # Loop through each particle type
#     for ihalo_key in ['subhalo','halo']:
#         for itype in parttypes:
#             collated_output_file_inflow_itype=collated_output_file[ihalo_key]['Inflow'].create_group(f'PartType{itype}')
#             collated_output_file_outflow_itype=collated_output_file[ihalo_key]['Outflow'].create_group(f'PartType{itype}')
#             #Save data from collated dictionary to output file
#             for new_field in new_outputs_inflow:
#                 collated_output_file_inflow_itype.create_dataset(name=new_field,data=summed_acc_data[f'{ihalo_key}/Inflow/PartType{itype}/'+new_field],dtype=np.float32)
#             for new_field in new_outputs_outflow:
#                 collated_output_file_outflow_itype.create_dataset(name=new_field,data=summed_acc_data[f'{ihalo_key}/Outflow/PartType{itype}/'+new_field],dtype=np.float32)
#     collated_output_file.close()

#     t2=time.time()
#     print(f'Finished collating files in {t2-t1} sec')
#     return None

# ########################### GENERATE DETAILED ACCRETION DATA ###########################

# def gen_accretion_data_detailed_serial(base_halo_data,snap=None,halo_index_list=None,pre_depth=1,post_depth=1,compression=False):
    
#     """

#     gen_accretion_data_detailed_serial : function
# 	----------

#     Generate and save accretion rates for each particle type by comparing particle lists from VELOCIraptor FOF outputs. 

#     ** note: particle histories, base_halo_data and halo particle data must have been generated as per gen_particle_history_serial (this file),
#              gen_base_halo_data in STFTools.py and dump_structure_particle_data in STFTools.py

# 	Parameters
# 	----------
#     base_halo_data : list of dictionaries
#         The minimal halo data list of dictionaries previously generated ("B1" is sufficient)

#     snap : int
#         The index in the base_halo_data for which to calculate accretion rates (should be actual snap index)
#         We will retrieve particle data based on the flags at this index
    
#     halo_index_list : dict
#         "iprocess": int
#         "indices: list of int
#         List of the halo indices for which to calculate accretion rates. If 'None',
#         find for all halos in the base_halo_data dictionary at the desired snapshot. 

#     pre_depth : int
#         How many snaps to skip back to when comparing particle lists.
#         Initial snap for calculation will be snap-pre_depth. 

#     pre_depth : int
#         How many snaps to skip back to when comparing particle lists.
#         Initial snap (s1) for calculation will be s1=snap-pre_depth, and we will check particle histories at s1.

# 	Returns
# 	----------
#     FOF_AccretionData_snap{snap2}_pre{pre_depth}_post{post_depth}_px.hdf5: hdf5 file with datasets
#         Header contains attributes:
#             "snap1"
#             "snap2"
#             "snap3"
#             "snap1_LookbackTime"
#             "snap2_LookbackTime"
#             "snap3_LookbackTime"
#             "ave_LookbackTime" (snap 1 -> snap 2)
#             "delta_LookbackTime" (snap 1 -> snap 2)
#             "snap1_z"
#             "snap2_z"
#             "snap3_z"
#             "ave_z (snap 1 -> snap 2)

#         There is a group for each halo: ihalo_xxxxxx
#             - we iterate through each SO region and find the corresponding halo. 
#             - return nan datasets if cannot find match for halo.

#         Each halo group with attributes:
#         "snapx_com"
#         "snapx_cminpot"
#         "snapx_cmbp"
#         "snapx_vmax"
#         "snapx_v"
#         "snapx_M_200crit"
#         "snapx_R_200mean"
#         "snapx_R_200crit"
        
#         Inflow:
#             For each particle type /PartTypeX/:
#                 Each of the following datasets will have n_in particles - the n_in initially selected as accretion "candidates" from being in the SO particle list at snap 2. 
#                 The particles are categorised by their type at SNAP 1 (i.e. prior to entering the halo)

#                 'ParticleIDs': ParticleID (in particle data for given type) of all accreted particles.
#                 'Masses': Mass of all accretion candidates at snap 1. 
                
#                 'snap1_Processed': How many snaps has this particle been part of any structure in the past.
#                 'snap1_FOF': Is the candidate particle included in the FOF at snap 1? 
#                 'snap1_Bound': Is the candidate particle bound (in the FOF list) at snap 1?
#                 'snap1_Coordinates': The absolute coordinates of the particle at snap 1. 
#                 'snap1_Velocity': The absolute instantaneous velocity of the particle at snap 1. 
#                 'snap1_r_xx': The relative coordinate of the particle at snap 1 relative to halo center from xx. 
#                 'snap1_rabs_xx': The radius of the particle at snap 1 relative to halo center from xx. 
#                 'snap1_vrad': The instantaneous radial velocity of the particle at snap 1 relative to halo (using com). 
#                 'snap1_vtan': The instantaneous tangential velocity of the particle at snap 1 relative to halo (using com). 
#                 'snap2_FOF': Is the candidate particle included in the FOF at snap 2? 
#                 'snap2_Bound': Is the candidate particle bound (in the FOF list) at snap 2?
#                 'snap2_Coordinates': The absolute coordinates of the particle at snap 2. 
#                 'snap2_Velocity': The absolute instantaneous velocity of the particle at snap 2. 
#                 'snap2_r_xx': The relative coordinate of the particle at snap 2 relative to halo center from xx. 
#                 'snap2_rabs_xx': The radius of the particle at snap 2 relative to halo center from xx. 
#                 'snap2_vrad': The instantaneous radial velocity of the particle at snap 2 relative to halo (using com). 
#                 'snap2_vtan': The instantaneous tangential velocity of the particle at snap 2 relative to halo (using com). 
#                 'snap3_FOF': Is the candidate particle included in the FOF at snap 3? 
#                 'snap3_Bound': Is the candidate particle bound (in the FOF list) at snap 3?
#                 'ave_vrad_xx': Average radial velocity from snap1_r -> snap2_r, where r is taken relative to halo center from xx. 
                
#                 Where xx can be from com, cminpot, or cmbp. 

#         Outflow: 
#             For each particle type /PartTypeX/:
#                 Each of the following datasets will have n_out particles - the n initially selected by being in the particle list at snap 1.
#                 The particles are categorised by their type at SNAP 2 (i.e. after entering the halo)

#                 'ParticleIDs': ParticleID (in particle data for given type) of all accreted particles.
#                 'Masses': Mass of all accretion candidates at snap 1. 
#                 'snap1_Processed': How many snaps has this particle been part of any structure in the past.
#                 'snap1_FOF': Is the candidate particle included in the FOF at snap 1? 
#                 'snap1_Bound': Is the candidate particle bound (in the FOF list) at snap 1?
#                 'snap1_Coordinates': The absolute coordinates of the particle at snap 1. 
#                 'snap1_Velocity': The absolute instantaneous velocity of the particle at snap 1. 
#                 'snap1_r_xx': The relative coordinate of the particle at snap 1 relative to halo center from xx. 
#                 'snap1_rabs_xx': The radius of the particle at snap 1 relative to halo center from xx. 
#                 'snap1_vrad': The instantaneous radial velocity of the particle at snap 1 relative to halo (using com). 
#                 'snap1_vtan': The instantaneous tangential velocity of the particle at snap 1 relative to halo (using com). 
#                 'snap2_FOF': Is the candidate particle included in the FOF at snap 2? 
#                 'snap2_Bound': Is the candidate particle bound (in the FOF list) at snap 2?
#                 'snap2_Coordinates': The absolute coordinates of the particle at snap 2. 
#                 'snap2_Velocity': The absolute instantaneous velocity of the particle at snap 2. 
#                 'snap2_r_xx': The relative coordinate of the particle at snap 2 relative to halo center from xx. 
#                 'snap2_rabs_xx': The radius of the particle at snap 2 relative to halo center from xx. 
#                 'snap2_vrad': The instantaneous radial velocity of the particle at snap 2 relative to halo (using com). 
#                 'snap2_vtan': The instantaneous tangential velocity of the particle at snap 2 relative to halo (using com). 
#                 'snap3_FOF': Is the candidate particle included in the FOF at snap 3? 
#                 'snap3_Bound': Is the candidate particle bound (in the FOF list) at snap 3?
#                 'ave_vrad_xx': Average radial velocity from snap1_r -> snap2_r, where r is taken relative to halo center from xx. 

#                 Where xx can be from com, cminpot, or cmbp. 

#         Where there will be num_total_halos ihalo datasets. 
    
#     """
    
    
#     # Initialising halo index list
#     t1_io=time.time()

#     if halo_index_list==None:
#         halo_index_list_snap2=list(range(len(base_halo_data[snap]["hostHaloID"])))#use all halos if not handed halo index list
#         iprocess="x"
#         num_processes=1
#         test=True
#     else:
#         try:
#             halo_index_list_snap2=halo_index_list["indices"] #extract index list from input dictionary
#             iprocess=str(halo_index_list["iprocess"]).zfill(2) #the process for this index list (this is just used for the output file name)
#             print(f'iprocess {iprocess} has {len(halo_index_list_snap2)} halo indices: {halo_index_list_snap2}')
#             num_processes=halo_index_list["np"]
#             test=halo_index_list["test"]
#         except:
#             print('Not parsed a valud halo index list. Exiting.')
#             return None

#     # Create log file and directories
#     acc_log_dir=f"job_logs/acc_logs/"
#     if not os.path.exists(acc_log_dir):
#         os.mkdir(acc_log_dir)
#     if test:
#         run_log_dir=f"job_logs/acc_logs/detailed_pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_test/"
#     else:
#         run_log_dir=f"job_logs/acc_logs/detailed_pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}/"

#     if not os.path.exists(run_log_dir):
#         try:
#             os.mkdir(run_log_dir)
#         except:
#             pass

#     run_snap_log_dir=run_log_dir+f'snap_{str(snap).zfill(3)}/'

#     if not os.path.exists(run_snap_log_dir):
#         try:
#             os.mkdir(run_snap_log_dir)
#         except:
#             pass
#     if test:
#         fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}_test.log"
#         print(f'iprocess {iprocess} will save progress to log file: {fname_log}')

#     else:
#         fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}.log"

#     if os.path.exists(fname_log):
#         os.remove(fname_log)
    
#     with open(fname_log,"a") as progress_file:
#         progress_file.write('Initialising and loading in data ...\n')
#     progress_file.close()

#     # Assigning snap
#     if snap==None:
#         snap=len(base_halo_data)-1#if not given snap, just use the last one

#     if compression==False:
#         compression=None
#     else:
#         compression='gzip'

#     # Find previous snap (to compare halo particles) and subsequent snap (to check accretion fidelity)
#     snap1=snap-pre_depth
#     snap2=snap
#     snap3=snap+post_depth
#     snaps=[snap1,snap2,snap3]

#     # Find the indices of halos at snap1 and snap3 (ordered by snap2 halo indices)
#     halo_index_list_snap1=[find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=pre_depth) for ihalo in halo_index_list_snap2]
#     halo_index_list_snap3=[find_descen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=post_depth) for ihalo in halo_index_list_snap2]

#     # Initialising outputs
#     if not os.path.exists('acc_data'):#create folder for outputs if doesn't already exist
#         os.mkdir('acc_data')
#     if test:
#         calc_dir=f'acc_data/detailed_pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_test/'
#     else:
#         calc_dir=f'acc_data/detailed_pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}/'

#     if not os.path.exists(calc_dir):#create folder for outputs if doesn't already exist
#         try:
#             os.mkdir(calc_dir)
#         except:
#             pass
#     calc_snap_dir=calc_dir+f'snap_{str(snap2).zfill(3)}/'
    
#     if not os.path.exists(calc_snap_dir):#create folder for outputs if doesn't already exist
#         try:
#             os.mkdir(calc_snap_dir)
#         except:
#             pass

#     run_outname=base_halo_data[snap]['outname']#extract output name (simulation name)
#     outfile_name=calc_snap_dir+'Detailed_AccretionData_pre'+str(pre_depth).zfill(2)+'_post'+str(post_depth).zfill(2)+'_snap'+str(snap).zfill(3)+'_p'+str(iprocess).zfill(3)+'.hdf5'
    
#     if not os.path.exists(outfile_name):#if the accretion file doesn't exists, initialise with header
#         print(f'Initialising output file at {outfile_name}...')
#         output_hdf5=h5py.File(outfile_name,"w")#initialise file object
#         # Make header for accretion data  based on base halo data 
#         header_hdf5=output_hdf5.create_group("Header")
#         lt_ave=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']+base_halo_data[snap2]['SimulationInfo']['LookbackTime'])/2
#         z_ave=(base_halo_data[snap1]['SimulationInfo']['z']+base_halo_data[snap2]['SimulationInfo']['z'])/2
#         dt=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']-base_halo_data[snap2]['SimulationInfo']['LookbackTime'])
#         t1=base_halo_data[snap1]['SimulationInfo']['LookbackTime']
#         t2=base_halo_data[snap2]['SimulationInfo']['LookbackTime']
#         t3=base_halo_data[snap3]['SimulationInfo']['LookbackTime']
#         z1=base_halo_data[snap1]['SimulationInfo']['z']
#         z2=base_halo_data[snap2]['SimulationInfo']['z']
#         z3=base_halo_data[snap3]['SimulationInfo']['z']
#         header_hdf5.attrs.create('ave_LookbackTime',data=lt_ave,dtype=np.float16)
#         header_hdf5.attrs.create('ave_z',data=z_ave,dtype=np.float16)
#         header_hdf5.attrs.create('delta_LookbackTime',data=dt,dtype=np.float16)
#         header_hdf5.attrs.create('snap1_LookbackTime',data=t1,dtype=np.float16)
#         header_hdf5.attrs.create('snap2_LookbackTime',data=t2,dtype=np.float16)
#         header_hdf5.attrs.create('snap3_LookbackTime',data=t3,dtype=np.float16)
#         header_hdf5.attrs.create('snap1_z',data=z1,dtype=np.float16)
#         header_hdf5.attrs.create('snap2_z',data=z2,dtype=np.float16)
#         header_hdf5.attrs.create('snap3_z',data=z3,dtype=np.float16)
#         header_hdf5.attrs.create('snap1',data=snap1,dtype=np.int16)
#         header_hdf5.attrs.create('snap2',data=snap2,dtype=np.int16)
#         header_hdf5.attrs.create('snap3',data=snap3,dtype=np.int16)
#         header_hdf5.attrs.create('pre_depth',data=snap2-snap1,dtype=np.int16)
#         header_hdf5.attrs.create('post_depth',data=snap3-snap2,dtype=np.int16)
#         header_hdf5.attrs.create('outname',data=np.string_(base_halo_data[snap2]['outname']))
#         header_hdf5.attrs.create('total_num_halos',data=base_halo_data[snap2]['Count'])

#     else:
#         print(f'Opening existing output file at {outfile_name} ...')
#         output_hdf5=h5py.File(outfile_name,"r+")#initialise file object

#     # Now find which simulation type we're dealing with
#     part_filetype=base_halo_data[snap]["Part_FileType"]
#     print(f'Particle data type: {part_filetype}')

#     # Standard particle type names from simulation
#     PartNames=['gas','DM','','','star','BH']

#     # Assign the particle types we're considering 
#     if part_filetype=='EAGLE':
#         PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
#         Mass_DM=base_halo_data[snap2]['SimulationInfo']['Mass_DM_Physical']
#         Mass_Gas=base_halo_data[snap2]['SimulationInfo']['Mass_Gas_Physical']

#     #Load in FOF particle lists: snap 1, snap 2, snap 3
#     FOF_Part_Data={}
#     FOF_Part_Data[str(snap1)]=get_FOF_particle_lists(base_halo_data,snap1,halo_index_list=halo_index_list_snap1)
#     FOF_Part_Data[str(snap2)]=get_FOF_particle_lists(base_halo_data,snap2,halo_index_list=halo_index_list_snap2)
#     FOF_Part_Data[str(snap3)]=get_FOF_particle_lists(base_halo_data,snap3,halo_index_list=halo_index_list_snap3)
#     FOF_Part_Data_fields=list(FOF_Part_Data[str(snap1)].keys())

#     #Particle data filepath
#     hval=base_halo_data[snap1]['SimulationInfo']['h_val'];scalefactors={}
#     scalefactors={str(snap):base_halo_data[snap]['SimulationInfo']['ScaleFactor'] for snap in snaps}
#     Part_Data_FilePaths={str(snap):base_halo_data[snap]['Part_FilePath'] for snap in snaps}
#     Part_Data_fields=['Coordinates','Velocity','Mass','ParticleIDs']
#     Part_Data_cube_fields=['Coordinates','Velocity','Mass','ParticleTypes']
#     Part_Data_comtophys={str(snap):{'Coordinates':scalefactors[str(snap)]/hval,'Velocity':scalefactors[str(snap)]/hval,'Mass':10.0**10/hval,'ParticleIDs':1} for snap in snaps}
    
#     #Load in particle histories: snap 1
#     print(f'Retrieving & organising particle histories for snap = {snap1} ...')
#     Part_Histories_File_snap1=h5py.File("part_histories/PartHistory_"+str(snap1).zfill(3)+"_"+run_outname+".hdf5",'r')
#     Part_Histories_IDs_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIDs'].value for parttype in PartTypes}
#     Part_Histories_Index_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIndex'].value for parttype in PartTypes}
#     Part_Histories_npart_snap1={str(parttype):len(Part_Histories_IDs_snap1[str(parttype)]) for parttype in PartTypes}
#     Part_Histories_HostStructure_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/HostStructure'].value for parttype in PartTypes}
#     Part_Histories_Processed_L1_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/Processed_L1'].value for parttype in [0,1]}
#     Part_Histories_Processed_L1_snap1[str(4)]=np.ones(Part_Histories_npart_snap1[str(4)]);Part_Histories_Processed_L1_snap1[str(5)]=np.ones(Part_Histories_npart_snap1[str(5)])

#     #Load in particle histories: snap 2
#     print(f'Retrieving & organising particle histories for snap = {snap2} ...')
#     Part_Histories_File_snap2=h5py.File("part_histories/PartHistory_"+str(snap2).zfill(3)+"_"+run_outname+".hdf5",'r')
#     Part_Histories_IDs_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIDs'].value for parttype in PartTypes}
#     Part_Histories_Index_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIndex'].value for parttype in PartTypes}
#     Part_Histories_HostStructure_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/HostStructure'].value for parttype in PartTypes}
#     Part_Histories_npart_snap2={str(parttype):len(Part_Histories_IDs_snap2[str(parttype)]) for parttype in PartTypes}
    
#     print()
#     t2_io=time.time()
#     print('*********************************************************')
#     print(f'Done with I/O in {(t2_io-t1_io):.2f} sec - entering main halo loop ...')
#     print('*********************************************************')

#     with open(fname_log,"a") as progress_file:
#         progress_file.write(f'Done with I/O in {(t2_io-t1_io):.2f} sec - entering main halo loop ...\n')
#     progress_file.close()


#     # Initialise summed outputs
#     # default params
#     r200_facs=[0.125,0.25,0.5,0.75,1,1.25,1.5,2]
#     vmax_facs=[-1,0,0.125,0.25,0.375,0.5,0.75,1]
#     cgm_r200_fac=1.5

#     num_halos_thisprocess=len(halo_index_list_snap2)
#     integrated_output_hdf5=output_hdf5.create_group('integrated_output')
#     integrated_output_hdf5.create_dataset('ihalo',data=halo_index_list_snap2)
    
#     output_fields={'Inflow':['Total_DeltaM_In','Field_DeltaM_In','CGM_DeltaM_In','Merger_DeltaM_In'],#for all halos
#                   'Outflow':['Total_DeltaM_Out','Field_DeltaM_Out','CGM_DeltaM_Out','Transfer_DeltaM_Out']}
    
#     for output_group in ['Inflow','Outflow']:
#         integrated_output_hdf5.create_group(output_group)
#         integrated_output_hdf5[output_group].create_group('FOF-haloscale');integrated_output_hdf5[output_group].create_group('FOF-subhaloscale')

#         for ir200_fac,r200_fac in enumerate(r200_facs):
#             integrated_output_hdf5[output_group].create_group(f'SO-r200_fac_{ir200_fac+1}')
#             integrated_output_hdf5[output_group][f'SO-r200_fac_{ir200_fac+1}'].attrs.create('r200_fac',data=r200_fac)

#         for ivmax_fac, vmax_fac in enumerate(vmax_facs):
#             ivmax_key=f'vmax_fac_{ivmax_fac+1}'
            
#             #haloscale
#             integrated_output_hdf5[output_group]['FOF-haloscale'].create_group(ivmax_key);integrated_output_hdf5[output_group]['FOF-haloscale'][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
#             for itype in parttypes:
#                 itype_key=f'PartType{itype}'
#                 integrated_output_hdf5[output_group][f'FOF-haloscale'][ivmax_key].create_group(itype_key)
#                 for dataset in output_fields[output_group]:
#                     integrated_output_hdf5[output_group][f'FOF-haloscale'][ivmax_key][itype_key].create_dataset(dataset,data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
#             #subhaloscale
#             integrated_output_hdf5[output_group]['FOF-subhaloscale'].create_group(ivmax_key);integrated_output_hdf5[output_group]['FOF-subhaloscale'][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
#             for itype in parttypes:
#                 itype_key=f'PartType{itype}'
#                 integrated_output_hdf5[output_group][f'FOF-subhaloscale'][ivmax_key].create_group(itype_key)
#                 for dataset in output_fields[output_group]:
#                     integrated_output_hdf5[output_group][f'FOF-subhaloscale'][ivmax_key][itype_key].create_dataset(dataset,data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
#             #SO
#             for ir200_fac,r200_fac in enumerate(r200_facs):
#                 ir200_key=f'SO-r200_fac_{ir200_fac+1}'
#                 integrated_output_hdf5[output_group][ir200_key].create_group(ivmax_key);integrated_output_hdf5[output_group][ir200_key][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
#                 for itype in parttypes:
#                     itype_key=f'PartType{itype}'
#                     integrated_output_hdf5[output_group][ir200_key][ivmax_key].create_group(itype_key)
#                     for dataset in output_fields[output_group]:
#                         integrated_output_hdf5[output_group][ir200_key][ivmax_key][itype_key].create_dataset(dataset,data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)

#     for iihalo,ihalo_s2 in enumerate(halo_index_list_snap2):# for each halo at snap 2
#         t1_halo=time.time()

#         #Create group for this halo in output file
#         try:
#             ihalo_hdf5=output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))
#             ihalo_hdf5.create_group('Inflow');ihalo_hdf5.create_group('Outflow')
#             ihalo_hdf5.create_group('Metadata')
#             for itype in PartTypes:
#                 ihalo_hdf5['Inflow'].create_group(f'PartType{itype}')
#                 ihalo_hdf5['Outflow'].create_group(f'PartType{itype}')
#         except:
#             ihalo_hdf5=output_hdf5['ihalo_'+str(ihalo_s2).zfill(6)]
#             ihalo_hdf5_inkeys=list(ihalo_hdf5['Inflow'].keys());ihalo_hdf5_outkeys=list(ihalo_hdf5['Outflow'].keys());ihalo_hdf5_mdkeys=list(ihalo_hdf5['Metadata'].keys())
#             for itype in PartTypes:
#                 for ihalo_hdf5_inkey in ihalo_hdf5_inkeys: del ihalo_hdf5['Inflow'][f'PartType{itype}'][ihalo_hdf5_inkey]
#                 for ihalo_hdf5_outkey in ihalo_hdf5_outkeys: del ihalo_hdf5['Outflow'][f'PartType{itype}'][ihalo_hdf5_inkey]
#                 for ihalo_hdf5_mdkey in ihalo_hdf5_inkeys: del ihalo_hdf5['Metadata'][f'PartType{itype}'][ihalo_hdf5_mdkey]
        
#         try:
#             # Find halo progenitor and descendants
#             ihalo_indices={str(snap1):halo_index_list_snap1[iihalo],str(snap2):ihalo_s2,str(snap3):halo_index_list_snap3[iihalo]}
            
#             #Record halo properties as attributes 
#             ihalo_tracked=(ihalo_indices[str(snap1)]>-1 and ihalo_indices[str(snap3)]>-1)#track if have both progenitor and descendant
#             structuretype=base_halo_data[snap2]["Structuretype"][ihalo_indices[str(snap2)]]#structure type
#             numsubstruct=base_halo_data[snap2]["numSubStruct"][ihalo_indices[str(snap2)]]
#             sublevel=int(np.floor((structuretype-0.01)/10))

#             # Print progress to terminal
#             print()
#             print('**********************************************')
#             print('Halo index: ',ihalo_s2,f' - {numsubstruct} substructures')
#             print(f'Progenitor: {ihalo_indices[str(snap1)]} | Descendant: {ihalo_indices[str(snap3)]}')
#             print('**********************************************')
#             print()

#             # Print progress to output file
#             with open(fname_log,"a") as progress_file:
#                 progress_file.write(f' \n')
#                 progress_file.write(f'Starting with ihalo {ihalo_s2} ... \n')
#             progress_file.close()
            
#             if ihalo_tracked:
#                 for isnap,snap in enumerate(snaps):
#                     ihalo_isnap=ihalo_indices[str(snap)]
#                     if ihalo_isnap>=0:
#                         ihalo_hdf5['Metadata'].create_dataset(f'snap{isnap+1}_com',data=[base_halo_data[snap]['Xc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Yc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zc'][ihalo_indices[str(snap)]]],dtype=np.float32,shape=(1,3))
#                         ihalo_hdf5['Metadata'].create_dataset(f'snap{isnap+1}_cminpot',data=[base_halo_data[snap]['Xcminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Ycminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zcminpot'][ihalo_indices[str(snap)]]],dtype=np.float32,shape=(1,3))
#                         ihalo_hdf5['Metadata'].create_dataset(f'snap{isnap+1}_vcom',data=[base_halo_data[snap]['VXc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VYc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VZc'][ihalo_indices[str(snap)]]],dtype=np.float32,shape=(1,3))
#                         ihalo_hdf5['Metadata'].create_dataset(f'snap{isnap+1}_R_200crit',data=base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]],dtype=np.float32)
#                         ihalo_hdf5['Metadata'].create_dataset(f'snap{isnap+1}_R_200mean',data=base_halo_data[snap]['R_200mean'][ihalo_indices[str(snap)]],dtype=np.float32)
#                         ihalo_hdf5['Metadata'].create_dataset(f'snap{isnap+1}_Mass_200crit',data=base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*10**10,dtype=np.float32)
#                         ihalo_hdf5['Metadata'].create_dataset(f'snap{isnap+1}_vmax',data=base_halo_data[snap]['Vmax'][ihalo_indices[str(snap)]],dtype=np.float32)
#                         ihalo_hdf5['Metadata'].create_dataset(f'snap{isnap+1}_vesc_crit',data=np.sqrt(2*base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*base_halo_data[snap]['SimulationInfo']['Gravity']/base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]]),dtype=np.float32)

#                 ihalo_hdf5['Metadata'].create_dataset('sublevel',data=sublevel,dtype=np.int16)
#                 ihalo_hdf5['Metadata'].create_dataset('ave_R_200crit',data=0.5*base_halo_data[snap1]['R_200crit'][ihalo_indices[str(snap1)]]+0.5*base_halo_data[snap2]['R_200crit'][ihalo_indices[str(snap2)]],dtype=np.int16)

#                 #Grab the FOF particle data
#                 ihalo_fof_particles={}
#                 for snap in snaps:
#                     ihalo_fof_particles[str(snap)]={field:FOF_Part_Data[str(snap)][field][str(ihalo_indices[str(snap)])] for field in FOF_Part_Data_fields}
#                     ihalo_fof_particles[str(snap)]['SortedIndices']=np.argsort(ihalo_fof_particles[str(snap)]['Particle_IDs'])
#                     ihalo_fof_particles[str(snap)]['SortedIDs']=ihalo_fof_particles[str(snap)]['Particle_IDs'][(ihalo_fof_particles[str(snap)]['SortedIndices'],)]
#                     ihalo_fof_particles[str(snap)]['ParticleIDs_set']=set(ihalo_fof_particles[str(snap)]['Particle_IDs'])

#                 #Grab/slice EAGLE datacubes
#                 print(f'Retrieving datacubes for ihalo {ihalo_s2} ...')
#                 use='cminpot'
#                 ihalo_com_physical={str(snap):ihalo_hdf5['Metadata'][f'snap{isnap+1}_{use}'].value for isnap,snap in enumerate(snaps)}
#                 ihalo_com_comoving={str(snap):ihalo_hdf5['Metadata'][f'snap{isnap+1}_{use}'].value/Part_Data_comtophys[str(snap)]['Coordinates'] for isnap,snap in enumerate(snaps)}
#                 ihalo_vcom_physical={str(snap):ihalo_hdf5['Metadata'][f'snap{isnap+1}_vcom'].value for isnap,snap in enumerate(snaps)}

#                 ihalo_cube_rfac=1.25
#                 ihalo_cuberadius_physical={str(snap):ihalo_hdf5['Metadata'][f'snap{isnap+1}_R_200mean'].value*ihalo_cube_rfac for snap in snaps}
#                 ihalo_cuberadius_comoving={str(snap):ihalo_hdf5['Metadata'][f'snap{isnap+1}_R_200mean'].value/Part_Data_comtophys[str(snap)]['Coordinates']*ihalo_cube_rfac for isnap,snap in enumerate(snaps)}
                
#                 #Read EAGLE datacubes
#                 ihalo_cube_particles={str(snap):{field:[] for field in Part_Data_fields} for snap in snaps}
#                 ihalo_cube_npart={str(snap):{} for snap in snaps}
#                 t1_slicing=time.time()
#                 for snap in snaps:

#                     ihalo_EAGLE_snap=read_eagle.EagleSnapshot(Part_Data_FilePaths[str(snap)])
#                     ihalo_EAGLE_snap.select_region(xmin=ihalo_com_comoving[str(snap)][0][0]-ihalo_cuberadius_comoving[str(snap)],xmax=ihalo_com_comoving[str(snap)][0][0]+ihalo_cuberadius_comoving[str(snap)],
#                                                 ymin=ihalo_com_comoving[str(snap)][0][1]-ihalo_cuberadius_comoving[str(snap)],ymax=ihalo_com_comoving[str(snap)][0][1]+ihalo_cuberadius_comoving[str(snap)],
#                                                 zmin=ihalo_com_comoving[str(snap)][0][2]-ihalo_cuberadius_comoving[str(snap)],zmax=ihalo_com_comoving[str(snap)][0][2]+ihalo_cuberadius_comoving[str(snap)])
#                     ihalo_EAGLE_types=[]
#                     for itype in PartTypes:       
#                         for ifield,field in enumerate(Part_Data_fields):
#                             if not (field=='Mass' and itype==1):
#                                 data=ihalo_EAGLE_snap.read_dataset(itype,field)*Part_Data_comtophys[str(snap)][field];ihalo_cube_npart[str(snap)][str(itype)]=len(data)     
#                             else:
#                                 data=np.ones(ihalo_cube_npart[str(snap)]['1'])*Mass_DM
#                             ihalo_cube_particles[str(snap)][field].extend(data)
#                         ihalo_EAGLE_types.extend((np.ones(ihalo_cube_npart[str(snap)][str(itype)])*itype).astype(int))
                    
#                     ihalo_cube_particles[str(snap)]['ParticleTypes']=np.array(ihalo_EAGLE_types)
#                     for field in Part_Data_fields:
#                         ihalo_cube_particles[str(snap)][field]=np.array(ihalo_cube_particles[str(snap)][field])

#                     ihalo_cube_particles[str(snap)]['SortedIndices']=np.argsort(ihalo_cube_particles[str(snap)]['ParticleIDs'])
#                     ihalo_cube_particles[str(snap)]['SortedIDs']=ihalo_cube_particles[str(snap)]['ParticleIDs'][(ihalo_cube_particles[str(snap)]['SortedIndices'],)]

#                 t2_slicing=time.time()
#                 print(f'Finished datacubes for ihalo {ihalo_s2} in {t2_slicing-t1_slicing:.2f} sec')
                
#                 ############################## INFLOW ##############################
#                 ####################################################################

#                 # SELECT INFLOW CANDIDATES AS THOSE WITHIN R200crit OR the FOF envelope at snap 2
#                 ihalo_ave_R_200crit_physical=(ihalo_hdf5['Metadata']['snap1_R_200crit'].value+ihalo_hdf5['Metadata']['snap2_R_200crit'].value)/2
#                 try:
#                     ihalo_cube_rminpot_snap2=np.sqrt(np.sum(np.square(ihalo_cube_particles[str(snap2)]['Coordinates']-ihalo_com_physical[str(snap2)]),axis=1))
#                     ihalo_cube_radialcut_snap2=np.where(ihalo_cube_rminpot_snap2<ihalo_ave_R_200crit_physical)
#                     ihalo_cube_inflow_candidate_data_snap2={field:ihalo_cube_particles[str(snap2)][field] for field in Part_Data_fields}
#                     ihalo_fof_inflow_candidate_data_snap2={field:ihalo_fof_particles[str(snap2)][field] for field in FOF_Part_Data_fields}
#                     ihalo_combined_inflow_candidate_IDs=np.concatenate([ihalo_fof_inflow_candidate_data_snap2['Particle_IDs'],ihalo_cube_inflow_candidate_data_snap2['ParticleIDs']])
#                     ihalo_combined_inflow_candidate_inFOF=np.concatenate([np.ones(len(ihalo_fof_inflow_candidate_data_snap2['Particle_IDs'])),np.zeros(len(ihalo_cube_inflow_candidate_data_snap2['ParticleIDs']))])
#                     ihalo_combined_inflow_candidate_IDs_unique,ihalo_combined_inflow_candidate_IDs_unique_indices=np.unique(ihalo_combined_inflow_candidate_IDs,return_index=True)
#                     ihalo_combined_inflow_candidate_IDs_unique=np.array(ihalo_combined_inflow_candidate_IDs_unique,dtype=np.int64)
#                     ihalo_combined_inflow_candidate_count=len(ihalo_combined_inflow_candidate_IDs_unique)
#                 except:
#                     print(f'Skipping ihalo {ihalo_s2} (couldnt retrieve data cube)')
#                     continue

#                 #GRAB DATA FOR EACH INFLOW CANDIDATE
#                 ihalo_combined_inflow_candidate_data={}

#                 # OUTPUTS FROM DATACUBE: Coordinates, Velocity, Mass, Type 
#                 ihalo_combined_inflow_candidate_cubeindices={}
#                 print(f'num inflow candidates for ihalo {ihalo_s2}: {ihalo_combined_inflow_candidate_count}')
#                 for isnap,snap in enumerate(snaps):
            
#                     #find the indices of the IDs in the (sorted) datacube for this halo (will return nan if not in the cube) - outputs index
#                     ihalo_combined_inflow_candidate_IDindices_temp=binary_search(ihalo_combined_inflow_candidate_IDs_unique,sorted_list=ihalo_cube_particles[str(snap)]['SortedIDs'],check_entries=True)
#                     #use the indices from the sorted IDs above to extract the cube indices (will return nan if not in the cube) - outputs index
#                     ihalo_combined_inflow_candidate_cubeindices[str(snap)]=mask_wnans(array=ihalo_cube_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_inflow_candidate_IDindices_temp)
#                     #use the cube indices to extract particle data, record which particles couldn't be found
                    

#                     #for each snap, grab detailed particle data
#                     for field in ['Coordinates','Velocity','Mass','ParticleIDs','ParticleTypes']:
#                         ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_cube_particles[str(snap)][field],indices=ihalo_combined_inflow_candidate_cubeindices[str(snap)])
                        
#                     #derive other cubdata outputs
#                     ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com']=ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Coordinates']-ihalo_com_physical[str(snap)]
#                     ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_rabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com']),axis=1))
#                     ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_runit_com']=np.divide(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com'],np.column_stack([ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_rabs_com']]*3))
#                     ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_v_com']=ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Velocity']-ihalo_vcom_physical[str(snap)]
#                     ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_v_com']),axis=1))
#                     ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vrad_com']=np.sum(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_runit_com']*ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_v_com'],axis=1)
#                     ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vtan_com']=np.sqrt(np.square(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vabs_com'])-np.square(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vrad_com']))
#                     ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Particle_InCube']=np.isfinite(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_ParticleTypes'])

#                 #include average radial velocity
#                 vel_conversion=978.462 #Mpc/Gyr to km/s
#                 ihalo_combined_inflow_candidate_data[f'ave_vrad_com']=(ihalo_combined_inflow_candidate_data[f'snap2_rabs_com']-ihalo_combined_inflow_candidate_data[f'snap1_rabs_com'])/dt*vel_conversion
            
#                 # OUTPUTS FROM FOF: InFOF, Bound
#                 ihalo_combined_inflow_candidate_fofindices={}
#                 for isnap,snap in enumerate(snaps):
#                     #find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
#                     ihalo_combined_inflow_candidate_IDindices_temp=binary_search(ihalo_combined_inflow_candidate_IDs_unique,sorted_list=ihalo_fof_particles[str(snap)]['SortedIDs'],check_entries=True)
#                     #use the indices from the sorted IDs above to extract the fof indices (will return nan if not in the fof) - outputs index
#                     ihalo_combined_inflow_candidate_fofindices[str(snap)]=mask_wnans(array=ihalo_fof_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_inflow_candidate_IDindices_temp)
                    
#                     #use the fof indices to extract particle data, record which particles couldn't be found
#                     ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']=np.isfinite(ihalo_combined_inflow_candidate_IDindices_temp)
#                     ihalo_combined_inflow_candidate_fofdata_notinfofmask=np.where(np.logical_not(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']))
#                     ihalo_combined_inflow_candidate_fofdata_notinfofmask_count=len(ihalo_combined_inflow_candidate_fofdata_notinfofmask[0])
#                     for field in ['Particle_Bound','Particle_InHost']:
#                         ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_fof_particles[str(snap)][field],indices=ihalo_combined_inflow_candidate_fofindices[str(snap)])
#                         ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_{field}'][ihalo_combined_inflow_candidate_fofdata_notinfofmask]=np.zeros(ihalo_combined_inflow_candidate_fofdata_notinfofmask_count)

#                 # OUTPUTS FROM HISTORIES: Processed, Structure (by particle type)
#                 ihalo_combined_inflow_candidate_partindices={}
#                 ihalo_combined_inflow_candidate_data['snap1_Structure']=np.zeros(ihalo_combined_inflow_candidate_count)
#                 ihalo_combined_inflow_candidate_data['snap1_Processed']=np.zeros(ihalo_combined_inflow_candidate_count)
                
#                 for itype in PartTypes:
#                     ihalo_combined_inflow_candidate_typemask_snap1=np.where(ihalo_combined_inflow_candidate_data['snap1_ParticleTypes']==itype)
#                     ihalo_combined_inflow_candidate_IDs_unique_itype=ihalo_combined_inflow_candidate_IDs_unique[ihalo_combined_inflow_candidate_typemask_snap1]
#                     #find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
#                     ihalo_combined_inflow_candidate_IDindices_temp=binary_search(ihalo_combined_inflow_candidate_IDs_unique_itype,sorted_list=Part_Histories_IDs_snap1[str(itype)],check_entries=False)
#                     #use the indices from the sorted IDs above to extract the partdata indices (will return nan if not in the fof) - outputs index
#                     ihalo_combined_inflow_candidate_partindices[str(itype)]=Part_Histories_Index_snap1[str(itype)][(ihalo_combined_inflow_candidate_IDindices_temp,)]
#                     #extract host structure and processing
#                     ihalo_combined_inflow_candidate_data['snap1_Structure'][ihalo_combined_inflow_candidate_typemask_snap1]=Part_Histories_HostStructure_snap1[str(itype)][ihalo_combined_inflow_candidate_partindices[str(itype)]]
#                     ihalo_combined_inflow_candidate_data['snap1_Processed'][ihalo_combined_inflow_candidate_typemask_snap1]=Part_Histories_Processed_L1_snap1[str(itype)][ihalo_combined_inflow_candidate_partindices[str(itype)]]

#                 # SAVE TO FILE 
#                 output_fields_dtype={}
#                 output_fields_float16=['ave_vrad_com',"r_com","rabs_com","vrad_com","vtan_com","Mass"]
#                 for field in output_fields_float16:
#                     output_fields_dtype[field]=np.float16

#                 output_fields_int64=["ParticleIDs","Structure"]
#                 for field in output_fields_int64:
#                     output_fields_dtype[field]=np.int64
                
#                 output_fields_int8=["Processed","Particle_InFOF","Particle_Bound","Particle_InHost","Particle_InCube"]
#                 for field in output_fields_int8:
#                     output_fields_dtype[field]=np.int8        

#                 for itype in PartTypes:
#                     itype_key=f'PartType{itype}'
                
#                     ### PARTICLE OUTPUTS
#                     # Candidate particle types and masses taken at snap 1 (before "entering" halo)
#                     ihalo_itype_mask=np.where(ihalo_combined_inflow_candidate_data["snap1_ParticleTypes"]==itype)
                    
#                     ihalo_hdf5['Inflow'][itype_key].create_dataset('ParticleIDs',data=ihalo_combined_inflow_candidate_IDs_unique[ihalo_itype_mask],dtype=output_fields_dtype["ParticleIDs"],compression=compression)
#                     ihalo_hdf5['Inflow'][itype_key].create_dataset('Mass',data=ihalo_combined_inflow_candidate_data['snap1_Mass'][ihalo_itype_mask],dtype=output_fields_dtype["Mass"],compression=compression)
#                     ihalo_hdf5['Inflow'][itype_key].create_dataset('ave_vrad_com',data=ihalo_combined_inflow_candidate_data['ave_vrad_com'][ihalo_itype_mask],dtype=output_fields_dtype["ave_vrad_com"],compression=compression)

#                     #Rest of fields: snap 1
#                     ihalo_snap1_inflow_outputs=["Structure","Processed","r_com","rabs_com","vrad_com","vtan_com","Particle_InFOF","Particle_Bound","Particle_InHost","Particle_InCube"]
#                     for ihalo_snap1_inflow_output in ihalo_snap1_inflow_outputs:
#                         ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap1_{ihalo_snap1_inflow_output}',data=ihalo_combined_inflow_candidate_data[f'snap1_{ihalo_snap1_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap1_inflow_output],compression=compression)
                    
#                     #Rest of fields: snap 2
#                     ihalo_snap2_inflow_outputs=["r_com","rabs_com","vrad_com","vtan_com","Particle_InFOF","Particle_Bound","Particle_InHost"]

#                     for ihalo_snap2_inflow_output in ihalo_snap2_inflow_outputs:
#                         ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap2_{ihalo_snap2_inflow_output}',data=ihalo_combined_inflow_candidate_data[f'snap2_{ihalo_snap2_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap2_inflow_output],compression=compression)
                    
#                     #Rest of fields: snap 3
#                     ihalo_snap3_inflow_outputs=["Particle_InFOF","Particle_Bound","Particle_InHost",'rabs_com']

#                     for ihalo_snap3_inflow_output in ihalo_snap3_inflow_outputs:
#                         ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap3_{ihalo_snap3_inflow_output}',data=ihalo_combined_inflow_candidate_data[f'snap3_{ihalo_snap3_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap3_inflow_output],compression=compression)
            
#                     ### SUMMED OUTPUTS
#                     ihalo_itype_inflow_masses=ihalo_combined_inflow_candidate_data['snap1_Mass'][ihalo_itype_mask]
 
#                     # masks
#                     ihalo_itype_inflow_vmax_masks={'vmax_fac_'+str(ivmax_fac+1):-ihalo_combined_inflow_candidate_data[f'snap1_vrad_com'][ihalo_itype_mask]>vmax_fac*ihalo_hdf5['Metadata']['ave_vmax'][ihalo_itype_mask] for ivmax_fac,vmax_fac in enumerate(vmax_facs)}
#                     ihalo_itype_inflow_origin_masks={'field':ihalo_combined_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]=-1,'merger':ihalo_combined_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]>0}
                    
#                     # FOF
#                     ihalo_itype_inflow_FOF_full_mask=np.logical_and(ihalo_combined_inflow_candidate_data["snap2_Particle_InFOF"][ihalo_itype_mask],np.logical_not(ihalo_combined_inflow_candidate_data["snap1_Particle_InFOF"][ihalo_itype_mask]))
#                     ihalo_itype_inflow_FOF_fullsnap3_mask=ihalo_combined_inflow_candidate_data["snap3_Particle_InFOF"][ihalo_itype_mask]
#                     ihalo_itype_inflow_FOF_central_mask=np.logical_and(ihalo_combined_inflow_candidate_data["snap2_Particle_InHost"][ihalo_itype_mask],np.logical_not(ihalo_combined_inflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask]))
#                     ihalo_itype_inflow_FOF_centralsnap3_mask=np.logical_and(ihalo_combined_inflow_candidate_data["snap2_Particle_InHost"][ihalo_itype_mask],np.logical_not(ihalo_combined_inflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask]))
#                     #cgm origin masks: full - anything not in the fof but within cgm_r200_fac*r200, central: anything not in the host 6dfof but within within cgm_r200_fac*r200
#                     ihalo_itype_inflow_FOF_full_cgm_mask=np.logical_and(ihalo_combined_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]==-1,ihalo_combined_inflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]<cgm_r200_fac*ihalo_hdf5['Metadata']['ave_R_200crit'].value)
#                     ihalo_itype_inflow_FOF_central_cgm_mask=np.logical_and(np.logical_or(ihalo_combined_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]==-1,ihalo_combined_inflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask]==0),ihalo_combined_inflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]<cgm_r200_fac*ihalo_hdf5['Metadata']['ave_R_200crit'].value)

#                     # SO
#                     ihalo_itype_inflow_r200_masks={'r200_fac_'+str(ir200_fac+1):np.logical_and(ihalo_combined_inflow_candidate_data["snap2_rabs_com"][ihalo_itype_mask]<r200_fac*ihalo_metadata['ave_R_200crit'],ihalo_combined_inflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_metadata['ave_R_200crit']) for ir200_fac,r200_fac in enumerate(r200_facs)}
#                     ihalo_itype_inflow_r200_snap3_masks={'r200_fac_'+str(ir200_fac+1):ihalo_combined_inflow_candidate_data["snap3_rabs_com"][ihalo_itype_mask]<r200_fac*ihalo_metadata['ave_R_200crit'] for ir200_fac,r200_fac in enumerate(r200_facs)}
#                     #cgm origin masks: anything outside r200fac*r200, but within cgm_r200_fac*r200
#                     ihalo_itype_inflow_r200_cgm_masks={'r200_fac_'+str(ir200_fac+1):np.logical_and(ihalo_combined_inflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_hdf5['Metadata']['ave_R_200crit'].value,ihalo_combined_inflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]<cgm_r200_fac*ihalo_hdf5['Metadata']['ave_R_200crit'].value) for ir200_fac,r200_fac in enumerate(r200_facs)}

#                     #ITERATE THROUGH VMAX CUTS
#                     for ivmax_fac, vmax_fac in enumerate(vmax_facs):
#                         ivmax_key=f'vmax_fac_{ivmax_fac+1}'
#                         ivmax_mask=ihalo_itype_inflow_vmax_masks[ivmax_key]

#                         #halo scale
#                         ivmax_fof_haloscale_inflow_mask=np.logical_and(ivmax_mask,ihalo_itype_inflow_FOF_full_mask)
#                         ivmax_fof_haloscale_stableinflow_mask=np.logical_and(ivmax_fof_haloscale_inflow_mask,ihalo_itype_inflow_FOF_fullsnap3_mask)
#                         total_mass_where=np.where(ivmax_fof_haloscale_inflow_mask);stable_total_mass_where=np.where(ivmax_fof_haloscale_stableinflow_mask)
#                         field_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_inflow_mask,ihalo_itype_inflow_origin_masks['field']));stable_field_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_stableinflow_mask,ihalo_itype_inflow_origin_masks['field']))
#                         merger_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_inflow_mask,ihalo_itype_inflow_origin_masks['merger']));stable_merger_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_stableinflow_mask,ihalo_itype_inflow_origin_masks['merger']))
#                         cgm_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_inflow_mask,ihalo_itype_inflow_FOF_full_cgm_mask));stable_cgm_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_stableinflow_mask,ihalo_itype_inflow_FOF_full_cgm_mask))

#                         integrated_output_hdf5['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['All_Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[total_mass_where])
#                         integrated_output_hdf5['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['All_Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[field_mass_where])
#                         integrated_output_hdf5['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['All_Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[merger_mass_where])
#                         integrated_output_hdf5['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['All_CGM_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[cgm_mass_where])                        
#                         integrated_output_hdf5['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Stable_Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[stable_total_mass_where])
#                         integrated_output_hdf5['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Stable_Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[stable_field_mass_where])
#                         integrated_output_hdf5['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Stable_Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[stable_merger_mass_where])
#                         integrated_output_hdf5['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Stable_CGM_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[stable_cgm_mass_where])
                        
#                         #subhalo scale
#                         if ihalo_numsubstruct>0 or ihalo_hostHaloID>0:

#                             ivmax_fof_subhaloscale_inflow_mask=np.logical_and(ivmax_mask,ihalo_itype_inflow_FOF_central_mask)
#                             ivmax_fof_subhaloscale_stableinflow_mask=np.logical_and(ivmax_fof_haloscale_inflow_mask,ihalo_itype_inflow_FOF_centralsnap3_mask)

#                             total_mass_where=np.where(ivmax_fof_subhaloscale_inflow_mask);stable_total_mass_where=np.where(ivmax_fof_subhaloscale_stableinflow_mask)
#                             field_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_inflow_mask,ihalo_itype_inflow_origin_masks['field']));stable_field_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_stableinflow_mask,ihalo_itype_inflow_origin_masks['field']))
#                             merger_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_inflow_mask,ihalo_itype_inflow_origin_masks['merger']));stable_merger_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_stableinflow_mask,ihalo_itype_inflow_origin_masks['merger']))
#                             cgm_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_inflow_mask,ihalo_itype_inflow_FOF_central_cgm_mask));stable_cgm_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_stableinflow_mask,ihalo_itype_inflow_FOF_central_cgm_mask))

#                             integrated_output_hdf5['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['All_Total_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[total_mass_where])
#                             integrated_output_hdf5['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['All_Field_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[field_mass_where])
#                             integrated_output_hdf5['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['All_Merger_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[merger_mass_where])
#                             integrated_output_hdf5['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['All_CGM_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[cgm_mass_where])
#                             integrated_output_hdf5['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Stable_Total_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[stable_total_mass_where])
#                             integrated_output_hdf5['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Stable_Field_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[stable_field_mass_where])
#                             integrated_output_hdf5['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Stable_Merger_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[stable_merger_mass_where])
#                             integrated_output_hdf5['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Stable_CGM_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[stable_cgm_mass_where])

#                         #SO scale
#                         for ir200_fac, r200_fac in enumerate(r200_facs):
#                             ir200_key='SO-r200_fac_'+str(ir200_fac+1)
#                             ir200_inflow_mask=ihalo_itype_inflow_r200_masks[ir200_key[3:]]
#                             ir200_cgm_mask=ihalo_itype_inflow_r200_cgm_masks[ir200_key[3:]]
#                             ir200_stable_mass_mask=ihalo_itype_inflow_r200_snap3_masks[ir200_key[3:]

#                             for ivmax_fac, vmax_fac in enumerate(vmax_facs):
#                                 ivmax_key=f'vmax_fac_{ivmax_fac+1}'
#                                 ivmax_mask=ihalo_itype_inflow_vmax_masks[ivmax_key]
#                                 ivmax_ir200_inflow_mask=np.logical_and(ivmax_mask,ir200_inflow_mask)
#                                 ivmax_ir200_stableinflow_mask=np.logical_and(ivmax_ir200_inflow_mask,ir200_stable_mass_mask)

#                                 total_mass_where=np.where(ivmax_ir200_inflow_mask);stable_total_mass_where=np.where(ivmax_ir200_stableinflow_mask)
#                                 field_mass_where=np.where(np.logical_and(ivmax_ir200_inflow_mask,ihalo_itype_inflow_origin_masks['field']));stable_field_mass_where=np.where(np.logical_and(ivmax_ir200_stableinflow_mask,ihalo_itype_inflow_origin_masks['field']))
#                                 merger_mass_where=np.where(np.logical_and(ivmax_ir200_inflow_mask,ihalo_itype_inflow_origin_masks['merger']));stable_merger_mass_where=np.where(np.logical_and(ivmax_ir200_stableinflow_mask,ihalo_itype_inflow_origin_masks['merger']))
#                                 cgm_mass_where=np.where(np.logical_and(ivmax_ir200_inflow_mask,ir200_cgm_mask));stable_cgm_mass_where=np.where(np.logical_and(ivmax_ir200_stableinflow_mask,ir200_cgm_mask))

#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['All_Total_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[total_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['All_Field_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[field_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['All_Merger_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[merger_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['All_CGM_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[cgm_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['Stable_Total_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[stable_total_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['Stable_Field_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[stable_field_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['Stable_Merger_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[stable_merger_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['Stable_CGM_DeltaM_In'][iihalo]=np.nansum(ihalo_itype_inflow_masses[stable_cgm_mass_where])
                
#                 ############################## OUTFLOW ##############################
#                 ####################################################################

#                 # SELECT OUFLOW CANDIDATES AS THOSE WITHIN R200crit OR the FOF envelope at snap 1
#                 ihalo_ave_R_200crit_physical=(ihalo_hdf5['Metadata']['snap1_R_200crit'].value+ihalo_hdf5['Metadata']['snap2_R_200crit'].value)/2
#                 ihalo_cube_rminpot_snap1=np.sqrt(np.sum(np.square(ihalo_cube_particles[str(snap1)]['Coordinates']-ihalo_com_physical[str(snap1)]),axis=1))
#                 ihalo_cube_radialcut_snap1=np.where(ihalo_cube_rminpot_snap1<ihalo_ave_R_200crit_physical)
#                 ihalo_cube_outflow_candidate_data_snap1={field:ihalo_cube_particles[str(snap1)][field] for field in Part_Data_fields}
#                 ihalo_fof_outflow_candidate_data_snap1={field:ihalo_fof_particles[str(snap1)][field] for field in FOF_Part_Data_fields}
#                 ihalo_combined_outflow_candidate_IDs=np.concatenate([ihalo_fof_outflow_candidate_data_snap1['Particle_IDs'],ihalo_cube_outflow_candidate_data_snap1['ParticleIDs']])
#                 ihalo_combined_outflow_candidate_IDs_unique=np.array(np.unique(ihalo_combined_outflow_candidate_IDs),dtype=np.int64)
#                 ihalo_combined_outflow_candidate_count=len(ihalo_combined_outflow_candidate_IDs_unique)

#                 # GRAB DATA FOR EACH OUTFLOW CANDIDATE
#                 ihalo_combined_outflow_candidate_data={}

#                 # OUTPUTS FROM DATACUBE: Coordinates, Velocity, Mass, Type 
#                 ihalo_combined_outflow_candidate_cubeindices={}
#                 print(f'num outflow candidates for ihalo {ihalo_s2}: {ihalo_combined_outflow_candidate_count}')
#                 for isnap,snap in enumerate(snaps):
#                     #find the indices of the IDs in the (sorted) datacube for this halo (will return nan if not in the cube) - outputs index
#                     ihalo_combined_outflow_candidate_IDindices_temp=binary_search(ihalo_combined_outflow_candidate_IDs_unique,sorted_list=ihalo_cube_particles[str(snap)]['SortedIDs'],check_entries=True)
#                     #use the indices from the sorted IDs above to extract the cube indices (will return nan if not in the cube) - outputs index
#                     ihalo_combined_outflow_candidate_cubeindices[str(snap)]=mask_wnans(array=ihalo_cube_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_outflow_candidate_IDindices_temp)
#                     #use the cube indices to extract particle data, record which particles couldn't be found

#                     for field in ['Coordinates','Velocity','Mass','ParticleIDs','ParticleTypes']:
#                         ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_cube_particles[str(snap)][field],indices=ihalo_combined_outflow_candidate_cubeindices[str(snap)])
                    
#                     #derive other cubdata outputs
#                     ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com']=ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Coordinates']-ihalo_com_physical[str(snap)]
#                     ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_rabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com']),axis=1))
#                     ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_runit_com']=np.divide(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com'],np.column_stack([ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_rabs_com']]*3))
#                     ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_v_com']=ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Velocity']-ihalo_vcom_physical[str(snap)]
#                     ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_v_com']),axis=1))
#                     ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vrad_com']=np.sum(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_runit_com']*ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_v_com'],axis=1)
#                     ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vtan_com']=np.sqrt(np.square(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vabs_com'])-np.square(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vrad_com']))
                
#                 #include average radial velocity
#                 vel_conversion=978.462 #Mpc/Gyr to km/s
#                 ihalo_combined_outflow_candidate_data[f'ave_vrad_com']=(ihalo_combined_outflow_candidate_data[f'snap2_rabs_com']-ihalo_combined_outflow_candidate_data[f'snap1_rabs_com'])/dt*vel_conversion
            
#                 # OUTPUTS FROM FOF: InFOF, Bound
#                 ihalo_combined_outflow_candidate_fofindices={}
#                 for isnap,snap in enumerate(snaps):
#                     #find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
#                     ihalo_combined_outflow_candidate_IDindices_temp=binary_search(ihalo_combined_outflow_candidate_IDs_unique,sorted_list=ihalo_fof_particles[str(snap)]['SortedIDs'],check_entries=True)
#                     #use the indices from the sorted IDs above to extract the fof indices (will return nan if not in the fof) - outputs index
#                     ihalo_combined_outflow_candidate_fofindices[str(snap)]=mask_wnans(array=ihalo_fof_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_outflow_candidate_IDindices_temp)
                    
#                     #use the fof indices to extract particle data, record which particles couldn't be found
#                     ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']=np.isfinite(ihalo_combined_outflow_candidate_IDindices_temp)
#                     ihalo_combined_outflow_candidate_fofdata_notinfofmask=np.where(np.logical_not(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']))
#                     ihalo_combined_outflow_candidate_fofdata_notinfofmask_count=len(ihalo_combined_outflow_candidate_fofdata_notinfofmask[0])
#                     for field in ['Particle_Bound','Particle_InHost']:
#                         ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_fof_particles[str(snap)][field],indices=ihalo_combined_outflow_candidate_fofindices[str(snap)])
#                         ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_{field}'][ihalo_combined_outflow_candidate_fofdata_notinfofmask]=np.zeros(ihalo_combined_outflow_candidate_fofdata_notinfofmask_count)

#                 # OUTPUTS FROM HISTORIES: Processed, Structure (by particle type) at snap 2
#                 ihalo_combined_outflow_candidate_partindices={}
#                 ihalo_combined_outflow_candidate_data['snap2_Structure']=np.zeros(ihalo_combined_outflow_candidate_count)
                
#                 for itype in PartTypes:
#                     ihalo_combined_outflow_candidate_typemask_snap2=np.where(ihalo_combined_outflow_candidate_data['snap2_ParticleTypes']==itype)
#                     ihalo_combined_outflow_candidate_IDs_unique_itype=ihalo_combined_outflow_candidate_IDs_unique[ihalo_combined_outflow_candidate_typemask_snap2]
#                     #find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
#                     ihalo_combined_outflow_candidate_IDindices_temp=binary_search(ihalo_combined_outflow_candidate_IDs_unique_itype,sorted_list=Part_Histories_IDs_snap2[str(itype)],check_entries=False)
#                     #use the indices from the sorted IDs above to extract the partdata indices (will return nan if not in the fof) - outputs index
#                     ihalo_combined_outflow_candidate_partindices[str(itype)]=Part_Histories_Index_snap2[str(itype)][(ihalo_combined_outflow_candidate_IDindices_temp,)]
#                     #extract host structure and processing
#                     ihalo_combined_outflow_candidate_data['snap2_Structure'][ihalo_combined_outflow_candidate_typemask_snap2]=Part_Histories_HostStructure_snap2[str(itype)][ihalo_combined_outflow_candidate_partindices[str(itype)]]

#                 # SAVE TO FILE 

#                 for itype in PartTypes:
#                     itype_key=f'PartType{itype}'

#                     ### PARTICLE OUTPUTS
#                     # Candidate particle types and masses taken at snap 2 (after "leaving" halo)
#                     ihalo_itype_mask=np.where(ihalo_combined_outflow_candidate_data["snap2_ParticleTypes"]==itype)
#                     ihalo_hdf5['Outflow'][itype_key].create_dataset('ParticleIDs',data=ihalo_combined_outflow_candidate_IDs_unique[ihalo_itype_mask],dtype=output_fields_dtype["ParticleIDs"],compression=compression)
#                     ihalo_hdf5['Outflow'][itype_key].create_dataset('Mass',data=ihalo_combined_outflow_candidate_data['snap2_Mass'][ihalo_itype_mask],dtype=output_fields_dtype["Mass"],compression=compression)
#                     ihalo_hdf5['Outflow'][itype_key].create_dataset('ave_vrad_com',data=ihalo_combined_outflow_candidate_data[f'ave_vrad_com'][ihalo_itype_mask],dtype=output_fields_dtype["ave_vrad_com"],compression=compression)

#                     #Rest of fields: snap 1
#                     ihalo_snap1_outflow_outputs=["r_com","rabs_com","vrad_com","vtan_com","Particle_InFOF","Particle_Bound","Particle_InHost"]
#                     for ihalo_snap1_outflow_output in ihalo_snap1_outflow_outputs:
#                         ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap1_{ihalo_snap1_outflow_output}',data=ihalo_combined_outflow_candidate_data[f'snap1_{ihalo_snap1_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap1_outflow_output],compression=compression)
                    
#                     #Rest of fields: snap 2
#                     ihalo_snap2_outflow_outputs=["r_com","rabs_com","vrad_com","vtan_com","Particle_InFOF","Particle_Bound","Particle_InHost","Particle_InCube","Structure"]

#                     for ihalo_snap2_outflow_output in ihalo_snap2_outflow_outputs:
#                         ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap2_{ihalo_snap2_outflow_output}',data=ihalo_combined_outflow_candidate_data[f'snap2_{ihalo_snap2_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap2_outflow_output],compression=compression)
                    
#                     #Rest of fields: snap 3
#                     ihalo_snap3_outflow_outputs=["Particle_InFOF","Particle_Bound","Particle_InHost","Particle_InCube","rabs_com"]

#                     for ihalo_snap3_outflow_output in ihalo_snap3_outflow_outputs:
#                         ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap3_{ihalo_snap3_outflow_output}',data=ihalo_combined_outflow_candidate_data[f'snap3_{ihalo_snap3_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap3_outflow_output],compression=compression)
                    
#                     ### SUMMED OUTPUTS
#                     ihalo_itype_outflow_masses=ihalo_combined_outflow_candidate_data['snap1_Mass'][ihalo_itype_mask]
 
#                     # masks
#                     ihalo_itype_outflow_vmax_masks={'vmax_fac_'+str(ivmax_fac+1):ihalo_combined_outflow_candidate_data[f'snap1_vrad_com'][ihalo_itype_mask]>vmax_fac*ihalo_hdf5['Metadata']['ave_vmax'][ihalo_itype_mask] for ivmax_fac,vmax_fac in enumerate(vmax_facs)}
#                     ihalo_itype_outflow_destination_masks={'field':ihalo_combined_outflow_candidate_data["snap2_Structure"][ihalo_itype_mask]=-1,'merger':ihalo_combined_outflow_candidate_data["snap2_Structure"][ihalo_itype_mask]>0}
                    
#                     # FOF
#                     ihalo_itype_outflow_FOF_full_mask=np.logical_and(ihalo_combined_outflow_candidate_data["snap1_Particle_InFOF"][ihalo_itype_mask],np.logical_not(ihalo_combined_outflow_candidate_data["snap2_Particle_InFOF"][ihalo_itype_mask]))
#                     ihalo_itype_outflow_FOF_central_mask=np.logical_and(ihalo_combined_outflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask],np.logical_not(ihalo_combined_outflow_candidate_data["snap2_Particle_InHost"][ihalo_itype_mask]))
#                     ihalo_itype_stableoutflow_FOF_full_mask=np.logical_and(ihalo_itype_outflow_FOF_full_mask,np.logical_not(ihalo_combined_outflow_candidate_data["snap3_Particle_InFOF"]))
#                     ihalo_itype_stableoutflow_FOF_central_mask=np.logical_and(ihalo_itype_outflow_FOF_central_mask,np.logical_not(ihalo_combined_outflow_candidate_data["snap3_Particle_InHost"]))

#                     #cgm destination masks: full - anything not in the fof but within cgm_r200_fac*r200, central: anything not in the host 6dfof but within within cgm_r200_fac*r200
#                     ihalo_itype_outflow_FOF_full_cgm_mask=np.logical_and(ihalo_combined_outflow_candidate_data["snap2_Structure"][ihalo_itype_mask]==-1,ihalo_combined_outflow_candidate_data["snap2_rabs_com"][ihalo_itype_mask]<cgm_r200_fac*ihalo_hdf5['Metadata']['ave_R_200crit'].value)
#                     ihalo_itype_outflow_FOF_central_cgm_mask=np.logical_and(np.logical_or(ihalo_combined_outflow_candidate_data["snap1_Structure"][ihalo_itype_mask]==-1,ihalo_combined_outflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask]==0),ihalo_combined_outflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]<cgm_r200_fac*ihalo_hdf5['Metadata']['ave_R_200crit'].value)

#                     # SO
#                     ihalo_itype_outflow_r200_masks={'r200_fac_'+str(ir200_fac+1):np.logical_and(ihalo_combined_outflow_candidate_data["snap2_rabs_com"][ihalo_itype_mask]<r200_fac*ihalo_metadata['ave_R_200crit'],ihalo_combined_outflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_metadata['ave_R_200crit']) for ir200_fac,r200_fac in enumerate(r200_facs)}
#                     ihalo_itype_stableoutflow_r200_masks={'r200_fac_'+str(ir200_fac+1):np.logical_and(ihalo_itype_outflow_r200_masks['r200_fac_'+str(ir200_fac+1)],ihalo_combined_outflow_candidate_data["snap3_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_metadata['ave_R_200crit']) for ir200_fac,r200_fac in enumerate(r200_facs)}
#                     #cgm origin masks: anything outside r200fac*r200, but within cgm_r200_fac*r200
#                     ihalo_itype_outflow_r200_cgm_masks={'r200_fac_'+str(ir200_fac+1):np.logical_and(ihalo_combined_outflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_hdf5['Metadata']['ave_R_200crit'].value,ihalo_combined_outflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]<cgm_r200_fac*ihalo_hdf5['Metadata']['ave_R_200crit'].value) for ir200_fac,r200_fac in enumerate(r200_facs)}

#                     #ITERATE THROUGH VMAX CUTS
#                     for ivmax_fac, vmax_fac in enumerate(vmax_facs):
#                         ivmax_key=f'vmax_fac_{ivmax_fac+1}'
#                         ivmax_mask=ihalo_itype_outflow_vmax_masks[ivmax_key]

#                         #halo scale
#                         ivmax_fof_haloscale_outflowmask=np.logical_and(ivmax_mask,ihalo_itype_outflow_FOF_full_mask)
#                         ivmax_fof_haloscale_stableoutflow_mask=np.logical_and(ivmax_mask,ihalo_itype_stableoutflow_FOF_full_mask)
                        
#                         total_mass_where=np.where(ivmax_fof_haloscale_outflow_mask);stable_total_mass_where=np.where(ivmax_fof_haloscale_stableoutflow_mask)
#                         field_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_outflow_mask,ihalo_itype_outflow_origin_masks['field']));stable_field_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_stableoutflow_mask,ihalo_itype_outflow_origin_masks['field']))
#                         merger_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_outflow_mask,ihalo_itype_outflow_origin_masks['merger']));stable_merger_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_stableoutflow_mask,ihalo_itype_outflow_origin_masks['merger']))
#                         cgm_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_outflow_mask,ihalo_itype_outflow_FOF_full_cgm_mask));stable_cgm_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_stableoutflow_mask,ihalo_itype_outflow_FOF_full_cgm_mask))

#                         integrated_output_hdf5['Outflow'][f'FOF-haloscale'][ivmax_key][itype_key]['All_Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[total_mass_where])
#                         integrated_output_hdf5['Outflow'][f'FOF-haloscale'][ivmax_key][itype_key]['All_Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[field_mass_where])
#                         integrated_output_hdf5['Outflow'][f'FOF-haloscale'][ivmax_key][itype_key]['All_Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[merger_mass_where])
#                         integrated_output_hdf5['Outflow'][f'FOF-haloscale'][ivmax_key][itype_key]['All_CGM_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[cgm_mass_where])
#                         integrated_output_hdf5['Outflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Stable_Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[stable_total_mass_where])
#                         integrated_output_hdf5['Outflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Stable_Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[stable_field_mass_where])
#                         integrated_output_hdf5['Outflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Stable_Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[stable_merger_mass_where])
#                         integrated_output_hdf5['Outflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Stable_CGM_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[stable_cgm_mass_where])
                        
#                         #subhalo scale
#                         if ihalo_numsubstruct>0 or ihalo_hostHaloID>0:

#                             ivmax_fof_subhaloscale_inflow_mask=np.logical_and(ivmax_mask,ihalo_itype_inflow_FOF_central_mask)
#                             ivmax_fof_subhaloscale_stableinflow_mask=np.logical_and(ivmax_mask,ihalo_itype_stableinflow_FOF_central_mask)
                            
#                             total_mass_where=np.where(ivmax_fof_subhaloscale_outflow_mask);stable_total_mass_where=np.where(ivmax_fof_subhaloscale_stableoutflow_mask)
#                             field_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_outflow_mask,ihalo_itype_outflow_origin_masks['field']));stable_field_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_stableoutflow_mask,ihalo_itype_outflow_origin_masks['field']))
#                             merger_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_outflow_mask,ihalo_itype_outflow_origin_masks['merger']));stable_merger_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_stableoutflow_mask,ihalo_itype_outflow_origin_masks['merger']))
#                             cgm_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_outflow_mask,ihalo_itype_outflow_FOF_full_cgm_mask));stable_cgm_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_stableoutflow_mask,ihalo_itype_outflow_FOF_full_cgm_mask))
                            
#                             integrated_output_hdf5['Outflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['All_Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[total_mass_where])
#                             integrated_output_hdf5['Outflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['All_Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[field_mass_where])
#                             integrated_output_hdf5['Outflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['All_Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[merger_mass_where])
#                             integrated_output_hdf5['Outflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['All_CGM_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[cgm_mass_where])
#                             integrated_output_hdf5['Outflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Stable_Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[stable_total_mass_where])
#                             integrated_output_hdf5['Outflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Stable_Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[stable_field_mass_where])
#                             integrated_output_hdf5['Outflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Stable_Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[stable_merger_mass_where])
#                             integrated_output_hdf5['Outflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Stable_CGM_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_outflow_masses[stable_cgm_mass_where])
                        
#                         #SO scale
#                         for ir200_fac, r200_fac in enumerate(r200_facs):
#                             ir200_key='SO-r200_fac_'+str(ir200_fac+1)
#                             ir200_outflow_mask=ihalo_itype_outflow_r200_masks[ir200_key[3:]]
#                             ir200_stableoutflow_mask=ihalo_itype_outflow_r200_masks[ir200_key[3:]]
#                             ir200_cgm_mask=ihalo_itype_inflow_r200_cgm_masks[ir200_key[3:]]

#                             for ivmax_fac, vmax_fac in enumerate(vmax_facs):
#                                 ivmax_key=f'vmax_fac_{ivmax_fac+1}'
#                                 ivmax_mask=ihalo_itype_inflow_vmax_masks[ivmax_key]
#                                 ivmax_ir200_inflow_mask=np.logical_and(ivmax_mask,ir200_inflow_mask)
                                
#                                 total_mass_where=np.where(ivmax_ir200_inflow_mask)
#                                 field_mass_where=np.where(np.logical_and(ivmax_ir200_inflow_mask,ihalo_itype_inflow_origin_masks['field']))
#                                 merger_mass_where=np.where(np.logical_and(ivmax_ir200_inflow_mask,ihalo_itype_inflow_origin_masks['merger']))
#                                 cgm_mass_where=np.where(np.logical_and(ivmax_ir200_inflow_mask,ir200_cgm_mask))

#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[total_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[field_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[merger_mass_where])
#                                 integrated_output_hdf5['Inflow'][ir200_key][ivmax_key][itype_key]['CGM_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[cgm_mass_where])

#             else:
#                 print(f'Skipping ihalo {ihalo_s2}')
#                 with open(fname_log,"a") as progress_file:
#                     progress_file.write(f"Skipping ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
#                     progress_file.write(f" \n")
#                 progress_file.close()
#             t2_halo=time.time()
#             print(f"Took {t2_halo-t1_halo:.2f} sec on ihalo {ihalo_s2}")
#             print()

#             with open(fname_log,"a") as progress_file:
#                 progress_file.write(f"Done with ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
#                 progress_file.write(f"[Took {t2_halo-t1_halo:.2f} sec]\n")
#                 progress_file.write(f" \n")
#             progress_file.close()

#         except:
#             print(f'Skipping ihalo {ihalo_s2}')
#             with open(fname_log,"a") as progress_file:
#                 progress_file.write(f"Skipping ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
#                 progress_file.write(f" \n")
#             progress_file.close()
#             continue
            
#     #Finished with output file
#     output_hdf5.close()
#     return None

# ########################### SUM DETAILED ACCRETION DATA ###########################

# def postprocess_accretion_data(base_halo_data,path):

#     """ sum from gen_accretion_data_detailed_serial """

#     if not path.endswith('/'): path=path+'/'
#     snap=path.split('snap_')[-1]
#     if snap.endswith('/'): snap=snap[:-1]
#     snap=int(snap)

#     allfiles=os.listdir(path)
#     accfile_paths=sorted([path+fname for fname in allfiles if ('AccretionData' in fname and 'summed' not in fname)])[::-1][:2]
#     nfiles=len(accfile_paths)
    
#     outfile_path=accfile_paths[0][:-10]+'_summed_2.hdf5'
#     if os.path.exists(outfile_path):
#         print(f'Removing {outfile_path}')
#         os.remove(outfile_path)

#     outlog_path=path+'progress.log'
#     if os.path.exists(outlog_path):
#         print(f'Removing {outlog_path}')
#         os.remove(outlog_path)

#     time.sleep(1)
#     #Initialise output file with  header
#     outfile=h5py.File(outfile_path,'w')
#     outfile.create_group('Header')
#     exfile=h5py.File(accfile_paths[0],'r+')
#     exfile_keys=list(exfile.keys());exfile_halokeys=[exfile_key for exfile_key in exfile_keys if 'ihalo' in exfile_key]
#     parttype_keys=list(exfile[exfile_halokeys[0]]['Inflow'].keys())
#     parttypes=[int(parttype_key.split('PartType')[-1]) for parttype_key in parttype_keys]
#     header_keys=list(exfile['Header'].attrs)
#     for header_key in header_keys:
#         outfile['Header'].attrs.create(header_key,exfile['Header'].attrs[header_key])

#     #params
#     r200_facs=[0.125,0.25,0.5,0.75,1,1.25,1.5,2]
#     vmax_facs=[-1,0,0.125,0.25,0.375,0.5,0.75,1]
#     cgm_r200_fac=1.5

#     #Initialise output file output groups and datasets
#     num_total_halos=base_halo_data[snap]['Count']
#     num_total_halos_infiles=len(flatten([list(h5py.File(accfile_path,'r').keys()) for accfile_path in accfile_paths]))-len(accfile_paths)

#     FOF_haloscale_inflow_output_fields=['Total_DeltaM_In','Field_DeltaM_In','CGM_DeltaM_In','Merger_DeltaM_In']#for all halos
#     FOF_subhaloscale_inflow_output_fields=['Total_DeltaM_In','Field_DeltaM_In','CGM_DeltaM_In','Merger_DeltaM_In']#just for substructure (host or satellites)
#     SO_inflow_outputs_fields=['Total_DeltaM_In','Field_DeltaM_In','Merger_DeltaM_In']#for all radial bins

#     outfile.create_group('Inflow')
#     outfile['Inflow'].create_group('FOF-haloscale');outfile['Inflow'].create_group('FOF-subhaloscale')
#     for ir200_fac,r200_fac in enumerate(r200_facs):
#         outfile['Inflow'].create_group(f'SO-r200_fac_{ir200_fac+1}')
#         outfile['Inflow'][f'SO-r200_fac_{ir200_fac+1}'].attrs.create('r200_fac',data=r200_fac)

#     for ivmax_fac, vmax_fac in enumerate(vmax_facs):
#         ivmax_key=f'vmax_fac_{ivmax_fac+1}'
#         outfile['Inflow']['FOF-haloscale'].create_group(ivmax_key);outfile['Inflow']['FOF-haloscale'][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
#         for itype in parttypes:
#             itype_key=f'PartType{itype}'
#             outfile['Inflow'][f'FOF-haloscale'][ivmax_key].create_group(itype_key)
#             for dataset in FOF_haloscale_inflow_output_fields:
#                 outfile['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key].create_dataset(dataset,data=np.zeros(num_total_halos)+np.nan,dtype=np.float32)
#         outfile['Inflow']['FOF-subhaloscale'].create_group(ivmax_key);outfile['Inflow']['FOF-subhaloscale'][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
#         for itype in parttypes:
#             itype_key=f'PartType{itype}'
#             outfile['Inflow'][f'FOF-subhaloscale'][ivmax_key].create_group(itype_key)
#             for dataset in FOF_subhaloscale_inflow_output_fields:
#                 outfile['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key].create_dataset(dataset,data=np.zeros(num_total_halos)+np.nan,dtype=np.float32)
#         for ir200_fac,r200_fac in enumerate(r200_facs):
#             ir200_key=f'SO-r200_fac_{ir200_fac+1}'
#             outfile['Inflow'][ir200_key].create_group(ivmax_key);outfile['Inflow'][ir200_key][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
#             for itype in parttypes:
#                 itype_key=f'PartType{itype}'
#                 outfile['Inflow'][ir200_key][ivmax_key].create_group(itype_key)
#                 for dataset in SO_inflow_outputs_fields:
#                     outfile['Inflow'][ir200_key][ivmax_key][itype_key].create_dataset(dataset,data=np.zeros(num_total_halos)+np.nan,dtype=np.float32)

#     iihalo=0
#     for accfile_path in accfile_paths:
#         print('Loading from ',accfile_path)
#         accfile=h5py.File(accfile_path,'r')
#         accfile_allkeys=list(accfile.keys())
#         accfile_halokeys=[key for key in accfile_allkeys if 'halo' in key]

#         for accfile_halokey in accfile_halokeys:
#             iihalo=iihalo+1
#             # Print progress to output file

#             with open(outlog_path,"a") as progress_file:
#                 progress_file.write(f' \n')
#                 progress_file.write(f'Starting with iihalo {iihalo} / {num_total_halos_infiles} ({iihalo/num_total_halos_infiles*100:.1f}% done)... \n')
#             progress_file.close()

#             if iihalo%10==0:
#                 print(f'{iihalo/num_total_halos_infiles*100:.1f} % done summing accretion data')

#             ihalo=int(accfile_halokey.split('ihalo_')[-1])
#             ihalo_metadata={field:accfile[accfile_halokey]['Metadata'][field].value for field in list(accfile[accfile_halokey]['Metadata'].keys())}
#             ihalo_numsubstruct=base_halo_data[snap]['numSubStruct'][ihalo]
#             ihalo_hostHaloID=base_halo_data[snap]['hostHaloID'][ihalo]

#             if len(list(ihalo_metadata.keys()))>0:
#                 ihalo_metadata['ave_R_200crit']=(ihalo_metadata['snap1_R_200crit']+ihalo_metadata['snap2_R_200crit'])*0.5
#                 ihalo_metadata['ave_vmax']=(ihalo_metadata['snap1_vmax']+ihalo_metadata['snap2_vmax'])*0.5
                
#                 for itype,itype_key in zip(parttypes,parttype_keys):

#                     ######## INFLOW ########
#                     ihalo_itype_inflow_group=accfile[accfile_halokey]["Inflow"][itype_key]
#                     try:
#                         ihalo_itype_inflow_masses=ihalo_itype_inflow_group["Mass"].value
#                     except:
#                         print(f'Skipping halo {ihalo} part type {itype}')
#                         continue

#                     try:
#                         if ihalo_itype_inflow_masses[0]<10**6:
#                             ihalo_itype_inflow_masses=ihalo_itype_inflow_masses*10**10
#                     except:
#                         pass
                    
#                     #masks
#                     ihalo_itype_inflow_vmax_masks={'vmax_fac_'+str(ivmax_fac+1):-ihalo_itype_inflow_group["snap1_vrad_com"].value>vmax_fac*ihalo_metadata['ave_vmax'] for ivmax_fac,vmax_fac in enumerate(vmax_facs)}
                    
#                     ihalo_itype_inflow_origin_masks={'field':ihalo_itype_inflow_group["snap1_Structure"].value==-1,'merger':ihalo_itype_inflow_group["snap1_Structure"].value>0,
#                                                      'cgm':np.logical_and(ihalo_itype_inflow_group["snap1_Structure"].value==-1,ihalo_itype_inflow_group["snap1_rabs_com"].value<cgm_r200_fac*ihalo_metadata['ave_R_200crit'])}
                    
#                     # FOF
#                     ihalo_itype_inflow_FOF_mask=np.logical_and(ihalo_itype_inflow_group["snap2_Particle_InFOF"].value,np.logical_not(ihalo_itype_inflow_group["snap1_Particle_InFOF"].value))
#                     ihalo_itype_inflow_FOF_central_mask=np.logical_and(ihalo_itype_inflow_group["snap2_Particle_InHost"].value,np.logical_not(ihalo_itype_inflow_group["snap1_Particle_InHost"].value))
                     
#                     # SO
#                     ihalo_itype_inflow_r200_masks={'r200_fac_'+str(ir200_fac+1):np.logical_and(ihalo_itype_inflow_group["snap2_rabs_com"].value<r200_fac*ihalo_metadata['ave_R_200crit'],ihalo_itype_inflow_group["snap1_rabs_com"].value>r200_fac*ihalo_metadata['ave_R_200crit']) for ir200_fac,r200_fac in enumerate(r200_facs)}
                    
#                     ### SAVE TO OUTPUT dictionary
#                     # FOFs
#                     for ivmax_fac, vmax_fac in enumerate(vmax_facs):
#                         ivmax_key=f'vmax_fac_{ivmax_fac+1}'
#                         ivmax_mask=ihalo_itype_inflow_vmax_masks[ivmax_key]

#                         #halo scale
#                         ivmax_fof_haloscale_inflow_mask=np.logical_and(ivmax_mask,ihalo_itype_inflow_FOF_mask)
#                         total_mass_where=np.where(ivmax_fof_haloscale_inflow_mask)
#                         field_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_inflow_mask,ihalo_itype_inflow_origin_masks['field']))
#                         cgm_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_inflow_mask,ihalo_itype_inflow_origin_masks['cgm']))
#                         merger_mass_where=np.where(np.logical_and(ivmax_fof_haloscale_inflow_mask,ihalo_itype_inflow_origin_masks['merger']))

#                         outfile['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[total_mass_where])
#                         outfile['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[field_mass_where])
#                         outfile['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['CGM_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[cgm_mass_where])
#                         outfile['Inflow'][f'FOF-haloscale'][ivmax_key][itype_key]['Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[merger_mass_where])

#                         if ihalo_numsubstruct>0 or ihalo_hostHaloID>0:


#                             #subhalo scale
#                             ivmax_fof_subhaloscale_inflow_mask=np.logical_and(ivmax_mask,ihalo_itype_inflow_FOF_central_mask)
#                             total_mass_where=np.where(ivmax_fof_subhaloscale_inflow_mask)
#                             field_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_inflow_mask,ihalo_itype_inflow_origin_masks['field']))
#                             cgm_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_inflow_mask,ihalo_itype_inflow_origin_masks['cgm']))
#                             merger_mass_where=np.where(np.logical_and(ivmax_fof_subhaloscale_inflow_mask,ihalo_itype_inflow_origin_masks['merger']))

#                             outfile['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[total_mass_where])
#                             outfile['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[field_mass_where])
#                             outfile['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['CGM_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[cgm_mass_where])
#                             outfile['Inflow'][f'FOF-subhaloscale'][ivmax_key][itype_key]['Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[merger_mass_where])

#                     #SOs
#                     for ir200_fac, r200_fac in enumerate(r200_facs):
#                         ir200_key='SO-r200_fac_'+str(ir200_fac+1)
#                         ir200_inflow_mask=ihalo_itype_inflow_r200_masks[ir200_key[3:]]
                        
#                         for ivmax_fac, vmax_fac in enumerate(vmax_facs):
#                             ivmax_key=f'vmax_fac_{ivmax_fac+1}'
#                             ivmax_mask=ihalo_itype_inflow_vmax_masks[ivmax_key]
#                             ivmax_ir200_inflow_mask=np.logical_and(ivmax_mask,ir200_inflow_mask)
                            
#                             total_mass_where=np.where(ivmax_ir200_inflow_mask)
#                             field_mass_where=np.where(np.logical_and(ivmax_ir200_inflow_mask,ihalo_itype_inflow_origin_masks['field']))
#                             cgm_mass_where=np.where(np.logical_and(ivmax_ir200_inflow_mask,ihalo_itype_inflow_origin_masks['cgm']))
#                             merger_mass_where=np.where(np.logical_and(ivmax_ir200_inflow_mask,ihalo_itype_inflow_origin_masks['merger']))

#                             outfile['Inflow'][ir200_key][ivmax_key][itype_key]['Total_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[total_mass_where])
#                             outfile['Inflow'][ir200_key][ivmax_key][itype_key]['Field_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[field_mass_where])
#                             outfile['Inflow'][ir200_key][ivmax_key][itype_key]['Merger_DeltaM_In'][ihalo]=np.nansum(ihalo_itype_inflow_masses[merger_mass_where])


#             else:
#                 print(f'Skipping halo {ihalo}')
#                 pass








