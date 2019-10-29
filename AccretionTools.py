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

from pandas import DataFrame as df

# VELOCIraptor python tools etc
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
    PartHistory_xxx-outname.hdf5 : hdf5 file with datasets

        '/PartTypeX/PartID' - SORTED particle IDs from simulation.
        '/PartTypeX/PartIndex' - Corresponding indices of particles. 
        '/PartTypeX/HostStructure' - Host structure (from STF) of particles. (-1: no host structure)
    
    Will save to file at: part_histories/PartTypeX_History_xxx-outname.dat

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

    {integrated_output}.hdf5

    And datasets:
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

        Inflow: 

        '/ihalo_xxxxxx/PartTypeX/Inflow/ParticleIDs': ParticleID (in particle data for given type) of all accreted particles.
        '/ihalo_xxxxxx/PartTypeX/Inflow/Masses': Mass (in particle data for given type) of all accreted particles.
        '/ihalo_xxxxxx/PartTypeX/Inflow/Fidelity': Whether this particle stayed in the halo at the given fidelity gap. 
        '/ihalo_xxxxxx/PartTypeX/Inflow/PreviousHost': Which structure was this particle host to (-1: not in any fof object, 0 if CGM (subhalos only), >0: ID of previous halo).
        '/ihalo_xxxxxx/PartTypeX/Inflow/Processed_L1': How many snaps has this particle been part of any structure in the past. 
        '/ihalo_xxxxxx/PartTypeX/Inflow/Processed_L2': How many snaps has this particle been part of halos with no substructure in the past. 

        and 

        '/ihalo_xxxxxx/PartTypeX/Outflow/ParticleIDs': ParticleID (in particle data for given type) of all outflow particles.
        '/ihalo_xxxxxx/PartTypeX/Outflow/Masses': Mass (in particle data for given type) of all outflow particles.
        '/ihalo_xxxxxx/PartTypeX/Outflow/Destination_S2': Where did the particle end up after outflow at snap 2 (-1: not in halo or group, 0: CGM (only subhalos), >1: ID of destination subhalo in same field halo)
        '/ihalo_xxxxxx/PartTypeX/Outflow/Destination_S3': Where did the particle end up after outflow at snap 3 (-1: not in halo or group, 0: CGM (only subhalos), 1: reaccreted, >1: ID of destination subhalo in same field halo)

        Where there will be n_halos ihalo datasets. 

        '/Header': Contains attributes: "t1","t2","dt","z_ave","lt_ave"

    
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
            num_processes=halo_index_list["np"]
            test=halo_index_list["test"]
        except:
            print('Not parsed a valud halo index list. Exiting.')
            return None


    acc_log_dir=f"job_logs/acc_logs/"
    if not os.path.exists(acc_log_dir):
        os.mkdir(acc_log_dir)
    if test:
        run_log_dir=f"job_logs/acc_logs/pre{pre_depth}_post{post_depth}_np{num_processes}_test/"
    else:
        run_log_dir=f"job_logs/acc_logs/pre{pre_depth}_post{post_depth}_np{num_processes}/"

    if not os.path.exists(run_log_dir):
        os.mkdir(run_log_dir)
    run_snap_log_dir=run_log_dir+f'snap_{str(snap).zfill(3)}/'
    if not os.path.exists(run_snap_log_dir):
        os.mkdir(run_snap_log_dir)
    if test:
        fname_log=run_snap_log_dir+f"progress_p{iprocess}_n{str(len(halo_index_list_snap2)).zfill(6)}_test.log"
    else:
        fname_log=run_snap_log_dir+f"progress_p{iprocess}_n{str(len(halo_index_list_snap2)).zfill(6)}.log"

    if os.path.exists(fname_log):
        os.remove(fname_log)

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
        calc_dir=f'acc_data/pre{pre_depth}_post{post_depth}_np{num_processes}_test/'
    else:
        calc_dir=f'acc_data/pre{pre_depth}_post{post_depth}_np{num_processes}/'

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
    outfile_name=calc_snap_dir+'FOF_AccretionData_pre'+str(pre_depth)+'_post'+str(post_depth)+'_snap'+str(snap).zfill(3)+'_p'+str(iprocess).zfill(3)+f'_n{str(len(halo_index_list_snap2)).zfill(6)}.hdf5'
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

    # Now find which simulation type we're dealing with
    part_filetype=base_halo_data[snap]["Part_FileType"]

    # Standard particle type names from simulation
    PartNames=['gas','DM','','','star','BH']
    
    # Assign the particle types we're considering 
    if part_filetype=='EAGLE':
        PartTypes=[0,1,4] #Gas, DM, Stars
        constant_mass={str(0):False,str(1):True,str(4):False}
    else:
        PartTypes=[0,1] #Gas, DM
        constant_mass={str(0):True,str(1):True}

    # Read in particle masses
    h_val=base_halo_data[snap2]['SimulationInfo']['h_val']
    if part_filetype=='EAGLE':# if an EAGLE snapshot
        print('Reading in EAGLE snapshot data ...')
        varying_mass=True
        EAGLE_boxsize=base_halo_data[snap1]['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap_1=read_eagle.EagleSnapshot(base_halo_data[snap1]['Part_FilePath'])
        EAGLE_Snap_1.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
        Part_Data_Masses_Snap1=dict();Part_Data_IDs_Snap1=dict()
        for itype in PartTypes:
            if not itype==1:#everything except DM
                try:
                    Part_Data_Masses_Snap1[str(itype)]=EAGLE_Snap_1.read_dataset(itype,"Mass")*10**10/h_val #CHECK THIS√
                except:
                    print('No particles of this type were found.')
                    Part_Data_Masses_Snap1[str(itype)]=[]
            else:#for DM, find particle data file and save 
                hdf5file=h5py.File(base_halo_data[snap1]['Part_FilePath'])#hdf5 file
                Part_Data_Masses_Snap1[str(itype)]=hdf5file['Header'].attrs['MassTable'][1]*10**10/h_val #CHECK THIS√
        print('Done reading in EAGLE snapshot data')
    else:#assuming constant mass
        constant_mass=True
        Part_Data_Masses_Snap1=dict()
        hdf5file=h5py.File(base_halo_data[snap1]['Part_FilePath'])
        MassTable=hdf5file["Header"].attrs["MassTable"]
        Part_Data_Masses_Snap1[str(1)]=MassTable[1]/h_val#CHECK THIS
        Part_Data_Masses_Snap1[str(0)]=MassTable[0]/h_val#CHECK THIS

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
    Part_Histories_HostStructure_snap2={str(parttype):Part_Histories_File_snap2["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes}

    #Load in particle lists from VR
    print('Retrieving VR halo particle lists ...')
    snap_1_halo_particles=get_particle_lists(base_halo_data[snap1],halo_index_list=halo_index_list_snap1,include_unbound=True,add_subparts_to_fofs=True)
    snap_2_halo_particles=get_particle_lists(base_halo_data[snap2],halo_index_list=halo_index_list_snap2,include_unbound=True,add_subparts_to_fofs=True)
    snap_2_halo_particles_nosubpart_all=get_particle_lists(base_halo_data[snap2],include_unbound=True,add_subparts_to_fofs=False)
    snap_2_halo_particles_withsubpart_all=get_particle_lists(base_halo_data[snap2],include_unbound=True,add_subparts_to_fofs=True)
    snap_3_halo_particles=get_particle_lists(base_halo_data[snap3],halo_index_list=halo_index_list_snap3,include_unbound=True,add_subparts_to_fofs=True)
    snap_3_halo_particles_nosubpart_all=get_particle_lists(base_halo_data[snap3],include_unbound=True,add_subparts_to_fofs=False)
    snap_3_halo_particles_withsubpart_all=get_particle_lists(base_halo_data[snap3],include_unbound=True,add_subparts_to_fofs=True)

    t2_io=time.time()
    print()
    print('*********************************************************')
    print(f'Done with I/O in {(t2_io-t1_io):.2f} sec - entering main halo loop ...')
    print('*********************************************************')

    count=0
    halos_done=0
    num_halos_thisprocess=len(halo_index_list_snap2)
    for iihalo,ihalo_s2 in enumerate(halo_index_list_snap2):# for each halo at snap 2
        
        t1_halo=time.time()
        t1_preamble=time.time()
        # Create group for this halo in output file
        halo_hdf5=output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))
        halo_in_hdf5=halo_hdf5.create_group('Inflow')
        halo_out_hdf5=halo_hdf5.create_group('Outflow')

        # Find halo progenitor and descendants 
        ihalo_s1=halo_index_list_snap1[iihalo]#find progenitor
        ihalo_s3=halo_index_list_snap3[iihalo]#find descendant
        try:
            idhalo_s1=base_halo_data[snap1]['ID'][ihalo_s1]
            idhalo_s3=base_halo_data[snap3]['ID'][ihalo_s3]
        except:
            idhalo_s1=np.nan
            idhalo_s3=np.nan

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
            
            # Returns mask for s2 of particles which are in s2 but not in s1
            print(f"Finding new particles to ihalo {ihalo_s2} ...")
            t1_new=time.time()
            new_particle_IDs_mask_snap2=np.isin(snap2_IDs_temp,snap1_IDs_temp,assume_unique=True,invert=True)
            t2_new=time.time()
            print(f"n(in) = {np.sum(new_particle_IDs_mask_snap2)}")

            # # Returns mask for s1 of particles which are in s1 but not in s2
            print(f"Finding particles which left ihalo {ihalo_s2} ...")
            t1_out=time.time()
            out_particle_IDs_mask_snap1=np.isin(snap1_IDs_temp,snap2_IDs_temp,assume_unique=True,invert=True)
            t2_out=time.time()
            print(f"n(out) = {np.sum(out_particle_IDs_mask_snap1)}")

            t1_itype=[];t2_itype=[]
            t1_typing=[];t2_typing=[]
            t1_indexing_in=[];t2_indexing_in=[]
            t1_indexing_out=[];t2_indexing_out=[]
            t1_inflow=[];t2_inflow=[]
            t1_outflow=[];t2_outflow=[]
            t1_print=[];t2_print=[]
            t1_save=[];t2_save=[]

            new_particle_IDs_itype_snap2={str(itype):[] for itype in PartTypes}
            out_particle_IDs_itype_snap1={str(itype):[] for itype in PartTypes}

            ihalo_itype_snap1_inflow_history_L1={str(itype):[] for itype in PartTypes}
            ihalo_itype_snap1_inflow_history_L2={str(itype):[] for itype in PartTypes}
            ihalo_itype_snap1_inflow_masses={str(itype):[] for itype in PartTypes}
            ihalo_itype_snap1_inflow_structure={str(itype):[] for itype in PartTypes}
            ihalo_itype_snap1_inflow_fidelity={str(itype):[] for itype in PartTypes}
            ihalo_itype_snap1_inflow_transformed={str(itype):[] for itype in PartTypes}

            ihalo_itype_snap2_outflow_transformed={str(itype):[] for itype in PartTypes}
            ihalo_itype_snap2_outflow_destination={str(itype):[] for itype in PartTypes}
            ihalo_itype_snap1_outflow_masses={str(itype):[] for itype in PartTypes}
            ihalo_itype_snap3_outflow_recycled={str(itype):[] for itype in PartTypes}

            # Now loop through each particle type and process accreted particle data 
            for iitype,itype in enumerate(PartTypes):
                t1_itype.append(time.time())#Time the full loop for this halo and particle type
                
                print('--------------------')
                print(f'{PartNames[itype]} particles')
                print('--------------------')

                # Finding particles of itype
                # print(f"Extracting new particles of type {itype} from halo list at snap 2 ...")
                t1_typing.append(time.time())
                new_particle_mask_itype=np.logical_and(new_particle_IDs_mask_snap2,snap2_Types_temp==itype)# Mask for particles in halo list at snap 2 which arrived and are of the correct type
                new_particle_IDs_itype_snap2[str(itype)]=list(np.compress(new_particle_mask_itype,snap2_IDs_temp))# Compress snap 2 list with above mask
                new_particle_count=len(new_particle_IDs_itype_snap2[str(itype)])# Count number of new particles
                # print(f"Extracting outflow particles of type {itype} from halo list at snap 1 ...")
                out_particle_mask_itype=np.logical_and(out_particle_IDs_mask_snap1,snap1_Types_temp==itype)# Mask for particles in halo list at snap 1 which outflowed and are of the correct type
                out_particle_IDs_itype_snap1[str(itype)]=np.compress(out_particle_mask_itype,snap1_IDs_temp)# Compress snap 1 list with above mask
                out_particle_count=len(out_particle_IDs_itype_snap1[str(itype)])# Count number of outflow particles
                t2_typing.append(time.time())

                # Use the above inflow IDs and find their index in particle histories                 
                
                ################################ this is the bottleneck in the code
                #indexing inflow particle IDs
                print(f"Finding relative particle index of accreted particles: n = {new_particle_count} ...")
                t1_indexing_in.append(time.time())
                if new_particle_count>0:
                    if itype == 4:#if not stars, we don't need to check if the IDs from snap 2 are actually present at snap 1
                        new_particle_IDs_itype_snap1_historyindex=binary_search(items=new_particle_IDs_itype_snap2[str(itype)],sorted_list=Part_Histories_IDs_snap1[str(itype)],check_entries=True)
                        print((1-np.sum(np.isfinite(new_particle_IDs_itype_snap1_historyindex))/len(new_particle_IDs_itype_snap1_historyindex))*100,'% accreted star particles were actually gas at previous snapshot')
                    else:
                        new_particle_IDs_itype_snap1_historyindex=binary_search(items=new_particle_IDs_itype_snap2[str(itype)],sorted_list=Part_Histories_IDs_snap1[str(itype)],check_entries=False)
                else:
                    new_particle_IDs_itype_snap1_historyindex=[]
                t2_indexing_in.append(time.time())

                #indexing outflow particle IDs (these are taken at snap1, so we don't need to check at all)
                print(f"Finding relative particle index of outflow particles: n = {out_particle_count} ... (both snap 1 and snap 2)")
                t1_indexing_out.append(time.time())
                if out_particle_count>0:
                    out_particle_IDs_itype_snap1_historyindex=binary_search(items=out_particle_IDs_itype_snap1[str(itype)],sorted_list=Part_Histories_IDs_snap1[str(itype)])#don't need to check snap 1
                    if itype==0:#if gas, we need to check whether they've been converted to stars at snap 2
                        out_particle_IDs_itype_snap2_historyindex=binary_search(items=out_particle_IDs_itype_snap1[str(itype)],sorted_list=Part_Histories_IDs_snap2[str(itype)],check_entries=True)
                    else:
                        out_particle_IDs_itype_snap2_historyindex=binary_search(items=out_particle_IDs_itype_snap1[str(itype)],sorted_list=Part_Histories_IDs_snap2[str(itype)],check_entries=False)
                else:
                    out_particle_IDs_itype_snap1_historyindex=[]
                    out_particle_IDs_itype_snap2_historyindex=[]
                t2_indexing_out.append(time.time())

                ################################ this is the bottleneck in the code

                ############## INFLOW PARTICLE PROCESSING ##############

                print(f'Retrieving histories (prev processing, prev host, masses) and checking fidelity of inflow particles...')
                t1_inflow.append(time.time())

                if itype==4:
                    new_particle_IDs_itype_snap2_star_truncated=new_particle_IDs_itype_snap2['4']

                for iipart_historyindex,ipart_historyindex in enumerate(new_particle_IDs_itype_snap1_historyindex):
                    ID=new_particle_IDs_itype_snap2[str(itype)][iipart_historyindex]
                    # we have to be careful with star particles - we have their index in ipart_historyindex IF they were a star at the previous snap, otherwise np.nan
                    if ipart_historyindex>=0: #if our calculated index is valid at snap1, just use this index for the current parttype 
                        #(i.e. the particle was found as this type at previous snap)
                        ihalo_itype_snap1_inflow_transformed[str(itype)].append(0)
                        # Fidelity
                        if ID in snap3_IDs_temp_set:#if still in halo at snap 3
                            ihalo_itype_snap1_inflow_fidelity[str(itype)].append(1)
                        else:
                            ihalo_itype_snap1_inflow_fidelity[str(itype)].append(0)

                        # Mass
                        if constant_mass[str(itype)]:# If this particle type has a constant mass
                            ipart_snap1_mass=Part_Data_Masses_Snap1[str(itype)]
                        else:# If this particle type has a varying mass
                            ipart_snap1_partdataindex=Part_Histories_Index_snap1[str(itype)][ipart_historyindex]
                            ipart_snap1_mass=Part_Data_Masses_Snap1[str(itype)][ipart_snap1_partdataindex]
                        
                        ihalo_itype_snap1_inflow_masses[str(itype)].append(ipart_snap1_mass)

                        # Processing history
                        if itype==0 or itype==1: #Gas or DM
                            ipart_snap1_history_L1=Part_Histories_Processed_L1_snap1[str(itype)][ipart_historyindex]
                            ipart_snap1_history_L2=Part_Histories_Processed_L2_snap1[str(itype)][ipart_historyindex]
                            ihalo_itype_snap1_inflow_history_L1[str(itype)].append(ipart_snap1_history_L1)
                            ihalo_itype_snap1_inflow_history_L2[str(itype)].append(ipart_snap1_history_L1)

                        # Previous host
                        ipart_snap1_prevhost=Part_Histories_HostStructure_snap1[str(itype)][ipart_historyindex]
                        if isub:
                            if ipart_snap1_prevhost==prev_hostgroupID:
                                ipart_snap1_prevhost=0#set previous host to ZERO if from CGM
                        ihalo_itype_snap1_inflow_structure[str(itype)].append(ipart_snap1_prevhost)

                    else: # the particle was transformed (i.e. was gas at snap 1 and star at snap 2) - we want this particle in the gas list not the star list so we add to this
                        new_particle_IDs_itype_snap2['0'].append(ID)#add ID to gas list
                        ihalo_itype_snap1_inflow_transformed['0'].append(1)#note this gas particle was transformed
                        
                        #remove the ID from the star accretion list 
                        ID_index_todelete=np.where(ID==new_particle_IDs_itype_snap2_star_truncated)[0][0]
                        new_particle_IDs_itype_snap2_star_truncated=np.delete(arr=new_particle_IDs_itype_snap2_star_truncated,obj=ID_index_todelete)

                        # Fidelity
                        if ID in snap3_IDs_temp_set:#if still in halo at snap 3
                            ihalo_itype_snap1_inflow_fidelity['0'].append(1)
                        else:
                            ihalo_itype_snap1_inflow_fidelity['0'].append(0)

                        # Find index in particle history
                        ipart_transformed_historyindex=bisect_left(a=Part_Histories_IDs_snap1['0'],x=ID)#search for this ID in the gas list
                        ipart_transformed_partdataindex=Part_Histories_Index_snap1['0'][ipart_transformed_historyindex]#index in gas particle data

                        # Mass
                        ipart_snap1_mass=Part_Data_Masses_Snap1['0'][ipart_transformed_partdataindex]
                        ihalo_itype_snap1_inflow_masses['0'].append(ipart_snap1_mass)

                        ipart_snap1_history_L1=Part_Histories_Processed_L1_snap1['0'][ipart_transformed_historyindex]
                        ipart_snap1_history_L2=Part_Histories_Processed_L2_snap1['0'][ipart_transformed_historyindex]
                        ihalo_itype_snap1_inflow_history_L1['0'].append(ipart_snap1_history_L1)#add processing history to gas list
                        ihalo_itype_snap1_inflow_history_L2['0'].append(ipart_snap1_history_L2)#add processing history to gas list

                        # Previous host
                        ipart_snap1_prevhost=Part_Histories_HostStructure_snap1['0'][ipart_transformed_historyindex]
                        if isub:
                            if ipart_snap1_prevhost==prev_hostgroupID:
                                ipart_snap1_prevhost=0#set previous host to ZERO if from CGM
                        ihalo_itype_snap1_inflow_structure['0'].append(ipart_snap1_prevhost)

                t2_inflow.append(time.time())
                
                if itype==4:#truncate list of inflow star particles to just those that were stars beforehand. 
                    new_particle_IDs_itype_snap2['4']=new_particle_IDs_itype_snap2_star_truncated

                ############## OUTFLOW PARTICLE PROCESSING ##############    
                
                print(f'Retrieving masses and fate of outflow particles...')
                t1_outflow.append(time.time())
                

                
                #Grab masses and recycling status from snap1 indices
                for iipart_historyindex,ipart_historyindex in enumerate(out_particle_IDs_itype_snap1_historyindex):
                    #All these indices will be valid as we took the list of particles from snap 1 directly
                    ID=out_particle_IDs_itype_snap1[str(itype)][iipart_historyindex]
                    # Mass
                    if constant_mass[str(itype)]:# If this particle type has a constant mass
                        ipart_snap1_mass=Part_Data_Masses_Snap1[str(itype)]
                    else:
                        ipart_snap1_partdataindex=Part_Histories_Index_snap1[str(itype)][ipart_historyindex]
                        ipart_snap1_mass=Part_Data_Masses_Snap1[str(itype)][ipart_snap1_partdataindex]
                    
                    ihalo_itype_snap1_outflow_masses[str(itype)].append(ipart_snap1_mass)
                    ihalo_itype_snap3_outflow_recycled[str(itype)].append(int(ID in snap3_IDs_temp_set))
    

                
                #Grab future destination and transofmration status from snap2 indices
                for iipart_historyindex,ipart_historyindex in enumerate(out_particle_IDs_itype_snap2_historyindex):
                    ID=out_particle_IDs_itype_snap1[str(itype)][iipart_historyindex]
                    if ipart_historyindex>=0:
                        ihalo_itype_snap2_outflow_transformed[str(itype)].append(0)
                        #Find destination
                        ipart_snap2_destination=Part_Histories_HostStructure_snap2[str(itype)][ipart_historyindex]
                        if isub:
                            if ipart_snap2_destination==current_hostgroupID:
                                ipart_snap2_destination=0
                        ihalo_itype_snap2_outflow_destination[str(itype)].append(ipart_snap2_destination)
                        
                    else:
                        #need to find the transformed gas ID from snap1 in the star list at snap 2
                        ihalo_itype_snap2_outflow_transformed[str(itype)].append(1)
                        ipart_transformed_historyindex=bisect_left(a=Part_Histories_IDs_snap2['4'],x=ID)#search for this ID in the star list at snap 2
                        ipart_transformed_ID_athistoryindex=Part_Histories_IDs_snap2['4'][ipart_transformed_historyindex]
                        
                        if ID!=ipart_transformed_ID_athistoryindex:
                            print(f"Couldn't find outflow particle {ID} at snap 2 - not in star list (probably turned into BH)")
                            ipart_snap2_destination=np.nan
                        else:
                            #Find destination
                            ipart_snap2_destination=Part_Histories_HostStructure_snap2['4'][ipart_transformed_historyindex]
                            if isub:
                                if ipart_snap2_destination==current_hostgroupID:
                                    ipart_snap2_destination=0
                                                                
                        ihalo_itype_snap2_outflow_destination[str(itype)].append(ipart_snap2_destination)

                t2_outflow.append(time.time())


                ############## PRINT RESULTS ##############

                t1_print.append(time.time())
                if not isub:#if a field halo, either cosmological accretion or from mergers ("clumpy")
                    print('-- INFLOW --')
                    print(f'Gross {PartNames[itype]} accretion: {np.sum(np.array(ihalo_itype_snap1_inflow_masses[str(itype)])):.2e} Msun')
                    print(f'Particles that stayed in halo at snap 3: {np.sum(ihalo_itype_snap1_inflow_fidelity[str(itype)])/len(ihalo_itype_snap1_inflow_fidelity[str(itype)])*100:.2f}%')
                    print(f'Accretion from field: {np.sum(np.array(ihalo_itype_snap1_inflow_structure[str(itype)])<0)/len(ihalo_itype_snap1_inflow_structure[str(itype)])*100:.2f}%')
                    print(f'Accretion from other halos: {np.sum(np.array(ihalo_itype_snap1_inflow_structure[str(itype)])>0)/len(ihalo_itype_snap1_inflow_structure[str(itype)])*100:.2f}%')#clumpy if prevhost>0
                    print('-- OUTFLOW --')
                    print(f'Gross {PartNames[itype]} outflow: {np.sum(np.array(ihalo_itype_snap1_outflow_masses[str(itype)])):.2e} Msun')
                    print(f'Outflow particles in other halos at snap 2: {np.sum(np.array(ihalo_itype_snap2_outflow_destination[str(itype)])>0)/len(ihalo_itype_snap2_outflow_destination[str(itype)])*100:.2f}%')
                    print(f'Outflow particles in field at snap 2: {np.sum(np.array(ihalo_itype_snap2_outflow_destination[str(itype)])<0)/len(ihalo_itype_snap2_outflow_destination[str(itype)])*100:.2f}%')
                    print(f'Outflow particles re-accreted at snap 3: {np.sum(np.array(ihalo_itype_snap3_outflow_recycled[str(itype)])==1)/len(ihalo_itype_snap3_outflow_recycled[str(itype)])*100:.2f}%')
                    
                else:

                    print('-- INFLOW --')
                    print(f'Gross {PartNames[itype]} accretion: {np.sum(np.array(ihalo_itype_snap1_inflow_masses[str(itype)])):.2e} Msun')
                    print(f'Particles that stayed in halo at snap 3: {np.sum(ihalo_itype_snap1_inflow_fidelity[str(itype)])/len(ihalo_itype_snap1_inflow_fidelity[str(itype)])*100:.2f}%')
                    print(f'Accretion from field: {np.sum(np.array(ihalo_itype_snap1_inflow_structure[str(itype)])<0)/len(ihalo_itype_snap1_inflow_structure[str(itype)])*100:.2f}%')
                    print(f'Accretion from CGM: {np.sum(np.array(ihalo_itype_snap1_inflow_structure[str(itype)])==0)/len(ihalo_itype_snap1_inflow_structure[str(itype)])*100:.2f}%')#CGM if prevhost==0
                    print(f'Accretion from other halos: {np.sum(np.array(ihalo_itype_snap1_inflow_structure[str(itype)])>0)/len(ihalo_itype_snap1_inflow_structure[str(itype)])*100:.2f}%')#clumpy if prevhost>0
                    print('-- OUTFLOW --')
                    print(f'Gross {PartNames[itype]} outflow: {np.sum(np.array(ihalo_itype_snap1_outflow_masses[str(itype)])):.2e} Msun')
                    print(f'Outflow particles in CGM at snap 2: {np.sum(np.array(ihalo_itype_snap2_outflow_destination[str(itype)])==0)/len(ihalo_itype_snap2_outflow_destination[str(itype)])*100:.2f}%')
                    print(f'Outflow particles in other halos at snap 2: {np.sum(np.array(ihalo_itype_snap2_outflow_destination[str(itype)])>0)/len(ihalo_itype_snap2_outflow_destination[str(itype)])*100:.2f}%')
                    print(f'Outflow particles in field at snap 2: {np.sum(np.array(ihalo_itype_snap2_outflow_destination[str(itype)])<0)/len(ihalo_itype_snap2_outflow_destination[str(itype)])*100:.2f}%')
                    print(f'Outflow particles re-accreted at snap 3: {np.sum(np.array(ihalo_itype_snap3_outflow_recycled[str(itype)])==1)/len(ihalo_itype_snap3_outflow_recycled[str(itype)])*100:.2f}%')

                t2_print.append(time.time())


            # Saving INFLOW data for this parttype of the halo to file 
            for iitype, itype in enumerate(PartTypes):
                t1_save.append(time.time())

                halo_in_parttype_hdf5=halo_in_hdf5.create_group('PartType'+str(itype))
                halo_in_parttype_hdf5.create_dataset('ParticleIDs',data=new_particle_IDs_itype_snap2[str(itype)],dtype=np.int64)#######
                halo_in_parttype_hdf5.create_dataset('Masses',data=ihalo_itype_snap1_inflow_masses[str(itype)],dtype=np.float64)
                halo_in_parttype_hdf5.create_dataset('Fidelity',data=ihalo_itype_snap1_inflow_fidelity[str(itype)],dtype=np.uint8)
                halo_in_parttype_hdf5.create_dataset('PreviousHost',data=ihalo_itype_snap1_inflow_structure[str(itype)],dtype=np.int64)
                if itype==0 or itype==1:
                    halo_in_parttype_hdf5.create_dataset('Processed_L1',data=ihalo_itype_snap1_inflow_history_L1[str(itype)],dtype=np.uint8)
                    halo_in_parttype_hdf5.create_dataset('Processed_L2',data=ihalo_itype_snap1_inflow_history_L2[str(itype)],dtype=np.uint8)
                if itype==0:
                    halo_in_parttype_hdf5.create_dataset('Transformed',data=ihalo_itype_snap1_inflow_transformed[str(itype)],dtype=np.uint8)

                # Saving OUTFLOW data for this parttype of the halo to file 
                halo_out_parttype_hdf5=halo_out_hdf5.create_group('PartType'+str(itype))
                halo_out_parttype_hdf5.create_dataset('ParticleIDs',data=out_particle_IDs_itype_snap1[str(itype)],dtype=np.int64)
                halo_out_parttype_hdf5.create_dataset('Masses',data=ihalo_itype_snap1_outflow_masses[str(itype)],dtype=np.float64)
                halo_out_parttype_hdf5.create_dataset('Destination',data=ihalo_itype_snap2_outflow_destination[str(itype)],dtype=np.int64)
                halo_out_parttype_hdf5.create_dataset('Recycled',data=ihalo_itype_snap3_outflow_recycled[str(itype)],dtype=np.uint8)
                if itype==0:
                    halo_out_parttype_hdf5.create_dataset('Transformed',data=ihalo_itype_snap2_outflow_transformed[str(itype)],dtype=np.uint8)
                t2_save.append(time.time())
                t2_itype.append(time.time())

        else:#if halo not tracked, return np.nan for fidelity, ids, prevhost
            for itype in PartTypes:
                # print(f'Saving {PartNames[itype]} data for ihalo {ihalo_s2} (not tracked) to hdf5 ...')
                halo_in_parttype_hdf5=halo_in_hdf5.create_group('PartType'+str(itype))
                halo_in_parttype_hdf5.create_dataset('ParticleIDs',data=np.nan,dtype=np.float16)
                halo_in_parttype_hdf5.create_dataset('Masses',data=np.nan,dtype=np.float16)
                halo_in_parttype_hdf5.create_dataset('Fidelity',data=np.nan,dtype=np.float16)
                halo_in_parttype_hdf5.create_dataset('PreviousHost',data=np.nan,dtype=np.float16)
                if itype==0 or itype==1:
                    halo_in_parttype_hdf5.create_dataset('Processed_L1',data=np.nan,dtype=np.float16)
                    halo_in_parttype_hdf5.create_dataset('Processed_L2',data=np.nan,dtype=np.float16)
                if itype==0:
                    halo_out_parttype_hdf5.create_dataset('Transformed',data=np.nan,dtype=np.float16)
                # Saving OUTFLOW data for this parttype of the halo to file 
                halo_out_parttype_hdf5=halo_out_hdf5.create_group('PartType'+str(itype))
                halo_out_parttype_hdf5.create_dataset('ParticleIDs',data=np.nan,dtype=np.float16)
                halo_out_parttype_hdf5.create_dataset('Masses',data=np.nan,dtype=np.float16)
                halo_out_parttype_hdf5.create_dataset('Destination',data=np.nan,dtype=np.float16)
                halo_out_parttype_hdf5.create_dataset('Recycled',data=np.nan,dtype=np.float16)
                if itype==0:
                    halo_out_parttype_hdf5.create_dataset('Transformed',data=np.nan,dtype=np.float16)
                
        t2_halo=time.time()


        # Print halo data for outputs 
        print()
        print(f'Done with halo {base_halo_data[snap2]["ID"][ihalo_s2]}!')
        print()
        print('-- PERFORMANCE --')
        print(f'Total particles in: {np.sum(np.array(new_particle_IDs_mask_snap2))}, total particles out: {np.sum(np.array(out_particle_IDs_mask_snap1))}')
        print(f'Total time spent on halo: {(t2_halo-t1_halo):.2f} sec')
        print(f'Total time spent on finding inflow particles: {(t2_new-t1_new):.2f} sec - {(t2_new-t1_new)/((t2_halo-t1_halo))*100:.2f} % of halo time')
        print(f'Total time spent on finding outflow particles: {(t2_out-t1_out):.2f} sec - {(t2_out-t1_out)/((t2_halo-t1_halo))*100:.2f} % of halo time')
        performance_ihalo=[]
        for iitype,itype in enumerate(PartTypes):
            itype_time=t2_itype[iitype]-t1_itype[iitype]
            print(f'Total time on {PartNames[itype]} particles: {itype_time:.2f} sec ({itype_time/(t2_halo-t1_halo)*100:.2f} % of halo time)')
            print(f'Breakdown of time on {PartNames[itype]} particles ...')
            performance_dict={}
            performance_dict['Indexing_in']=(t2_indexing_in[iitype]-t1_indexing_in[iitype])
            performance_dict['Indexing_out']=(t2_indexing_out[iitype]-t1_indexing_out[iitype])
            performance_dict['Inflow']=(t2_inflow[iitype]-t1_inflow[iitype])
            performance_dict['Outflow']=(t2_outflow[iitype]-t1_outflow[iitype])
            performance_dict['Saving']=(t2_save[iitype]-t1_save[iitype])
            performance_dict=df(performance_dict,index=[0])
            print(performance_dict)
            performance_ihalo.append(performance_dict)
        halos_done=halos_done+1
        
        with open(fname_log,"a") as progress_file:
            progress_file.write(" \n")
            progress_file.write(f"Done with ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
            progress_file.write(f"ihalo {ihalo_s2} took {t2_halo-t1_halo} sec\n")
            progress_file.write(f"Particles in = {np.sum(new_particle_IDs_mask_snap2)}\n")
            progress_file.write(f"Particles out = {np.sum(out_particle_IDs_mask_snap1)}\n")
            for iitype,itype in enumerate(PartTypes):
                progress_file.write(f'PartType{itype} - [n(in)= {np.sum(np.logical_and(snap2_Types_temp==itype,new_particle_IDs_mask_snap2))}, n(out)={np.sum(np.logical_and(snap1_Types_temp==itype,out_particle_IDs_mask_snap1))}]: timings (sec)')
                progress_file.write(" \n")
                progress_file.write(performance_ihalo[iitype].to_string())
                progress_file.write(" \n")
        progress_file.close()

        print('----------------')
        print()


    #Close the output file, finish up
    output_hdf5.close()

########################### POSTPROCESS/SUM ACCRETION DATA ###########################

def postprocess_acc_data_serial(path):
    """

    postprocess_acc_data_serial : function
	----------

    Collate and post process all the accretion data in the provided directory (which must only contain the required data).

	Parameters
	----------
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
    acc_data_filelist_trunc=[filename for filename in acc_data_filelist if (('px' not in filename) and ('FOF' in filename) and ('DS' not in filename))]
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
    total_num_halos=np.sum([len(list(ifile.keys()))-1 for ifile in acc_data_hdf5files])#total number of halos from file
    if total_num_halos<1000:
        print(f'Using array size {3*10**5}')
        total_num_halos=3*10**5
    print(f'Collating data for {total_num_halos} halos')
    
    # Copy over header information from first file
    acc_data_hdf5files_header=acc_data_hdf5files[0]['Header']
    acc_data_hdf5files_header_attrs=list(acc_data_hdf5files_header.attrs)
    collated_output_file_header=collated_output_file.create_group('Header')

    print("Attributes of accretion calculation: ")
    for attribute in acc_data_hdf5files_header_attrs:
        collated_output_file_header.attrs.create(attribute,data=acc_data_hdf5files_header.attrs[attribute])
        print(attribute,collated_output_file_header.attrs[attribute])

    collated_output_file_header.attrs.create('total_num_halos',data=total_num_halos)

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
        print(f"Reading from file {ifile}: {acc_data_filetemp}")
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

def add_gas_particle_data(base_halo_data,accdata_path,datasets=None):
    """

    add_gas_particle_data : function 
	----------

    Add EAGLE particle data to the accretion files. 

	Parameters
	----------
    base_halo_data: dict
        The base halo data dictionary (encodes particle data filepath, snap, particle histories).

    # part_histories: dict or None
    #     An optional dictionary with keys 'snap2_gas_datasets', 'snap1_gas_datasets', 'snap2_star_datasets' - loaded before runtime.
    #     If not provided, will load. 
    #     (saves memory if parallelizing)

    # part_datasets: dict or None
    #     An optional dictionary with keys 'snap2_gas_datasets', 'snap1_gas_datasets', 'snap2_star_datasets' - loaded before runtime.
    #     If not provided, will load. 
    #     (saves memory if parallelizing)

    accdata_path : str
        The file path to the base hdf5 accretion data file. 

    datasets: list 
        List of keys for datasets to extract. See Schaye+15 for full description. 

    Returns
	----------
        Requested gas datasets for snap 1 and snap 2, saved to file at accdata_path. 

    """
    
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
    snap2=int(acc_filename.split('snap')[-1][:3])
    pre_depth=int(acc_filename.split('pre')[-1][:1])
    snap1=snap2-pre_depth
    print(acc_filename)
    iprocess=int(acc_filename.split('_p')[-1][:3])

    acc_file=h5py.File(accdata_path,'r+')
    ihalo_groups=sorted(list(acc_file.keys()))
    ihalo_groups_trunc=[ihalo_group for ihalo_group in ihalo_groups if 'ihalo_' in ihalo_group]
    ihalo_count=len(ihalo_groups_trunc)

    run_snap_log_dir='job_logs/acc_logs/'+calc_dir+f'/snap_{str(snap2).zfill(3)}/'
    if test:
        fname_log=run_snap_log_dir+f'partdata_p{str(iprocess).zfill(2)}_n{str(ihalo_count).zfill(6)}_test.log'
    else:
        fname_log=run_snap_log_dir+f'partdata_p{str(iprocess).zfill(2)}_n{str(ihalo_count).zfill(6)}.log'

    if os.path.exists(fname_log):
        os.remove(fname_log)

    with open(fname_log,"a") as progress_file:
        progress_file.write('Loading in data ...')
    progress_file.close()

    t1_io=time.time()

    partdata_filetype=base_halo_data[snap2]['Part_FileType']
    partdata_outname=base_halo_data[snap2]['outname']
    parthist_file_snap2=h5py.File(f'part_histories/PartHistory_{str(snap2).zfill(3)}_{partdata_outname}.hdf5','r')
    parthist_file_snap1=h5py.File(f'part_histories/PartHistory_{str(snap1).zfill(3)}_{partdata_outname}.hdf5','r')
    
    #Load particle histories
    parthist_gas_IDs_snap1=parthist_file_snap1['PartType0']['ParticleIDs'].value
    parthist_gas_indices_snap1=parthist_file_snap1['PartType0']['ParticleIndex'].value
    parthist_gas_IDs_snap2=parthist_file_snap2['PartType0']['ParticleIDs'].value
    parthist_gas_indices_snap2=parthist_file_snap2['PartType0']['ParticleIndex'].value    
    parthist_star_IDs_snap2=parthist_file_snap2['PartType4']['ParticleIDs'].value
    parthist_star_indices_snap2=parthist_file_snap2['PartType4']['ParticleIndex'].value
    parthist_star_count_snap2=len(parthist_star_IDs_snap2)

    #Load particle data
    gas_particle_datasets_snap1={dataset:[] for dataset in datasets}
    gas_particle_datasets_snap2={dataset:[] for dataset in datasets}
    star_particle_datasets_snap2={dataset:[] for dataset in datasets}

    if 'EAGLE' in partdata_filetype:
        print('Reading in EAGLE snapshot data ...')
        EAGLE_boxsize=base_halo_data[snap1]['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap_1=read_eagle.EagleSnapshot(base_halo_data[snap1]['Part_FilePath'])
        EAGLE_Snap_1.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
        EAGLE_Snap_2=read_eagle.EagleSnapshot(base_halo_data[snap2]['Part_FilePath'])
        EAGLE_Snap_2.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)

        for dataset in datasets:
            gas_particle_datasets_snap1[dataset]=EAGLE_Snap_1.read_dataset(0,dataset)
            gas_particle_datasets_snap2[dataset]=EAGLE_Snap_2.read_dataset(0,dataset)
            try:
                star_particle_datasets_snap2[dataset]=EAGLE_Snap_2.read_dataset(4,dataset)
            except:
                pass#cannot get the dataset for stars (will be empty list)

    else:#non-eagle file -- GADGET OR SWIFT (don't have read routine)
        PartFile_Snap_1=h5py.File(base_halo_data[snap1]['Part_FilePath'],'r')
        PartFile_Snap_2=h5py.File(base_halo_data[snap2]['Part_FilePath'],'r')

        for dataset in datasets:
            gas_particle_datasets_snap1[dataset]=PartFile_Snap_1['PartType0'][dataset].value
            gas_particle_datasets_snap2[dataset]=PartFile_Snap_2['PartType1'][dataset].value
            try:
                star_particle_datasets_snap2[dataset]=PartFile_Snap_2['PartType4'][dataset].value
            except:
                pass#cannot get the dataset for stars (will be empty list)
    
    t2_io=time.time()

    with open(fname_log,"a") as progress_file:
        progress_file.write(" \n")
        progress_file.write(f'Finished I/O in {t2_io-t1_io}! Entering main halo loop... \n')
    progress_file.close()

    for iihalo,ihalo_group in enumerate(ihalo_groups_trunc):
        t1_halo=time.time()
        print(f'Processing {ihalo_group}')
        with open(fname_log,"a") as progress_file:
            progress_file.write(" \n")
            progress_file.write(f'Processing halo {ihalo_group} ({iihalo+1} out of {ihalo_count}) \n')
        progress_file.close()

        ihalo_datasets_inflow={}
        ihalo_datasets_outflow={}
        for dataset in datasets:#initialise empty halo datasets
            ihalo_datasets_inflow[f'snap2_{dataset}']=[]
            ihalo_datasets_inflow[f'snap1_{dataset}']=[]            
            ihalo_datasets_outflow[f'snap2_{dataset}']=[]
            ihalo_datasets_outflow[f'snap1_{dataset}']=[]        

        gas_IDs_in_snap1=acc_file[ihalo_group]['Inflow']['PartType0']['ParticleIDs'].value
        gas_IDs_out_snap1=acc_file[ihalo_group]['Outflow']['PartType0']['ParticleIDs'].value

        if np.size(gas_IDs_in_snap1)==1 and type(gas_IDs_in_snap1)==np.float16:
            print(f'Not processing {ihalo_group}')
            for dataset in datasets:
                ihalo_datasets_inflow[f'snap1_{dataset}']=np.nan
                ihalo_datasets_outflow[f'snap1_{dataset}']=np.nan
                ihalo_datasets_inflow[f'snap2_{dataset}']=np.nan
                ihalo_datasets_outflow[f'snap2_{dataset}']=np.nan
        else:#valid halo
            transformed_in=np.array(acc_file[ihalo_group]['Inflow']['PartType0']['Transformed'])==1
            transformed_out=np.array(acc_file[ihalo_group]['Outflow']['PartType0']['Transformed'])==1

            #Find indices of gas particles for snap1
            ihalo_gas_inflow_history_indices_snap1=binary_search(items=gas_IDs_in_snap1,sorted_list=parthist_gas_IDs_snap1,check_entries=False)
            ihalo_gas_outflow_history_indices_snap1=binary_search(items=gas_IDs_out_snap1,sorted_list=parthist_gas_IDs_snap1,check_entries=False)
            ihalo_gas_inflow_partdata_indices_snap1=[parthist_gas_indices_snap1[index] for index in ihalo_gas_inflow_history_indices_snap1]
            ihalo_gas_outflow_partdata_indices_snap1=[parthist_gas_indices_snap1[index] for index in ihalo_gas_outflow_history_indices_snap1]

            for dataset in datasets:
                ihalo_datasets_inflow[f'snap1_{dataset}']=[gas_particle_datasets_snap1[dataset][index] for index in ihalo_gas_inflow_partdata_indices_snap1]
                ihalo_datasets_outflow[f'snap1_{dataset}']=[gas_particle_datasets_snap1[dataset][index] for index in ihalo_gas_outflow_partdata_indices_snap1]

            #Find indices of gas particles for snap2
            ihalo_gas_inflow_history_indices_snap2=binary_search(items=gas_IDs_in_snap1,sorted_list=parthist_gas_IDs_snap2,check_entries=False)
            ihalo_gas_outflow_history_indices_snap2=binary_search(items=gas_IDs_out_snap1,sorted_list=parthist_gas_IDs_snap2,check_entries=False)
            
            #Save the shape of each dataset for each particle
            dataset_shapes={}
            dataset_types={}
            for dataset in datasets:
                dataset_shapes[dataset]=np.size(gas_particle_datasets_snap2[dataset][0])
                dataset_types[dataset]=np.float32
            dataset_types['ParticleIDs']=np.int64

            #inflow
            for iipartID_in,ipartID_in in enumerate(gas_IDs_in_snap1):
                star_at_snap2=transformed_in[iipartID_in]
                if not star_at_snap2:
                    history_index=ihalo_gas_inflow_history_indices_snap2[iipartID_in]
                    partdata_index=parthist_gas_indices_snap2[history_index]
                    for dataset in datasets:
                        ipart_idataset=gas_particle_datasets_snap2[dataset][partdata_index]
                        ihalo_datasets_inflow[f'snap2_{dataset}'].append(ipart_idataset)
                else:
                    history_index=bisect_left(a=parthist_star_IDs_snap2,x=ipartID_in,lo=0,hi=parthist_star_count_snap2)
                    if parthist_star_IDs_snap2[history_index]==ipartID_in:
                        partdata_index=parthist_star_indices_snap2[history_index]
                    else:
                        partdata_index=np.nan
                    
                    for dataset in datasets:
                        dataset_shape=dataset_shapes[dataset]
                        dataset_type=dataset_types[dataset]
                        if partdata_index>=0:
                            try:
                                ihalo_datasets_inflow[f'snap2_{dataset}'].append(star_particle_datasets_snap2[dataset][partdata_index])
                            except:
                                # print(f'Couldnt get {dataset} data for stars.')
                                if dataset_type==np.float32:
                                    if dataset_shape==1:
                                        ipart_idataset=np.nan
                                    else:
                                        ipart_idataset=[np.nan for i in range(dataset_shape)]
                                else:
                                    if dataset_shape==1:
                                        ipart_idataset=-1
                                    else:
                                        ipart_idataset=[-1 for i in range(dataset_shape)]
                                ihalo_datasets_inflow[f'snap2_{dataset}'].append(ipart_idataset)
                        else:
                            if dataset_type==np.float32:
                                if dataset_shape==1:
                                    ipart_idataset=np.nan
                                else:
                                    ipart_idataset=[np.nan for i in range(dataset_shape)]
                            else:
                                if dataset_shape==1:
                                    ipart_idataset=-1
                                else:
                                    ipart_idataset=[-1 for i in range(dataset_shape)]
                            ihalo_datasets_inflow[f'snap2_{dataset}'].append(ipart_idataset)

            #outflow
            for iipartID_out,ipartID_out in enumerate(gas_IDs_out_snap1):
                star_at_snap2=transformed_out[iipartID_out]
                if not star_at_snap2:
                    history_index=ihalo_gas_outflow_history_indices_snap2[iipartID_out]
                    partdata_index=parthist_gas_indices_snap2[history_index]
                    for dataset in datasets:
                        ihalo_datasets_outflow[f'snap2_{dataset}'].append(gas_particle_datasets_snap2[dataset][partdata_index])
                else:
                    history_index=bisect_left(a=parthist_star_IDs_snap2,x=ipartID_out,lo=0,hi=parthist_star_count_snap2)
                    if parthist_star_IDs_snap2[history_index]==ipartID_out:
                        partdata_index=parthist_star_indices_snap2[history_index]
                    else:
                        partdata_index=np.nan
                    
                    for dataset in datasets:
                        dataset_shape=dataset_shapes[dataset]
                        dataset_type=dataset_types[dataset]
                        if partdata_index>=0:
                            try:
                                ihalo_datasets_outflow[f'snap2_{dataset}'].append(star_particle_datasets_snap2[dataset][partdata_index])
                            except:
                                # print(f'Couldnt get {dataset} data for stars.')
                                if dataset_type==np.float32:
                                    if dataset_shape==1:
                                        ipart_idataset=np.nan
                                    else:
                                        ipart_idataset=[np.nan for i in range(dataset_shape)]
                                else:
                                    if dataset_shape==1:
                                        ipart_idataset=-1
                                    else:
                                        ipart_idataset=[-1 for i in range(dataset_shape)]
                                
                                ihalo_datasets_outflow[f'snap2_{dataset}'].append(ipart_idataset)
                        else:
                            if dataset_type==np.float32:
                                if dataset_shape==1:
                                    ipart_idataset=np.nan
                                else:
                                    ipart_idataset=[np.nan for i in range(dataset_shape)]
                            else:
                                if dataset_shape==1:
                                    ipart_idataset=-1
                                else:
                                    ipart_idataset=[-1 for i in range(dataset_shape)]
                            ihalo_datasets_outflow[f'snap2_{dataset}'].append(ipart_idataset)

        for dataset in datasets:
            try:
                ihalo_datasets_inflow[f'snap2_{dataset}']=np.array(ihalo_datasets_inflow[f'snap2_{dataset}'],dtype=dataset_types[dataset])
                ihalo_datasets_inflow[f'snap1_{dataset}']=np.array(ihalo_datasets_inflow[f'snap1_{dataset}'],dtype=dataset_types[dataset])
                ihalo_datasets_outflow[f'snap2_{dataset}']=np.array(ihalo_datasets_outflow[f'snap2_{dataset}'],dtype=dataset_types[dataset])
                ihalo_datasets_outflow[f'snap1_{dataset}']=np.array(ihalo_datasets_outflow[f'snap1_{dataset}'],dtype=dataset_types[dataset])
            except:
                ihalo_datasets_inflow[f'snap2_{dataset}']=np.array(ihalo_datasets_inflow[f'snap2_{dataset}'],dtype=np.float32)
                ihalo_datasets_inflow[f'snap1_{dataset}']=np.array(ihalo_datasets_inflow[f'snap1_{dataset}'],dtype=np.float32)
                ihalo_datasets_outflow[f'snap2_{dataset}']=np.array(ihalo_datasets_outflow[f'snap2_{dataset}'],dtype=np.float32)
                ihalo_datasets_outflow[f'snap1_{dataset}']=np.array(ihalo_datasets_outflow[f'snap1_{dataset}'],dtype=np.float32)

            try:
                acc_file[ihalo_group]['Inflow']['PartType0'].require_dataset(f'snap2_{dataset}',data=ihalo_datasets_inflow[f'snap2_{dataset}'],dtype=dataset_types[dataset],shape=np.shape(ihalo_datasets_inflow[f'snap2_{dataset}']),exact=False)
                acc_file[ihalo_group]['Inflow']['PartType0'].require_dataset(f'snap1_{dataset}',data=ihalo_datasets_inflow[f'snap1_{dataset}'],dtype=dataset_types[dataset],shape=np.shape(ihalo_datasets_inflow[f'snap1_{dataset}']),exact=False)
                acc_file[ihalo_group]['Outflow']['PartType0'].require_dataset(f'snap2_{dataset}',data=ihalo_datasets_outflow[f'snap2_{dataset}'],dtype=dataset_types[dataset],shape=np.shape(ihalo_datasets_outflow[f'snap2_{dataset}']),exact=False)
                acc_file[ihalo_group]['Outflow']['PartType0'].require_dataset(f'snap1_{dataset}',data=ihalo_datasets_outflow[f'snap1_{dataset}'],dtype=dataset_types[dataset],shape=np.shape(ihalo_datasets_outflow[f'snap1_{dataset}']),exact=False)
    
            except:
                del acc_file[ihalo_group]['Inflow']['PartType0'][f'snap2_{dataset}']
                del acc_file[ihalo_group]['Inflow']['PartType0'][f'snap1_{dataset}']
                del acc_file[ihalo_group]['Outflow']['PartType0'][f'snap2_{dataset}']
                del acc_file[ihalo_group]['Outflow']['PartType0'][f'snap1_{dataset}']
                acc_file[ihalo_group]['Inflow']['PartType0'].create_dataset(f'snap2_{dataset}',data=ihalo_datasets_inflow[f'snap2_{dataset}'])
                acc_file[ihalo_group]['Inflow']['PartType0'].create_dataset(f'snap1_{dataset}',data=ihalo_datasets_inflow[f'snap1_{dataset}'])
                acc_file[ihalo_group]['Outflow']['PartType0'].create_dataset(f'snap2_{dataset}',data=ihalo_datasets_outflow[f'snap2_{dataset}'])
                acc_file[ihalo_group]['Outflow']['PartType0'].create_dataset(f'snap1_{dataset}',data=ihalo_datasets_outflow[f'snap1_{dataset}'])

        t2_halo=time.time()

        with open(fname_log,"a") as progress_file:
            progress_file.write(" \n")
            progress_file.write(f'Done processing halo {ihalo_group} ({iihalo+1} out of {ihalo_count}) - took {t2_halo-t1_halo} \n')
            progress_file.write(f'Progress: {(iihalo+1)/ihalo_count*100:.1f}%\n')
        progress_file.close()

    acc_file.close()

########################### READ ALL ACC DATA ###########################

def get_particle_acc_data(snap,halo_index_list,path='',fields_in=["Fidelity","ParticleIDs","Masses","Processed_L1","Processed_L2"],fields_out=['Particle_IDs','Masses',"Destination_S2","Destination_S3"],itype=None):
    if type(halo_index_list)==int:
        halo_index_list=[halo_index_list]
    else:
        halo_index_list=list(halo_index_list)
    
    print('Indexing halos ...')
    t1=time.time()
    if path=='':
        directory='acc_data/snap_'+str(snap).zfill(3)+'/'
    else:
        directory=path+'/acc_data/snap_'+str(snap).zfill(3)+'/'

    accdata_filelist=os.listdir(directory)
    accdata_filelist_trunc=sorted([directory+accfile for accfile in accdata_filelist if (('summed' not in accfile) and ('px' not in accfile) and ('DS' not in accfile))])
    accdata_files=[h5py.File(accdata_filename,'r') for accdata_filename in accdata_filelist_trunc]
    accdata_halo_lists=[list(accdata_file.keys()) for accdata_file in accdata_files]
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
    print(f'Done in {t2-t1}')
    
    if itype==None:
        parttypes=[0,1]
    else:
        parttypes=[itype]
    partfields_in=fields_in
    partfields_out=fields_out
    particle_acc_data_in={f"PartType{itype}":{field: [[] for i in range(desired_num_halos)] for field in partfields_in} for itype in parttypes}
    particle_acc_data_out={f"PartType{itype}":{field: [[] for i in range(desired_num_halos)] for field in partfields_out} for itype in parttypes}
    particle_acc_files=[]    

    print('Now retrieving halo data from file ...')
    t1=time.time()
    for iihalo,ihalo in enumerate(halo_index_list):
        ihalo_name='ihalo_'+str(ihalo).zfill(6)
        ifile=ihalo_files[iihalo]
        particle_acc_files.append(accdata_filelist_trunc[int(ifile)])
        for parttype in parttypes:
            for field in partfields_in:
                ihalo_itype_ifield=accdata_files[int(ihalo_files[iihalo])][ihalo_name+f'/Inflow/PartType{parttype}/'+field].value
                particle_acc_data_in[f'PartType{parttype}'][field][iihalo]=ihalo_itype_ifield
            for field in partfields_out:
                ihalo_itype_ifield=accdata_files[int(ihalo_files[iihalo])][ihalo_name+f'/Outflow/PartType{parttype}/'+field].value
                particle_acc_data_out[f'PartType{parttype}'][field][iihalo]=ihalo_itype_ifield

    particle_acc_data={"Inflow":particle_acc_data_in,"Outflow":particle_acc_data_out}
    t2=time.time()
    print(f'Done in {t2-t1}')

    return particle_acc_files,particle_acc_data
 
########################### READ SUMMED ACC DATA ###########################

def get_summed_acc_data(path):

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
    hdf5file=h5py.File(path,'r')

    # Load in metadata
    acc_metadata=dict()
    hdf5header_attrs=list(hdf5file['/Header'].attrs)
    for attribute in hdf5header_attrs:
        acc_metadata[attribute]=hdf5file['/Header'].attrs[attribute]

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
            acc_data_inflow[part_group_name][dataset]=hdf5file['Inflow/'+part_group_name+'/'+dataset].value

    for part_group_name in part_group_list:
        for dataset in acc_fields_outflow:
            acc_data_outflow[part_group_name][dataset]=hdf5file['Outflow/'+part_group_name+'/'+dataset].value    
    
    acc_data={'Inflow':acc_data_inflow,'Outflow':acc_data_outflow}
    
    return acc_metadata, acc_data

