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
import pickle
import astropy.units as u
import read_eagle
import time

from astropy.cosmology import FlatLambdaCDM,z_at_value
from scipy.spatial import KDTree
from pandas import DataFrame as df
from os import path

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
                Particle_IDs_Unsorted_itype=EAGLE_Snap.read_dataset(itype,"ParticleIDs")
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
            all_Structure_IDs_itype_partindex=binary_search_1(sorted_array=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"],elements=all_Structure_IDs_itype)
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

########################### GENERATE ACCRETION DATA ###########################

def gen_accretion_data_serial(base_halo_data,snap=None,halo_index_list=None,pre_depth=1,post_depth=1):
    
    """

    gen_accretion_data_serial : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists with appropriate kwargs. 

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
    
    AccretionData_snap{snap2}_pre{pre_depth}_post{post_depth}_px.hdf5: hdf5 file with datasets

        '/PartTypeX/ihalo_xxxxxx/ParticleID': ParticleID (in particle data for given type) of all accreted particles (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/Masses': ParticleID (in particle data for given type) of all accreted particles (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/Fidelity': Whether this particle stayed at the given fidelity gap (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/PreviousHost': Which structure was this particle host to (-1 if not in any fof object) (length: n_new_particles)
            ....etc

        Where there will be n_halos ihalo datasets. 

        '/Header': Contains attributes: "t1","t2","dt","z_ave","lt_ave"

    
    """

    # Initialising halo index list
    if halo_index_list==None:
        halo_index_list_snap2=list(range(len(base_halo_data[snap]["hostHaloID"])))#use all halos if not handed halo index list
        iprocess="x"
    else:
        halo_index_list_snap2=halo_index_list["indices"] #extract index list from input dictionary
        iprocess=str(halo_index_list["iprocess"]).zfill(2) #the process for this index list (this is just used for the output file name)

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
    run_outname=base_halo_data[snap]['outname']#extract output name (simulation name)
    outfile_name='acc_data/AccretionData_snap'+str(snap).zfill(3)+'_pre'+str(pre_depth)+'_post'+str(post_depth)+'_p'+iprocess+'.hdf5'
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
    header_hdf5.attrs.create('snap3_z',data=z3,dtype=np.float16)

    # Now find which simulation type we're dealing with
    part_filetype=base_halo_data[snap]["Part_FileType"]

    # Standard particle type names from simulation
    PartNames=['gas','DM','','','star','BH']
    
    # Assign the particle types we're considering 
    if part_filetype=='EAGLE':
        PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
        SimType='EAGLE'
    else:
        PartTypes=[0,1] #Gas, DM
        SimType='OtherHydro'

    # Read in particle IDs and masses
    h_val=base_halo_data[snap2]['SimulationInfo']['h_val']
    if part_filetype=='EAGLE':# if an EAGLE snapshot
        print('Reading in EAGLE snapshot data ...')
        EAGLE_boxsize=base_halo_data[snap]['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap_2=read_eagle.EagleSnapshot(base_halo_data[snap2]['Part_FilePath'])
        EAGLE_Snap_2.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
        PartData_Masses_Snap2=dict()
        PartData_IDs_Snap2=dict()
        for itype in PartTypes:
            if not itype==1:#everything except DM
                PartData_Masses_Snap2[str(itype)]=EAGLE_Snap_2.read_dataset(itype,"Mass")*10**10 #read the particle masses directly
                PartData_IDs_Snap2[str(itype)]=EAGLE_Snap_2.read_dataset(itype,"ParticleIDs") #read the particle IDs directly
            else:#for DM, find particle data file and save 
                hdf5file=h5py.File(base_halo_data[snap2]['Part_FilePath'])#hdf5 file
                dm_mass=hdf5file['Header'].attrs['MassTable'][1]*10**10
                PartData_Masses_Snap2[str(itype)]=dm_mass*np.ones(hdf5file['Header'].attrs['NumPart_Total'][1])
                PartData_IDs_Snap2[str(itype)]=EAGLE_Snap_2.read_dataset(itype,"ParticleIDs")
        print('Done reading in EAGLE snapshot data')
       
    else:#assuming constant mass
        PartData_Masses_Snap2=dict()
        PartData_IDs_Snap2=dict()
        hdf5file=h5py.File(base_halo_data[snap2]['Part_FilePath'])
        MassTable=hdf5file["Header"].attrs["MassTable"]
        PartData_Masses_Snap2[str(1)]=MassTable[1]*np.ones(hdf5file["Header"].attrs["NumPart_Total"][0])/h_val
        PartData_Masses_Snap2[str(0)]=MassTable[0]*np.ones(hdf5file["Header"].attrs["NumPart_Total"][1])/h_val

    #Load in particle histories: snap 1
    print(f'Retrieving & organising particle histories for snap = {snap1} ...')
    Part_Histories_File_snap1=h5py.File("part_histories/PartHistory_"+str(snap1).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap1=[Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes]
    Part_Histories_Index_snap1=[Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes]
    Part_Histories_HostStructure_snap1=[Part_Histories_File_snap1["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes]
    print(f'Done retrieving & organising particle histories for snap = {snap1}')

    #Load in particle histories: snap 2
    print(f'Retrieving & organising particle histories for snap = {snap2} ...')
    Part_Histories_File_snap2=h5py.File("part_histories/PartHistory_"+str(snap2).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap2=[Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes]
    Part_Histories_Index_snap2=[Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes]
    Part_Histories_HostStructure_snap2=[Part_Histories_File_snap2["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes]
    print(f'Done retrieving & organising particle histories for snap = {snap2}')

    #Load in particle lists from VR
    print('Retrieving VR halo particle lists ...')
    snap_1_halo_particles=get_particle_lists(base_halo_data[snap1],halo_index_list=halo_index_list_snap1,include_unbound=True,add_subparts_to_fofs=True)
    snap_2_halo_particles=get_particle_lists(base_halo_data[snap2],halo_index_list=halo_index_list_snap2,include_unbound=True,add_subparts_to_fofs=True)
    snap_3_halo_particles=get_particle_lists(base_halo_data[snap3],halo_index_list=halo_index_list_snap3,include_unbound=True,add_subparts_to_fofs=True)
    print('Done loading VR halo particle lists')

    count=0#
    subhalos=set(np.where(base_halo_data[snap]['hostHaloID']>0)[0])
    fieldhalos=set(np.where(base_halo_data[snap]['hostHaloID']>0)[0])

    #outputs: IDs, Masses, Fidelity, PreviousHost
    #prev_host: -1: cosmological, 0: from CGM (highest level group) - this won't happen for groups/clusters, >0: from another halo/subhalo at the same level (that subhalo's ID)
        
    for iihalo,ihalo_s2 in enumerate(halo_index_list_snap2):# for each halo at snap 2
        # Create group for this halo in output file
        halo_hdf5=output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))

        # Find halo progenitor and descendants 
        ihalo_s1=halo_index_list_snap1[iihalo]#find progenitor
        ihalo_s3=halo_index_list_snap3[iihalo]#find descendant
        ihalo_tracked=(ihalo_s1>-1 and ihalo_s3>-1)#track if have both progenitor and descendant
        structuretype=base_halo_data[snap2]["Structuretype"][ihalo_s2]#structure type

        # If we have a subhalo, find its progenitor host group (for CGM accretion)
        if structuretype>10:
            isub=True
            ifield=False
            try:
                prev_subhaloindex=find_progen_index(base_halo_data,index2=ihalo_s2,snap2=snap,depth=1) #subhalo index at previous snapshot 
                prev_hostHaloID=base_halo_data[snap1]["hostHaloID"][prev_subhaloindex] #the host halo ID of this subhalo at the previous snapshot
            except:#if can't find progenitor, don't try to compare for CGM accretion
                prev_hostHaloID=np.nan
                print("Couldn't find the progenitor group - not checking for CGM accretion")
        else:
            isub=False
            ifield=True
            prev_hostHaloID=np.nan

        # Print halo data for outputs 
        print('**********************************')
        if ifield:
            print('Halo index: ',ihalo_s2,f' - field halo')
        if isub:
            print('Halo index: ',ihalo_s2,f' - sub halo')
            print(f'Host halo at previous snap: {prev_hostHaloID}')
        print(f'Progenitor: {ihalo_s1} | Descendant: {ihalo_s3}')
        print('**********************************')

        # If this halo is going to be tracked (and is not a subsubhalo) then we continue
        if ihalo_tracked and structuretype<25:# if we found both the progenitor and the descendent (and it's not a subsubhalo)
            snap1_IDs_temp=snap_1_halo_particles['Particle_IDs'][iihalo]#IDs in the halo at the previous snap
            snap1_Types_temp=snap_1_halo_particles['Particle_Types'][iihalo]#Types of particles in the halo at the previous snap
            snap2_IDs_temp=snap_2_halo_particles['Particle_IDs'][iihalo]#IDs in the halo at the current snap
            snap2_Types_temp=snap_2_halo_particles['Particle_Types'][iihalo]# Types of particles in the halo at the current snap
            snap3_IDs_temp=set(snap_3_halo_particles['Particle_IDs'][iihalo])# Set of IDs in the halo at the subsequent snapshot (to compare with)

            # Returns mask for s2 of particles which were not in s1
            print(f"Finding new particles to ihalo {ihalo_s2} ...")
            new_particle_IDs_mask_snap2=np.in1d(snap2_IDs_temp,snap1_IDs_temp,invert=True)

            # Now loop through each particle type and process accreted particle data 
            for iitype,itype in enumerate(PartTypes):
                
                # Finding particles of itype∂
                print(f"Compressing for new particles of type {itype} ...")
                new_particle_mask_itype=np.logical_and(new_particle_IDs_mask_snap2,snap2_Types_temp==itype)
                new_particle_IDs_itype_snap2=np.compress(new_particle_mask_itype,snap2_IDs_temp)#compress for just the IDs of particles of this type

                print(f"Finding relative particle index of accreted particles in halo {ihalo_s2} of type {PartNames[itype]}: n = {len(new_particle_IDs_itype_snap2)} ...")
                new_particle_IDs_itype_snap2_historyindex=np.searchsorted(a=Part_Histories_IDs_snap2[iitype],v=new_particle_IDs_itype_snap2)#index of the new IDs in particle histories snap 2
                new_particle_IDs_itype_snap1_historyindex=np.searchsorted(a=Part_Histories_IDs_snap1[iitype],v=new_particle_IDs_itype_snap2)#index of the new IDs in particle histories snap 1
                    
                # Retrieve relevant particle masses
                print(f"Retrieving mass of accreted particles in halo {ihalo_s2} of type {PartNames[itype]}: n = {len(new_particle_IDs_itype_snap2)} ...")
                if itype==1:#if dm, just use the masstable value
                    new_particle_masses=np.ones(len(new_particle_IDs_itype_snap2))*PartData_Masses_Snap2[str(itype)][0]   
                else:#otherwise, read explicitly
                    new_particle_masses=[PartData_Masses_Snap2[str(itype)][Part_Histories_Index_snap2[iitype][history_index]] for history_index in new_particle_IDs_itype_snap2_historyindex]

                # Checking the previous state of the newly accreted particles
                print(f"Checking previous state of accreted particles in halo {ihalo_s2} of type {PartNames[itype]}: n = {len(new_particle_IDs_itype_snap2)} ...")
                previous_structure=[Part_Histories_HostStructure_snap1[iitype][history_index] for history_index in new_particle_IDs_itype_snap1_historyindex]
                if not isub:#if a field halo, either cosmological accretion or from mergers ("clumpy")
                    new_previous_structure=previous_structure
                    print(f'Cosmological {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)<0)/len(new_previous_structure)*100}%')
                    print(f'Clumpy {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)>0)/len(new_previous_structure)*100}%')
                else:#if subhalo, could be from CGM 
                    new_previous_structure=[]
                    for previous_halo_id in previous_structure:#check if the previous host structure was the enclosing group
                        if previous_halo_id==prev_hostHaloID:
                            new_previous_structure.append(0)#if so, previous host structure is 0
                        else:
                            new_previous_structure.append(previous_halo_id)
                    new_previous_structure=np.array(new_previous_structure)
                    print(f'Cosmological {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)<0)/len(new_previous_structure)*100}%')#cosmological if prevhost==-1
                    print(f'CGM {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)==0)/len(new_previous_structure)*100}%')#CGM if prevhost==0
                    print(f'Clumpy {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)>0)/len(new_previous_structure)*100}%')#clumpy if prevhost>0
                
                # Checking the future state of the newly accreted particles
                print(f"Checking which accreted particles stayed in halo {ihalo_s2} of type {PartNames[itype]}: n = {len(new_particle_IDs_itype_snap2)} ...")
                new_particle_stayed_snap3=[int(ipart in snap3_IDs_temp) for ipart in new_particle_IDs_itype_snap2]#if the particle is in the descendant halo at snap3, set fidelity of particle to 1
                print(f'Done, {np.sum(new_particle_stayed_snap3)/len(new_particle_stayed_snap3)*100}% stayed')

                # Saving data for this parttype of the halo to file 
                print(f'Saving {PartNames[itype]} data for ihalo {ihalo_s2} to hdf5 ...')
                halo_parttype_hdf5=halo_hdf5.create_group('PartType'+str(itype))
                halo_parttype_hdf5.create_dataset('ParticleIDs',data=new_particle_IDs_itype_snap2,dtype=np.int64)
                halo_parttype_hdf5.create_dataset('Masses',data=new_particle_masses,dtype=np.float64)
                halo_parttype_hdf5.create_dataset('Fidelity',data=new_particle_stayed_snap3,dtype=np.int8)
                halo_parttype_hdf5.create_dataset('PreviousHost',data=new_previous_structure,dtype=np.int32)
                print(f'Done with {PartNames[itype]} for ihalo {ihalo_s2}!')

        else:#if halo not tracked, return np.nan for fidelity, ids, prevhost
            print(f'Saving {PartNames[itype]} data for ihalo {ihalo_s2} (not tracked) to hdf5 ...')
            halo_parttype_hdf5=halo_hdf5.create_group('PartType'+str(itype))
            halo_parttype_hdf5.create_dataset('ParticleIDs',data=np.nan,dtype=np.float16)
            halo_parttype_hdf5.create_dataset('Masses',data=np.nan,dtype=np.float16)
            halo_parttype_hdf5.create_dataset('Fidelity',data=np.nan,dtype=np.float16)
            halo_parttype_hdf5.create_dataset('PreviousHost',data=np.nan,dtype=np.float16)
            print(f'Done with {PartNames[itype]} for ihalo {ihalo_s2}!')

    #Close the output file, finish up
    output_hdf5.close()

########################### POSTPROCESS ACCRETION DATA ###########################

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
    
    AccretionData_snap{snap2}_pre{pre_depth}_post{post_depth}.hdf5: hdf5 file with datasets:

        verbose outputs
        ---------------
        '/PartTypeX/ihalo_xxxxxx/ParticleID': ParticleID (in particle data for given type) of all accreted particles (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/Masses': ParticleID (in particle data for given type) of all accreted particles (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/Fidelity': Whether this particle stayed at the given fidelity gap (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/PreviousHost': Which structure was this particle host to (-1 if not in any fof object) (length: n_new_particles)
        
        summed outputs
        ---------------
        '/PartTypeX/ihalo_xxxxxx/All_TotalDeltaN': Total number of particles of type X new to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/All_TotalDeltaM': Total mass of particles of type X new to the halo  (length: 1)
        '/PartTypeX/ihalo_xxxxxx/All_CosmologicalDeltaN': Total number of cosmological origin particles of type X new to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/All_CosmologicalDeltaM': Total mass of cosmological origin particles of type X new to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/All_CGMDeltaN': Total number of CGM origin particles of type X new to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/All_CGMDeltaM': Total mass of CGM origin particles of type X new to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/All_ClumpyDeltaN': Total number of clumpy origin particles of type X new to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/All_ClumpyDeltaM': Total mass of clumpy origin particles of type X new to the halo (length: 1)
        
        '/PartTypeX/ihalo_xxxxxx/Stable_TotalDeltaN': Total number of particles of type X new (and LOYAL) to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/Stable_TotalDeltaM': Total mass of particles of type X new (and LOYAL) to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/Stable_CosmologicalDeltaN': Total number of cosmological origin particles of type X new (and LOYAL) to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/Stable_CosmologicalDeltaM': Total mass of cosmological origin particles of type X new (and LOYAL) to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/Stable_CGMDeltaN': Total number of CGM origin particles of type X new (and LOYAL) to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/Stable_CGMDeltaM': Total mass of CGM origin particles of type X new (and LOYAL) to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/Stable_ClumpyDeltaN': Total number of clumpy origin particles of type X new (and LOYAL) to the halo (length: 1)
        '/PartTypeX/ihalo_xxxxxx/Stable_ClumpyDeltaM': Total mass of clumpy origin particles of type X new (and LOYAL) to the halo (length: 1)
        
        Where there will be n_halos ihalo datasets. 

        '/Header' contains attributes: 
        't1'
        't2'
        'dt'
        'z_ave'
        'lt_ave'
    
    """
    # List the contents of the provided directory
    acc_data_filelist=os.listdir(directory)
    acc_data_outfile_name=acc_data_filelist[0][-9:]+'.hdf5'
    print(f'Output file name: {acc_data_outfile_name}')
    
    # Initialise output file
    collated_output_file=h5py.File('acc_data/'+acc_data_outfile_name,'w')
    
    # Open existing files in list structure
    acc_data_hdf5files=[h5py.File('acc_data/'+acc_data_file,'r') for acc_data_file in acc_data_filelist]
    total_num_halos=np.sum([len(list(ifile.keys()))-1 for ifile in acc_data_hdf5files])#total number of halos from file

    # Copy over header information from first file
    acc_data_hdf5files_header=dict(acc_data_hdf5files[0]['Header'].attrs)
    collated_output_file_header=collated_output_file.create_group('Header')
    for attribute in list(acc_data_hdf5files_header.keys()):
        collated_output_file_header.attrs.create(attribute,data=acc_data_hdf5files_header[attribute],dtype=np.float16)
    collated_output_file_header.attrs.create('total_num_halos',data=total_num_halos)

    # Names of the new outputs
    new_outputs=["All_TotalDeltaM","All_TotalDeltaN",
    "All_CosmologicalDeltaN",'All_CosmologicalDeltaM',
    'All_CGMDeltaN','All_CGMDeltaM',
    'All_ClumpyDeltaN','All_ClumpyDeltaM',
    "Stable_TotalDeltaM","Stable_TotalDeltaN",
    "Stable_CosmologicalDeltaN",'Stable_CosmologicalDeltaM',
    'Stable_CGMDeltaN','Stable_CGMDeltaM',
    'Stable_ClumpyDeltaN','Stable_ClumpyDeltaM']

    # Loop through groups (halos), subgroups (particle types) and then each field and copy over to unified file
    # In addition, calculate new summed outputs
    print('Starting to collate files ...')
    t1=time.time()
    iihalo=0
    for ifile_hdf5 in acc_data_hdf5files:#for each output file
        ifile_halo_keys_all=list(ifile_hdf5.keys())[1:]#list the halo keys
        ifile_halo_keys=[ifile_halo_key for  ifile_halo_key in ifile_halo_keys if ifile_halo_key.startswith('ihalo')]# just halos, not the header
        for ihalo_group in ifile_halo_keys:# for each halo 
            outfile_ihalo_group=collated_output_file.create_group(ihalo_group)
            ihalo_partkeys=list(ifile_hdf5[ihalo_group].keys())
            for ihalo_partkey in ihalo_partkeys:#for each parttype
                outfile_ihalo_partkey_group=outfile_ihalo_group.create_group(ihalo_partkey)#subgroup for parttype
            
                # The output, verbose datasets from gen_accretion_data
                masses=ifile_hdf5[ihalo_group][ihalo_partkey]['Masses'].value
                ids=ifile_hdf5[ihalo_group][ihalo_partkey]['ParticleIDs'].value
                fidelity=ifile_hdf5[ihalo_group][ihalo_partkey]['Fidelity'].value
                prevhost=ifile_hdf5[ihalo_group][ihalo_partkey]['PreviousHost'].value

                # Copy these outputs to our new file 
                outfile_ihalo_partkey_group.create_dataset('Masses',data=masses,dtype=np.float32)
                outfile_ihalo_partkey_group.create_dataset('Fidelity',data=fidelity,dtype=np.float32)
                outfile_ihalo_partkey_group.create_dataset('ParticleIDs',data=ids,dtype=np.int64)
                outfile_ihalo_partkey_group.create_dataset('PreviousHost',data=prevhost,dtype=np.int32)
                
                # If the halo was not tracked the fidelity/ other datasets will be np.nan, check sum to see if this is the case
                if not np.isfinite(np.sum(fidelity)):#if this is a nan halo
                    for new_output in new_outputs:#go through all the new summed outputs and set them to also nan
                        outfile_ihalo_partkey_group.create_dataset(new_output,data=np.nan,dtype=np.float32)

                # If the halo is valid and was tracked, continue with the calculations
                else:
                    # Create masks for new outputs
                    stable_mask=fidelity#the particles which stayed
                    cosmological_mask=prevhost<0#the particles of cosmological origin
                    cgm_mask=prevhost==0#the particles of CGM origin
                    clumpy_mask=prevhost>0#the particles of other origin

                    # Compress the masses for combinations of the above masks
                    all_masses=masses#all new masses
                    stable_masses=np.compress(stable_mask,masses)#new masses which stayed
                    all_cosmological_masses=np.compress(cosmological_mask,masses)#all new cosmological masses
                    all_cgm_masses=np.compress(cgm_mask,masses)#all new cgm masses
                    all_clumpy_masses=np.compress(clumpy_mask,masses)#all new clumpy masses
                    stable_cosmological_masses=np.compress(np.logical_and(stable_mask,cosmological_mask),masses)#new cosmological masses that stayed
                    stable_cgm_masses=np.compress(np.logical_and(stable_mask,cgm_mask),masses)#new cgm masses that stayed
                    stable_clumpy_masses=np.compress(np.logical_and(stable_mask,clumpy_mask),masses)#clumpy masses that stayed

                    # Write outputs ...
                    outfile_ihalo_partkey_group.create_dataset('All_TotalDeltaM',data=np.sum(masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('All_CosmologicalDeltaN',data=np.size(all_cosmological_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('All_CosmologicalDeltaM',data=np.sum(all_cosmological_masses),dtype=np.float32)                   
                    outfile_ihalo_partkey_group.create_dataset('All_CGMDeltaN',data=np.size(all_cgm_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('All_CGMDeltaM',data=np.sum(all_cgm_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('All_ClumpyDeltaN',data=np.size(all_clumpy_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('All_ClumpyDeltaM',data=np.sum(all_clumpy_masses),dtype=np.float32)
                    
                    outfile_ihalo_partkey_group.create_dataset('Stable_TotalDeltaN',data=np.size(stable_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('Stable_TotalDeltaM',data=np.sum(stable_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('Stable_CosmologicalDeltaN',data=np.size(stable_cosmological_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('Stable_CosmologicalDeltaM',data=np.sum(stable_cosmological_masses),dtype=np.float32)                   
                    outfile_ihalo_partkey_group.create_dataset('Stable_CGMDeltaN',data=np.size(stable_cgm_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('Stable_CGMDeltaM',data=np.sum(stable_cgm_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('Stable_ClumpyDeltaN',data=np.size(stable_clumpy_masses),dtype=np.float32)
                    outfile_ihalo_partkey_group.create_dataset('Stable_ClumpyDeltaM',data=np.sum(stable_clumpy_masses),dtype=np.float32)

            iihalo=iihalo+1
            if iihalo%500==0:
                print(iihalo/total_num_halos*100,'% done')

    t2=time.time()
    collated_output_file.close()
    print(f'Finished collating files in {t2-t1} sec')
    return None

########################### READ EAGLE DATA FROM IDs ###########################

def read_eagle_from_IDs(base_halo_data_snap,itype=0,ParticleIDs=[],datasets=[]):
    """

    read_eagle_from_IDs : function 
	----------

    Return a dictionary of datasets (from 'datasets' argument) with the data ordered and specifically
    selected based on the provided particleIDs. 

	Parameters
	----------
    base_halo_data_snap : dict
        The base halo data dictionary for this snap (encodes particle data filepath, snap, particle histories).

    itype : int 
        [0 (gas),1 (DM), 4 (stars), 5 (BH)]
        The particle type we want to read EAGLE data for. 
        *** todo MAKE THIS A LIST AS PER BELOW *** (so we don't have to read the EAGLE snap multiple times for multiple halos)

    particleIDs: list
        The list of particle IDs for which we want to extract data for. 
        *** todo ALLOW FOR LISTS OF LISTS *** (so we don't have to read the EAGLE snap multiple times for multiple halos)

    datasets: list 
        List of keys for datasets to extract. See Schaye+15 for full description. 

    Returns
	----------
    output_datasets : dict
        Dictionary of the different datasets requested. 
        *** todo MAKE THE OUTPUT A LIST OF DICTIONARIES *** (so we don't have to read the EAGLE snap multiple times for multiple halos)

    """
    # Load the relevant EAGLE snapshot
  
    partdata_filepath=base_halo_data_snap["Part_FilePath"]
    EAGLE_Snap=read_eagle.EagleSnapshot(partdata_filepath)
    EAGLE_boxsize=base_halo_data_snap['SimulationInfo']['BoxSize_Comoving']
    EAGLE_Snap.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)

    # Read the relevant datasets
    EAGLE_datasets={dataset:EAGLE_Snap.read_dataset(itype,dataset) for dataset in datasets}

    # Load in the particle histories
    part_histories=h5py.File("part_histories/PartHistory_"+str(base_halo_data_snap["Snap"]).zfill(3)+'_'+base_halo_data_snap["outname"]+".hdf5",'r')
    sorted_IDs=part_histories["PartType"+str(itype)+"/ParticleIDs"].value
    sorted_IDs_indices=part_histories["PartType"+str(itype)+"/ParticleIndex"]

    # Find the indices of our particleIDs in the particle histories
    history_indices=np.searchsorted(v=ParticleIDs,a=sorted_IDs)
    
    # Find the data for each particle here
    output_datasets={dataset:[] for dataset in datasets}#initialise outputs
    for history_index in history_indices:#for each index in the histories (i.e. every particle)
        if history_index>=0:#if we have a valid index (i.e. not np.nan)
            particle_index=sorted_IDs_indices[history_index]#identify the index in the eagle snapshots
            for dataset in datasets:#for each dataset, add the data for this particle
                output_datasets[dataset].append(EAGLE_datasets[dataset][particle_index])
        else:#if the index was invalid, just record np.nan for all the datasets for this particle
            for dataset in datasets:
                output_datasets[dataset].append(np.nan)
    
    # Return the dictionary of particle data
    return output_datasets
        
