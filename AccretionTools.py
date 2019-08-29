########################### CREATE PARTICLE HISTORIES ###########################
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

#  python tools 
from VRPythonTools import *
from STFTools import *
from GenPythonTools import *


def gen_particle_history_serial(base_halo_data,snaps=[],verbose=1):

    """

    gen_particle_history_serial : function
	----------

    Generate and save particle history data from velociraptor property and particle files.

	Parameters
	----------
    base_halo_data : list of dictionaries
        The halo data list of dictionaries previously generated (by gen_base_halo_data). Should contain the type of particle file we'll be reading. 

    snaps : list of ints
        The list of absolute snaps (corresponding to index in base_halo_data) for which we will add 
        particles in halos or subhalos (and save accordingly). The running lists will build on the previous snap. 

	Returns
	----------
    PartHistory_xxx-outname.hdf5 : hdf5 file with datasets

        '/PartTypeX/PartID'
        '/PartTypeX/PartIndex'
        '/PartTypeX/HostStructure'

	"""

    # Will save to file at: part_histories/PartTypeX_History_xxx-outname.dat
    # Snaps
    if snaps==[]:
        snaps=list(range(len(base_halo_data)))

    try:
        valid_snaps=[len(base_halo_data[snap].keys())>3 for snap in snaps] #which indices of snaps are valid
        valid_snaps=np.compress(valid_snaps,snaps)
        run_outname=base_halo_data[valid_snaps[0]]['outname']

    except:
        print("Couldn't validate snaps")
        return []

    # if the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    
    PartNames=['gas','DM','','','star','BH']

    if base_halo_data[valid_snaps[0]]['Part_FileType']=='EAGLE':
        PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
        SimType='EAGLE'
    else:
        PartTypes=[0,1] #Gas, DM
        SimType='OtherHydro'

    isnap=0
    # Iterate through snapshots and flip switches as required
    for snap in valid_snaps:

        if not os.path.isdir("part_histories"):
            os.mkdir("part_histories")
        outfile_name="part_histories/PartHistory_"+str(snap).zfill(3)+"_"+run_outname+".hdf5"
        if os.path.exists(outfile_name):
            os.remove(outfile_name)
        outfile=h5py.File(outfile_name,'w')

        #Load the EAGLE data for this snapshot
        EAGLE_boxsize=base_halo_data[snap]['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap=read_eagle.EagleSnapshot(base_halo_data[snap]['Part_FilePath'])
        EAGLE_Snap.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)

        Particle_History_Flags=dict()

        #Load the Halo particle lists for this snapshot for each particle type
        t1=time.time()
        snap_Halo_Particle_Lists=get_particle_lists(base_halo_data[snap],include_unbound=True,add_subparts_to_fofs=False)
        n_halos=len(snap_Halo_Particle_Lists["Particle_IDs"])
        n_halo_particles=[len(snap_Halo_Particle_Lists["Particle_IDs"][ihalo]) for ihalo in range(n_halos)]
        allhalo_Particle_hosts=np.concatenate([np.ones(n_halo_particles[ihalo],dtype='int64')*haloid for ihalo,haloid in enumerate(base_halo_data[snap]['ID'])])
        
        #anyhalo==l1
        structure_Particles=df({'ParticleIDs':np.concatenate(snap_Halo_Particle_Lists['Particle_IDs']),'ParticleTypes':np.concatenate(snap_Halo_Particle_Lists['Particle_Types']),"HostStructureID":allhalo_Particle_hosts},dtype=np.int64).sort_values(["ParticleIDs"])
        structure_Particles_bytype={str(itype):np.array(structure_Particles[["ParticleIDs","HostStructureID"]].loc[structure_Particles["ParticleTypes"]==itype]) for itype in PartTypes}
        n_structure_particles=np.sum([len(structure_Particles_bytype[str(itype)][:,0]) for itype in PartTypes])
        t2=time.time()
        print(f"Loaded, concatenated and sorted halo particle lists for snap {snap} in {t2-t1} sec")
        print(f"There are {np.sum(n_structure_particles)} particles in structure (L1)")

        # map IDs to indices from EAGLE DATA and initialise array
        
        for itype in PartTypes:
            
            t1=time.time()
            #load new snap data
            if SimType=='EAGLE': 
                Particle_IDs_Unsorted_itype=EAGLE_Snap.read_dataset(itype,"ParticleIDs")
                N_Particles_itype=len(Particle_IDs_Unsorted_itype)
            else:
                h5py_Snap=h5py.File(base_halo_data[snap]['Part_FilePath'])
                Particle_IDs_Unsorted_itype=h5py_Snap['PartType'+str(itype)+'/ParticleIDs']
                N_Particles_itype=len(Particle_IDs_Unsorted_itype)

            #initialise flag data structure with mapped IDs
            print(f"Mapping IDs to indices for all {PartNames[itype]} particles at snap {snap} ...")
            Particle_History_Flags[str(itype)]={"ParticleIDs_Sorted":np.sort(Particle_IDs_Unsorted_itype),"ParticleIndex_Original":np.argsort(Particle_IDs_Unsorted_itype),"HostStructureID":np.ones(N_Particles_itype,dtype='int64')-np.int64(2)}
            t2=time.time()
            print(f"Mapped IDs to indices for all {PartNames[itype]} particles at snap {snap} in {t2-t1} sec")
            
            #flip switches of new particles
            print("Adding host indices ...")
            t1=time.time()
            ipart_switch=0
            
            all_Structure_IDs_itype=structure_Particles_bytype[str(itype)][:,0]
            all_Structure_HostStructureID_itype=structure_Particles_bytype[str(itype)][:,1]

            all_Structure_IDs_itype_partindex=binary_search_1(sorted_array=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"],elements=all_Structure_IDs_itype)

            for ipart_switch, ipart_index in enumerate(all_Structure_IDs_itype_partindex):
                if ipart_switch%100000==0:
                    print(ipart_switch/len(all_Structure_IDs_itype_partindex)*100,f'% done adding host halos for {PartNames[itype]} particles')
                Particle_History_Flags[str(itype)]["HostStructureID"][ipart_index]=np.int64(all_Structure_HostStructureID_itype[ipart_switch])

            t2=time.time()
            print(f"Added host halos in {t2-t1} sec for {PartNames[itype]} particles")

        print(f'Dumping data to file')
        t1=time.time()

        for itype in PartTypes:
            dset_write=outfile.create_dataset(f'/PartType{itype}/ParticleIDs',dtype=np.int64,compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"])
            dset_write=outfile.create_dataset(f'/PartType{itype}/ParticleIndex',dtype=np.int32,compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIndex_Original"])
            dset_write=outfile.create_dataset(f'/PartType{itype}/HostStructure',dtype=np.int64,compression='gzip',data=Particle_History_Flags[str(itype)]["HostStructureID"])
        
        outfile.close()
        t2=time.time()

        print(f'Dumped snap {snap} data to file in {t2-t1} sec')

        isnap+=1

    return Particle_History_Flags


def gen_accretion_data_serial(base_halo_data,snap=None,halo_index_list=None,pre_depth=1,post_depth=1,verbose=1):
    
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
    
    halo_index_list : list
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
    
    AccretionData_snap{snap2}_pre{pre_depth}_post{post_depth}_ihalo_xxxxxx_xxxxxx_outname.hdf5: hdf5 file with datasets

        '/PartTypeX/ihalo_xxxxxx/ParticleID': ParticleID (in particle data for given type) of all accreted particles (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/Masses': ParticleID (in particle data for given type) of all accreted particles (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/Fidelity': Whether this particle stayed at the given fidelity gap (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/PreviousHost': Which structure was this particle host to (-1 if not in any fof object) (length: n_new_particles)
            ....etc

        Where there will be n_halos ihalo datasets. 

        '/Header': Contains attributes: "t1","t2","dt","z_ave","lt_ave"

    
    """

    #Initialising halo index list
    if halo_index_list==None:
        halo_index_list_snap2=list(range(len(base_halo_data[snap]["hostHaloID"])))#use all halos if not handed halo index list
    else:
        halo_index_list_snap2=halo_index_list

    #Assigning snap
    if snap==None:
        snap=len(base_halo_data)-1#if not given snap, just use the last one
    

    snap1=snap-pre_depth
    snap2=snap
    snap3=snap+post_depth

    halo_index_list_snap1=[find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=pre_depth) for ihalo in halo_index_list_snap2]
    halo_index_list_snap3=[find_descen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=post_depth) for ihalo in halo_index_list_snap2]

    #Initialising outputs
    run_outname=base_halo_data[snap]['outname']
    if not os.path.exists('acc_data'):
        os.mkdir('acc_data')
    outfile_name='acc_data/AccretionData_snap'+str(snap).zfill(3)+'_pre'+str(pre_depth)+'_post'+str(post_depth)+f'_ihalo_'+str(halo_index_list_snap2[0]).zfill(6)+'_'+str(halo_index_list_snap2[1]).zfill(6)+'.hdf5'
    
    output_hdf5=h5py.File(outfile_name,"w")
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

    part_filetype=base_halo_data[snap]["Part_FileType"]

    # Particle types from sim type
    PartNames=['gas','DM','','','star','BH']
    if part_filetype=='EAGLE':
        PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
        SimType='EAGLE'
    else:
        PartTypes=[0,1] #Gas, DM
        SimType='OtherHydro'

    if part_filetype=='EAGLE':
        print('Reading in EAGLE snapshot data ...')
        EAGLE_boxsize=base_halo_data[snap]['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap_2=read_eagle.EagleSnapshot(base_halo_data[snap2]['Part_FilePath'])
        EAGLE_Snap_2.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)

        snap_2_masses=dict()
        snap_2_ids=dict()
        for itype in PartTypes:
            if not itype==1:#everything except DM
                snap_2_masses[str(itype)]=EAGLE_Snap_2.read_dataset(itype,"Mass")*10**10
                snap_2_ids[str(itype)]=EAGLE_Snap_2.read_dataset(itype,"ParticleIDs")
            else:#DM
                hdf5file=h5py.File(base_halo_data[snap2]['Part_FilePath'])
                dm_mass=hdf5file['Header'].attrs['MassTable'][1]*10**10
                n_part=hdf5file['Header'].attrs['NumPart_Total'][1]
                snap_2_masses[str(itype)]=dm_mass*np.ones(n_part)
                snap_2_ids[str(itype)]=EAGLE_Snap_2.read_dataset(itype,"ParticleIDs")
        print('Done reading in EAGLE snapshot data')
       
    else:#assuming constant mass (convert to physical!)
        snap_2_masses=dict()
        snap_2_ids=dict()

        hdf5file=h5py.File(base_halo_data[snap2]['Part_FilePath'])
        masses_0=hdf5file["Header"].attrs["MassTable"][0]
        masses_1=hdf5file["Header"].attrs["MassTable"][1]
        n_part_0=hdf5file["Header"].attrs["NumPart_Total"][0]
        n_part_1=hdf5file["Header"].attrs["NumPart_Total"][1]
        snap_2_masses[str(1)]=masses_1*np.ones(n_part_0)
        snap_2_masses[str(0)]=masses_0*np.ones(n_part_1)

    #Load in particle histories
    print(f'Retrieving & organising particle histories for snap = {snap1} ...')

    Part_Histories_File_snap1=h5py.File("part_histories/PartHistory_"+str(snap1).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap1=[Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes]
    Part_Histories_Index_snap1=[Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes]
    Part_Histories_HostStructure_snap1=[Part_Histories_File_snap1["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes]
    print(f'Done retrieving & organising particle histories for snap = {snap1}')

    print(f'Retrieving & organising particle histories for snap = {snap2} ...')

    Part_Histories_File_snap2=h5py.File("part_histories/PartHistory_"+str(snap2).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap2=[Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes]
    Part_Histories_Index_snap2=[Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes]
    Part_Histories_HostStructure_snap2=[Part_Histories_File_snap2["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes]
    print(f'Done retrieving & organising particle histories for snap = {snap2}')

    print(f'Retrieving & organising particle histories for snap = {snap3} ...')

    Part_Histories_File_snap3=h5py.File("part_histories/PartHistory_"+str(snap3).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap3=[Part_Histories_File_snap3["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes]
    Part_Histories_Index_snap3=[Part_Histories_File_snap3["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes]
    Part_Histories_HostStructure_snap3=[Part_Histories_File_snap3["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes]
    print(f'Done retrieving & organising particle histories for snap = {snap3}')


    #Load in particle lists from VR
    print('Retrieving VR halo particle lists ...')
    snap_1_halo_particles=get_particle_lists(base_halo_data[snap1],halo_index_list=halo_index_list_snap1,include_unbound=True,add_subparts_to_fofs=True)
    snap_2_halo_particles=get_particle_lists(base_halo_data[snap2],halo_index_list=halo_index_list_snap2,include_unbound=True,add_subparts_to_fofs=True)
    snap_3_halo_particles=get_particle_lists(base_halo_data[snap3],halo_index_list=halo_index_list_snap3,include_unbound=True,add_subparts_to_fofs=True)
    print('Done loading VR halo particle lists')

    count=0    
    subhalos=set(np.where(base_halo_data[snap]['hostHaloID']>0)[0])
    fieldhalos=set(np.where(base_halo_data[snap]['hostHaloID']>0)[0])

    #outputs: IDs, Masses, Fidelity, PreviousHost
    #prev_host: -1: cosmological, 0: from CGM (highest level group) - this won't happen for groups/clusters, >0: from another halo/subhalo at the same level (that subhalo's ID)
        
    for iihalo,ihalo_s2 in enumerate(halo_index_list):# for each halo at snap 2
        
        halo_hdf5=output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))

        #if a subhalo, find its group at the previous snapshot

        ihalo_s1=halo_index_list_snap1[iihalo]
        ihalo_s3=halo_index_list_snap3[iihalo]
        ihalo_tracked=(ihalo_s1>-1 and ihalo_s3>-1)
        structuretype=base_halo_data[snap2]["Structuretype"][ihalo_s2]

        if structuretype>10:
            isub=True
            ifield=False
            try:
                prev_subhaloindex=find_progen_index(base_halo_data,index2=ihalo_s2,snap2=snap,depth=1) #subhalo index at previous snapshot 
                prev_hostHaloID=base_halo_data[snap1]["hostHaloID"][prev_subhaloindex] #the host halo ID of this subhalo at the previous snapshot
            except:
                prev_hostHaloID=np.nan
        else:
            isub=False
            ifield=True
            prev_hostHaloID=np.nan

        print('**********************************')
        if ifield:
            print('Halo index: ',ihalo_s2,f' - field halo')
        if isub:
            print('Halo index: ',ihalo_s2,f' - sub halo')
            print(f'Host halo at previous snap: {prev_hostHaloID}')
        print(f'Progenitor: {ihalo_s1} | Descendant: {ihalo_s3}')
        print('**********************************')

        if ihalo_tracked and structuretype<25:# if we found both the progenitor and the descendent (and it's not a subsubhalo)
            count=count+1
            snap1_IDs_temp=snap_1_halo_particles['Particle_IDs'][iihalo]
            snap1_Types_temp=snap_1_halo_particles['Particle_Types'][iihalo]
            snap2_IDs_temp=snap_2_halo_particles['Particle_IDs'][iihalo]
            snap2_Types_temp=snap_2_halo_particles['Particle_Types'][iihalo]
            snap3_IDs_temp=set(snap_3_halo_particles['Particle_IDs'][iihalo])

            #returns mask for s2 of particles which were not in s1
            print(f"Finding new particles to ihalo {ihalo_s2} ...")
            new_particle_IDs_mask_snap2=np.in1d(snap2_IDs_temp,snap1_IDs_temp,invert=True)

            for iitype,itype in enumerate(PartTypes):

                print(f"Compressing for new particles of type {itype} ...")
                new_particle_mask_itype=np.logical_and(new_particle_IDs_mask_snap2,snap2_Types_temp==itype)
                new_particle_IDs_itype_snap2=np.compress(new_particle_mask_itype,snap2_IDs_temp)

                print(f"Finding relative particle index of accreted particles in halo {ihalo_s2} of type {PartNames[itype]}: n = {len(new_particle_IDs_itype_snap2)} ...")
                new_particle_IDs_itype_snap2_historyindex=np.searchsorted(a=Part_Histories_IDs_snap2[iitype],v=new_particle_IDs_itype_snap2)
                new_particle_IDs_itype_snap1_historyindex=np.searchsorted(a=Part_Histories_IDs_snap1[iitype],v=new_particle_IDs_itype_snap2)
                    
                #particle_masses
                print(f"Retrieving mass of accreted particles in halo {ihalo_s2} of type {PartNames[itype]}: n = {len(new_particle_IDs_itype_snap2)} ...")
                if itype==1:#DM:
                    new_particle_masses=np.ones(len(new_particle_IDs_itype_snap2))*snap_2_masses[str(itype)][0]   
                else:
                    new_particle_masses=[snap_2_masses[str(itype)][Part_Histories_Index_snap2[iitype][history_index]] for history_index in new_particle_IDs_itype_snap2_historyindex]

                #add any other properties to check here...
                x=1

                #checking previous snap
                print(f"Checking previous state of accreted particles in halo {ihalo_s2} of type {PartNames[itype]}: n = {len(new_particle_IDs_itype_snap2)} ...")
                previous_structure=[Part_Histories_HostStructure_snap1[iitype][history_index] for history_index in new_particle_IDs_itype_snap1_historyindex]
                if not isub:
                    new_previous_structure=previous_structure
                    print(f'Cosmological {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)<0)/len(new_previous_structure)*100}%')
                    print(f'Clumpy {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)>0)/len(new_previous_structure)*100}%')
                else:
                    new_previous_structure=[]
                    for previous_halo_id in previous_structure:
                        if previous_halo_id==prev_hostHaloID:
                            new_previous_structure.append(0)
                        else:
                            new_previous_structure.append(previous_halo_id)
                    new_previous_structure=np.array(new_previous_structure)
                    print(f'Cosmological {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)<0)/len(new_previous_structure)*100}%')
                    print(f'CGM {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)==0)/len(new_previous_structure)*100}%')
                    print(f'Clumpy {PartNames[itype]} accretion: {np.sum(np.array(new_previous_structure)>0)/len(new_previous_structure)*100}%')

                #fidelity
                print(f"Checking which accreted particles stayed in halo {ihalo_s2} of type {PartNames[itype]}: n = {len(new_particle_IDs_itype_snap2)} ...")
                new_particle_stayed_snap3=[int(ipart in snap3_IDs_temp) for ipart in new_particle_IDs_itype_snap2]
                print(f'Done, {np.sum(new_particle_stayed_snap3)/len(new_particle_stayed_snap3)*100}% stayed')

                print(f'Saving {PartNames[itype]} data for ihalo {ihalo_s2} to hdf5 ...')
                halo_parttype_hdf5=halo_hdf5.create_group('PartType'+str(itype))
                halo_parttype_hdf5.create_dataset('ParticleIDs',data=new_particle_IDs_itype_snap2,dtype=np.int64)
                halo_parttype_hdf5.create_dataset('Masses',data=new_particle_masses,dtype=np.float64)
                halo_parttype_hdf5.create_dataset('Fidelity',data=new_particle_stayed_snap3,dtype=np.int8)
                halo_parttype_hdf5.create_dataset('PreviousHost',data=new_previous_structure,dtype=np.int32)
                print(f'Done with {PartNames[itype]} for ihalo {ihalo_s2}!')

        else:
            #### return nan accretion rate
            print(f'Saving {PartNames[itype]} data for ihalo {ihalo_s2} to hdf5 ...')
            halo_parttype_hdf5=halo_hdf5.create_group('PartType'+str(itype))
            halo_parttype_hdf5.create_dataset('ParticleIDs',data=np.nan,dtype=np.float16)
            halo_parttype_hdf5.create_dataset('Masses',data=np.nan,dtype=np.float16)
            halo_parttype_hdf5.create_dataset('Fidelity',data=np.nan,dtype=np.float16)
            halo_parttype_hdf5.create_dataset('PreviousHost',data=np.nan,dtype=np.float16)
            print(f'Done with {PartNames[itype]} for ihalo {ihalo_s2}!')

    output_hdf5.close()



def postprocess_acc_data_serial(directory):

    acc_data_filelist=os.listdir(directory)
    acc_data_outfile_name=acc_data_filelist[0].split('_ihalo')[0]+'.hdf5'
    print(f'Output file name: {acc_data_outfile_name}')
    
    collated_output_file=h5py.File('acc_data/'+acc_data_outfile_name,'w')
    
    acc_data_hdf5files=[h5py.File('acc_data/'+acc_data_file,'r') for acc_data_file in acc_data_filelist]
    acc_data_hdf5files_header=dict(acc_data_hdf5files[0]['Header'].attrs)

    collated_output_file_header=collated_output_file.create_group('Header')
    for attribute in list(acc_data_hdf5files_header.keys()):
        collated_output_file_header.attrs.create(attribute,data=acc_data_hdf5files_header[attribute],dtype=np.float16)

    total_num_halos=np.sum([len(list(ifile.keys()))-1 for ifile in acc_data_hdf5files])
    new_outputs=["All_TotalDeltaM","All_TotalDeltaN","All_CosmologicalDeltaN",'All_CosmologicalDeltaM','All_CGMDeltaN','All_CGMDeltaM','All_ClumpyDeltaN','All_ClumpyDeltaM',"Stable_TotalDeltaM","Stable_TotalDeltaN","Stable_CosmologicalDeltaN",'Stable_CosmologicalDeltaM','Stable_CGMDeltaN','Stable_CGMDeltaM','Stable_ClumpyDeltaN','Stable_ClumpyDeltaM']

    print('Starting to collate files ...')
    t1=time.time()
    iihalo=0
    for ifile_hdf5 in acc_data_hdf5files:
        ifile_halo_keys=list(ifile_hdf5.keys())[1:]
        for ihalo_group in ifile_halo_keys:# for each halo 
            outfile_ihalo_group=collated_output_file.create_group(ihalo_group)
            ihalo_partkeys=list(ifile_hdf5[ihalo_group].keys())
            for ihalo_partkey in ihalo_partkeys:#for each parttype
                outfile_ihalo_partkey_group=outfile_ihalo_group.create_group(ihalo_partkey)
                ihalo_partkey_datasets=list(ifile_hdf5[ihalo_group][ihalo_partkey].keys())
            
                ####new datasets
                masses=ifile_hdf5[ihalo_group][ihalo_partkey]['Masses'].value
                ids=ifile_hdf5[ihalo_group][ihalo_partkey]['ParticleIDs'].value
                fidelity=ifile_hdf5[ihalo_group][ihalo_partkey]['Fidelity'].value
                prevhost=ifile_hdf5[ihalo_group][ihalo_partkey]['PreviousHost'].value

                outfile_ihalo_partkey_group.create_dataset('Masses',data=masses,dtype=np.float32)
                outfile_ihalo_partkey_group.create_dataset('Fidelity',data=fidelity,dtype=np.float32)
                outfile_ihalo_partkey_group.create_dataset('ParticleIDs',data=ids,dtype=np.int64)
                outfile_ihalo_partkey_group.create_dataset('PreviousHost',data=prevhost,dtype=np.int32)

                try:
                    npart=len(masses)
                except:
                    npart=0
                
                if npart>0:
                    stable_mask=fidelity
                    cosmological_mask=prevhost<1
                    cgm_mask=prevhost==0
                    clumpy_mask=prevhost>0

                    if not np.isfinite(np.sum(fidelity)):#if this is a nan halo
                        for new_output in new_outputs:
                            outfile_ihalo_partkey_group.create_dataset(new_output,data=np.nan,dtype=np.float32)

                    else:
                        stable_masses=np.compress(stable_mask,masses)
                        all_cosmological_masses=np.compress(cosmological_mask,masses)
                        all_cgm_masses=np.compress(cgm_mask,masses)
                        all_clumpy_masses=np.compress(clumpy_mask,masses)
                        stable_cosmological_masses=np.compress(np.logical_and(stable_mask,cosmological_mask),masses)
                        stable_cgm_masses=np.compress(np.logical_and(stable_mask,cgm_mask),masses)
                        stable_clumpy_masses=np.compress(np.logical_and(stable_mask,clumpy_mask),masses)

                        outfile_ihalo_partkey_group.create_dataset('All_TotalDeltaM',data=np.sum(masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('All_CosmologicalDeltaN',data=len(all_cosmological_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('All_CosmologicalDeltaM',data=np.sum(all_cosmological_masses),dtype=np.float32)                   
                        outfile_ihalo_partkey_group.create_dataset('All_CGMDeltaN',data=len(all_cgm_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('All_CGMDeltaM',data=np.sum(all_cgm_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('All_ClumpyDeltaN',data=len(all_clumpy_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('All_ClumpyDeltaM',data=np.sum(all_clumpy_masses),dtype=np.float32)
                        
                        outfile_ihalo_partkey_group.create_dataset('Stable_TotalDeltaN',data=len(stable_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('Stable_TotalDeltaM',data=np.sum(stable_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('Stable_CosmologicalDeltaN',data=len(stable_cosmological_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('Stable_CosmologicalDeltaM',data=np.sum(stable_cosmological_masses),dtype=np.float32)                   
                        outfile_ihalo_partkey_group.create_dataset('Stable_CGMDeltaN',data=len(stable_cgm_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('Stable_CGMDeltaM',data=np.sum(stable_cgm_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('Stable_ClumpyDeltaN',data=len(stable_clumpy_masses),dtype=np.float32)
                        outfile_ihalo_partkey_group.create_dataset('Stable_ClumpyDeltaM',data=np.sum(stable_clumpy_masses),dtype=np.float32)

                else:#if no new particles
                    for new_output in new_outputs:
                        outfile_ihalo_partkey_group.create_dataset(new_output,data=0,dtype=np.float32)
            iihalo=iihalo+1

            if iihalo%500==0:
                print(iihalo/total_num_halos*100,'% done')
    t2=time.time()
    print(f'Finished collating files in {t2-t1} sec')

def read_eagle_fromIDs(base_halo_data_snap,EAGLE_Snap=None,itype=0,ParticleIDs=[],datasets=[]):

    #Load eagle snapshot if needed
    if EAGLE_Snap==None:    
        partdata_filepath=base_halo_data_snap["Part_FilePath"]
        EAGLE_Snap=read_eagle.EagleSnapshot(partdata_filepath)
        EAGLE_boxsize=base_halo_data_snap['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
    
    EAGLE_datasets={dataset:EAGLE_Snap.read_dataset(itype,dataset) for dataset in datasets}

    #Load in the particle histories
    part_histories=h5py.File("part_histories/PartHistory_"+str(base_halo_data_snap["Snap"]).zfill(3)+'_'+base_halo_data_snap["outname"]+".hdf5",'r')
    sorted_IDs=part_histories["PartType"+str(itype)+"/ParticleIDs"].value
    sorted_IDs_indices=part_histories["PartType"+str(itype)+"/ParticleIndex"]
    history_indices=np.searchsorted(v=ParticleIDs,a=sorted_IDs)
    output_datasets={dataset:[] for dataset in datasets}

    for history_index in history_indices:
        if history_index>=0:
            particle_index=sorted_IDs_indices[history_index]
            for dataset in datasets:
                output_datasets[dataset].append(EAGLE_datasets[dataset][particle_index])
        else:
            for dataset in datasets:
                output_datasets[dataset].append(np.nan)
    
    return output_datasets
        
