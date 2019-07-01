#########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data & Halo Properties ##########################################
#########################################################################################################################################################################

#*** Preamble ***
import os
import numpy as np
import h5py
import pickle
from pandas import DataFrame as df
import astropy.units as u
from multiprocessing import Pool,cpu_count
from astropy.cosmology import FlatLambdaCDM,z_at_value

# VELOCIraptor python tools 
from VRPythonTools import *


########################### CREATE HALO DATA ###########################

def gen_halo_data_all(snaps=[],tf_treefile="",vr_directory="",vr_prefix="snap_",vr_files_type=2,vr_files_nested=False,vr_files_lz=4,extra_halo_fields=[],halo_TEMPORALHALOIDVAL=[],verbose=1):
    
    """

    gen_halo_data_all : function
	----------

    Generate halo data from velociraptor property and particle files.

	Parameters
	----------
	snaps : list or int
		The snapshots to load halo data from. 
            If this is an empty list, we find all snap data.
            If this is a non-empty list, we use these snaps.
            If this is an integer, we find data for all snaps up to this integer.

    tf_treefile : string
        The full path to the text file in which the tree data files are listed.
    
    vr_directory : string
        The path to the directory in which the VELOCIraptor data is found.
    
    vr_prefix : string
        The prefix to the VELOCIraptor files.

    vr_files_type : int
        The filetype of the VELOCIraptor inputs: (2 = hdf5)

    vr_files_nested : bool
        Boolean flag as to whether the VELOCIraptor files are nested in a file structure.

    vr_files_lz : int
        The number of digits defining the snapshot in the VELOCIraptor file names.

    extra_halo_fields : list of strings
        List of any extra halo fields we may desire. 

    halo_TEMPORALHALOIDVAL : int
        The multiplier used by VELOCIraptor to create unique temporal halo IDs. 

    verbose : bool
        Flag indicating how verbose we want the code to be when we run.

    Returns
	-------
	halo_data_all : list

        A list (for each snap desired) of dictionaries which contain halo data with the following fields:
        'ID'
        'hostHaloID'
        'numSubStruct'
        'Mass_tot'
        'Mass_200crit'
        'Mass_200mean'
        'M_gas'
        'M_gas_500c'
        'Xc'
        'Yc'
        'Zc'
        'R_200crit'
    
        (any extra halo fields)
        'Snap'
        'SimulationInfo'
            'h_val'
            'Hubble_unit'
            'Omega_Lambda'
            'ScaleFactor'
            'z'
            'LookbackTime'
        
        'UnitInfo'
        'FilePath'
        'FileType'
	
	"""

    ###### WILL SAVE TO FILE THE HALO DATA WITH FORM: halo_data_all.dat

    halo_data_all=[]

    ### input processing
    
    # extra halo fields
    try:
        halo_fields=['ID','hostHaloID','numSubStruct','Mass_tot','Mass_200crit','Mass_200mean','M_gas','M_gas_500c','Xc','Yc','Zc','R_200crit']#default halo fields
        halo_fields.extend(extra_halo_fields)
    except:
        print('Please enter valid extra halo fields (should be a list of strings')
    
    # snapshots
    if snaps==[]: #if no snaps specified, find them all
        sim_snaps=list(range(1000))
        if verbose:
            print("Looking for snaps up to 1000")
    elif type(snaps)==list: #if we're given a non-empty list
        sim_snaps=snaps
    elif type(snaps)==int: #if we're given an integer
        sim_snaps=list(range(snaps))

    print('Reading halo data using VR python tools')

    err=0
    found=0
    #for each snap specified, we will generate halo data
    for isnap,snap in enumerate(sim_snaps):
        if verbose:
            print('Searching for halo data at snap = ',snap)
            if vr_files_nested:
                print('File: '+vr_directory+vr_prefix+str(snap).zfill(vr_files_lz)+"/"+vr_prefix+str(snap).zfill(vr_files_lz))
            else:
                print('File: '+vr_directory+vr_prefix+str(snap).zfill(vr_files_lz))

        #use VR python tools to load in halo data for this snap
        if vr_files_nested:
            halo_data_snap=ReadPropertyFile(vr_directory+vr_prefix+str(snap).zfill(vr_files_lz)+"/"+vr_prefix+str(snap).zfill(vr_files_lz),ibinary=vr_files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True)
        else:
            halo_data_snap=ReadPropertyFile(vr_directory+vr_prefix+str(snap).zfill(vr_files_lz),ibinary=vr_files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True)

        #if data is found
        if not halo_data_snap==[]:
            halo_data_all.append(halo_data_snap)
            halo_data_all[isnap][0]['Snap']=snap
            found=found+1

        #if data is not found
        else:
            err=err+1
            if verbose:
                print("Couldn't find velociraptor files for snap = ",snap)
        
            if err>2 and found<2:#not finding files -- don't bother continuing
                print("Failed to find file on multiple occasions, terminating")
                return []

            if err>2 and found>1:#reached end of snaps
                print("Reached end of snapshots, total number of snaps found = ",len(halo_data_all))
                break

    # List of number of halos detected for each snap and list isolated data dictionary for each snap (in dictionaries)
    halo_data_counts=[item[1] for item in halo_data_all]
    halo_data_all=[item[0] for item in halo_data_all]

    snap_no=len(halo_data_all)
    sim_snaps=[halo_data_all[isnap]['Snap'] for isnap in range(snap_no)]

    # Add halo count to halo data at each snap
    for isnap,snap in enumerate(sim_snaps):
        halo_data_all[isnap]['Count']=halo_data_counts[isnap]

    # List sim info and unit info for each snap (in dictionaries)
    halo_siminfo=[halo_data_all[snap]['SimulationInfo'] for snap in sim_snaps]
    halo_unitinfo=[halo_data_all[snap]['UnitInfo'] for snap in sim_snaps]
    
    # Import tree data from TreeFrog, build temporal head/tails from descendants -- adds to halo_data_all (all halo data)
    print('Now assembling descendent tree using VR python tools')

    # Read in tree data
    halo_tree=ReadHaloMergerTreeDescendant(tf_treefile,ibinary=vr_files_type,iverbose=verbose+1,imerit=True,inpart=False)

    # Now build trees and add onto halo data array
    if halo_TEMPORALHALOIDVAL==[]:#if not given halo TEMPORALHALOIVAL, use the vr default
        BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=verbose)
    else:
        BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=verbose,TEMPORALHALOIDVAL=halo_TEMPORALHALOIDVAL)
    
    print('Finished assembling descendent tree using VR python tools')

    if verbose==1:
        print('Adding timesteps & filepath information')
    
    # Adding timesteps and final bits of information 
    H0=halo_data_all[0]['SimulationInfo']['h_val']*halo_data_all[0]['SimulationInfo']['Hubble_unit']
    Om0=halo_data_all[0]['SimulationInfo']['Omega_Lambda']
    cosmo=FlatLambdaCDM(H0=H0,Om0=Om0)

    for isnap,snap in enumerate(sim_snaps):
        scale_factor=halo_data_all[isnap]['SimulationInfo']['ScaleFactor']
        redshift=z_at_value(cosmo.scale_factor,scale_factor,zmin=-0.5)
        lookback_time=cosmo.lookback_time(redshift).value

        halo_data_all[isnap]['SimulationInfo']['z']=redshift
        halo_data_all[isnap]['SimulationInfo']['LookbackTime']=lookback_time

        if vr_files_nested:
            halo_data_all[isnap]['FilePath']=vr_directory+vr_prefix+str(snap).zfill(vr_files_lz)+"/"+vr_prefix+str(snap).zfill(vr_files_lz)
            halo_data_all[isnap]['FileType']=vr_files_type
        else:
            halo_data_all[isnap]['FilePath']=vr_directory+vr_prefix+str(snap).zfill(vr_files_lz)
            halo_data_all[isnap]['FileType']=vr_files_type

    with open('halo_data_all.dat', 'wb') as halo_data_file:
        pickle.dump(halo_data_all, halo_data_file)
        halo_data_file.close()

    print('Done generating base halo data')

    return halo_data_all

########################### RETRIEVE PARTICLE LISTS ###########################

def get_particle_lists(snap,halo_data_snap,add_subparts_to_fofs=False,verbose=1):
    
    """

    gen_particle_history : function
	----------

    Retrieve the particle lists for each halo from velociraptor particle files at a given snapshot.

	Parameters
	----------
    snap : int
        The snapshot for which we want the particle lists.
    
    halo_data_snap : dictionary
        The halo data dictoinary for the relevant snapshot.

    add_subparts_to_fof : bool
        Flag as to whether to add subhalo particles to their fof halos.

    verbose : bool
        Flag indicating how verbose we want the code to be when we run.

    Returns
    ----------
    part_data_temp : dictionary 
        The particle IDs, Types, and counts for the given snapshot in a dictionary
        Keys: 
            "Particle_IDs" - list (for each halo) of lists of particle IDs
            "Particle_Types" - list (for each halo) of lists of particle Types
            "Npart" - list (for each halo) of the number of particles belonging to the object

	"""

    ### input checking
    # snapshot
    try:
        snap=int(snap)
    except:
        print('Snapshot not a valid integer')

    if verbose:
        print('Reading particle lists for snap = ',snap)

    # particle data
    try:
        part_data_temp=ReadParticleDataFile(halo_data_snap['FilePath'],ibinary=halo_data_snap['FileType'],iverbose=0,iparttypes=1)
        
        if part_data_temp==[]:
            part_data_temp={"Npart":[],"Npart_unbound":[],'Particle_IDs':[],'Particle_Types':[]}
            print('Particle data not found for snap = ',snap)
            print('Used directory: ',vr_directory+vr_prefix+str(snap).zfill(vr_files_lz))
            return part_data_temp

    except: #if we can't load particle data
        if verbose:
            print('Particle data not included in hdf5 file for snap = ',snap)
        part_data_temp={"Npart":[],"Npart_unbound":[],'Particle_IDs':[],'Particle_Types':[]}
        return part_data_temp

    if add_subparts_to_fofs:

        if verbose==1:
            print('Appending FOF particle lists with substructure')
        
        field_halo_indices_temp=np.where(halo_data_snap['hostHaloID']==-1)[0]#find field/fof halos

        for i_field_halo,field_halo_ID in enumerate(halo_data_snap['ID'][field_halo_indices_temp]):#go through each field halo
            
            sub_halos_temp=(np.where(halo_data_snap['hostHaloID']==field_halo_ID)[0])#find the indices of its subhalos

            if len(sub_halos_temp)>0:#where there is substructure

                field_halo_temp_index=field_halo_indices_temp[i_field_halo]
                field_halo_plist=part_data_temp['Particle_IDs'][field_halo_temp_index]
                field_halo_tlist=part_data_temp['Particle_Types'][field_halo_temp_index]
                
                sub_halos_plist=np.concatenate([part_data_temp['Particle_IDs'][isub] for isub in sub_halos_temp])#list all particles IDs in substructure
                sub_halos_tlist=np.concatenate([part_data_temp['Particle_Types'][isub] for isub in sub_halos_temp])#list all particles types substructure

                part_data_temp['Particle_IDs'][field_halo_temp_index]=np.concatenate([field_halo_plist,sub_halos_plist])#add particles to field halo particle list
                part_data_temp['Particle_Types'][field_halo_temp_index]=np.concatenate([field_halo_tlist,sub_halos_tlist])#add particles to field halo particle list
                part_data_temp['Npart'][field_halo_temp_index]=len(part_data_temp['Particle_IDs'][field_halo_temp_index])#update Npart for each field halo

        if verbose==1:
            print('Finished appending FOF particle lists with substructure')

    return part_data_temp

#################### PARTICLE HISTORIES WORKER FUNCTION ###########################

def calc_particle_history(halo_index_list,sub_bools,particle_IDs_subset,verbose=1):

    if len(halo_index_list)==len(particle_IDs_subset):
        pass
    else:
        return []

    n_halos_parsed=len(halo_index_list)
    
    if np.sum(sub_bools)<2:
        sub_halos_plist=[]
    else:
        sub_halos_plist=[]
        for ihalo,plist in enumerate(particle_IDs_subset):
            if sub_bools[ihalo]:
                sub_halos_plist.append(plist)
        sub_halos_plist=np.concatenate(sub_halos_plist)


    all_halos_plist=np.concatenate(particle_IDs_subset)

    return [all_halos_plist,sub_halos_plist]

########################### CREATE PARTICLE HISTORIES ###########################

def gen_particle_history_serial(halo_data_all,npart,min_snap=0,verbose=1):

    """

    gen_particle_history_serial : function
	----------

    Generate and save particle history data from velociraptor property and particle files.

	Parameters
	----------
    halo_data_all : list of dictionaries
        The halo data list of dictionaries previously generated.

    npart : int
        The integer total number of particles in the simulation.

    min_snap : int
        The snap after which to save particle histories.

	Returns
	----------
    {'all_ids':running_list_all,'sub_ids':running_list_sub} : dict
        Dictionary of particle lists which have ever been part of any halo and those which 
        have been part of a subhalo at any point up to the last snap in the halo_data_all array.

        This data is saved for each snapshot on the way in a np.pickle file in the directory "part_histories"

	"""

    ###### WILL SAVE TO FILE PARTICLE HISTORIES WITH FORM: part_histories/snap_xxx_parthistory_all.dat and part_histories/snap_xxx_parthistory_sub.dat

    ### Input checks
    # Snaps
    try:
        no_snaps=len(halo_data_all)
    except:
        print("Invalid halo data")

    # if the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    if not os.path.isdir("part_histories"):
        os.mkdir("part_histories")

    print('Generating particle histories up to snap = ',no_snaps)

    running_list_all=[]
    running_list_sub=[]
    sub_part_hist=np.zeros(npart)
    all_part_hist=np.zeros(npart)

    # for each snapshot get the particle data and add to the running list

    for isnap in range(no_snaps):

        #Load particle data for this snapshot
        new_particle_data=get_particle_lists(snap=isnap,halo_data_snap=halo_data_all[isnap],add_subparts_to_fofs=False,verbose=verbose)

        #if no halos or no new particle data
        if len(new_particle_data['Particle_IDs'])==0 or len(halo_data_all[isnap]['hostHaloID'])<2:
            if verbose:
                print('Either no particle data or no halos for snap = ',isnap)
            continue
        #if particle data is valid, continue
        else:
            if verbose:
                print('Have particle lists for snap = ',isnap)

            n_halos_snap=len(halo_data_all[isnap]['hostHaloID'])# Number of halos at this snap
            sub_halos_snap=halo_data_all[isnap]['hostHaloID']>0 #Boolean mask indicating which halos are subhalos

            # Implement the worker function to grab the particle lists of particles in all structure and substructure
            temp_result_array=calc_particle_history(halo_index_list=list(range(n_halos_snap)),sub_bools=sub_halos_snap,particle_IDs_subset=new_particle_data["Particle_IDs"])
            all_halos_plist=temp_result_array[0]
            sub_halos_plist=temp_result_array[1]

            # Find the particles new to structure or substructure
            new_structure_indices=np.array(np.compress(np.logical_not(np.in1d(all_halos_plist,running_list_all)),all_halos_plist))
            new_substructure_indices=np.array(np.compress(np.logical_not(np.in1d(sub_halos_plist,running_list_sub)),sub_halos_plist))

            # Add all these particles to the running list from all the previous snaps
            running_list_all=np.concatenate([running_list_all,all_halos_plist])
            running_list_sub=np.concatenate([running_list_sub,sub_halos_plist])

            # Make sure we're not repeating particles in the running list
            running_list_all=np.unique(running_list_all)
            running_list_sub=np.unique(running_list_sub)

            #Iterate through the newly identified particles and set their index to True
            for new_part_structure in new_structure_indices:
                all_part_hist[int(new_part_structure)]=1

            for new_part_substructure in new_substructure_indices:
                sub_part_hist[int(new_part_substructure)]=1

            # Now if our snapshot is above the minimum snap set at the outset
            # we save the boolean lists (of length npart) for this snapshot and move on
            if isnap in range(no_snaps):
                if isnap>min_snap:
                    parthist_filename_all="part_histories/snap_"+str(isnap).zfill(3)+"_parthistory_all.dat"
                    parthist_filename_sub="part_histories/snap_"+str(isnap).zfill(3)+"_parthistory_sub.dat"

                    if verbose:
                        print('Saving histories for snap = ',str(isnap),'to .dat file')

                    with open(parthist_filename_all, 'wb') as parthist_file:
                        pickle.dump(all_part_hist, parthist_file)
                        parthist_file.close()
                    with open(parthist_filename_sub, 'wb') as parthist_file:
                        pickle.dump(sub_part_hist, parthist_file)
                        parthist_file.close()

                    if verbose:                    
                        print('Done saving histories for snap = ',str(isnap),'to .dat file')

    print('Unique particle histories created')
    return {'all_ids':all_part_hist,'sub_part_ids':sub_part_hist}

########################### ACCRETION WORKER FUNCTION ###########################

def calc_accretion_rate(halo_index_list,field_bools,part_IDs_1,part_IDs_2,part_Types_2,particle_history=[],verbose=1):
    """

    calc_accretion_rate : function
	----------

    Worker function of gen_accretion_rate.
    Used to generate the number of new particles to halos from initial and final particle lists.

	Parameters
	----------
	halo_index_list : list of int
		The list of halo indices corresponding to the particle lists. 
        Non-essential for functionality, but should be of length part_IDs_1 and part_IDs_2.

    field_bools : list of bool
        List of length part_IDs_1 (and part_IDs_2, part_Types_2, halo_index_list)
        containing boolean flags (True/False) indicating whether the halo at its index is a field halo.
    
    part_IDs_1 : list of lists
        A list of length part_IDs_2, part_Types_2, field_bools, halo_index_list which at each index
        contains the list of particle IDs in the halo at that index at the initial snapshot to be
        considered for accretion calculations.

    part_IDs_2 : list of lists
        A list of length part_IDs_1, part_Types_2, field_bools, halo_index_list which at each index
        contains the list of particle IDs in the halo at that index at the final snapshot to be
        considered for accretion calculations.

    part_Types_2 : list of lists
        A list of length part_IDs_1, part_IDs_2, field_bools, halo_index_list which at each index
        contains the list of particle types in the halo at that index at the final snapshot to be
        considered for accretion calculations.

    verbose : bool
        Flag indicating how verbose we want the code to be when we run.

    Returns
	-------

    np.column_stack((halo_index_list,delta_n0,delta_n1)) : np.ndarray
        N_halo x 3 array
        Each row contains 
            [0]: halo_index from halo_index_list
            [1]: delta_n0 (number of new type 0 particles)
            [2]: delta_n1 (number of new type 1 particles)
    
    """
    #### Input checks
    # Ensure that the arrays fed are all of the correct dimension
    if len(part_IDs_1)==len(part_IDs_2):
        try:
            test=part_IDs_1[0]
            n_halos_parsed=len(halo_index_list)
        except:
            print('Particle data is not a list of lists, terminating')
            return []
        if verbose:
            print(f'Accretion rate calculator parsed {n_halos_parsed} halos')
    else:
        print('An unequal number of particle lists and/or halo indices were parsed, terminating')
        return []
    
    # Initialise outputs
    delta_n0=[]
    delta_n1=[]
    halo_indices_abs=[]

    #### Main halo loop
    for ihalo,ihalo_abs in enumerate(halo_index_list):
        #ihalo is counter, ihalo_abs is absolute halo index (at final snap)
        if verbose:
            print(f'Finding particles new to halo {ihalo_abs}')

        part_IDs_init=part_IDs_1[ihalo]
        part_IDs_final=part_IDs_2[ihalo]
        part_Types_final=part_Types_2[ihalo]
        
        part_count_1=len(part_IDs_init)
        part_count_2=len(part_IDs_final)
        # Verifying particle counts are adequate
        if part_count_2<100 or part_count_1<100:
            if verbose:
                print(f'Particle count in halo {ihalo_abs} is less than 100 - not processing')
            # if <100 particles at initial or final snap, then don't calculate accretion rate to this halo
            delta_n0.append(np.nan)
            delta_n1.append(np.nan)

        # If particle counts are adequate, then continue with calculation. 
        else:
            if verbose:
                print(f'Particle count in halo {ihalo_abs} is adequate for accretion rate calculation')

            #Finding list of particles new to the halo 
            new_particle_IDs=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_IDs_final))#list of particles new to halo
            new_particle_Types=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_Types_final))#list of particle types new to halo

            if verbose:
                print('Number of new particles to halo: ',len(new_particle_IDs))

            #Trimming particles which have been part of structure in the past (i.e. those which are already in halos)    
            
            if particle_history==[]:#if the particle history argument is empty then we are not trimming particles
                trim_particles=False
            else: #if we are given particle history, then we are trimming particles
                trim_particles=True
                allstructure_history=particle_history[0]
                substructure_history=particle_history[1]

            if trim_particles:#if we have the particle histories
                
                if len(substructure_history)<100:#if the particle history is of insufficient length then skip
                    print('Failed to find particle histories for trimming at snap = ',snap-depth-1)
                    delta_n0.append(np.nan)
                    delta_n1.append(np.nan)
                
                else:#if our particle history is valid
                    t1=time.time()

                    #reset lists which count whether a particle is valid or not (based on what its history is)
                    field_mask_good=[]
                    sub_mask_good=[]

                    if field_bools[ihalo]==True:#if a field halo then we check whether each particle has been part of ANY structure
                        for ipart in new_particle_IDs:#iterate through each new particle to the halo
                            if allstructure_history[ipart]==1:#if the particle has been part of structure, note this by invalidating
                                field_mask_good.append(False)
                            else:#if the particle is genuinely new to being in any structure, not its index as valid
                                field_mask_good.append(True)
                        if verbose:
                            print('Done cross checking particles for field halo, now compressing - keeping ',np.sum(field_mask_good),' of ',len(new_particle_IDs),' particles')
                        
                        #reduce list to the genuinely unprocessed particles
                        new_particle_Types=np.compress(field_mask_good,new_particle_Types)

                    else:#if a subhalo
                        for ipart in new_particle_IDs:
                            if substructure_history[ipart]==1:
                                sub_mask_good.append(False)
                            else:
                                sub_mask_good.append(True)
                        if verbose:
                            print('Done cross checking particles for sub halo, now compressing - keeping ',np.sum(sub_mask_good),' of ',len(new_particle_IDs),' particles')
                        
                        #reduce list to unprocessed particles
                        new_particle_Types=np.compress(sub_mask_good,new_particle_Types)

            #### Now we simply count the number of new particles of each type

            delta_n0_temp=np.sum(new_particle_Types==0)
            delta_n1_temp=np.sum(new_particle_Types==1)
            delta_n0.append(delta_n0_temp) #append the result to our final array
            delta_n1.append(delta_n1_temp) #append the result to our final array 

    return np.column_stack((halo_index_list,delta_n0,delta_n1))

########################### GENERATE ACCRETION RATES ###########################

def gen_accretion_rate(halo_data_all,snap,npart,mass_table,halo_index_list=[],depth=5,trim_particles=True,verbose=1): 
    
    """

    gen_accretion_rate : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists and (maybe) trimming particles.
    The snapshot for which this is calculated represents the final snapshot in the calculation. 

    ** note: if trimming particles, part_histories must have been generated

	Parameters
	----------
    halo_data_all : list of dictionaries
        The halo data list of dictionaries previously generated.

    snap : int
        The snapshot for which to calculate accretion rates. This will be the final snap.

    mass_table : list
        List of the particle masses in order (directly from simulation, unconverted).
    
    halo_index_list : list
        List of the halo indices for which to calculate accretion rates.

    depth : int
        How many snaps to skip back to when comparing particle lists.
        Initial snap for calculation will be snap-depth. 
    
    trim_particles: bool
        Boolean flag as to indicating whether or not to remove the particles which have previously been 
        part of structure or substructure in our accretion rate calculation. 

	Returns
	----------
    delta_m : dictionary
        Dictionary of accretion rates for each particle type to each halo at the desired snap.
        Keys: 
            "DM_Acc"
            "Gas_Acc"
            "dt"
        This data is saved for each snapshot on the way in a np.pickle file in the directory "/acc_rates"

	"""
    
    ################## Input Checks ##################

    n_halos_tot=len(halo_data_all[snap]['hostHaloID'])

    # Snap
    try:
        snap=int(snap)
    except:
        print('Invalid snap')
        return []
    
    # If the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    if not os.path.isdir("acc_rates"):
        os.mkdir("acc_rates")

    # If trimming the accretion rates we have to load the particle histories
    if trim_particles:#load particle histories if we need to
        snap_reqd=snap-depth-1#the snap before our initial snap
        try:#check if the files have already been generated
            print('Trying to find particle histories at snap = ',snap_reqd)
            parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all.dat"
            parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub.dat"
            with open(parthist_filename_all, 'rb') as parthist_file:
                allstructure_history=pickle.load(parthist_file)
                parthist_file.close()
            with open(parthist_filename_sub, 'rb') as parthist_file:
                substructure_history=pickle.load(parthist_file)
                parthist_file.close()
            print('Found particle histories')
        except:#if they haven't, generate them and load the required snap
            try:
                print(f'Did not find particle histories at snap {snap_reqd}, generating now')
                #generate particles which have been part of structure for all snaps (saved to file)
                gen_particle_history_2(halo_data_all=halo_data_all,npart=npart,snap_list=list(range(snap_reqd,len(halo_data_all))),verbose=1)
                parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all.dat"
                parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub.dat"
                with open(parthist_filename_all, 'rb') as parthist_file:
                    allstructure_history=pickle.load(parthist_file)
                    parthist_file.close()
                with open(parthist_filename_sub, 'rb') as parthist_file:
                    substructure_history=pickle.load(parthist_file)
                    parthist_file.close()             
            except:
                print('Failed to find particle histories for trimming at snap = ',snap-depth-1,', terminating')
                return []
        particle_history=[allstructure_history,substructure_history]

    else:
        particle_history=[]

    ################## Finding initial and final particle lists; organising ##################

    if verbose:
        print('Now generating accretion rates for snap = ',snap,' at depth = ',depth,' trimming = ',trim_particles)
    
    # Find progenitor index subfunction
    def find_progen_index(index_0,snap,depth):
        id_0=halo_data_all[snap]['ID'][index_0]#the original id
        tail_id=halo_data_all[snap]['Tail'][index_0]#the tail id
        for idepth in range(1,depth+1,1):
            new_id=tail_id #the new id from tail in last snap
            if new_id in halo_data_all[snap-idepth]['ID']:
                new_index=np.where(halo_data_all[snap-idepth]['ID']==new_id)[0][0] #what index in the previous snap does the new_id correspond to
                tail_id=halo_data_all[snap-idepth]['Tail'][new_index] #the new id for next loop
            else:
                new_index=np.nan
                return new_index
             #new index at snap-depth
        return new_index
    
    # If we aren't given a halo_index_list, then just calculate for all 
    if halo_index_list==[]:
        halo_index_list=list(range(n_halos_tot))

    # Find and load FINAL snap particle data
    part_data_2=get_particle_lists(snap,halo_data_snap=halo_data_all[snap],add_subparts_to_fofs=True,verbose=0)
    part_data_2_ordered_IDs=[part_data_2['Particle_IDs'][ihalo] for ihalo in halo_index_list] #just retrieve the halos we want
    part_data_2_ordered_Types=[part_data_2['Particle_Types'][ihalo] for ihalo in halo_index_list] #just retrieve the halos we want

    # Find and load INITIAL snap particle data (and ensuring they exist)
    part_data_1=get_particle_lists(snap-depth,halo_data_snap=halo_data_all[snap-depth],add_subparts_to_fofs=True,verbose=0)
    if snap-depth<0 or part_data_1["Npart"]==[]:# if we can't find initial particles
        print('Initial particle lists not found at required depth (snap = ',snap-depth,')')
        return []

    # Organise initial particle lists
    print('Organising initial particle lists')
    t1=time.time()

    part_data_1_ordered_IDs=[]#initialise empty initial particle lists
    # Iterate through each final halo and find its progenitor particle lists at the desired depth
    for ihalo_abs in halo_index_list:
        progen_index=find_progen_index(ihalo_abs,snap=snap,depth=depth)#finds progenitor index at desired snap

        if progen_index>-1:#if progenitor index is valid
            part_data_1_ordered_IDs.append(part_data_1['Particle_IDs'][progen_index])

        else:#if progenitor can't be found, make particle lists for this halo (both final and initial) empty to avoid confusion
            part_data_1_ordered_IDs.append([])
            part_data_2['Particle_IDs']=[]
            part_data_2['Particle_Types']=[]

    t2=time.time()

    print(f'Organised initial particle lists in {t2-t1} sec')

    ############################# Distributing halos to multiprocessing pool #############################

    n_halos_tot=len(halo_data_all[snap]['hostHaloID'])#number of total halos at the final snapshot in the halo_data_all dictionary
    n_halos_desired=len(halo_index_list)#number of halos for calculation desired
    field_bools=(halo_data_all[snap]['hostHaloID']==-1)#boolean mask of halos which are field

    t1=time.time()
    temp_accretion_result_array=calc_accretion_rate(halo_index_list,field_bools,part_data_1_ordered_IDs,part_data_2_ordered_IDs,part_data_2_ordered_Types,particle_history,1)
    t2=time.time()

    if verbose:
        print(f'Calculated accretion rate to {n_halos_desired} halos in {t2-t1} sec')

    delta_n0=temp_accretion_result_array[:,1]
    delta_n1=temp_accretion_result_array[:,2]

    ############################# Post-processing accretion calc results #############################

    sim_unit_to_Msun=halo_data_all[0]['UnitInfo']['Mass_unit_to_solarmass']#Simulation mass units in Msun
    m_0=mass_table[0]*sim_unit_to_Msun #parttype0 mass in Msun
    m_1=mass_table[1]*sim_unit_to_Msun #parttype1 mass in Msun
    lt2=halo_data_all[snap]['SimulationInfo']['LookbackTime']#final lookback time
    lt1=halo_data_all[snap-depth]['SimulationInfo']['LookbackTime']#initial lookback time
    delta_t=abs(lt1-lt2)#lookback time change from initial to final snapshot (Gyr)

    # Find which particle type is more massive (i.e. DM) and save accretion rates in dictionary
    # 'DM_Acc', 'Gas_Acc' and 'dt' as Msun/Gyr and dt accordingly
    if mass_table[0]>mass_table[1]:
        delta_m={'DM_Acc':np.array(delta_n0)*m_0/delta_t,'Gas_Acc':np.array(delta_n1)*m_1/delta_t,'dt':delta_t}
    else:
        delta_m={'DM_Acc':np.array(delta_n1)*m_1/delta_t,'Gas_Acc':np.array(delta_n0)*m_0/delta_t,'dt':delta_t}

    # Now save all these accretion rates to file (in directory where run /acc_rates) 
    # (with filename depending on exact calculation parameters)

    print('Saving accretion rates to .dat file.')

    if trim_particles:
        with open('acc_rates/snap_'+str(snap).zfill(3)+'_accretion_trimmed_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'w') as acc_data_file:
            pickle.dump(delta_m,acc_data_file)
            acc_data_file.close()
    else:
        with open('acc_rates/snap_'+str(snap).zfill(3)+'_accretion_base_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'w') as acc_data_file:
            pickle.dump(delta_m,acc_data_file)
            acc_data_file.close()

    #return the delta_m dictionary. 
    return delta_m

########################### HALO INDEX LISTS GENERATOR ###########################

def gen_halo_indices_mp(all_halo_indices,n_processes):
    """

    gen_halo_indices_mp : function
	----------

    Generate list of lists of halo indices divided amongst a given amount of processes.

	Parameters
	----------
    all_halo_indices : list or int
        If list, a list of integer halo indices to divide.
        If int, a list of integer halo indices up to the int is generated.

    n_processes : int
        Number of processes (likely number of cores) to distribute halo indices across. 

    Returns
	----------
    halo_index_lists : list of lists
        The resulting halo index lists for each process. 

    """
    # Create halo index list from integer or provided list
    if type(all_halo_indices)==int:
        all_halo_indices=list(range(all_halo_indices))
    else:
        all_halo_indices=list(all_halo_indices)

    n_halos=len(all_halo_indices)
    halo_rem=n_halos%n_processes
    n_halos_per_process=int(n_halos/n_processes)

    #initialising loop variables
    last_index=0
    index_lists=[]
    halo_index_lists=[]

    #loop for each process to generate halo index lists
    for iprocess in range(n_processes):
        if halo_rem==0: #if there's an exact multiple of halos as cpu cores then distribute evenly
            indices_temp=list(range(iprocess*n_halos_per_process,(iprocess+1)*n_halos_per_process))
            index_lists.append(indices_temp)
            halo_index_list_temp=[all_halo_indices[index_temp] for index_temp in indices_temp]
            halo_index_lists.append(halo_index_list_temp)

        else: #otherwise split halos evenly except last process
            if iprocess<halo_rem:
                indices_temp=list(range(last_index,last_index+n_halos_per_process+1))
                index_lists.append(indices_temp)
                last_index=indices_temp[-1]+1
                halo_index_list_temp=[all_halo_indices[index_temp] for index_temp in indices_temp]
                halo_index_lists.append(halo_index_list_temp)

            else:
                indices_temp=list(range(last_index,last_index+n_halos_per_process))
                index_lists.append(indices_temp)
                last_index=indices_temp[-1]+1
                halo_index_list_temp=[all_halo_indices[index_temp] for index_temp in indices_temp]
                halo_index_lists.append(halo_index_list_temp)

    return halo_index_lists

########################### ACCRETION RATE FILE HANDLER ###########################

def gen_filename_dataframe(directory):


    """

    gen_filename_dataframe : function
	----------

    Generates a pandas dataframe of the filenames and associated characteristics of saved accretion rate files. 

	Parameters
	----------
    directory : str
        Where to search for accretion rate files. 

    Returns
	----------
    filename_dataframe : pd.DataFrame
        DataFrame containing the keys listed below.

        Keys

            'filename': filename string
            'type': 0 (base), 1 (trimmed)
            'depth': snap gap
            'span': n_halos per process in generation
            'index1': first halo index
            'index2': final halo index

    """
    
    desired_file_list=os.listdir(directory)
    is_data_file=[]

    for filename in desired_file_list:
        is_data_file.append(filename.endswith('.dat'))
    desired_file_list=np.compress(is_data_file,desired_file_list) #the data files

    #initialise results
    snaps=[]
    base_or_trim=[]
    depths=[]
    halo_range_1=[]
    halo_range_2=[]
    spans=[]

    #iterate through each of the data files
    for filename in desired_file_list:
        file_split=filename.split('_')
        snaps.append(int(file_split[1]))
        base_or_trim.append(int(file_split[3][0]=="t"))
        depths.append(int(file_split[4][-1]))
        halo_range_temp=np.array(file_split[5][:-4].split('-')).astype(int)
        halo_range_1.append(halo_range_temp[0])
        halo_range_2.append(halo_range_temp[1])
        spans.append(halo_range_temp[1]-halo_range_temp[0]+1)

    #create data frame and order according to type, depth, then snap
    filename_dataframe={'filename':desired_file_list,'snap':snaps,'type':base_or_trim,'depth':depths,'index1':halo_range_1,'index2':halo_range_2,'span':spans}
    filename_dataframe=df(filename_dataframe)
    filename_dataframe=filename_dataframe.sort_values(by=['type','depth','snap','span','index1'])

    return filename_dataframe

########################### ACCRETION RATE LOADER ###########################
def load_accretion_rate(directory,calc_type,snap,depth,span,verbose=1):

    filename_dataframe=gen_filename_dataframe(directory)
    relevant_files=list(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==snap,filename_dataframe['depth']==depth,filename_dataframe['span']==span))]['filename'])
    index1=list(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==snap,filename_dataframe['depth']==depth,filename_dataframe['span']==span))]['index1'])
    index2=list(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==snap,filename_dataframe['depth']==depth,filename_dataframe['span']==span))]['index2'])
    
    print(filename_dataframe)
    print(relevant_files)

    if verbose:
        print(f'Found {len(relevant_files)} accretion rate files (snap = {snap}, type = {calc_type}, depth = {depth}, span = {span})')

    acc_rate_dataframe=df({'ihalo':[],'DM_Acc':[],'Gas_Acc':[],'dt':[]})

    for ifile,ifilename in enumerate(relevant_files):

        print(directory+ifilename)
        halo_indices=list(range(index1[ifile],index2[ifile]))
        print(len(halo_indices))
        with open(directory+ifilename,'rb') as acc_rate_file:
            dataframe_temp=pickle.load(acc_rate_file)
            dataframe_temp=df(dataframe_temp)
            dataframe_temp['ihalo']=halo_indices
            acc_rate_file.close()
        acc_rate_dataframe.append(dataframe_temp)

    return acc_rate_dataframe
        


