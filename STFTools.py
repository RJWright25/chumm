#########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data & Halo Properties ##########################################
#########################################################################################################################################################################

#*** Preamble ***
import os
import numpy as np
import h5py
import pickle
import astropy.units as u
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

    with open('halo_data_base.dat', 'wb') as halo_data_file:
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

########################### CREATE PARTICLE HISTORIES ###########################

def gen_particle_history(halo_data_all,verbose=1):

    """

    gen_particle_history : function
	----------

    Generate and save particle history data from velociraptor property and particle files.

	Parameters
	----------
    halo_data_all : list of dictionaries
        The halo data list of dictionaries previously generated.

	Returns
	----------
    {'all_ids':running_list_all,'sub_ids':running_list_sub} : dict
        Dictionary of particle lists which have ever been part of any halo and those which 
        have been part of a subhalo at any point up to the last snap in the halo_data_all array.

        This data is saved for each snapshot on the way in a np.pickle file in the directory "part_histories"

	"""
    ### input checks
    #snaps
    try:
        no_snaps=len(halo_data_all)
    except:
        print("Invalid halo data")
        return []

    # if the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    if not os.path.isdir("part_histories"):
        os.mkdir("part_histories")

    print('Generating particle histories up to snap = ',no_snaps)

    running_list_all=[]
    running_list_sub=[]
    sub_part_ids=[]
    all_part_ids=[]

    # for each snapshot get the particle data and add to the running list

    for isnap in range(no_snaps):

        new_particle_data=get_particle_lists(snap=isnap,halo_data_snap=halo_data_all[isnap],add_subparts_to_fofs=False,verbose=verbose)
        
        if len(new_particle_data['Particle_IDs'])==0 or len(halo_data_all[isnap]['hostHaloID'])<2:#if no halos or no new particle data
            if verbose:
                print('Either no particle data or no halos for snap = ',isnap)
            continue

        else:
            if verbose:
                print('Have particle lists for snap = ',isnap)
                        
            sub_halos_temp=(np.where(halo_data_all[isnap]['hostHaloID']>0)[0])#find the indices all subhalos

            if len(sub_halos_temp)>1:
                all_halos_plist=np.concatenate(new_particle_data['Particle_IDs'])
                sub_halos_plist=np.concatenate([new_particle_data['Particle_IDs'][isub] for isub in sub_halos_temp])#list all particles IDs in substructure
                    
                running_list_all=np.concatenate([running_list_all,all_halos_plist])
                running_list_sub=np.concatenate([running_list_sub,sub_halos_plist])

                running_list_all=np.unique(running_list_all)
                running_list_sub=np.unique(running_list_sub)

                parthist_filename_all="part_histories/snap_"+str(isnap).zfill(3)+"_parthistory_all.dat"
                parthist_filename_sub="part_histories/snap_"+str(isnap).zfill(3)+"_parthistory_sub.dat"

                if verbose:
                    print('Saving histories for snap = ',str(isnap),'to .dat file.')

                with open(parthist_filename_all, 'wb') as parthist_file:
                    pickle.dump(running_list_all, parthist_file)
                    parthist_file.close()
                with open(parthist_filename_sub, 'wb') as parthist_file:
                    pickle.dump(running_list_sub, parthist_file)
                    parthist_file.close()                    
                
    print('Unique particle histories created')
    return {'all_ids':running_list_all,'sub_ids':running_list_sub}

########################### GENERATE ACCRETION RATES ###########################

def gen_accretion_rate(halo_data_all,snap,mass_table,halo_cap=[],halo_index_list=[],depth=5,trim_particles=True,verbose=1): 
    
    """

    gen_accretion_rate : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists and (maybe) trimming particles.
    note: if trimming particles, part_histories must have been generated

	Parameters
	----------
    halo_data_all : list of dictionaries
        The halo data list of dictionaries previously generated.

    snap : int
        The snapshot for which to calculate accretion rates

    mass_table : list
        List of the particle masses in order (directly from simulation, unconverted)
    
    halo_index_list : list
        List of the halo indices for which to calculate accretion rates.

    depth : int
        How many snaps to skip back to when comparing particle lists. 
    
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
    
    ###input checks
    #snap
    try:
        snap=int(snap)
    except:
        print('Invalid snap')
        return []
    
    #if the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    if not os.path.isdir("acc_rates"):
        os.mkdir("acc_rates")

    #converting mass table to msun
    sim_unit_to_Msun=halo_data_all[0]['UnitInfo']['Mass_unit_to_solarmass']
    m_0=mass_table[0]*sim_unit_to_Msun #MSol
    m_1=mass_table[1]*sim_unit_to_Msun #MSol

    if trim_particles:#load particle histories if we need to
        snap_reqd=snap-depth-1
        try:##check if the files have already been generated
            print('Trying to find particle histories at snap = ',snap_reqd)
            parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all.dat"
            parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub.dat"
            with open(parthist_filename_all, 'rb') as parthist_file:
                allstructure_history=pickle.load(parthist_file)
                parthist_file.close()
            with open(parthist_filename_sub, 'rb') as parthist_file:
                substructure_history=pickle.load(parthist_file)
                parthist_file.close()
            print('Found them!')

        except:#if they haven't, generate them and load the required snap
            try:
                print('Did not find particle histories -- generating them now')       
                gen_particle_history(halo_data_all=halo_data_all,verbose=0)#generate particles which have been part of structure for all snaps (saved to file)
                parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all.dat"
                parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub.dat"
                with open(parthist_filename_all, 'rb') as parthist_file:
                    allstructure_history=pickle.load(parthist_file)
                    parthist_file.close()
                with open(parthist_filename_sub, 'rb') as parthist_file:
                    substructure_history=pickle.load(parthist_file)
                    parthist_file.close()             
            except:
                print('Failed to find particle histories for trimming at snap = ',snap-depth-1,'. Terminating.')
                return []

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

    isnap=-1
    isnap=isnap+1    

    if verbose:
        print('Now generating accretion rates for snap = ',snap,' at depth = ',depth,' trimming = ',trim_particles)

    #find final snap particle data
    part_data_2=get_particle_lists(snap,halo_data_snap=halo_data_all[snap],add_subparts_to_fofs=True,verbose=0)
    if not halo_index_list==[]:
        halo_index_list=halo_index_list
        n_halos_2=len(halo_index_list)
    elif halo_cap==[]:
        print("Finding accretion rates for all halos")
        n_halos_2=len(part_data_2["Npart"])
        halo_index_list=list(range(n_halos_2))
    else:
        n_halos_2=halo_cap
        halo_index_list=list(range(n_halos_2))

    if n_halos_2==0:# if we can't find final particles or there are no halos
        print('Final particle lists not found at snap = ',snap)
        return []

    #find initial snap particle data
    part_data_1=get_particle_lists(snap-depth,halo_data_snap=halo_data_all[snap-depth],add_subparts_to_fofs=True,verbose=0)
    if snap-depth<0 or part_data_1["Npart"]==[]:# if we can't find initial particles
        print('Initial particle lists not found at required depth (snap = ',snap-depth,')')
        return []

    delta_m0=[]
    delta_m1=[]

    for ihalo in halo_index_list:#for each halo

        if verbose:
            print('Done with accretion rates for ',ihalo,' halos out of ',n_halos_2)

        progen_index=find_progen_index(index_0=ihalo,snap=snap,depth=depth)#find progen index
        if progen_index>-1:#if progen_index is valid
            part_IDs_init=part_data_1['Particle_IDs'][progen_index]
            part_IDs_final=part_data_2['Particle_IDs'][ihalo]
            part_Types_init=part_data_1['Particle_Types'][progen_index]
            part_Types_final=part_data_2['Particle_Types'][ihalo]
        else:
            delta_m0.append(np.nan)
            delta_m1.append(np.nan)
            continue

        new_particle_IDs=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_IDs_final))#list of particles new to halo
        new_particle_Types=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_Types_final))#list of particle types new to halo

        if verbose:
            print('Number of new particles to halo: ',len(new_particle_IDs))

        ################# TRIMMING PARTICLES #################
        #get particle histories for the snap depth (minus 1)
        if trim_particles:
            if len(substructure_history)<100:
                print('Failed to find particle histories for trimming at snap = ',snap-depth-1)

            t1=time.time()
            if halo_data_all[snap]['hostHaloID'][ihalo]==-1:#if a field halo
                field_mask_good=np.in1d(new_particle_IDs,allstructure_history,invert=True)
                if verbose:
                    print('Done cross checking, now compressing')
                new_particle_Types=np.compress(field_mask_good,new_particle_Types)

            else:#if a subhalo
                sub_mask_good=np.in1d(new_particle_IDs,substructure_history,invert=True)
                if verbose:
                    print('Done cross checking, now compressing')
                new_particle_Types=np.compress(sub_mask_good,new_particle_Types)
            t2=time.time()

        ########### NOW WE HAVE THE DESIRED NEW (UNIQUE) PARTICLES FOR EACH HALO ###########
        delta_m0_temp=np.sum(new_particle_Types==0)*m_0
        delta_m1_temp=np.sum(new_particle_Types==1)*m_1
        delta_m0.append(delta_m0_temp)
        delta_m1.append(delta_m1_temp)
        ####################################################################################

    lt2=halo_data_all[snap]['SimulationInfo']['LookbackTime']
    lt1=halo_data_all[snap-depth]['SimulationInfo']['LookbackTime']
    delta_t=abs(lt1-lt2)#Gyr

    if mass_table[0]>mass_table[1]:#make sure m_dm is more massive (the more massive particle should be the dm particle)
        delta_m={'DM_Acc':np.array(delta_m0)/delta_t,'Gas_Acc':np.array(delta_m1)/delta_t,'dt':delta_t}
    else:
        delta_m={'DM_Acc':np.array(delta_m1)/delta_t,'Gas_Acc':np.array(delta_m0)/delta_t,'dt':delta_t}

    #### save to file.
    print('Saving accretion rates to .dat file.')
    if trim_particles:
        with open('acc_rates/snap_'+str(snap).zfill(3)+'_accretion_trimmed_'+str(depth)+'_'+str(n_halos_2)+'.dat', 'wb') as acc_data_file:
            pickle.dump(delta_m,acc_data_file)
            acc_data_file.close()
    else:
        with open('acc_rates/snap_'+str(snap).zfill(3)+'_accretion_base_'+str(depth)+'_'+str(n_halos_2)+'.dat', 'wb') as acc_data_file:
            pickle.dump(delta_m,acc_data_file)
            acc_data_file.close()
    return delta_m





########################### CREATE PARTICLE HISTORIES NEW ###########################
def gen_particle_history_2(halo_data_all,npart,verbose=1):

    """

    gen_particle_history_2 : function
	----------

    Generate and save particle history data from velociraptor property and particle files.

	Parameters
	----------
    halo_data_all : list of dictionaries
        The halo data list of dictionaries previously generated.

	Returns
	----------
        Dictionary of particle lists (each length n_particles) indicating true/false as to whether 
        the particle at that index has ever been part of any halo or subhalo.

        This data is saved for each snapshot on the way in a np.pickle file in the directory "part_histories"

	"""
    ### input checks
    #snaps
    try:
        no_snaps=int(halo_data_all[-1]['Snap'])
    except:
        print("Invalid halo data")
        return []

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

        new_particle_data=get_particle_lists(snap=isnap,halo_data_snap=halo_data_all[isnap],add_subparts_to_fofs=False,verbose=verbose)
        
        if len(new_particle_data['Particle_IDs'])==0 or len(halo_data_all[isnap]['hostHaloID'])<2:#if no halos or no new particle data
            if verbose:
                print('Either no particle data or no halos for snap = ',isnap)
            continue

        else:
            if verbose:
                print('Have particle lists for snap = ',isnap)
                        
            sub_halos_temp=(np.where(halo_data_all[isnap]['hostHaloID']>0)[0])#find the indices all subhalos

            if len(sub_halos_temp)>1:
                all_halos_plist=np.concatenate(new_particle_data['Particle_IDs'])
                sub_halos_plist=np.concatenate([new_particle_data['Particle_IDs'][isub] for isub in sub_halos_temp])#list all particles IDs in substructure
                new_structure_indices=np.array(np.compress(np.logical_not(np.in1d(all_halos_plist,running_list_all)),all_halos_plist)).astype(str)
                new_substructure_indices=np.array(np.compress(np.logical_not(np.in1d(sub_halos_plist,running_list_sub)),sub_halos_plist)).astype(str)

                running_list_all=np.concatenate([running_list_all,all_halos_plist])
                running_list_sub=np.concatenate([running_list_sub,sub_halos_plist])
                running_list_all=np.unique(running_list_all)
                running_list_sub=np.unique(running_list_sub)

                for new_part_structure in new_structure_indices:
                    all_part_hist[int(new_part_structure)]=1

                for new_part_substructure in new_substructure_indices:
                    sub_part_hist[int(new_part_substructure)]=1

                parthist_filename_all="part_histories/snap_"+str(isnap).zfill(3)+"_parthistory_all_2.dat"
                parthist_filename_sub="part_histories/snap_"+str(isnap).zfill(3)+"_parthistory_sub_2.dat"

                if verbose:
                    print('Saving histories for snap = ',str(isnap),'to .dat file.')

                with open(parthist_filename_all, 'wb') as parthist_file:
                    pickle.dump(all_part_hist, parthist_file)
                    parthist_file.close()
                with open(parthist_filename_sub, 'wb') as parthist_file:
                    pickle.dump(sub_part_hist, parthist_file)
                    parthist_file.close()                    
                
    print('Unique particle histories created')
    return {'all_ids':all_part_hist,'sub_part_ids':sub_part_hist}




########################### GENERATE ACCRETION RATES 2 ###########################

def gen_accretion_rate_2(halo_data_all,snap,npart,mass_table,halo_cap=[],halo_index_list=[],depth=5,trim_particles=True,verbose=1): 
    
    """

    gen_accretion_rate : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists and (maybe) trimming particles.
    note: if trimming particles, part_histories must have been generated

	Parameters
	----------
    halo_data_all : list of dictionaries
        The halo data list of dictionaries previously generated.

    snap : int
        The snapshot for which to calculate accretion rates

    mass_table : list
        List of the particle masses in order (directly from simulation, unconverted)
    
    halo_index_list : list
        List of the halo indices for which to calculate accretion rates.

    depth : int
        How many snaps to skip back to when comparing particle lists. 
    
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
    
    ###input checks
    #snap
    try:
        snap=int(snap)
    except:
        print('Invalid snap')
        return []
    
    #if the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    if not os.path.isdir("acc_rates"):
        os.mkdir("acc_rates")

    #converting mass table to msun
    sim_unit_to_Msun=halo_data_all[0]['UnitInfo']['Mass_unit_to_solarmass']
    m_0=mass_table[0]*sim_unit_to_Msun #MSol
    m_1=mass_table[1]*sim_unit_to_Msun #MSol

    if trim_particles:#load particle histories if we need to
        snap_reqd=snap-depth-1
        try:##check if the files have already been generated
            print('Trying to find particle histories at snap = ',snap_reqd)
            parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all_2.dat"
            parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub_2.dat"
            with open(parthist_filename_all, 'rb') as parthist_file:
                allstructure_history=pickle.load(parthist_file)
                parthist_file.close()
            with open(parthist_filename_sub, 'rb') as parthist_file:
                substructure_history=pickle.load(parthist_file)
                parthist_file.close()
            print('Found them!')

        except:#if they haven't, generate them and load the required snap
            #try:
            print('Did not find particle histories -- generating them now')       
            gen_particle_history_2(halo_data_all=halo_data_all,npart=npart,verbose=1)#generate particles which have been part of structure for all snaps (saved to file)
            parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all_2.dat"
            parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub_2.dat"
            with open(parthist_filename_all, 'rb') as parthist_file:
                allstructure_history=pickle.load(parthist_file)
                parthist_file.close()
            with open(parthist_filename_sub, 'rb') as parthist_file:
                substructure_history=pickle.load(parthist_file)
                parthist_file.close()             
            # except:
            #     print('Failed to find particle histories for trimming at snap = ',snap-depth-1,'. Terminating.')
            #     return []

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

    isnap=-1
    isnap=isnap+1    

    if verbose:
        print('Now generating accretion rates for snap = ',snap,' at depth = ',depth,' trimming = ',trim_particles)

    #find final snap particle data
    part_data_2=get_particle_lists(snap,halo_data_snap=halo_data_all[snap],add_subparts_to_fofs=True,verbose=0)
    if not halo_index_list==[]:
        halo_index_list=halo_index_list
        n_halos_2=len(halo_index_list)
    elif halo_cap==[]:
        print("Finding accretion rates for all halos")
        n_halos_2=len(part_data_2["Npart"])
        halo_index_list=list(range(n_halos_2))
    else:
        n_halos_2=halo_cap
        halo_index_list=list(range(n_halos_2))

    if n_halos_2==0:# if we can't find final particles or there are no halos
        print('Final particle lists not found at snap = ',snap)
        return []

    #find initial snap particle data
    part_data_1=get_particle_lists(snap-depth,halo_data_snap=halo_data_all[snap-depth],add_subparts_to_fofs=True,verbose=0)
    if snap-depth<0 or part_data_1["Npart"]==[]:# if we can't find initial particles
        print('Initial particle lists not found at required depth (snap = ',snap-depth,')')
        return []

    delta_m0=[]
    delta_m1=[]

    for ihalo in halo_index_list:#for each halo

        if verbose:
            print('Done with accretion rates for ',ihalo,' halos out of ',n_halos_2)

        progen_index=find_progen_index(index_0=ihalo,snap=snap,depth=depth)#find progen index
        if progen_index>-1:#if progen_index is valid
            part_IDs_init=part_data_1['Particle_IDs'][progen_index]
            part_IDs_final=part_data_2['Particle_IDs'][ihalo]
            part_Types_init=part_data_1['Particle_Types'][progen_index]
            part_Types_final=part_data_2['Particle_Types'][ihalo]
        else:
            delta_m0.append(np.nan)
            delta_m1.append(np.nan)
            continue

        new_particle_IDs=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_IDs_final))#list of particles new to halo
        new_particle_Types=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_Types_final))#list of particle types new to halo

        if verbose:
            print('Number of new particles to halo: ',len(new_particle_IDs))

        ################# TRIMMING PARTICLES #################
        #get particle histories for the snap depth (minus 1)

        if trim_particles:
            if len(substructure_history)<100:
                print('Failed to find particle histories for trimming at snap = ',snap-depth-1)

            t1=time.time()
            field_mask_good=[]
            sub_mask_good=[]

            if halo_data_all[snap]['hostHaloID'][ihalo]==-1:#if a field halo
                for ipart in new_particle_IDs:
                    if allstructure_history[ipart]==1:
                        field_mask_good.append(False)
                    else:
                        field_mask_good.append(True)
                if verbose:
                    print('Done cross checking particles for field halo, now compressing - keeping ',np.sum(field_mask_good),' of ',len(new_particle_IDs),' particles')
                new_particle_Types=np.compress(field_mask_good,new_particle_Types)

            else:#if a subhalo
                for ipart in new_particle_IDs:
                    if substructure_history[ipart]==1:
                        sub_mask_good.append(False)
                    else:
                        sub_mask_good.append(True)
                if verbose:
                    print('Done cross checking particles for sub halo, now compressing - keeping ',np.sum(sub_mask_good),' of ',len(new_particle_IDs),' particles')
                new_particle_Types=np.compress(sub_mask_good,new_particle_Types)
            t2=time.time()

        ########### NOW WE HAVE THE DESIRED NEW (UNIQUE) PARTICLES FOR EACH HALO ###########
        delta_m0_temp=np.sum(new_particle_Types==0)*m_0
        delta_m1_temp=np.sum(new_particle_Types==1)*m_1
        delta_m0.append(delta_m0_temp)
        delta_m1.append(delta_m1_temp)
        ####################################################################################

    lt2=halo_data_all[snap]['SimulationInfo']['LookbackTime']
    lt1=halo_data_all[snap-depth]['SimulationInfo']['LookbackTime']
    delta_t=abs(lt1-lt2)#Gyr

    if mass_table[0]>mass_table[1]:#make sure m_dm is more massive (the more massive particle should be the dm particle)
        delta_m={'DM_Acc':np.array(delta_m0)/delta_t,'Gas_Acc':np.array(delta_m1)/delta_t,'dt':delta_t}
    else:
        delta_m={'DM_Acc':np.array(delta_m1)/delta_t,'Gas_Acc':np.array(delta_m0)/delta_t,'dt':delta_t}

    #### save to file.
    print('Saving accretion rates to .dat file.')
    if trim_particles:
        with open('acc_rates/snap_'+str(snap).zfill(3)+'_accretion_trimmed_'+str(depth)+'_'+str(n_halos_2)+'_2.dat', 'wb') as acc_data_file:
            pickle.dump(delta_m,acc_data_file)
            acc_data_file.close()
    else:
        with open('acc_rates/snap_'+str(snap).zfill(3)+'_accretion_base_'+str(depth)+'_'+str(n_halos_2)+'_2.dat', 'wb') as acc_data_file:
            pickle.dump(delta_m,acc_data_file)
            acc_data_file.close()
    return delta_m

