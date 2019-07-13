#########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data & Halo Properties ##########################################
#########################################################################################################################################################################

#*** Preamble ***
import os
from os import path
import numpy as np
import h5py
import pickle
from pandas import DataFrame as df
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM,z_at_value
from scipy.spatial import KDTree

# VELOCIraptor python tools 
from VRPythonTools import *

########################### CREATE BASE HALO DATA ###########################

def gen_base_halo_data(snaps=[],outname='',vr_filelist="",tf_filelist="",vr_files_type=2,halo_TEMPORALHALOIDVAL=[],verbose=1):
    
    """

    gen_base_halo_data : function
	----------

    Generate halo data from velociraptor property and particle files.

	Parameters
	----------
    snaps: list of str
        The list of (absolute) snaps we are creating halo data for (same length/order as vr_filelist, tf_filelist)

    outname : str
        Suffix for output file. 

    vr_filelist : string
        The full path to the text file in which the velociraptor data files are listed.

    tf_filelist : string
        The full path to the text file in which the tree data files are listed.
    
    vr_files_type : int
        The filetype of the VELOCIraptor inputs: (2 = hdf5)

    halo_TEMPORALHALOIDVAL : int
        The multiplier used by VELOCIraptor to create unique temporal halo IDs. 

    verbose : bool
        Flag indicating how verbose we want the code to be when we run.

    Returns
	-------
	base1_vrhalodata_outname : list
	base2_vrhalodata_outname : list

        A list (for each snap desired) of dictionaries which contain halo data with the following fields:
        'ID'
        'hostHaloID'
        'Snap'
        'Head'*
        'Tail'
        'HeadSnap'*
        'TailSnap'*
        'RootHead'*
        'RootTail'*
        'RootHeadSnap'*
        'RootTailSnap'*
        'HeadRank'*
        'Num_descen'*
        'Num_progen'*
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

    * items will be removed from base1 file.

	"""

    ###### WILL SAVE TO FILE THE HALO DATA WITH FORM: halo_data_all.dat

    halo_data_all=[]

    halo_fields=['ID','hostHaloID']#default halo fields

    # velociraptor lists
    with open(vr_filelist,'rb') as vr_file:
        vr_list=np.loadtxt(vr_filelist,dtype=str)
        vr_file.close()

    # treefrog lsits
    with open(tf_filelist,'rb') as tf_file:
        tf_list=np.loadtxt(tf_filelist,dtype=str)
        tf_file.close()

    # check whether we have the same amount of TF files as VR files
    if len(vr_list)==len(tf_list):
        pass
    else:
        print("VELOCIraptor files and TreeFrog files don't match in length")
        print(f"VR: {vr_list}")
        print(f"TF: {tf_list}")
        return([])
    
    if snaps==[]: #if no snaps specified, just use th length of the VR/TF catalogue 
        sim_snaps=list(range(len(vr_list)))
        if verbose:
            print("Looking for snaps up to 1000")
    elif type(snaps)==list: #if we're given a non-empty list
        sim_snaps=snaps

    snap_no=len(sim_snaps)

    print('Reading halo data using VR python tools')
    #for each snap specified, we will generate halo data
    for isnap,snap in enumerate(sim_snaps):
        if verbose:
            print('Searching for halo data at snap = ',snap)
            print('File: '+vr_list[isnap])
           
        #use VR python tools to load in halo data for this snap
        halo_data_snap=ReadPropertyFile(vr_list[isnap],ibinary=vr_files_type,iseparatesubfiles=0,iverbose=0, desiredfields=halo_fields, isiminfo=True, iunitinfo=True)
        
        #if data is found
        if not halo_data_snap==[]:
            halo_data_all.append(halo_data_snap)
            halo_data_all[isnap][0]['Snap']=snap

        #if data is not found
        else:
            if verbose:
                print("Couldn't find velociraptor files for snap = ",snap)

    # List of number of halos detected for each snap and list isolated data dictionary for each snap (in dictionaries)
    halo_data_counts=[item[1] for item in halo_data_all]
    halo_data_all=[item[0] for item in halo_data_all]

    # Add halo count to halo data at each snap
    for isnap,snap in enumerate(sim_snaps):
        halo_data_all[isnap]['Count']=halo_data_counts[isnap]

    # List sim info and unit info for each snap (in dictionaries)
    halo_siminfo=[halo_data_all[isnap]['SimulationInfo'] for isnap in range(snap_no)]
    halo_unitinfo=[halo_data_all[isnap]['UnitInfo'] for isnap in range(snap_no)]

    # Import tree data from TreeFrog, build temporal head/tails from descendants -- adds to halo_data_all (all halo data)
    print('Now assembling descendent tree using VR python tools')

    # Read in tree data
    halo_tree=ReadHaloMergerTreeDescendant(tf_filelist,ibinary=vr_files_type,iverbose=verbose+1,imerit=True,inpart=False)

    # Now build trees and add onto halo data array
    if halo_TEMPORALHALOIDVAL==[]:#if not given halo TEMPORALHALOIVAL, use the vr default
        BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=verbose)
    else:
        BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=verbose,TEMPORALHALOIDVAL=halo_TEMPORALHALOIDVAL)
    
    print('Finished assembling descendent tree using VR python tools')

    if verbose==1:
        print('Adding timesteps & filepath information')
    
    # Adding timesteps and filepath information 
    H0=halo_data_all[0]['SimulationInfo']['h_val']*halo_data_all[0]['SimulationInfo']['Hubble_unit']
    Om0=halo_data_all[0]['SimulationInfo']['Omega_Lambda']
    cosmo=FlatLambdaCDM(H0=H0,Om0=Om0)

    for isnap,snap in enumerate(sim_snaps):
        scale_factor=halo_data_all[isnap]['SimulationInfo']['ScaleFactor']
        redshift=z_at_value(cosmo.scale_factor,scale_factor,zmin=-0.5)
        lookback_time=cosmo.lookback_time(redshift).value

        halo_data_all[isnap]['SimulationInfo']['z']=redshift
        halo_data_all[isnap]['SimulationInfo']['LookbackTime']=lookback_time

        halo_data_all[isnap]['FilePath']=vr_list[isnap]
        halo_data_all[isnap]['FileType']=vr_files_type

    print('Saving base2 halo data to file (contains detailed TreeFrog data)')

    ###### SAVE all data (with detailed TF) to file
    if path.exists('base2_vrhalodata_'+outname+'.dat'):
        if verbose:
            print('Overwriting existing base2 halo data ...')
        os.remove('base2_vrhalodata_'+outname+'.dat')

    with open('base2_vrhalodata_'+outname+'.dat', 'wb') as halo_data_file:
        pickle.dump(halo_data_all, halo_data_file)
        halo_data_file.close()

    print('Saving base1 halo data to file (removing detailed TreeFrog data)')

    ###### Remove superfluous data for acc_rate calcss

    fields_to_keep=['Count','ID','hostHaloID','Tail','FilePath','FileType','Snap','UnitInfo','SimulationInfo']
    halo_data_all_truncated=[]
    for isnap,halo_data_snap in enumerate(halo_data_all):
        halo_data_all_truncated_snap={}
        for field in fields_to_keep:
            halo_data_all_truncated_snap[field]=halo_data_snap[field]
        halo_data_all_truncated.append(halo_data_all_truncated_snap)

    ###### SAVE trimmed data to file
    if path.exists('base1_vrhalodata_'+outname+'.dat'):
        if verbose:
            print('Overwriting existing base1 halo data ...')
        os.remove('base1_vrhalodata_'+outname+'.dat')

    with open('base1_vrhalodata_'+outname+'.dat', 'wb') as halo_data_file:
        pickle.dump(halo_data_all_truncated, halo_data_file)
        halo_data_file.close()

    print('Done generating base halo data')

    return halo_data_all

########################### ADD DETAILED HALO DATA ###########################

def gen_detailed_halo_data(base_halo_data,extra_halo_fields=[],outname='',verbose=True):
    
    """
    
    gen_detailed_halo_data : function
	----------

    Add detailed halo data to base halo data from property files.

    Parameters
    ----------

    base_halo_data : list of dicts

        List (for each snap) of dictionaries containing basic halo data generated from gen_base_halo_data. 

    extra_halo_fields : list of str

        List of dictionary keys for halo properties to be added to the base halo data. 

    outname : str

        Suffix for halo data to be saved as. 

    Returns
    --------

    detailed_vrhalodata_outname : list of dict

    A list (for each snap desired) of dictionaries which contain halo data with the following fields:
        'ID'
        'hostHaloID'
        'Snap'
        'Head'
        'Tail'

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

        AND ANY EXTRAS from extra_halo_fields

	"""

    if extra_halo_fields==[]:
        property_filename=base_halo_data[-1]['FilePath']+".properties.0"
        property_file=h5py.File(property_filename)
        all_props=list(property_file.keys())
        all_props.remove('ID')
        all_props.remove('hostHaloID')
        extra_halo_fields=all_props

    new_halo_data=[]

    #loop through each snap and add the extra fields
    for isnap,base_halo_data_snap in enumerate(base_halo_data):

        if verbose:
            print(f'Adding detailed halo data for snap = ',isnap)

        new_halo_data_snap={}
        halo_data_snap=ReadPropertyFile(base_halo_data_snap['FilePath'],ibinary=base_halo_data_snap["FileType"],iseparatesubfiles=0,iverbose=0, desiredfields=extra_halo_fields, isiminfo=True, iunitinfo=True)[0]

        for halo_field in extra_halo_fields:
            new_halo_data_snap[halo_field]=halo_data_snap[halo_field]
        
        for halo_field in list(base_halo_data_snap.keys()):
            new_halo_data_snap[halo_field]=base_halo_data_snap[halo_field]

        if verbose:
            print('Adding R_rel information for subhalos and number densities')

        n_halos_snap=len(new_halo_data_snap['ID'])
        new_halo_data_snap['R_rel']=np.zeros(len(new_halo_data_snap['ID']))+np.nan
        new_halo_data_snap['N_peers']=np.zeros(len(new_halo_data_snap['ID']))+np.nan

        if n_halos_snap>2:
            for ihalo in range(n_halos_snap):
                hostID_temp=new_halo_data_snap['hostHaloID'][ihalo]
                if not hostID_temp==-1:
                    #if we have a subhalo
                    N_peers=np.sum(new_halo_data_snap['hostHaloID']==hostID_temp)-1
                    new_halo_data_snap['N_peers'][ihalo]=N_peers   
            
                    hostindex_temp=np.where(new_halo_data_snap['ID']==hostID_temp)[0][0]
                    host_radius=new_halo_data_snap['R_200crit'][hostindex_temp]
                    host_xyz=np.array([new_halo_data_snap['Xc'][hostindex_temp],new_halo_data_snap['Yc'][hostindex_temp],new_halo_data_snap['Zc'][hostindex_temp]])
                    sub_xy=np.array([new_halo_data_snap['Xc'][ihalo],new_halo_data_snap['Yc'][ihalo],new_halo_data_snap['Zc'][ihalo]])
                    group_centric_r=np.sqrt(np.sum((host_xyz-sub_xy)**2))
                    r_rel_temp=group_centric_r/host_radius
                    new_halo_data_snap['R_rel'][ihalo]=r_rel_temp

            if verbose:
                print('Done with R_rel')

            new_halo_data_snap['N_2Mpc']=np.zeros(n_halos_snap)+np.nan
            fieldhalos_snap=new_halo_data_snap['hostHaloID']==-1
            fieldhalos_snap_indices=np.where(fieldhalos_snap)[0]
            subhalos_snap=np.logical_not(fieldhalos_snap)
            subhalos_snap_indices=np.where(subhalos_snap)[0]

            all_halos_xyz=np.column_stack([new_halo_data_snap['Xc'],new_halo_data_snap['Yc'],new_halo_data_snap['Zc']])
            field_halos_xyz=np.compress(fieldhalos_snap,all_halos_xyz,axis=0)
            sub_halos_xyz=np.compress(subhalos_snap,all_halos_xyz,axis=0)

            if verbose:
                print(f"Adding number densities to field halos for snap = {isnap}")

            for i_field_halo,field_halo_xyz_temp in enumerate(field_halos_xyz):
                field_halo_index_temp=fieldhalos_snap_indices[i_field_halo]
                field_halos_xyz_rel_squared=(field_halos_xyz-field_halo_xyz_temp)**2
                field_halos_xyz_rel_dist=np.sqrt(np.sum(field_halos_xyz_rel_squared,axis=1))
                field_halos_xyz_rel_dist_2mpc_count=np.sum(field_halos_xyz_rel_dist<2)
                new_halo_data_snap['N_2Mpc'][field_halo_index_temp]=field_halos_xyz_rel_dist_2mpc_count

            if verbose:
                print(f"Adding number densities to subhalos for snap = {isnap}")

            for i_sub_halo,sub_halo_xyz_temp in enumerate(sub_halos_xyz):
                sub_halo_index_temp=subhalos_snap_indices[i_sub_halo]
                sub_halos_xyz_rel_squared=(sub_halos_xyz-sub_halo_xyz_temp)**2
                sub_halos_xyz_rel_dist=np.sqrt(np.sum(sub_halos_xyz_rel_squared,axis=1))
                sub_halos_xyz_rel_dist_2mpc_count=np.sum(sub_halos_xyz_rel_dist<2)
                new_halo_data_snap['N_2Mpc'][sub_halo_index_temp]=sub_halos_xyz_rel_dist_2mpc_count

            print(f"Done with number densities for snap = {isnap}")

        new_halo_data.append(new_halo_data_snap)

    if verbose:
        print('Saving full halo data to file')

    outfilename='base3_vrhalodata_'+outname+'.dat'

    ###### SAVE data to file
    if path.exists(outfilename):
        if verbose:
            print('Overwriting existing base3 halo data ...')
        os.remove(outfilename)

    with open(outfilename, 'wb') as halo_data_file:
        pickle.dump(new_halo_data, halo_data_file)
        halo_data_file.close()

    return new_halo_data

########################### RETRIEVE PARTICLE LISTS ###########################

def get_particle_lists(base_halo_data_snap,add_subparts_to_fofs=False,verbose=1):
    
    """

    get_particle_lists : function
	----------

    Retrieve the particle lists for each halo for the provided halo data dictionary 
    (and corresponding snapshot) from velociraptor.

	Parameters
    ----------

    base_halo_data_snap : dictionary
        The halo data dictionary for the relevant snapshot.

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
    snap=int(base_halo_data_snap["Snap"])

    if verbose:
        print('Reading particle lists for snap = ',snap)

    # particle data
    try:
        part_data_temp=ReadParticleDataFile(base_halo_data_snap['FilePath'],ibinary=base_halo_data_snap['FileType'],iverbose=0,iparttypes=1)
        
        if part_data_temp==[]:
            part_data_temp={"Npart":[],"Npart_unbound":[],'Particle_IDs':[],'Particle_Types':[]}
            print('Particle data not found for snap = ',snap)
            return part_data_temp

    except: #if we can't load particle data
        if verbose:
            print('Particle data not included in hdf5 file for snap = ',snap)
        part_data_temp={"Npart":[],"Npart_unbound":[],'Particle_IDs':[],'Particle_Types':[]}
        return part_data_temp

    if add_subparts_to_fofs:

        if verbose==1:
            print('Appending FOF particle lists with substructure')
        
        field_halo_indices_temp=np.where(base_halo_data_snap['hostHaloID']==-1)[0]#find field/fof halos

        for i_field_halo,field_halo_ID in enumerate(base_halo_data_snap['ID'][field_halo_indices_temp]):#go through each field halo
            
            sub_halos_temp=(np.where(base_halo_data_snap['hostHaloID']==field_halo_ID)[0])#find the indices of its subhalos

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

def gen_particle_history_serial(base_halo_data,min_snap=0,verbose=1):

    """

    gen_particle_history_serial : function
	----------

    Generate and save particle history data from velociraptor property and particle files.

	Parameters
	----------
    base_halo_data : list of dictionaries
        The halo data list of dictionaries previously generated. The particle histories 
        are created as per the snaps contained in this list. 

    min_snap : int
        The snap after which to save particle histories (here to save memory).

	Returns
	----------
    {'all_ids':running_list_all,'sub_ids':running_list_sub} : dict (of dict)
        Dictionary of particle lists which have ever been part of any halo and those which 
        have been part of a subhalo at any point up to the last snap in the base_halo_data array.
        This data is saved for each snapshot on the way in a np.pickle file in the directory "part_histories"

	"""

    ###### WILL SAVE TO FILE PARTICLE HISTORIES WITH FORM: part_histories/snap_xxx_parthistory_all.dat and part_histories/snap_xxx_parthistory_sub.dat

    ### Input checks
    # Snaps
    try:
        no_snaps=len(base_halo_data)
    except:
        print("Invalid halo data")

    # if the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    if not os.path.isdir("part_histories"):
        os.mkdir("part_histories")

    print('Generating particle histories up to snap = ',no_snaps)

    running_list_all=[]
    running_list_sub=[]
    sub_part_hist={}
    all_part_hist={}

    # for each snapshot which is included in base_halo_data get the particle data and add to the running list

    for isnap in range(no_snaps):

        #Load particle data for this snapshot
        new_particle_data=get_particle_lists(base_halo_data_snap=base_halo_data[isnap],add_subparts_to_fofs=False,verbose=verbose)
        
        snap_abs=base_halo_data[isnap]["Snap"]
        #if no halos or no new particle data
        if len(new_particle_data['Particle_IDs'])==0 or len(base_halo_data[isnap]['hostHaloID'])<2:
            if verbose:
                print('Either no particle data or no halos for snap = ',snap_abs)
            continue
        #if particle data is valid, continue
        else:
            if verbose:
                print('Have particle lists for snap = ',snap_abs)

            n_halos_snap=len(base_halo_data[isnap]['hostHaloID'])# Number of halos at this snap
            sub_bools=base_halo_data[isnap]['hostHaloID']>0 #Boolean mask indicating which halos are subhalos

            if np.sum(sub_bools)<2:
                sub_halos_plist=[]
            else:
                sub_halos_plist=[]
                for ihalo,plist in enumerate(new_particle_data['Particle_IDs']):
                    if sub_bools[ihalo]:
                        sub_halos_plist.append(plist)
                sub_halos_plist=np.concatenate(sub_halos_plist)

            all_halos_plist=np.concatenate(new_particle_data['Particle_IDs'])

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
                all_part_hist[str(int(new_part_structure))]=1

            for new_part_substructure in new_substructure_indices:
                sub_part_hist[str(int(new_part_substructure))]=1

            # Now if our snapshot is above the minimum snap set at the outset
            # we save the boolean lists (of length npart) for this snapshot and move on
            if isnap in range(no_snaps):
                if isnap>=min_snap:
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
    return [all_part_hist,sub_part_hist]

########################### GENERATE ACCRETION RATES: constant MASS ###########################

def gen_accretion_rate_constant_mass(base_halo_data,isnap,mass_table=[],halo_index_list=[],depth=5,trim_particles=True,verbose=1): 
    
    """

    gen_accretion_rate : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists and (maybe) trimming particles.
    The snapshot for which this is calculated represents the final snapshot in the calculation. 

    ** note: if trimming particles, part_histories must have been generated

	Parameters
	----------
    base_halo_data : list of dictionaries
        The minimal halo data list of dictionaries previously generated ("base1" is sufficient)

    isnap : int
        The index in the base_halo_data list for which to calculate accretion rates.
        (May be different to actual snap).

    mass_table : list
        List of the particle masses in order (directly from simulation, unconverted).
    
    halo_index_list : list
        List of the halo indices for which to calculate accretion rates. If not provided,
        find for all halos in the base_halo_data dictionary at the desired snapshot. 

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
            "halo_index_list"
        This data is saved for each snapshot on the way in a np.pickle file in the directory "/acc_rates"

	"""

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

def load_accretion_rate(directory,calc_type,snap,depth,span=[],halo_data_snap=[],append_fields=[],verbose=1):
    """

    load_accretion_rate : function
	----------

    Generates a pandas dataframe of accretion rates from file with given calculation parameters. 

	Parameters
	----------
    directory : str
        Where to search for accretion rate files.

    calc_type : int or bool
        0: base
        1: trimmed
    
    snap : int
        Snapshot of accretion rate calculation.

    span : int
        The span in halo_indices of the calculation (normally n_halos/n_processes).

    halo_data_snap : dict
        Halo data dictionary at this snapshot (to add relevant fields).
    
    append_fields : list of str
        List of halo data fields to append to accretion dataframe. 
    
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

    filename_dataframe=gen_filename_dataframe(directory)
    if span==[]:
        correct_snap_spans=np.array(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==snap))]['span'])
        span_new=np.nanmax(correct_snap_spans)
        correct_span=np.absolute(filename_dataframe['span']-span_new)<10
    else:
        span_new==span
        correct_span=filename_dataframe['span']==span

    relevant_files=list(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==snap,filename_dataframe['depth']==depth,correct_span))]['filename'])
    index1=list(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==snap,filename_dataframe['depth']==depth,correct_span))]['index1'])
    index2=list(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==snap,filename_dataframe['depth']==depth,correct_span))]['index2'])
    
    if verbose:
        print(f'Found {len(relevant_files)} accretion rate files (snap = {snap}, type = {calc_type}, depth = {depth}, span = {span_new})')
    
    acc_rate_dataframe={'DM_Acc':[],'Gas_Acc':[],'Tot_Acc':[],'fb':[],'dt':[],'halo_index_list':[]}

    if halo_data_snap==[]:
        append_fields=[]

    acc_rate_dataframe=df(acc_rate_dataframe)

    for ifile,ifilename in enumerate(relevant_files):
        halo_indices=list(range(index1[ifile],index2[ifile]+1))
        with open(directory+ifilename,'rb') as acc_rate_file:
            dataframe_temp=pickle.load(acc_rate_file)
            dataframe_temp=df(dataframe_temp)
            dataframe_temp['Tot_Acc']=np.array(dataframe_temp['DM_Acc'])+np.array(dataframe_temp['Gas_Acc'])
            dataframe_temp['fb']=np.array(dataframe_temp['Gas_Acc'])/(np.array(dataframe_temp['DM_Acc'])+np.array(dataframe_temp['Gas_Acc']))
            acc_rate_dataframe=acc_rate_dataframe.append(dataframe_temp)
            acc_rate_file.close()
    acc_rate_dataframe=acc_rate_dataframe.sort_values(by=['halo_index_list'])

    return acc_rate_dataframe
        


