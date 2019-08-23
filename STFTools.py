#########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data & Halo Properties ##########################################
#########################################################################################################################################################################

#*** Preamble ***
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

# VELOCIraptor python tools 
from VRPythonTools import *

########################### CREATE BASE HALO DATA ###########################

def gen_base_halo_data(partdata_filelist,partdata_filetype,vr_filelist,vr_filetype,tf_filelist,outname='',temporal_idval=10**12,verbose=1):
    
    """

    gen_base_halo_data : function
	----------

    Generate halo data from velociraptor property and particle files.

	Parameters
	----------

    outname : str
        Suffix for output file. 

    partdata_filelist : list of str
        List of the particle data file paths. None if if we don't have data for a certain isnap.
        This file needs to be PADDED with None to be of the same length as the actual snaps. 

    partdata_filetype : str
        Type of particle data we are using. 
        One of "EAGLE", "GADGET", "SWIFT" (so far)

    vr_filelist : list of str
        List of the velociraptor data file paths. None if if we don't have data for a certain isnap.
        This file needs to be PADDED with None to be of the same length as the actual snaps. 

    vr_filetype : int
        The filetype of the VELOCIraptor inputs: (2 = hdf5)

    tf_filelist : list of str
        List of the treefrog data file paths. None if if we don't have data for a certain isnap.
        This file needs to be PADDED with None to be of the same length as the actual snaps. 

    temporal_idval : int
        The multiplier used by TreeFrog to create unique temporal halo IDs. 

    verbose : bool
        Flag indicating how verbose we want the code to be when we run.

    Returns
	-------
	V1_HaloData_outname.dat : list
	V2_HaloData_outname.dat : list

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
        'VR_FilePath'
        'VR_FileType'
        'Part_FilePath'
        'Part_FileType'
        'outname'

    * items will be removed from V1 file

	"""

    ###### WILL SAVE TO FILE THE HALO DATA WITH FORM: VX_HaloData_outname.dat

    halo_data_all=[]
    base_fields=['ID','hostHaloID']#default halo fields

    # file lists
    part_list=partdata_filelist
    vr_list=vr_filelist
    tf_list=tf_filelist

    # check here

    sim_snaps=list(range(len(part_list)))
    have_halo_data=[]

    print('Reading halo data using VR python tools ...')
    #for each snap in the above lists we will generate halo data
    for snap in sim_snaps:
        if not vr_list[snap].startswith('/'):
            have_halo_data.append(False)
            if verbose:
                print(f'No halo data for snap {snap} (not given a file)')
            continue
        if verbose:
            print(f'Searching for halo data at snap {snap} ...')
            print(f'[File: {vr_list[snap]}]')
           
        #use VR python tools to load in halo data for this snap
        halo_data_snap=ReadPropertyFile(vr_list[snap],ibinary=vr_filetype,iseparatesubfiles=0,iverbose=0, desiredfields=base_fields, isiminfo=True, iunitinfo=True,)
        
        #if data is found
        if not halo_data_snap==[]:
            halo_data_all.append(halo_data_snap)
            have_halo_data.append(True)

        #if data is not found
        else:
            if verbose:
                print("Couldn't find velociraptor files for snap = ",snap)
            return []

    # List of number of halos detected for each snap and list isolated data dictionary for each snap (in dictionaries)
    halo_data_counts=[item[1] for item in halo_data_all]
    halo_data_all=[item[0] for item in halo_data_all]


    # Import tree data from TreeFrog, build temporal head/tails from descendants -- adds to halo_data_all (all halo data)
    print('Now assembling descendent tree using VR python tools')
    tf_filelist=np.compress(have_halo_data,tf_filelist)

    for isnap,item in enumerate(halo_data_all):
        halo_data_all[isnap]['Count']=halo_data_counts[isnap]
        if item["ID"][0]<temporal_idval:
            #read in IDs from TreeFrog
            treefile_compressed_isnap=tf_filelist[isnap]
            print(treefile_compressed_isnap)
            print(h5py.File(treefile_compressed_isnap,'r').keys())
            # treefrog_ids=h5py.File(treefile_compressed_isnap)["ID"]
            # halo_data_all[isnap]["ID"]=treefrog_ids
            # print(treefrog_ids)

    snap_no=len(tf_filelist)
    np.savetxt('tf_filelist_compressed.txt',tf_filelist,fmt='%s')
    tf_filelist="tf_filelist_compressed.txt"
    # Read in tree data
    halo_tree=ReadHaloMergerTreeDescendant(tf_filelist,ibinary=vr_filetype,iverbose=verbose+1,imerit=True,inpart=False)

    # Now build trees and add onto halo data array

    BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=verbose,TEMPORALHALOIDVAL=temporal_idval)
    
    print('Finished assembling descendent tree using VR python tools')

    if verbose==1:
        print('Adding timesteps & filepath information')
    
    # Adding timesteps and filepath information
    first_true_index=np.where(have_halo_data)[0][0]
    H0=halo_data_all[first_true_index]['SimulationInfo']['h_val']*halo_data_all[first_true_index]['SimulationInfo']['Hubble_unit']
    Om0=halo_data_all[first_true_index]['SimulationInfo']['Omega_Lambda']
    cosmo=FlatLambdaCDM(H0=H0,Om0=Om0)

    halo_data_output=[]
    isnap=-1
    for snap in sim_snaps:
        if have_halo_data[snap]:
            isnap=isnap+1
            scale_factor=halo_data_all[isnap]['SimulationInfo']['ScaleFactor']
            redshift=z_at_value(cosmo.scale_factor,scale_factor,zmin=-0.5)
            lookback_time=cosmo.lookback_time(redshift).value
            halo_data_all[isnap]['SimulationInfo']['z']=redshift
            halo_data_all[isnap]['SimulationInfo']['LookbackTime']=lookback_time
            halo_data_all[isnap]['VR_FilePath']=vr_list[snap]
            halo_data_all[isnap]['VR_FileType']=vr_filetype
            halo_data_all[isnap]['Part_FilePath']=part_list[snap]
            halo_data_all[isnap]['Part_FileType']=partdata_filetype
            halo_data_all[isnap]['outname']=outname
            halo_data_all[isnap]['Snap']=snap
            halo_data_all[isnap]['SimulationInfo']['BoxSize_Comoving']=halo_data_all[isnap]['SimulationInfo']['Period']/scale_factor
            halo_data_output.append(halo_data_all[isnap])
        else:
            halo_data_output.append({'Snap':snap,'Part_FilePath':part_list[snap],'Part_FileType':partdata_filetype})

    print('Saving B2 halo data to file (contains detailed TreeFrog data)')

    ###### SAVE all data (with detailed TF) to file
    if path.exists('B2_HaloData_'+outname+'.dat'):
        if verbose:
            print('Overwriting existing V2 halo data ...')
        os.remove('B2_HaloData_'+outname+'.dat')

    with open('B2_HaloData_'+outname+'.dat', 'wb') as halo_data_file:
        pickle.dump(halo_data_output, halo_data_file)
        halo_data_file.close()

    print('Saving B1 halo data to file (removing detailed TreeFrog data)')
    ###### Remove superfluous data for acc_rate calcs
    fields_to_keep=['Count','Snap','ID','hostHaloID','Tail','Head','VR_FilePath','VR_FileType','Part_FilePath','Part_FileType','UnitInfo','SimulationInfo','outname']
    halo_data_all_truncated=[]
    for snap,halo_data_snap in enumerate(halo_data_output):
        if have_halo_data[snap]:
            halo_data_all_truncated_snap={}
            for field in fields_to_keep:
                halo_data_all_truncated_snap[field]=halo_data_snap[field]
        else:
            halo_data_all_truncated_snap={'Snap':snap,'Part_FilePath':part_list[snap],'Part_FileType':partdata_filetype}
        halo_data_all_truncated.append(halo_data_all_truncated_snap)

    ###### Save the trimmed data to file
    if path.exists('B1_HaloData_'+outname+'.dat'):
        if verbose:
            print('Overwriting existing V1 halo data ...')
        os.remove('B1_HaloData_'+outname+'.dat')

    with open('B1_HaloData_'+outname+'.dat', 'wb') as halo_data_file:
        pickle.dump(halo_data_all_truncated, halo_data_file)
        halo_data_file.close()

    print('Done generating base halo data')

    return halo_data_all

########################### ADD DETAILED HALO DATA ###########################

def gen_detailed_halo_data(base_halo_data,vr_halo_fields=[],extra_halo_fields=[],verbose=True):
    
    """
    
    gen_detailed_halo_data : function
	----------

    Add detailed halo data to base halo data from property files.

    Parameters
    ----------

    base_halo_data : list of dicts

        List (for each snap) of dictionaries containing basic halo data generated from gen_base_halo_data. 

    vr_property_fields : list of str

        List of dictionary keys for halo properties (from velociraptor) to be added to the base halo data. 

    extra_halo_fields : list of str

        List of keys to add to halo data. Currently just supports 'R_rel', 'N_Peers'.

    Returns
    --------

    V3_HaloData_outname.dat : list of dict

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
        'VR_FilePath'
        'VR_FileType'
        'Part_FilePath'
        'Part_FileType'

        AND ANY EXTRAS from vr_property_fields

	"""
    no_snaps_tot=len(base_halo_data)

    # If we're not given vr halo fields, read all the available halo data
    if vr_halo_fields==[]:
        snap_try=-1 
        found=False
        if verbose:
            print('Not explicitly given extra halo fields, using all halo fields from last available snap')
        while found==False and snap_try>-200: # Loop to find the keys in the last halo data file
            if verbose:
                print(f'Searching for fields at snap = {no_snaps_tot+snap_try}')
            try:
                property_filename=base_halo_data[snap_try]['VR_FilePath']+".properties.0"
                property_file=h5py.File(property_filename)
                all_props=list(property_file.keys())
                vr_halo_fields=all_props
                found=True
                if verbose:
                    print(f'Found data at snap = {no_snaps_tot+snap_try}')
            except:
                snap_try=snap_try-1
                if verbose:
                    print(f"Didn't find data at snap = {no_snaps_tot+snap_try+1}")
    if verbose:
        print('Adding the following fields from properties file:')
        print(np.array(vr_halo_fields))

    new_halo_data=[]
    base_fields=list(base_halo_data[snap_try].keys())
    fields_needed=np.compress(np.logical_not(np.in1d(base_fields,vr_halo_fields)),base_fields)

    if verbose:
        print('Will also collect the following fields from base halo data:')
        print(np.array(fields_needed))

    # Loop through each snap and add the extra fields
    for snap,base_halo_data_snap in enumerate(base_halo_data):
        # Rirst check if we have a padded snapshot
        if len(base_halo_data_snap.keys())<4: 
            if verbose:
                print(f'Skipping padded snap ',snap)
            new_halo_data.append(base_halo_data_snap)
            continue

        n_halos_snap=len(base_halo_data[snap]['ID'])

        # Read new halo data
        if verbose:
            print(f'Adding detailed halo data for snap ',snap,' where there are ',n_halos_snap,' halos')

        new_halo_data_snap=ReadPropertyFile(base_halo_data_snap['VR_FilePath'],ibinary=base_halo_data_snap["VR_FileType"],iseparatesubfiles=0,iverbose=0, desiredfields=vr_halo_fields, isiminfo=True, iunitinfo=True)[0]

        # Adding old halo data from V1 calcs
        if verbose:
            print(f'Adding fields from base halo data')

        for field in fields_needed:
            new_halo_data_snap[field]=base_halo_data[snap][field]
        
        # Add extra halo fields -- post-process velociraptor files   
        if n_halos_snap>0:
            if 'R_rel' in extra_halo_fields: #Relative radius to host
                if verbose:
                    print('Adding R_rel information for subhalos')
                new_halo_data_snap['R_rel']=np.zeros(n_halos_snap)+np.nan #initialise to nan if field halo
                for ihalo in range(n_halos_snap):
                    hostID_temp=new_halo_data_snap['hostHaloID'][ihalo]
                    if not hostID_temp==-1:
                        #if we have a subhalo 
                        hostindex_temp=np.where(new_halo_data_snap['ID']==hostID_temp)[0][0]
                        host_radius=new_halo_data_snap['R_200crit'][hostindex_temp]
                        host_xyz=np.array([new_halo_data_snap['Xc'][hostindex_temp],new_halo_data_snap['Yc'][hostindex_temp],new_halo_data_snap['Zc'][hostindex_temp]])
                        sub_xy=np.array([new_halo_data_snap['Xc'][ihalo],new_halo_data_snap['Yc'][ihalo],new_halo_data_snap['Zc'][ihalo]])
                        group_centric_r=np.sqrt(np.sum((host_xyz-sub_xy)**2))
                        r_rel_temp=group_centric_r/host_radius
                        new_halo_data_snap['R_rel'][ihalo]=r_rel_temp
                if verbose:
                    print('Done with R_rel')

            if 'N_peers' in extra_halo_fields: #Number of peer subhalos
                if verbose:
                    print('Adding N_peers information for subhalos')
                new_halo_data_snap['N_peers']=np.zeros(len(new_halo_data_snap['ID']))+np.nan #initialise to nan if field halo
                for ihalo in range(n_halos_snap):
                    hostID_temp=new_halo_data_snap['hostHaloID'][ihalo]
                    if not hostID_temp==-1:
                        #if we have a subhalo
                        N_peers=np.sum(new_halo_data_snap['hostHaloID']==hostID_temp)-1
                        new_halo_data_snap['N_peers'][ihalo]=N_peers           
                if verbose:
                    print('Done with N_peers')

        else: #if insufficient halos at snap
            if verbose:
                print('Skipping adding the extra halo fields for this snap (insufficient halo count)')
        
        #Append our new halo data to the running list
        new_halo_data.append(new_halo_data_snap)

    if verbose:
        print('Saving full halo data to file ...')

    outfilename='B3_HaloData_'+base_halo_data[snap_try]['outname']+'.dat'

    # Save data to file (remove if path already exists)
    if path.exists(outfilename):
        if verbose:
            print('Overwriting existing V3 halo data ...')
        os.remove(outfilename)

    with open(outfilename, 'wb') as halo_data_file:
        pickle.dump(new_halo_data, halo_data_file)
        halo_data_file.close()

    return new_halo_data

########################### RETRIEVE PARTICLE LISTS ###########################

def get_particle_lists(base_halo_data_snap,include_unbound=True,add_subparts_to_fofs=False,verbose=1):
    
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
        print('Reading halo particle lists for snap = ',snap)

    # particle data
    try:
        if include_unbound:
            part_data_temp=ReadParticleDataFile(base_halo_data_snap['VR_FilePath'],ibinary=base_halo_data_snap['VR_FileType'],iverbose=0,iparttypes=1,unbound=True)
        else: 
            part_data_temp=ReadParticleDataFile(base_halo_data_snap['VR_FilePath'],ibinary=base_halo_data_snap['VR_FileType'],iverbose=0,iparttypes=1,unbound=False)
        
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


# def find_progen_index(base_halo_data,index2,snap2,snap1): ### given halo index2 at snap 2, find progenitor index at snap 1
    
