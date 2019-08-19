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
import read_eagle

# VELOCIraptor python tools 
from VRPythonTools import *

########################### CREATE BASE HALO DATA ###########################

def gen_base_halo_data(partdata_filelist,partdata_filetype,vr_filelist,vr_filetype,tf_filelist,outname='',temporal_idval=[],verbose=1):
    
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
        The multiplier used by VELOCIraptor to create unique temporal halo IDs. 

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
        halo_data_snap=ReadPropertyFile(vr_list[snap],ibinary=vr_filetype,iseparatesubfiles=0,iverbose=0, desiredfields=base_fields, isiminfo=True, iunitinfo=True)
        
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

    for isnap,item in enumerate(halo_data_all):
        halo_data_all[isnap]['Count']=halo_data_counts[isnap]

    # Import tree data from TreeFrog, build temporal head/tails from descendants -- adds to halo_data_all (all halo data)
    print('Now assembling descendent tree using VR python tools')
    tf_filelist=np.compress(have_halo_data,tf_filelist)  
    snap_no=len(tf_filelist)
    np.savetxt('tf_filelist_compressed.txt',tf_filelist,fmt='%s')
    tf_filelist="tf_filelist_compressed.txt"

    # Read in tree data
    halo_tree=ReadHaloMergerTreeDescendant(tf_filelist,ibinary=vr_filetype,iverbose=verbose+1,imerit=True,inpart=False)

    # Now build trees and add onto halo data array
    if temporal_idval==[]:#if not given halo TEMPORALHALOIVAL, use the vr default
        BuildTemporalHeadTailDescendant(snap_no,halo_tree,halo_data_counts,halo_data_all,iverbose=verbose)
    else:
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
            print(isnap)
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
            halo_data_all[isnap]['BoxSize_Comoving']=halo_data_all[isnap]['SimulationInfo']['Period']/scale_factor
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

    print('Saving V1 halo data to file (removing detailed TreeFrog data)')

    ###### Remove superfluous data for acc_rate calcs
    fields_to_keep=['Count','Snap','ID','hostHaloID','Tail','VR_FilePath','VR_FileType','Part_FilePath','Part_FileType','UnitInfo','SimulationInfo','outname']
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
        print('Reading particle lists for snap = ',snap)

    # particle data
    try:
        if include_unbound:
            part_data_temp=ReadParticleDataFile(base_halo_data_snap['FilePath'],ibinary=base_halo_data_snap['FileType'],iverbose=0,iparttypes=1,unbound=True)
        else: 
            part_data_temp=ReadParticleDataFile(base_halo_data_snap['FilePath'],ibinary=base_halo_data_snap['FileType'],iverbose=0,iparttypes=1,unbound=False)
        
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

# def gen_particle_history_serial(base_halo_data,snaps=[],verbose=1):

#     """

#     gen_particle_history_serial : function
# 	----------

#     Generate and save particle history data from velociraptor property and particle files.

# 	Parameters
# 	----------
#     base_halo_data : list of dictionaries
#         The halo data list of dictionaries previously generated (by gen_base_halo_data). Should contain the type of particle file we'll be reading. 

#     snaps : list of ints
#         The list of absolute snaps (corresponding to index in base_halo_data) for which we will add 
#         particles in halos or subhalos (and save accordingly). The running lists will build on the previous snap. 

# 	Returns
# 	----------
#     PartHistory_xxx-outname.hdf5 : hdf5 file with datasets

#         '/PartTypeX/PartID'
#         '/PartTypeX/PartIndex'
#         '/PartTypeX/Processed_L1'
#         '/PartTypeX/Processed_L2'

# 	"""

#     # Will save to file at: part_histories/PartTypeX_History_xxx-outname.dat
#     # Snaps
#     if snaps==[]:
#         snaps=snaps=list(range(no_snaps))

#     try:
#         valid_snaps=[len(base_halo_data[snap].keys())>3 for snap in snaps] #which indices of snaps are valid
#         valid_snaps=np.compress(valid_snaps,snaps)
#         outname=base_halo_data[valid_snaps[0]]['outname']

#     except:
#         print("Couldn't validate snaps")
#         return []

#     # if the directory with particle histories doesn't exist yet, make it (where we have run the python script)
#     if not os.path.isdir("part_histories"):
#         os.mkdir("part_histories")
    
#     if base_halo_data[valid_snaps[0]]['Part_FileType']='EAGLE':
#         PartTypes=[0,1,4] #Gas, DM, Stars
#     else:
#         PartTypes=[0,1] #Gas, DM

#     isnap=0
    
#     # for the desired snapshots in base_halo_data, get the particle data and add to the running list
#     for snap in valid_snaps:
#         #load new snap data
#         if base_halo_data[snap]['Part_FileType']=='EAGLE':
#             EAGLE_boxsize=
#             EAGLE_Snap=read_eagle.EagleSnapshot(base_halo_data[snap]['Part_FilePath'])
#             EAGLE_Snap.select_region()
#             Particle_IDs=[EAGLE_Snap.read_dataset(itype,"ParticleIDs") for itype in PartTypes]
#         else:
#             h5py_Snap=h5py.File(base_halo_data[snap]['Part_FilePath'])
#             Particle_IDs=[h5py_Snap['PartType0/ParticleIDs'],h5py_Snap['PartType1/ParticleIDs'],[],[]]

#         npart_Snap=[len(Particle_IDs[i]) for i in range(len(PartTypes))]

#         if isnap>0:#save old snap data if we can
#             PartHistory_Flags_Snap_Old=PartHistory_Flags_Snap

#         PartHistory_Flags_Snap=dict()#initialise our new snap data 

#         for itype,typenum in enumerate(PartTypes):
#             if npart_Snap[itype]>0:#if we have particles of this type at this snap
#                 PartHistory_Flags_Snap[str(itype)]=df({"ParticleID":np.sort(Particle_IDs[itype]),"ParticleIndex":np.argsort(Particle_IDs[itype]),"L1_B":np.zeros(npart_Snap[itype]),"L1_U":np.zeros(npart_Snap[itype]),"L2_B":np.zeros(npart_Snap[itype]),"L2_U":np.zeros(npart_Snap[itype]),"Is_New":np.zeros(npart_Snap[itype])},dtype=np.uint64)
                
#                 if isnap>0:#if we have past data
#                     #for each new id, check it was in the last.
#                     for ipart,New_ID_itype in enumerate(PartHistory_Flags_Snap[str(itype)]['ParticleID']):
#                         try:
#                             Old_Index_itype=np.searchsorted(PartHistory_Flags_Snap_Old[str(itype)]['ParticleID'],New_ID_itype)
#                             #if hasn't broken yet, we proceed to fill out the data
#                         except:
#                             PartHistory_Flags_Snap[str(itype)]['Is_New'][ipart]=1
                    
#                     Old_L1_B_IDs=PartHistory_Flags_Snap_Old[str(itype)]['ParticleID'][PartHistory_Flags_Snap_Old[str(itype)]["L1_B"]]


#                     ####make a list of (sorted) IDs for each level which need to be flagged


#                 else:#if we don't have past data, initialise empty
#                     pass
#             else:#if we have no particles of this type at this snap
#                 PartHistory_Flags_Snap[str(itype)]=None
#             pass
#         else: #if we are on the first snap and have no previous history

#         isnap=isnap+1

#     print('Unique particle histories created')

# # ########################### GENERATE ACCRETION RATES: constant MASS ###########################

# # def gen_accretion_rate_constant_mass(base_halo_data,isnap,mass_table=[],halo_index_list=[],depth=5,trim_particles=True,trim_unbound=True,include_unbound=True,verbose=1): 
    
# #     """

# #     gen_accretion_rate : function
# # 	----------

# #     Generate and save accretion rates for each particle type by comparing particle lists and (maybe) trimming particles.
# #     The snapshot for which this is calculated represents the final snapshot in the calculation. 

# #     ** note: if trimming particles, part_histories must have been generated

# # 	Parameters
# # 	----------
# #     base_halo_data : list of dictionaries
# #         The minimal halo data list of dictionaries previously generated ("base1" is sufficient)

# #     isnap : int
# #         The index in the base_halo_data list for which to calculate accretion rates.
# #         (May be different to actual snap).

# #     mass_table : list
# #         List of the particle masses in order (directly from simulation, unconverted).
    
# #     halo_index_list : list
# #         List of the halo indices for which to calculate accretion rates. If not provided,
# #         find for all halos in the base_halo_data dictionary at the desired snapshot. 

# #     depth : int
# #         How many snaps to skip back to when comparing particle lists.
# #         Initial snap for calculation will be snap-depth. 
    
# #     trim_particles: bool
# #         Boolean flag as to indicating whether or not to remove the particles which have previously been 
# #         part of structure or substructure in our accretion rate calculation. 

# # 	Returns
# # 	----------
# #     delta_m : dictionary
# #         Dictionary of accretion rates for each particle type to each halo at the desired snap.
# #         Keys: 
# #             "DM_Acc"
# #             "Gas_Acc"
# #             "dt"
# #             "halo_index_list"
# #         This data is saved for each snapshot on the way in a np.pickle file in the directory "/acc_rates"

# # 	"""

# #     ################## Input Checks ##################

# #     n_halos_tot=len(base_halo_data[isnap]['hostHaloID'])

# #     # Snap
# #     try:
# #         isnap=int(isnap)
# #     except:
# #         print('Invalid snap')
# #         return []
    
# #     # If the directory with particle histories doesn't exist yet, make it (where we have run the python script)
# #     if not os.path.isdir("acc_rates"):
# #         os.mkdir("acc_rates")

# #     # If trimming the accretion rates we have to load the particle histories
# #     if trim_particles:#load particle histories if we need to
# #         snap_reqd=isnap-depth-1#the snap before our initial snap
# #         try:#check if the files have already been generated
# #             print('Trying to find particle histories at isnap = ',snap_reqd)
# #             if trim_unbound:
# #                 parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all.dat"
# #                 parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub.dat"
# #             else:
# #                 parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all_boundonly.dat"
# #                 parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub_boundonly.dat"

# #             with open(parthist_filename_all, 'rb') as parthist_file:
# #                 allstructure_history=pickle.load(parthist_file)
# #                 parthist_file.close()
# #             with open(parthist_filename_sub, 'rb') as parthist_file:
# #                 substructure_history=pickle.load(parthist_file)
# #                 parthist_file.close()
# #             print('Found particle histories')
# #         except:#if they haven't, generate them and load the required snap
# #                 print('Failed to find particle histories for trimming at snap = ',isnap-depth-1,', terminating')
# #                 return []

# #     ################## Finding initial and final particle lists; organising ##################

# #     if verbose:
# #         print('Now generating accretion rates for isnap = ',isnap,' at depth = ',depth,' trimming = ',trim_particles,', using unbound = ',include_unbound)
    
# #     # Find progenitor index subfunction
# #     def find_progen_index(index_0,isnap,depth):
# #         id_0=base_halo_data[isnap]['ID'][index_0]#the original id
# #         tail_id=base_halo_data[isnap]['Tail'][index_0]#the tail id
# #         for idepth in range(1,depth+1,1):
#             new_id=tail_id #the new id from tail in last snap
#             if new_id in base_halo_data[isnap-idepth]['ID']:
#                 new_index=np.where(base_halo_data[isnap-idepth]['ID']==new_id)[0][0] #what index in the previous snap does the new_id correspond to
#                 tail_id=base_halo_data[isnap-idepth]['Tail'][new_index] #the new id for next loop
#             else:
#                 new_index=np.nan
#                 return new_index
#              #new index at snap-depth
#         return new_index
    
#     # If we aren't given a halo_index_list, then just calculate for all 
#     if halo_index_list==[]:
#         halo_index_list=list(range(n_halos_tot))

#     # Find and load FINAL snap particle data
#     if include_unbound:
#         part_data_2=get_particle_lists(base_halo_data_snap=base_halo_data[isnap],add_subparts_to_fofs=True,include_unbound=True,verbose=0)
#     else:
#         part_data_2=get_particle_lists(base_halo_data_snap=base_halo_data[isnap],add_subparts_to_fofs=True,include_unbound=False,verbose=0)

#     part_data_2_ordered_IDs=[part_data_2['Particle_IDs'][ihalo] for ihalo in halo_index_list] #just retrieve the halos we want
#     part_data_2_ordered_Types=[part_data_2['Particle_Types'][ihalo] for ihalo in halo_index_list] #just retrieve the halos we want

#     # Find and load INITIAL snap particle data (and ensuring they exist)
#     if include_unbound:
#         part_data_1=get_particle_lists(base_halo_data_snap=base_halo_data[isnap-depth],include_unbound=True,add_subparts_to_fofs=True,verbose=0)
#     else:
#         part_data_1=get_particle_lists(base_halo_data_snap=base_halo_data[isnap-depth],include_unbound=False,add_subparts_to_fofs=True,verbose=0)

#     if isnap-depth<0 or part_data_1["Npart"]==[]:# if we can't find initial particles
#         print('Initial particle lists not found at required depth (isnap = ',isnap-depth,')')
#         return []

#     # Organise initial particle lists
#     print('Organising initial particle lists')
#     t1=time.time()

#     part_data_1_ordered_IDs=[]#initialise empty initial particle lists
#     # Iterate through each final halo and find its progenitor particle lists at the desired depth
#     for ihalo_abs in halo_index_list:
#         progen_index=find_progen_index(ihalo_abs,isnap=isnap,depth=depth)#finds progenitor index at desired snap

#         if progen_index>-1:#if progenitor index is valid
#             part_data_1_ordered_IDs.append(part_data_1['Particle_IDs'][progen_index])

#         else:#if progenitor can't be found, make particle lists for this halo (both final and initial) empty to avoid confusion
#             part_data_1_ordered_IDs.append([])
#             part_data_2['Particle_IDs']=[]
#             part_data_2['Particle_Types']=[]

#     t2=time.time()

#     print(f'Organised initial particle lists in {t2-t1} sec')

#     n_halos_tot=len(base_halo_data[isnap]['hostHaloID'])#number of total halos at the final snapshot in the halo_data_all dictionary
#     n_halos_desired=len(halo_index_list)#number of halos for calculation desired
#     field_bools=(base_halo_data[isnap]['hostHaloID']==-1)#boolean mask of halos which are field

#     if len(part_data_1_ordered_IDs)==len(part_data_2_ordered_IDs):
#         if verbose:
#             print(f'Accretion rate calculator parsed {n_halos_desired} halos')
#     else:
#         print('An unequal number of particle lists and/or halo indices were parsed, terminating')
#         return []
    
#     # Initialise outputs
#     delta_n0=[]
#     delta_n1=[]
#     halo_indices_abs=[]

#     #### Main halo loop
#     for ihalo,ihalo_abs in enumerate(halo_index_list):
#         #ihalo is counter, ihalo_abs is absolute halo index (at final snap)
#         if verbose:
#             print(f'Finding particles new to halo {ihalo_abs}')

#         part_IDs_init=part_data_1_ordered_IDs[ihalo]
#         part_IDs_final=part_data_2_ordered_IDs[ihalo]
#         part_Types_final=part_data_2_ordered_Types[ihalo]
        
#         part_count_1=len(part_IDs_init)
#         part_count_2=len(part_IDs_final)

#         # Verifying particle counts are adequate
#         if part_count_2<2 or part_count_1<2:
#             if verbose:
#                 print(f'Particle count in halo {ihalo_abs} is less than 2 - not processing')
#             # if <2 particles at initial or final snap, then don't calculate accretion rate to this halo
#             delta_n0.append(np.nan)
#             delta_n1.append(np.nan)

#         # If particle counts are adequate, then continue with calculation. 
#         else:
#             if verbose:
#                 print(f'Particle count in halo {ihalo_abs} is adequate for accretion rate calculation')

#             #Finding list of particles new to the halo 
#             new_particle_IDs=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_IDs_final))#list of particles new to halo
#             new_particle_Types=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_Types_final))#list of particle types new to halo

#             if verbose:
#                 print('Number of new particles to halo: ',len(new_particle_IDs))

#             #Trimming particles which have been part of structure in the past (i.e. those which are already in halos)    

#             if trim_particles:#if we have the particle histories
                
#                 if len(substructure_history)<100:#if the particle history is of insufficient length then skip
#                     print('Failed to find particle histories for trimming at isnap = ',isnap-depth-1)
#                     delta_n0.append(np.nan)
#                     delta_n1.append(np.nan)
                
#                 else:#if our particle history is valid
#                     t1=time.time()

#                     #reset lists which count whether a particle is valid or not (based on what its history is)
#                     field_mask_good=[]
#                     sub_mask_good=[]

#                     if field_bools[ihalo]==True:#if a field halo then we check whether each particle has been part of ANY structure
#                         for i,ipart in enumerate(new_particle_IDs):
#                             try:
#                                 allstructure_history[str(ipart)]==1#if the particle has been part of structure, note this by invalidating
#                                 field_mask_good.append(False)

#                             except:#if the particle is genuinely new to being in any structure, not its index as valid
#                                 field_mask_good.append(True)
#                         if verbose:
#                             print('Done cross checking particles for field halo, now compressing - keeping ',np.sum(field_mask_good),' of ',len(new_particle_IDs),' particles')
                        
#                         #reduce list to the genuinely unprocessed particles
#                         new_particle_Types=np.compress(field_mask_good,new_particle_Types)
#                         new_particle_IDs=np.compress(field_mask_good,new_particle_IDs)

#                     else:#if a subhalo
#                         for i,ipart in enumerate(new_particle_IDs):
#                             try:
#                                 substructure_history[str(ipart)]==1
#                                 sub_mask_good.append(False)
#                             except:
#                                 sub_mask_good.append(True)
#                         if verbose:
#                             print('Done cross checking particles for sub halo, now compressing - keeping ',np.sum(sub_mask_good),' of ',len(new_particle_IDs),' particles')
                        
#                         #reduce list to unprocessed particles
#                         new_particle_Types=np.compress(sub_mask_good,new_particle_Types)
#                         new_particle_IDs=np.compress(sub_mask_good,new_particle_IDs)

#             #### Now we simply count the number of new particles of each type

#             delta_n0_temp=int(np.sum(new_particle_Types==0))
#             delta_n1_temp=int(np.sum(new_particle_Types==1))
#             delta_n0.append(delta_n0_temp) #append the result to our final array
#             delta_n1.append(delta_n1_temp) #append the result to our final array 

#     ############################# Post-processing accretion calc results #############################
#     sim_unit_to_Msun=base_halo_data[0]['UnitInfo']['Mass_unit_to_solarmass']#Simulation mass units in Msun
#     h=base_halo_data[isnap]['SimulationInfo']['h_val']
#     m_0=mass_table[0]*sim_unit_to_Msun/h #parttype0 mass in Msun (PHYSICAL)
#     m_1=mass_table[1]*sim_unit_to_Msun/h #parttype1 mass in Msun (PHYSICAL)
#     lt2=base_halo_data[isnap]['SimulationInfo']['LookbackTime']#final lookback time
#     lt1=base_halo_data[isnap-depth]['SimulationInfo']['LookbackTime']#initial lookback time
#     delta_t=abs(lt1-lt2)#lookback time change from initial to final snapshot (Gyr)

#     # Find which particle type is more massive (i.e. DM) and save accretion rates in dictionary
#     # 'DM_Acc', 'Gas_Acc' and 'dt' as Msun/Gyr and dt accordingly
#     if mass_table[0]>mass_table[1]:
#         delta_m={'DM_Acc':np.array(delta_n0)*m_0/delta_t,'DM_Acc_n':delta_n0,'Gas_Acc':np.array(delta_n1)*m_1/delta_t,'Gas_Acc_n':delta_n1,'dt':delta_t,'halo_index_list':halo_index_list}
#     else:
#         delta_m={'DM_Acc':np.array(delta_n1)*m_1/delta_t,'DM_Acc_n':delta_n1,'Gas_Acc':np.array(delta_n0)*m_0/delta_t,'Gas_Acc_n':delta_n0,'dt':delta_t,'halo_index_list':halo_index_list}
    
#     # Now save all these accretion rates to file (in directory where run /acc_rates)
#     # (with filename depending on exact calculation parameters) - snap is the index in halo data
#     # will overwrite existing file (first deletes)

#     print('Saving accretion rates to .dat file.')
#     if trim_particles:
#         if include_unbound and trim_unbound:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#         elif not include_unbound and not trim_unbound:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#         elif include_unbound and not trim_unbound:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed3_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed3_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed3_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed3_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#     else:
#         if include_unbound:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#         else:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#     #return the delta_m dictionary. 
#     return delta_m

# ########################### GENERATE ACCRETION RATES: VARYING MASS ###########################

# def gen_accretion_rate_eagle(base_halo_data,isnap,halo_index_list=[],depth=5,trim_particles=True,trim_unbound=True,include_unbound=True,verbose=1): 
    
#     """

#     gen_accretion_rate : function
# 	----------

#     Generate and save accretion rates for each particle type by comparing particle lists and (maybe) trimming particles.
#     The snapshot for which this is calculated represents the final snapshot in the calculation. 

#     ** note: if trimming particles, part_histories must have been generated

# 	Parameters
# 	----------
#     base_halo_data : list of dictionaries
#         The minimal halo data list of dictionaries previously generated ("base1" is sufficient)

#     isnap : int
#         The index in the base_halo_data list for which to calculate accretion rates.
#         (May be different to actual snap).
    
#     halo_index_list : list
#         List of the halo indices for which to calculate accretion rates. If not provided,
#         find for all halos in the base_halo_data dictionary at the desired snapshot. 

#     depth : int
#         How many snaps to skip back to when comparing particle lists.
#         Initial snap for calculation will be snap-depth. 
    
#     trim_particles: bool
#         Boolean flag as to indicating whether or not to remove the particles which have previously been 
#         part of structure or substructure in our accretion rate calculation. 

# 	Returns
# 	----------
#     delta_m : dictionary
#         Dictionary of accretion rates for each particle type to each halo at the desired snap.
#         Keys: 
#             "DM_Acc"
#             "Gas_Acc"
#             "dt"
#             "halo_index_list"
#         This data is saved for each snapshot on the way in a np.pickle file in the directory "/acc_rates"

# 	"""

#     n_halos_tot=len(base_halo_data[isnap]['hostHaloID'])

#     if verbose:
#         print("Loading in mass data ... ")

#     with open('mass_data/isnap_'+str(isnap).zfill(3)+'_mass_data.dat','rb') as mass_file:
#         mass_table=pickle.load(mass_file)
#         mass_file.close()

#     gas_mass_dict=mass_table[0]

#     # Snap
#     try:
#         isnap=int(isnap)
#     except:
#         print('Invalid snap')
#         return []
    
#     # If the directory with particle histories doesn't exist yet, make it (where we have run the python script)
#     if not os.path.isdir("acc_rates"):
#         os.mkdir("acc_rates")

#     # If trimming the accretion rates we have to load the particle histories
#     if trim_particles:#load particle histories if we need to
#         snap_reqd=isnap-depth-1#the snap before our initial snap
#         try:#check if the files have already been generated
#             print('Trying to find particle histories at isnap = ',snap_reqd)
#             if trim_unbound:
#                 parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all.dat"
#                 parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub.dat"
#             else:
#                 parthist_filename_all="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_all_boundonly.dat"
#                 parthist_filename_sub="part_histories/snap_"+str(snap_reqd).zfill(3)+"_parthistory_sub_boundonly.dat"

#             with open(parthist_filename_all, 'rb') as parthist_file:
#                 allstructure_history=pickle.load(parthist_file)
#                 parthist_file.close()
#             with open(parthist_filename_sub, 'rb') as parthist_file:
#                 substructure_history=pickle.load(parthist_file)
#                 parthist_file.close()
#             print('Found particle histories')
#         except:#if they haven't, generate them and load the required snap
#                 print('Failed to find particle histories for trimming at snap = ',isnap-depth-1,', terminating')
#                 return []

#     ################## Finding initial and final particle lists; organising ##################

#     if verbose:
#         print('Now generating accretion rates for isnap = ',isnap,' at depth = ',depth,' trimming = ',trim_particles)
    
#     # Find progenitor index subfunction
#     def find_progen_index(index_0,isnap,depth):
#         id_0=base_halo_data[isnap]['ID'][index_0]#the original id
#         tail_id=base_halo_data[isnap]['Tail'][index_0]#the tail id
#         for idepth in range(1,depth+1,1):
#             new_id=tail_id #the new id from tail in last snap
#             if new_id in base_halo_data[isnap-idepth]['ID']:
#                 new_index=np.where(base_halo_data[isnap-idepth]['ID']==new_id)[0][0] #what index in the previous snap does the new_id correspond to
#                 tail_id=base_halo_data[isnap-idepth]['Tail'][new_index] #the new id for next loop
#             else:
#                 new_index=np.nan
#                 return new_index
#              #new index at snap-depth
#         return new_index
    
#     # If we aren't given a halo_index_list, then just calculate for all 
#     if halo_index_list==[]:
#         halo_index_list=list(range(n_halos_tot))

#     # Find and load FINAL snap particle data
#     if include_unbound:
#         part_data_2=get_particle_lists(base_halo_data_snap=base_halo_data[isnap],add_subparts_to_fofs=True,include_unbound=True,verbose=0)
#     else:
#         part_data_2=get_particle_lists(base_halo_data_snap=base_halo_data[isnap],add_subparts_to_fofs=True,include_unbound=False,verbose=0)

#     part_data_2_ordered_IDs=[part_data_2['Particle_IDs'][ihalo] for ihalo in halo_index_list] #just retrieve the halos we want
#     part_data_2_ordered_Types=[part_data_2['Particle_Types'][ihalo] for ihalo in halo_index_list] #just retrieve the halos we want

#     # Find and load INITIAL snap particle data (and ensuring they exist)
#     if include_unbound:
#         part_data_1=get_particle_lists(base_halo_data_snap=base_halo_data[isnap-depth],add_subparts_to_fofs=True,include_unbound=True,verbose=0)
#     else:
#         part_data_1=get_particle_lists(base_halo_data_snap=base_halo_data[isnap-depth],add_subparts_to_fofs=True,include_unbound=False,verbose=0)

#     if isnap-depth<0 or part_data_1["Npart"]==[]:# if we can't find initial particles
#         print('Initial particle lists not found at required depth (isnap = ',isnap-depth,')')
#         return []

#     # Organise initial particle lists
#     print('Organising initial particle lists')
#     t1=time.time()

#     part_data_1_ordered_IDs=[]#initialise empty initial particle lists
#     # Iterate through each final halo and find its progenitor particle lists at the desired depth
#     for ihalo_abs in halo_index_list:
#         progen_index=find_progen_index(ihalo_abs,isnap=isnap,depth=depth)#finds progenitor index at desired snap

#         if progen_index>-1:#if progenitor index is valid
#             part_data_1_ordered_IDs.append(part_data_1['Particle_IDs'][progen_index])

#         else:#if progenitor can't be found, make particle lists for this halo (both final and initial) empty to avoid confusion
#             part_data_1_ordered_IDs.append([])
#             part_data_2['Particle_IDs']=[]
#             part_data_2['Particle_Types']=[]

#     t2=time.time()

#     print(f'Organised initial particle lists in {t2-t1} sec')

#     n_halos_tot=len(base_halo_data[isnap]['hostHaloID'])#number of total halos at the final snapshot in the halo_data_all dictionary
#     n_halos_desired=len(halo_index_list)#number of halos for calculation desired
#     field_bools=(base_halo_data[isnap]['hostHaloID']==-1)#boolean mask of halos which are field

#     if len(part_data_1_ordered_IDs)==len(part_data_2_ordered_IDs):
#         if verbose:
#             print(f'Accretion rate calculator parsed {n_halos_desired} halos')
#     else:
#         print('An unequal number of particle lists and/or halo indices were parsed, terminating')
#         return []
    
#     # Initialise outputs
#     delta_m0=[]
#     delta_m1=[]
#     delta_n0=[]
#     delta_n1=[]

#     halo_indices_abs=[]

#     #### Main halo loop
#     for ihalo,ihalo_abs in enumerate(halo_index_list):
#         #ihalo is counter, ihalo_abs is absolute halo index (at final snap)
#         if verbose:
#             print(f'Finding particles new to halo {ihalo_abs}')

#         part_IDs_init=part_data_1_ordered_IDs[ihalo]
#         part_IDs_final=part_data_2_ordered_IDs[ihalo]
#         part_Types_final=part_data_2_ordered_Types[ihalo]
        
#         part_count_1=len(part_IDs_init)
#         part_count_2=len(part_IDs_final)

#         # Verifying particle counts are adequate
#         if part_count_2<2 or part_count_1<2:
#             if verbose:
#                 print(f'Particle count in halo {ihalo_abs} is less than 2 - not processing')
#             # if <2 particles at initial or final snap, then don't calculate accretion rate to this halo
#             delta_m0.append(np.nan)
#             delta_m1.append(np.nan)
#             delta_n0.append(np.nan)
#             delta_n1.append(np.nan)

#         # If particle counts are adequate, then continue with calculation. 
#         else:
#             if verbose:
#                 print(f'Particle count in halo {ihalo_abs} is adequate for accretion rate calculation')

#             #Finding list of particles new to the halo 
#             new_particle_IDs=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_IDs_final))#list of particles new to halo
#             new_particle_Types=np.array(np.compress(np.logical_not(np.in1d(part_IDs_final,part_IDs_init)),part_Types_final))#list of particle types new to halo

#             if verbose:
#                 print('Number of new particles to halo: ',len(new_particle_IDs))

#             #Trimming particles which have been part of structure in the past (i.e. those which are already in halos)    

#             if trim_particles:#if we have the particle histories
                
#                 if len(substructure_history)<10:#if the particle history is of insufficient length then skip
#                     print('Failed to find particle histories for trimming at isnap = ',isnap-depth-1)
#                     delta_m0.append(np.nan)
#                     delta_m1.append(np.nan)
#                     delta_n0.append(np.nan)
#                     delta_n1.append(np.nan)
                
#                 else:#if our particle history is valid
#                     t1=time.time()

#                     #reset lists which count whether a particle is valid or not (based on what its history is)
#                     field_mask_good=[]
#                     sub_mask_good=[]

#                     if field_bools[ihalo]==True:#if a field halo then we check whether each particle has been part of ANY structure
#                         for ipart in new_particle_IDs:#iterate through each new particle to the halo
#                             try:
#                                 allstructure_history[str(ipart)]==1#if the particle has been part of structure, note this by invalidating
#                                 field_mask_good.append(False)
#                                 print('found the bugger')

#                             except:#if the particle is genuinely new to being in any structure, not its index as valid
#                                 field_mask_good.append(True)
#                         if verbose:
#                             print('Done cross checking particles for field halo, now compressing - keeping ',np.sum(field_mask_good),' of ',len(new_particle_IDs),' particles')
                        
#                         #reduce list to the genuinely unprocessed particles
#                         print('Previous length of particles:',len(new_particle_IDs))
#                         new_particle_Types=np.compress(field_mask_good,new_particle_Types)
#                         new_particle_IDs=np.compress(field_mask_good,new_particle_IDs)
#                         print('Trimmed length of particles:',len(new_particle_IDs))

#                     else:#if a subhalo
#                         for ipart in new_particle_IDs:
#                             try:
#                                 substructure_history[str(ipart)]==1
#                                 sub_mask_good.append(False)
#                             except:
#                                 sub_mask_good.append(True)
#                         if verbose:
#                             print('Done cross checking particles for sub halo, now compressing - keeping ',np.sum(sub_mask_good),' of ',len(new_particle_IDs),' particles')
                        
#                         #reduce list to unprocessed particles
#                         print('Previous length of particles:',len(new_particle_IDs))
#                         new_particle_Types=np.compress(sub_mask_good,new_particle_Types)
#                         new_particle_IDs=np.compress(sub_mask_good,new_particle_IDs)
#                         print('Trimmed length of particles:',len(new_particle_IDs))

#             #### Now we simply count the number of new particles of each type
#             delta_n1_temp=int(np.sum(new_particle_Types==1))
#             delta_m1_temp=delta_n1_temp*mass_table[1]
#             print('New DM Mass: ',delta_m1_temp)
#             new_gas_mask=new_particle_Types==0
#             new_IDs_Baryon=np.compress(new_gas_mask,new_particle_IDs)
#             delta_n0_temp=len(new_IDs_Baryon)
#             delta_n0.append(delta_n0_temp) #append the result to our final array
#             delta_n1.append(delta_n1_temp) #append the result to our final array 

#             print('Calculating new gas mass ...')
#             new_Mass_Baryon=0
#             for new_IDs_Baryon_temp in new_IDs_Baryon:
#                 new_Mass_Baryon=new_Mass_Baryon+gas_mass_dict[str(new_IDs_Baryon_temp)]

#             delta_m0_temp=new_Mass_Baryon
#             delta_m0.append(delta_m0_temp) #append the result to our final array
#             delta_m1.append(delta_m1_temp) #append the result to our final array 

#     ############################# Post-processing accretion calc results #############################
#     lt2=base_halo_data[isnap]['SimulationInfo']['LookbackTime']#final lookback time
#     lt1=base_halo_data[isnap-depth]['SimulationInfo']['LookbackTime']#initial lookback time
#     delta_t=abs(lt1-lt2)#lookback time change from initial to final snapshot (Gyr)

#     # Find which particle type is more massive (i.e. DM) and save accretion rates in dictionary
#     # 'DM_Acc', 'Gas_Acc' and 'dt' as Msun/Gyr and dt accordingly

#     delta_m={'DM_Acc':np.array(delta_m1)/delta_t,'DM_Acc_n':delta_n1,'Gas_Acc':np.array(delta_m0)/delta_t,'Gas_Acc_n':delta_n0,'dt':delta_t,'halo_index_list':halo_index_list}

#     # Now save all these accretion rates to file (in directory where run /acc_rates)
#     # (with filename depending on exact calculation parameters) - snap is the index in halo data
#     # will overwrite existing file (first deletes)

#     print('Saving accretion rates to .dat file.')
#     if trim_particles:
#         if include_unbound and trim_unbound:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#         elif not include_unbound and not trim_unbound:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#         elif include_unbound and not trim_unbound:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed3_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed3_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed3_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_trimmed3_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#     else:
#         if include_unbound:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#         else:
#             if path.exists('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat'):
#                 if verbose:
#                     print('Overwriting existing accretion data ...')
#                 os.remove('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#             with open('acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat', 'wb') as acc_data_file:
#                 print('Saving to acc_rates/snap_'+str(isnap).zfill(3)+'_accretion_base2_depth'+str(depth)+'_'+str(halo_index_list[0])+'-'+str(halo_index_list[-1])+'.dat')
#                 pickle.dump(delta_m,acc_data_file)
#                 acc_data_file.close()
#     #return the delta_m dictionary. 
#     return delta_m

# ########################### HALO INDEX LISTS GENERATOR ###########################

# def gen_halo_indices_mp(all_halo_indices,n_processes):
#     """

#     gen_halo_indices_mp : function
# 	----------

#     Generate list of lists of halo indices divided amongst a given amount of processes.

# 	Parameters
# 	----------
#     all_halo_indices : list or int
#         If list, a list of integer halo indices to divide.
#         If int, a list of integer halo indices up to the int is generated.

#     n_processes : int
#         Number of processes (likely number of cores) to distribute halo indices across. 

#     Returns
# 	----------
#     halo_index_lists : list of lists
#         The resulting halo index lists for each process. 

#     """
#     # Create halo index list from integer or provided list
#     if type(all_halo_indices)==int:
#         all_halo_indices=list(range(all_halo_indices))
#     else:
#         all_halo_indices=list(all_halo_indices)

#     n_halos=len(all_halo_indices)
#     halo_rem=n_halos%n_processes
#     n_halos_per_process=int(n_halos/n_processes)

#     #initialising loop variables
#     last_index=0
#     index_lists=[]
#     halo_index_lists=[]

#     #loop for each process to generate halo index lists
#     for iprocess in range(n_processes):
#         if halo_rem==0: #if there's an exact multiple of halos as cpu cores then distribute evenly
#             indices_temp=list(range(iprocess*n_halos_per_process,(iprocess+1)*n_halos_per_process))
#             index_lists.append(indices_temp)
#             halo_index_list_temp=[all_halo_indices[index_temp] for index_temp in indices_temp]
#             halo_index_lists.append(halo_index_list_temp)

#         else: #otherwise split halos evenly except last process
#             if iprocess<halo_rem:
#                 indices_temp=list(range(last_index,last_index+n_halos_per_process+1))
#                 index_lists.append(indices_temp)
#                 last_index=indices_temp[-1]+1
#                 halo_index_list_temp=[all_halo_indices[index_temp] for index_temp in indices_temp]
#                 halo_index_lists.append(halo_index_list_temp)

#             else:
#                 indices_temp=list(range(last_index,last_index+n_halos_per_process))
#                 index_lists.append(indices_temp)
#                 last_index=indices_temp[-1]+1
#                 halo_index_list_temp=[all_halo_indices[index_temp] for index_temp in indices_temp]
#                 halo_index_lists.append(halo_index_list_temp)

#     return halo_index_lists

# ########################### ACCRETION RATE FILE HANDLER ###########################

# def gen_filename_dataframe(directory):


#     """

#     gen_filename_dataframe : function
# 	----------

#     Generates a pandas dataframe of the filenames and associated characteristics of saved accretion rate files. 

# 	Parameters
# 	----------
#     directory : str
#         Where to search for accretion rate files. 

#     Returns
# 	----------
#     filename_dataframe : pd.DataFrame
#         DataFrame containing the keys listed below.

#         Keys

#             'filename': filename string
#             'type': 0 (base), 1 (trimmed)
#             'depth': snap gap
#             'span': n_halos per process in generation
#             'index1': first halo index
#             'index2': final halo index

#     """
    
#     desired_file_list=os.listdir(directory)
#     is_data_file=[]

#     for filename in desired_file_list:
#         is_data_file.append(filename.endswith('.dat'))
#     desired_file_list=np.compress(is_data_file,desired_file_list) #the data files

#     #initialise results
#     snaps=[]
#     calctype=[]
#     depths=[]
#     halo_range_1=[]
#     halo_range_2=[]
#     spans=[]

#     #iterate through each of the data files
#     for filename in desired_file_list:
#         file_split=filename.split('_')
#         snaps.append(int(file_split[1]))
#         if file_split[3]=="base":
#             calctype_temp=0
#         elif file_split[3]=="trimmed":
#             calctype_temp=1
#         elif file_split[3]=="base2":
#             calctype_temp=2
#         elif file_split[3]=="trimmed2":
#             calctype_temp=3
#         elif file_split[3]=="trimmed3":
#             calctype_temp=4
#         calctype.append(calctype_temp)
#         depths.append(int(file_split[4][-1]))
#         halo_range_temp=np.array(file_split[5][:-4].split('-')).astype(int)
#         halo_range_1.append(halo_range_temp[0])
#         halo_range_2.append(halo_range_temp[1])
#         spans.append(halo_range_temp[1]-halo_range_temp[0]+1)

#     #create data frame and order according to type, depth, then snap
#     filename_dataframe={'filename':desired_file_list,'snap':snaps,'type':calctype,'depth':depths,'index1':halo_range_1,'index2':halo_range_2,'span':spans}
#     filename_dataframe=df(filename_dataframe)
#     filename_dataframe=filename_dataframe.sort_values(by=['type','depth','snap','span','index1'])

#     return filename_dataframe

# ########################### ACCRETION RATE LOADER ###########################

# def load_accretion_rate(directory,calc_type,isnap,depth,span=[],verbose=1):
#     """

#     load_accretion_rate : function
# 	----------

#     Generates a pandas dataframe of accretion rates from file with given calculation parameters. 

# 	Parameters
# 	----------
#     directory : str
#         Where to search for accretion rate files.

#     calc_type : int
#         0: base
#         1: trimmed
#         2: base (bound only)
#         3: trimmed (bound only)
    
#     isnap : int
#         Snapshot index (in halo data) of accretion rate calculation.

#     span : int
#         The span in halo_indices of the calculation (normally n_halos/n_processes).


#     Returns
# 	----------
#     filename_dataframe : pd.DataFrame
#         DataFrame containing the keys listed below.

#         Keys

#             'filename': filename string
#             'type': 0 (base), 1 (trimmed)
#             'depth': snap gap
#             'span': n_halos per process in generation
#             'index1': first halo index
#             'index2': final halo index

#     """

#     filename_dataframe=gen_filename_dataframe(directory)
#     if span==[]:
#         correct_snap_spans=np.array(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==isnap))]['span'])
#         span_new=np.nanmax(correct_snap_spans)
#         correct_span=np.absolute(filename_dataframe['span']-span_new)<10
#     else:
#         span_new==span
#         correct_span=filename_dataframe['span']==span

#     relevant_files=list(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==isnap,filename_dataframe['depth']==depth,correct_span))]['filename'])
#     index1=list(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==isnap,filename_dataframe['depth']==depth,correct_span))]['index1'])
#     index2=list(filename_dataframe.iloc[np.logical_and.reduce((filename_dataframe['type']==calc_type,filename_dataframe['snap']==isnap,filename_dataframe['depth']==depth,correct_span))]['index2'])
    
#     if verbose:
#         print(f'Found {len(relevant_files)} accretion rate files (snap = {isnap}, type = {calc_type}, depth = {depth}, span = {span_new})')
    
#     acc_rate_dataframe={'DM_Acc':[],'DM_Acc_n':[],'Gas_Acc':[],'Gas_Acc_n':[],'Tot_Acc':[],'fb':[],'dt':[],'halo_index_list':[]}

#     acc_rate_dataframe=df(acc_rate_dataframe)

#     for ifile,ifilename in enumerate(relevant_files):
#         halo_indices=list(range(index1[ifile],index2[ifile]+1))
#         with open(directory+ifilename,'rb') as acc_rate_file:
#             dataframe_temp=pickle.load(acc_rate_file)
#             dataframe_temp=df(dataframe_temp)
#             dataframe_temp['Tot_Acc']=np.array(dataframe_temp['DM_Acc'])+np.array(dataframe_temp['Gas_Acc'])
#             dataframe_temp['fb']=np.array(dataframe_temp['Gas_Acc'])/(np.array(dataframe_temp['DM_Acc'])+np.array(dataframe_temp['Gas_Acc']))
#             acc_rate_dataframe=acc_rate_dataframe.append(dataframe_temp)
#             acc_rate_file.close()
#     acc_rate_dataframe=acc_rate_dataframe.sort_values(by=['halo_index_list'])

#     return acc_rate_dataframe
        
