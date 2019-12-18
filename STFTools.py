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
                                                                                                                                                                  
                                                                                                                                                                  
# STFTools.py - Python routines to read and process VELOCIraptor (Elahi+19) and TreeFrog (Elahi+19) outputs. 
# Author: RUBY WRIGHT 

# PREAMBLE
import os
import numpy as np
import h5py
import pickle
import astropy.units as u
import time
import read_eagle

from astropy.cosmology import FlatLambdaCDM,z_at_value
from os import path
from GenPythonTools import *
from ParticleTools import *
from VRPythonTools import *

########################### CREATE BASE HALO DATA ###########################

def gen_base_halo_data(partdata_filelist,partdata_filetype,vr_filelist,vr_filetype,tf_filelist,outname='',temporal_idval=10**12):
    
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

    Returns
	-------
    base_halo_data: list of dicts...
        A list (for each snap desired) of dictionaries which contain halo data with the following fields:
        'ID'
        'hostHaloID'
        'Xc'
        'Yc'
        'Zc'
        'VXc'
        'VYc'
        'VZc'
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
    
    Will save to file at: 
    B1_HaloData_outname.dat 
    B2_HaloData_outname.dat 

	"""

    if not os.path.exists('job_logs'):
        os.mkdir('job_logs')

    base_fields=['ID','hostHaloID','Structuretype',"numSubStruct",'Xc','Yc','Zc','Xcminpot','Ycminpot','Zcminpot','Xcmbp','Ycmbp','Zcmbp','VXc','VYc','VZc','R_200crit','R_200mean','Mass_200crit','Vmax']#default halo fields

    # File lists
    part_list=partdata_filelist#particle data filepaths -- padded with None for snaps we don't have
    vr_list=vr_filelist#velociraptor data filepaths -- padded with None for snaps we don't have
    tf_list=tf_filelist#treefrog data filepaths -- padded with None for snaps we don't have

    # Get snapshot indices from number of particle data files 
    sim_snaps=list(range(len(part_list)))
    halo_data_all=[]#initialise halo data list
    have_halo_data=[]#initialise flag list indicating existence of halo data at given snaps
    
    print('Reading halo data using VR python tools ...')
    for snap in sim_snaps:
        try:#attempt to find vr file from file list -if passes test, continues
            vr_list[snap].startswith('/')
            print(f'Searching for halo data at snap {snap} ...')
            print(f'[File: {vr_list[snap]}]')
        except:#if can't find vr file, skip this iteration and save empty halo data 
            have_halo_data.append(False)
            print(f'No halo data for snap {snap} (not given a file)')
            continue
           
        #use VR python tools to load in halo data for this snap
        halo_data_snap=ReadPropertyFile(vr_list[snap],ibinary=vr_filetype,iseparatesubfiles=0,iverbose=0, desiredfields=base_fields, isiminfo=True, iunitinfo=True)
        halo_data_snap[0]["Snap"]=snap
        #if data is found
        if not halo_data_snap==[]:
            halo_data_all.append(halo_data_snap)#will be length n_valid_snaps
            have_halo_data.append(True)#will be length n_ALL_snaps

        #if data is not found
        else:
            print("Couldn't find velociraptor files for snap = ",snap)
            return []#exit program if can't find vr files

    # List of number of halos detected for each snap and list isolated data dictionary for each snap (in dictionaries)
    halo_data_counts=[item[1] for item in halo_data_all]#will be length n_valid_snaps
    halo_data_all=[item[0] for item in halo_data_all]#will be length n_valid_snaps

    # Use TreeFrog IDs and convert hostHaloIDs if we don't have the temporal IDval integrated
    for isnap,halo_data_snap in enumerate(halo_data_all):#for the valid snaps
        halo_data_all[isnap]['Count']=halo_data_counts[isnap]#n_halos at this snap
        snap=halo_data_all[isnap]['Snap']
        try:
            if halo_data_snap["ID"][0]<temporal_idval:#if the first ID is less than the temporal IDval then do the conversion
                #read in IDs from TreeFrog
                treefile_compressed_isnap=tf_filelist[snap]+'.tree'
                treefile_isnap=h5py.File(treefile_compressed_isnap,'r+')
                treefile_ids=treefile_isnap["/ID"].value
                halo_data_all[isnap]["ID"]=treefile_ids
                treefile_isnap.close()

            if np.nanmax(halo_data_snap["hostHaloID"])<temporal_idval:#if the largest hostHaloID is less than the temporal IDval then do the conversion
                #read in IDs from TreeFrog
                for ihalo,hosthaloid in enumerate(halo_data_all[isnap]["hostHaloID"]):
                    if hosthaloid<0:
                        halo_data_all[isnap]["hostHaloID"][ihalo]=-1
                    else:
                        halo_data_all[isnap]["hostHaloID"][ihalo]=np.int64(isnap*temporal_idval)+hosthaloid
        except:
            pass

    # We have halo data, now load the trees
    # Import tree data from TreeFrog, build temporal head/tails from descendants -- adds to halo_data_all (all halo data)
    print('Now assembling descendent tree using VR python tools')
    no_tf_files=np.sum(have_halo_data)
    tf_filelist=np.compress(have_halo_data,tf_filelist)#compressing the TreeFrog filelist to valid snaps only 
    np.savetxt('job_logs/tf_filelist_compressed.txt',tf_filelist,fmt='%s')
    tf_filelist="job_logs/tf_filelist_compressed.txt"

    # Read in tree data
    halo_tree=ReadHaloMergerTreeDescendant(tf_filelist,ibinary=vr_filetype,iverbose=1,imerit=True,inpart=False)

    # Now build trees and add onto halo data array (for the valid, unpadded snaps)
    BuildTemporalHeadTailDescendant(no_tf_files,halo_tree,halo_data_counts,halo_data_all,iverbose=1,TEMPORALHALOIDVAL=temporal_idval)
    
    print('Finished assembling descendent tree using VR python tools')
    print('Adding timesteps & filepath information')
    
    # Adding timesteps and filepath information
    first_true_index=np.where(have_halo_data)[0][0]#finding first valid snap index to extract simulation data 
    H0=halo_data_all[first_true_index]['SimulationInfo']['h_val']*halo_data_all[first_true_index]['SimulationInfo']['Hubble_unit']#extract hubble constant
    Om0=halo_data_all[first_true_index]['SimulationInfo']['Omega_Lambda']#extract omega_lambda
    cosmo=FlatLambdaCDM(H0=H0,Om0=Om0)

    # Now tidy up and add extra details for output. 
    halo_data_output=[]
    isnap=-1
    for snap in sim_snaps:#for valid snaps, return the halo data dictionary and extra information
        if have_halo_data[snap]:
            isnap=isnap+1
            scale_factor=halo_data_all[isnap]['SimulationInfo']['ScaleFactor']
            redshift=z_at_value(cosmo.scale_factor,scale_factor,zmin=-0.5)
            lookback_time=cosmo.lookback_time(redshift).value
            halo_data_all[isnap]['SimulationInfo']['z']=redshift
            halo_data_all[isnap]['SimulationInfo']['LookbackTime']=lookback_time
            halo_data_all[isnap]['SimulationInfo']['Omega_b_Planck']=0.157332
            halo_data_all[isnap]['SimulationInfo']['BoxSize_Comoving']=halo_data_all[isnap]['SimulationInfo']['Period']*halo_data_all[isnap]['SimulationInfo']['h_val']/scale_factor
            if part_list[snap]:
                try:
                    halo_data_all[isnap]['SimulationInfo']['Mass_DM_Physical']=(h5py.File(part_list[snap],'r+')['Header'].attrs['MassTable'])[1]/halo_data_all[isnap]['SimulationInfo']['h_val']*10**10
                    halo_data_all[isnap]['SimulationInfo']['Mass_Gas_Physical']=halo_data_all[isnap]['SimulationInfo']['Mass_DM_Physical']*halo_data_all[isnap]['SimulationInfo']['Omega_b_Planck']/(1-halo_data_all[isnap]['SimulationInfo']['Omega_b_Planck'])
                except:
                    halo_data_all[isnap]['SimulationInfo']['Mass_DM_Physical']=9.70*10**6
                    halo_data_all[isnap]['SimulationInfo']['Mass_Gas_Physical']=1.81*10**6
                    
            halo_data_all[isnap]['VR_FilePath']=vr_list[snap]
            halo_data_all[isnap]['VR_FileType']=vr_filetype
            halo_data_all[isnap]['Part_FilePath']=part_list[snap]
            halo_data_all[isnap]['Part_FileType']=partdata_filetype
            halo_data_all[isnap]['PartHist_FilePath']=f'part_histories/PartHistory_{str(snap).zfill(3)}_{outname}.hdf5'
            halo_data_all[isnap]['outname']=outname
            halo_data_all[isnap]['Snap']=snap

            halo_data_output.append(halo_data_all[isnap])
        else:
            halo_data_output.append({'Snap':snap,'Part_FilePath':part_list[snap],'Part_FileType':partdata_filetype,'outname':outname})#for padded snaps, return particle data and snapshot 

    # Now save all the data (with detailed TreeFrog fields) as "B2"
    print('Saving B2 halo data to file (contains detailed TreeFrog data)')
    if path.exists('B2_HaloData_'+outname+'.dat'):
        print('Overwriting existing V2 halo data ...')
        os.remove('B2_HaloData_'+outname+'.dat')
    with open('B2_HaloData_'+outname+'.dat', 'wb') as halo_data_file:
        pickle.dump(halo_data_output, halo_data_file)
        halo_data_file.close()

    # Now save all the data (with detailed TreeFrog fields removed) as "B1" (saves memory for accretion calculations)
    fields_to_keep=['Count','Snap','Structuretype','numSubStruct','ID','hostHaloID','Tail','Head','VR_FilePath','VR_FileType','Part_FilePath','Part_FileType','PartHist_FilePath','UnitInfo','SimulationInfo','outname','Xc','Yc','Zc','Xcminpot','Ycminpot','Zcminpot','VXc','VYc','VZc','R_200crit','R_200mean','Mass_200crit','Vmax']
    halo_data_all_truncated=[]
    for snap,halo_data_snap in enumerate(halo_data_output):
        if have_halo_data[snap]:
            halo_data_all_truncated_snap={}
            for field in fields_to_keep:
                halo_data_all_truncated_snap[field]=halo_data_snap[field]
        else:
            halo_data_all_truncated_snap={'Snap':snap,'Part_FilePath':part_list[snap],'Part_FileType':partdata_filetype}
        halo_data_all_truncated.append(halo_data_all_truncated_snap)

    print('Saving B1 halo data to file (removing detailed TreeFrog data)')
    if path.exists('B1_HaloData_'+outname+'.dat'):
        print('Overwriting existing V1 halo data ...')
        os.remove('B1_HaloData_'+outname+'.dat')
    with open('B1_HaloData_'+outname+'.dat', 'wb') as halo_data_file:
        pickle.dump(halo_data_all_truncated, halo_data_file)
        halo_data_file.close()
    print('Done generating base halo data')

    return halo_data_output #returns the B2 version

########################### ADD DETAILED HALO DATA ###########################

def gen_detailed_halo_data(base_halo_data,snaps,vr_halo_fields=None,extra_halo_fields=None):
    
    """
    
    gen_detailed_halo_data : function
	----------

    Add detailed halo data to base halo data from property files.

    Parameters
    ----------

    base_halo_data_snap : list of dicts

        Dictionary for a snap containing basic halo data generated from gen_base_halo_data. 

    vr_halo_fields : list of str

        List of dictionary keys for halo properties (from velociraptor) to be added to the base halo data. 

    extra_halo_fields : list of str

        List of keys to add to halo data. Currently just supports 'R_rel', 'N_Peers', 'Subhalo_rank'

    Returns
    --------
    None
    
    saves to file:
    B3_HaloData_outname_snap.dat : list of dict

    Dictionaries for the parsed snaps which contain halo data with the following fields:

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
    t1=time.time()
    
    if not os.path.exists('halo_data'):
        os.mkdir('halo_data')

    if extra_halo_fields==None:
        extra_halo_fields=[]

    isnaps=snaps['indices']
    iprocess=snaps['iprocess']
    print(f'iprocess {iprocess} has snaps {isnaps}')

    for isnap in isnaps:
        base_halo_data_snap=base_halo_data[isnap]
        snap=base_halo_data_snap["Snap"]

        outfilename='halo_data/B3_HaloData_'+base_halo_data_snap['outname']+f'_{str(snap).zfill(3)}.dat'

        # If we're not given vr halo fields, find all of the available data fields
        if vr_halo_fields==None:
            try:
                print('Grabbing detailed halo data for snap',snap)
                property_filename=base_halo_data_snap['VR_FilePath']+".properties.0"
                property_file=h5py.File(property_filename)
                all_props=list(property_file.keys())
                vr_halo_fields=all_props
                if path.exists(outfilename):
                    print('Will overwrite existing B3 halo data ...')
                    os.remove(outfilename)
            except:
                print(f'Skipping padded snap ',snap)
                new_halo_data_snap=base_halo_data_snap
                dump_pickle(data=new_halo_data_snap, path=outfilename)
                continue
        
                
        print('Adding the following fields from properties file:')
        print(np.array(vr_halo_fields))

        base_fields=list(base_halo_data_snap.keys())
        fields_needed_from_prop=np.compress(np.logical_not(np.in1d(vr_halo_fields,base_fields)),vr_halo_fields)

        print('Will also collect the following fields from base halo data:')
        print(np.array(base_fields))

        # Loop through each snap and add the extra fields
        t1=time.time()    
        n_halos_snap=len(base_halo_data_snap['ID'])#number of halos at this snap

        # Read new halo data
        print(f'Adding detailed halo data for snap ',snap,' where there are ',n_halos_snap,' halos')
        new_halo_data_snap=ReadPropertyFile(base_halo_data_snap['VR_FilePath'],ibinary=base_halo_data_snap["VR_FileType"],iseparatesubfiles=0,iverbose=0, desiredfields=fields_needed_from_prop, isiminfo=True, iunitinfo=True)[0]

        # Adding old halo data from V1 calcs
        print(f'Adding fields from base halo data')
        for field in base_fields:
            new_halo_data_snap[field]=base_halo_data_snap[field]
        print('Done adding base fields')
        
        #Converting to physical 
        for new_field in list(new_halo_data_snap.keys()):
            if ('ass_' in new_field or 'M_' in new_field) and ('R_' not in new_field and 'rhalfmass' not in new_field):
                print(f'Converting {new_field} values to physical')
                new_halo_data_snap[new_field]=new_halo_data_snap[new_field]*10**10
            else:
                print(f'Not converting {new_field}')

        # Add extra halo fields -- post-process velociraptor files   
        if n_halos_snap>0:
            if 'R_rel' in extra_halo_fields: #Relative radius to host
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
                print('Done with R_rel')

            if 'M_rel' in extra_halo_fields:
                print('Adding M_rel information for subhalos')
                new_halo_data_snap['M_rel']=np.zeros(n_halos_snap)+np.nan #initialise to nan if field halo
                for ihalo in range(n_halos_snap):
                    hostID_temp=new_halo_data_snap['hostHaloID'][ihalo]
                    if not hostID_temp==-1:
                        #if we have a subhalo 
                        hostindex_temp=np.where(new_halo_data_snap['ID']==hostID_temp)[0][0]
                        host_M=new_halo_data_snap['Mass_FOF'][hostindex_temp]
                        sub_M=new_halo_data_snap['Mass_200crit'][ihalo]
                        M_rel_temp=sub_M/host_M
                        new_halo_data_snap['M_rel'][ihalo]=M_rel_temp
                print('Done with M_rel')

            if 'N_peers' in extra_halo_fields: #Number of peer subhalos
                print('Adding N_peers information for subhalos')
                new_halo_data_snap['N_peers']=np.zeros(len(new_halo_data_snap['ID']))+np.nan #initialise to nan if field halo
                for ihalo in range(n_halos_snap):
                    hostID_temp=new_halo_data_snap['hostHaloID'][ihalo]
                    if not hostID_temp==-1:
                        #if we have a subhalo
                        N_peers=np.sum(new_halo_data_snap['hostHaloID']==hostID_temp)-1
                        new_halo_data_snap['N_peers'][ihalo]=N_peers           
                print('Done with N_peers')

            if 'halotype' in extra_halo_fields:#0 = group, 1 = field halo, 2 =subhalo
                print('Adding halotype information ')
                new_halo_data_snap['halotype']=np.zeros(len(new_halo_data_snap['ID']))+np.nan #initialise to nan if field halo
                for ihalo in range(n_halos_snap):
                    hostID_temp=new_halo_data_snap['hostHaloID'][ihalo]
                    numsubstruct=new_halo_data_snap['numSubStruct'][ihalo]
                    if hostID_temp>=0:
                        new_halo_data_snap['halotype'][ihalo]=2
                    elif numsubstruct==0:
                        new_halo_data_snap['halotype'][ihalo]=1
                    else:
                        new_halo_data_snap['halotype'][ihalo]=0
                print('Done with halotype')

            if 'Subhalo_rank' in extra_halo_fields:# mass ordered rank for subhalos in a group/cluster
                print('Adding Subhalo_rank information for subhalos')
                new_halo_data_snap['Subhalo_rank']=np.zeros(len(new_halo_data_snap['ID']))
                processed_hostIDs=[]
                for ihalo in range(n_halos_snap):
                    hostID_temp=new_halo_data_snap['hostHaloID'][ihalo]
                    #if we have a subhalo
                    if not hostID_temp==-1:
                        if hostID_temp not in processed_hostIDs:
                            processed_hostIDs.append(hostID_temp)
                            mass=new_halo_data_snap['Mass_200crit'][ihalo]
                            peer_indices=np.where(new_halo_data_snap['hostHaloID']==hostID_temp)[0]
                            peer_ranks=rank_list([new_halo_data_snap['Mass_200crit'][ihalo_peer] for ihalo_peer in peer_indices])
                            for ipeer_index,peer_index in enumerate(peer_indices):
                                new_halo_data_snap["Subhalo_rank"][peer_index]=peer_ranks[ipeer_index]
                print('Done with Subhalo_rank')

        else: #if insufficient halos at snap
            print('Skipping adding the extra halo fields for this snap (insufficient halo count)')

        t2=time.time()

        # Save data to file
        print(f'Saving halo data for snap {snap} to file ...')
        dump_pickle(data=new_halo_data_snap, path=outfilename)
    
    return None

########################### COLLATE DETAILED HALO DATA ###########################

def postprocess_detailed_halo_data(path=None):
    """
    
    postprocess_detailed_halo_data : function
	----------

    Collates all the detailed snapshot halo data in given path into one combined file. 

    Parameters
    ----------

    path : str

        Path which contains the detailed halo data files for each snapshot. 

    Returns
    --------
    None

    saves to file:
    B3_HaloData_outname.dat : list of dict

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

    if path==None:
        path='halo_data/'
    
    if not path.endswith('/'):
        path=path+'/'
    
    halo_data_files=sorted(os.listdir(path))
    halo_data_files_trunc=[halo_data_file for halo_data_file in halo_data_files if 'HaloData' in halo_data_file]
    halo_data_files_wdir=[path+halo_data_file for halo_data_file in halo_data_files_trunc]
    outfilename=halo_data_files_trunc[-1][:-8]+'.dat'
    if os.path.exists(outfilename):
        print('Removing existing detailed halo data ...')
        os.remove(outfilename)
    print('Will save full halo data to: ',outfilename)
    print(f'Number of halo data snaps: {len(halo_data_files_wdir)}')
    full_halo_data=[[] for i in range(len(halo_data_files_wdir))]
    for isnap,halo_data_file in enumerate(halo_data_files_wdir):
        print(f'Adding to full halo data for isnap {isnap}')
        halo_data_snap=open_pickle(halo_data_file)
        full_halo_data[isnap]=halo_data_snap
        
    dump_pickle(data=full_halo_data,path=outfilename)
    return full_halo_data

########################### COMPRESS DETAILED HALO DATA ###########################

def compress_detailed_halo_data(detailed_halo_data,fields=None):
        
    """
    
    compress_halo_data : function
	----------

    Compress halo data list of dicts for desired fields. 

    Parameters
    ----------

    detailed_halo_data : list of dicts

        List (for each snap) of dictionaries containing full halo data generated from gen_detailed_halo_data. 

    fields : list of str

        List of dictionary keys for halo properties (from velociraptor) to be saved to the compressed halo data. 

    Returns
    --------
    None

    saves to file:
    B4_HaloData_outname.dat : list of dict

    A list (for each snap desired) of dictionaries which contain halo data with the desired fields, which by default will always contain:
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
        'outname'
        
        And any extras -- defaults:
        "Mass_tot"
        "Mass_gas"
        "Mass_200crit"
        "Mass_200mean"
        "Npart"
        
        """
    print('Compressing detailed halo data ...')

    #process fields to include defaults + those desired
    default_fields=['outname',
    'Part_FileType',
    'Part_FilePath',
    'PartHist_FilePath',
    'VR_FilePath',
    'VR_FileType',
    'ID',
    'hostHaloID',
    'Snap',
    'Head',
    'Tail',
    'SimulationInfo',
    'UnitInfo',
    'outname',
    "npart",
    "Mass_tot",
    'Mass_FOF',
    "M_gas",
    "Mass_200crit",
    "Mass_200mean",
    "Mass_200crit_excl",
    "Aperture_mass_100_kpc",
    "Aperture_mass_10_kpc",
    "Aperture_mass_30_kpc",
    "Aperture_mass_50_kpc",
    "Aperture_mass_5_kpc",
    "Aperture_mass_gas_100_kpc",
    "Aperture_mass_gas_10_kpc",
    "Aperture_mass_gas_30_kpc",
    "Aperture_mass_gas_50_kpc",
    "Aperture_mass_gas_5_kpc",
    "Aperture_mass_gas_nsf_100_kpc",
    "Aperture_mass_gas_nsf_10_kpc",
    "Aperture_mass_gas_nsf_30_kpc",
    "Aperture_mass_gas_nsf_50_kpc",
    "Aperture_mass_gas_nsf_5_kpc",
    "Aperture_mass_gas_sf_100_kpc",
    "Aperture_mass_gas_sf_10_kpc",
    "Aperture_mass_gas_sf_30_kpc",
    "Aperture_mass_gas_sf_50_kpc",
    "Aperture_mass_gas_sf_5_kpc",
    "Aperture_mass_star_100_kpc",
    "Aperture_mass_star_10_kpc",
    "Aperture_mass_star_30_kpc",
    "Aperture_mass_star_50_kpc",
    "Aperture_mass_star_5_kpc",
    "Aperture_rhalfmass_100_kpc",
    "Aperture_rhalfmass_10_kpc",
    "Aperture_rhalfmass_30_kpc",
    "Aperture_rhalfmass_50_kpc",
    "Aperture_rhalfmass_5_kpc",
    "Aperture_rhalfmass_gas_100_kpc",
    "Aperture_rhalfmass_gas_10_kpc",
    "Aperture_rhalfmass_gas_30_kpc",
    "Aperture_rhalfmass_gas_50_kpc",
    "Aperture_rhalfmass_gas_5_kpc",
    "Aperture_rhalfmass_gas_nsf_100_kpc",
    "Aperture_rhalfmass_gas_nsf_10_kpc",
    "Aperture_rhalfmass_gas_nsf_30_kpc",
    "Aperture_rhalfmass_gas_nsf_50_kpc",
    "Aperture_rhalfmass_gas_nsf_5_kpc",
    "Aperture_rhalfmass_gas_sf_100_kpc",
    "Aperture_rhalfmass_gas_sf_10_kpc",
    "Aperture_rhalfmass_gas_sf_30_kpc",
    "Aperture_rhalfmass_gas_sf_50_kpc",
    "Aperture_rhalfmass_gas_sf_5_kpc",
    "Aperture_rhalfmass_star_100_kpc",
    "Aperture_rhalfmass_star_10_kpc",
    "Aperture_rhalfmass_star_30_kpc",
    "Aperture_rhalfmass_star_50_kpc",
    "Aperture_rhalfmass_star_5_kpc",
    "R_HalfMass",
    "R_HalfMass_gas",
    "R_HalfMass_gas_nsf",
    "R_HalfMass_gas_sf",
    "R_HalfMass_star",
    "R_size",
    "R_200crit",
    "R_200mean",
    "Xc",
    "Yc",
    "Zc",
    "Xcminpot",
    "Ycminpot",
    "Zcminpot",
    "Xcmbp",
    "Ycmbp",
    "Zcmbp",
    "Rmax",
    "SFR_gas",
    "M_rel",
    "Subhalo_rank",
    "R_rel",
    "halotype",
    ]

    if fields==None:
        fields=default_fields

    no_snaps=len(detailed_halo_data)
    snap_mask=[len(detailed_halo_data_snap)>5 for detailed_halo_data_snap in detailed_halo_data]

    output_halo_data=[{field:[] for field in fields} for isnap in range(no_snaps)]
    outname=detailed_halo_data[-1]['outname']

    for snap, detailed_halo_data_snap in enumerate(detailed_halo_data):
        if snap_mask[snap]:
            print(f'Processing halo data for snap {snap} ({outname}) ...')
            for field in fields:
                print(f'Field: {field}')
                try:
                    output_halo_data[snap][field]=detailed_halo_data_snap[field]
                except:
                    print(f"Couldn't get {field} data")
                    pass
        else:
            output_halo_data[snap]=detailed_halo_data_snap
    
    file_outname=f'B4_HaloData_{outname}.dat'
    if os.path.exists(file_outname):
        os.remove(file_outname)
    dump_pickle(path=file_outname,data=output_halo_data)
    return output_halo_data

########################### FIND PROGENITOR AT DEPTH ###########################

def find_progen_index(base_halo_data,index2,snap2,depth): ### given halo index2 at snap 2, find progenitor index at snap1=snap2-depth
    
 
    """

    find_progen_index : function
	----------

    Find the index of the best matching progenitor halo at the previous snap. 

	Parameters
    ----------

    base_halo_data : dictionary
        The halo data dictionary for the relevant snapshot.

    index2 : int
        The index of the halo at the current (accretion) snap. 

    snap2 : int
        The snapshot index of the current (accretion) snap.
    
    depth : int
        The number of snapshots for which to scroll back. 

    Returns
    ----------
    index1 : int
        The index of the best matched halo at the desired snap. 

	"""

    padding=np.sum([len(base_halo_data[isnap])<5 for isnap in range(len(base_halo_data))])
    index_idepth=index2
    for idepth in range(depth):
        current_ID=base_halo_data[snap2-idepth]["ID"][index_idepth]
        tail_ID=base_halo_data[snap2-idepth]["Tail"][index_idepth]
        index_idepth=np.where(base_halo_data[snap2-idepth-1]["ID"]==tail_ID)[0]
        if len(index_idepth)==0:
            index_idepth=np.nan
            break
        else:
            index_idepth=index_idepth[0]
            if idepth==depth-1:
                return index_idepth
    return index_idepth

########################### FIND DESCENDANT AT DEPTH ###########################

def find_descen_index(base_halo_data,index2,snap2,depth): ### given halo index2 at snap 2, find descendant index at snap3=snap2+depth
    
    """

    find_descen_index : function
	----------

    Find the index of the best matching descendent halo at the following snap. 

	Parameters
    ----------

    base_halo_data : dictionary
        The halo data dictionary for the relevant snapshot.

    index2 : int
        The index of the halo at the current (accretion) snap. 

    snap2 : int
        The snapshot index of the current (accretion) snap.
    
    depth : int
        The number of snapshots for which to scroll forward. 

    Returns
    ----------
    index3 : int
        The index of the best matched halo at the desired snap. 

	"""

    padding=np.sum([len(base_halo_data[isnap])<5 for isnap in range(len(base_halo_data))])
    index_idepth=index2
    for idepth in range(depth):
        current_ID=base_halo_data[snap2+idepth]["ID"][index_idepth]
        head_ID=base_halo_data[snap2+idepth]["Head"][index_idepth]
        index_idepth=np.where(base_halo_data[snap2+idepth+1]["ID"]==head_ID)[0]
        if len(index_idepth)==0:
            index_idepth=np.nan
            break
        else:
            index_idepth=index_idepth[0]
            if idepth==depth-1:
                return index_idepth
    return index_idepth

########################### GET FOF PARTICLE LISTS into DICTIONARY ###########################

def get_FOF_particle_lists(base_halo_data,snap,halo_index_list=None):

    keys_FOF=["Npart","Npart_unbound",'Particle_IDs','Particle_Types',"Particle_Bound"]
    num_tot_halos=len(base_halo_data[snap]['ID'])
    if halo_index_list==None:
        halo_index_list=list(range(num_tot_halos))

    try:
        print('Reading FOF halo particle lists for snap = ',snap)
        part_data_temp_FOF=ReadParticleDataFile(base_halo_data[snap]['VR_FilePath'],iparttypes=1,ibinary=base_halo_data[snap]['VR_FileType'],iverbose=0)
        print('Read FOF halo particle lists for snap = ',snap)
        part_data_temp_FOF_dict={field: {} for field in keys_FOF}
        for field in keys_FOF:
            part_data_temp_FOF_dict[field]={str(ihalo):part_data_temp_FOF[field][ihalo] for ihalo in range(num_tot_halos)}
        part_data_temp_FOF=part_data_temp_FOF_dict
        part_data_temp_FOF['Particle_InHost']={}
        for ihalo in range(num_tot_halos):
            part_data_temp_FOF['Particle_InHost'][str(ihalo)]=np.ones(part_data_temp_FOF["Npart"][str(ihalo)])

    except: #if we can't load particle data
        print('Couldnt get FOF particle data for snap = ',snap)
        return None
    halo_index_list_keys=list(part_data_temp_FOF['Npart'].keys())
    halo_index_list_indices=[int(halo_index_list_key) for halo_index_list_key in halo_index_list_keys]

    print('Appending FOF particle lists with substructure')
    field_halo_indices_temp=np.where(base_halo_data[snap]['hostHaloID']==-1)[0]#find field/fof halos
    for i_field_halo,field_halo_ID in enumerate(base_halo_data[snap]['ID'][field_halo_indices_temp]):#go through each field halo
        sub_halos_temp=(np.where(base_halo_data[snap]['hostHaloID']==field_halo_ID)[0])#find the indices of its subhalos
        if len(sub_halos_temp)>0:#where there is substructure
            field_halo_temp_index=field_halo_indices_temp[i_field_halo]
            field_halo_plist=part_data_temp_FOF['Particle_IDs'][str(field_halo_temp_index)]
            field_halo_hlist=np.ones(len(field_halo_plist)).astype(int)
            field_halo_tlist=part_data_temp_FOF['Particle_Types'][str(field_halo_temp_index)]
            field_halo_blist=part_data_temp_FOF['Particle_Bound'][str(field_halo_temp_index)]
            sub_halos_plist=np.concatenate([part_data_temp_FOF['Particle_IDs'][str(isub)] for isub in sub_halos_temp])#list all particles IDs in substructure
            sub_halos_hlist=np.zeros(len(sub_halos_plist)).astype(int)
            sub_halos_tlist=np.concatenate([part_data_temp_FOF['Particle_Types'][str(isub)] for isub in sub_halos_temp])#list all particles types substructure
            sub_halos_blist=np.concatenate([part_data_temp_FOF['Particle_Bound'][str(isub)] for isub in sub_halos_temp])#list all particles bound 
            part_data_temp_FOF['Particle_IDs'][str(field_halo_temp_index)],unique_indices=np.unique(np.concatenate([field_halo_plist,sub_halos_plist]),return_index=True)#add particles to field halo particle list
            part_data_temp_FOF['Particle_InHost'][str(field_halo_temp_index)]=np.concatenate([field_halo_hlist,sub_halos_hlist])[unique_indices]#add particles to field halo particle list
            part_data_temp_FOF['Particle_Types'][str(field_halo_temp_index)]=np.concatenate([field_halo_tlist,sub_halos_tlist])[unique_indices]#add particles to field halo particle list
            part_data_temp_FOF['Particle_Bound'][str(field_halo_temp_index)]=np.concatenate([field_halo_blist,sub_halos_blist])[unique_indices]#add particles to field halo particle list
            part_data_temp_FOF['Npart'][str(field_halo_temp_index)]=len(part_data_temp_FOF['Particle_IDs'][str(field_halo_temp_index)])#update Npart for each field halo
    
    print('Finished appending FOF particle lists with substructure and ensuring unique')

    keys_FOF_all=["Npart","Npart_unbound",'Particle_IDs','Particle_Types',"Particle_Bound","Particle_InHost"]

    part_data_temp_FOF_trunc={}
    for field in keys_FOF_all:
        part_data_temp_FOF_trunc[field]={str(ihalo):part_data_temp_FOF[field][str(ihalo)] for ihalo in halo_index_list if ihalo>=0}
    
    return part_data_temp_FOF_trunc

########################### DUMP FOF/SO PARTICLE DATA ###########################

def dump_structure_particle_data(base_halo_data,snaps,add_partdata=True,ifofs=True,isos=True):
    
    """

    dump_structure_particle_data : function
	----------

    Retrieve the particle lists for each halo and so region for the provided halo data dictionary 
    (and corresponding snapshot) from velociraptor.

	Parameters
    ----------

    base_halo_data : list of dictionaries 
        The halo data list for the relevant simulation.
    
    snap : int
        The snap to retrieve particle lists for. 

    add_partdata : bool
        Whether to add particle data to the result. 

    ifof : bool
        Whether to add data for FOFs (default = True). 

    iso : bool
        Whether to add data for SOs (default = True). 

    Returns
    ----------
    part_data_temp : dictionary 
        The particle IDs, Types, and counts for the given snapshot in a dictionary
        Keys: 
            "Particle_IDs" - list (for each halo) of lists of particle IDs
            "Particle_Types" - list (for each halo) of lists of particle Types
            "Particle_Bound" - list (for each halo) of lists of the bound status of each particle
            "Particle_InHost" - list (for each halo) of lists of the bound status of each particle
            "Npart" - list (for each halo) of the number of particles belonging to the object

            and, if add_partdata:
            "Coordinates"
            "Velicity"

	"""

    if snaps==None:
        snap_indices=[len(base_halo_data)-1]
    else:
        snap_indices=snaps["indices"] #extract index list from input dictionary

    if base_halo_data[-1]['Part_FileType']=='EAGLE':
        parttypes=[0,1,4,5]
    else:
        parttypes=[0,1]

    outfolder='halo_data/part_data/'
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    runname=base_halo_data[-1]["outname"]
    
    part_fields=['Coordinates','Velocity','Mass']

    for snap in snap_indices:
        print(f'****************************************************')
        print(f'Generating data for particles in structure at snap {snap} ...')
        print(f'****************************************************')

        halo_index_list=list(range(len(base_halo_data[snap]['ID'])))
        so_index_list=list(range(len(base_halo_data[snap]['ID'])))[22000:]


        # Use VR python tools to grab FOF particle data
        if ifofs:
            keys_FOF=["Npart","Npart_unbound",'Particle_IDs','Particle_Types',"Particle_Bound"]
            try:
                print('Reading FOF halo particle lists for snap = ',snap)
                part_data_temp_FOF=ReadParticleDataFile(base_halo_data[snap]['VR_FilePath'],iparttypes=1,ibinary=base_halo_data[snap]['VR_FileType'],iverbose=0)
                print('Read FOF halo particle lists for snap = ',snap)
                num_tot_halos=len(part_data_temp_FOF["Npart"])
                part_data_temp_FOF_dict={field: {} for field in keys_FOF}
                for field in keys_FOF:
                    part_data_temp_FOF_dict[field]={str(ihalo):part_data_temp_FOF[field][ihalo] for ihalo in range(num_tot_halos)}
                part_data_temp_FOF=part_data_temp_FOF_dict
                part_data_temp_FOF['Particle_InHost']={}
                for ihalo in range(num_tot_halos):
                    part_data_temp_FOF['Particle_InHost'][str(ihalo)]=np.ones(part_data_temp_FOF["Npart"][str(ihalo)])

            except: #if we can't load particle data
                print('Couldnt get FOF particle data for snap = ',snap)
                return None
            halo_index_list_keys=list(part_data_temp_FOF['Npart'].keys())
            halo_index_list_indices=[int(halo_index_list_key) for halo_index_list_key in halo_index_list_keys][:100]

            print('Appending FOF particle lists with substructure')
            field_halo_indices_temp=np.where(base_halo_data[snap]['hostHaloID']==-1)[0]#find field/fof halos
            for i_field_halo,field_halo_ID in enumerate(base_halo_data[snap]['ID'][field_halo_indices_temp]):#go through each field halo
                sub_halos_temp=(np.where(base_halo_data[snap]['hostHaloID']==field_halo_ID)[0])#find the indices of its subhalos
                if len(sub_halos_temp)>0:#where there is substructure
                    field_halo_temp_index=field_halo_indices_temp[i_field_halo]
                    field_halo_plist=part_data_temp_FOF['Particle_IDs'][str(field_halo_temp_index)]
                    field_halo_hlist=np.ones(len(field_halo_plist)).astype(int)
                    field_halo_tlist=part_data_temp_FOF['Particle_Types'][str(field_halo_temp_index)]
                    field_halo_blist=part_data_temp_FOF['Particle_Bound'][str(field_halo_temp_index)]
                    sub_halos_plist=np.concatenate([part_data_temp_FOF['Particle_IDs'][str(isub)] for isub in sub_halos_temp])#list all particles IDs in substructure
                    sub_halos_hlist=np.zeros(len(sub_halos_plist)).astype(int)
                    sub_halos_tlist=np.concatenate([part_data_temp_FOF['Particle_Types'][str(isub)] for isub in sub_halos_temp])#list all particles types substructure
                    sub_halos_blist=np.concatenate([part_data_temp_FOF['Particle_Bound'][str(isub)] for isub in sub_halos_temp])#list all particles bound 
                    part_data_temp_FOF['Particle_IDs'][str(field_halo_temp_index)],unique_indices=np.unique(np.concatenate([field_halo_plist,sub_halos_plist]),return_index=True)#add particles to field halo particle list
                    part_data_temp_FOF['Particle_InHost'][str(field_halo_temp_index)]=np.concatenate([field_halo_hlist,sub_halos_hlist])[unique_indices]#add particles to field halo particle list
                    part_data_temp_FOF['Particle_Types'][str(field_halo_temp_index)]=np.concatenate([field_halo_tlist,sub_halos_tlist])[unique_indices]#add particles to field halo particle list
                    part_data_temp_FOF['Particle_Bound'][str(field_halo_temp_index)]=np.concatenate([field_halo_blist,sub_halos_blist])[unique_indices]#add particles to field halo particle list
                    part_data_temp_FOF['Npart'][str(field_halo_temp_index)]=len(part_data_temp_FOF['Particle_IDs'][str(field_halo_temp_index)])#update Npart for each field halo
            print('Finished appending FOF particle lists with substructure and ensuring unique')
        
            if add_partdata:        
                #Load particle histories    
                print('Loading particle history data ...')
                PartHistories_Snap_File=h5py.File(base_halo_data[snap]['PartHist_FilePath'],'r')
                PartHistories_Snap_IDs={str(itype):PartHistories_Snap_File[f"PartType{itype}"]["ParticleIDs"].value for itype in parttypes}
                PartHistories_Snap_Indices={str(itype):PartHistories_Snap_File[f"PartType{itype}"]["ParticleIndex"].value for itype in parttypes}

                #Load basic sim data for this snap
                h_val=base_halo_data[snap]['SimulationInfo']['h_val']
                scalefactor_Snap=base_halo_data[snap]['SimulationInfo']['ScaleFactor']
                dm_mass=base_halo_data[snap]['SimulationInfo']['Mass_DM_Physical']*h_val

                #Load simulation data if needed
                print('Loading simulation data ...')
                if base_halo_data[snap]['Part_FileType']=='EAGLE':
                    EAGLE_boxsize=base_halo_data[snap]['SimulationInfo']['BoxSize_Comoving']
                    EAGLE_Snap=read_eagle.EagleSnapshot(base_halo_data[snap]['Part_FilePath'])
                    EAGLE_Snap.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
                    PartData_Datasets_Snap_FULL={}
                    for field in part_fields:
                        PartData_Datasets_Snap_FULL[field]={}
                        for itype in parttypes:
                            if not field=='Mass':
                                PartData_Datasets_Snap_FULL[field][str(itype)]=EAGLE_Snap.read_dataset(itype,field)
                            else:
                                if itype==1:
                                    PartData_Datasets_Snap_FULL[field][str(itype)]=np.ones(len(PartData_Datasets_Snap_FULL['Coordinates']['1']))*dm_mass
                                else:
                                    PartData_Datasets_Snap_FULL[field][str(itype)]=EAGLE_Snap.read_dataset(itype,field)*10**10
                else:
                    PartData_Datasets_Snap_FULL={field:{str(itype):h5py.File(base_halo_data[snap]['Part_FilePath'],'r')[f'PartType{itype}'][field] for itype in parttypes} for field in part_fields}

            # Process FOF lists
            #initialise outputs
            fields_FOF=list(part_data_temp_FOF.keys())
            if add_partdata:
                fields_FOF.extend(part_fields)
                for field in part_fields:
                    part_data_temp_FOF[field]={}

            #output file
            outpath_FOF=outfolder+f'ParticleData_{runname}_{str(snap).zfill(3)}.hdf5'
            if os.path.exists(outpath_FOF):
                outfile_FOF=h5py.File(outpath_FOF,'r+')
            else:
                outfile_FOF=h5py.File(outpath_FOF,'w')

            print('Entering main halo loop to grab extra particle data ...')
            for iihalo,ihalo in enumerate(halo_index_list):
                if ihalo<10:
                    print(f'Processing ihalo {ihalo} ({iihalo/len(halo_index_list)*100:.1f}% done)')
                elif iihalo%100==0:
                    print(f'Processing ihalo {ihalo} ({iihalo/len(halo_index_list)*100:.1f}% done)')

                ifield=base_halo_data[snap]['hostHaloID'][ihalo]==-1
                ihalo_FOF_IDlist=part_data_temp_FOF["Particle_IDs"][str(ihalo)]
                ihalo_FOF_typelist=part_data_temp_FOF["Particle_Types"][str(ihalo)]
                ihalo_Npart=len(ihalo_FOF_IDlist)

                ihalo_types,ihalo_historyindices,ihalo_partindices=get_particle_indices(base_halo_data,
                                                                    IDs_sorted=PartHistories_Snap_IDs,
                                                                    indices_sorted=PartHistories_Snap_Indices,
                                                                    IDs_taken=ihalo_FOF_IDlist,
                                                                    types_taken=ihalo_FOF_typelist,
                                                                    snap_taken=snap,
                                                                    snap_desired=snap)
                if add_partdata:
                    for field in part_fields:
                        if field=='Coordinates' or field =='Velocity':
                            conversion_to_physical_fac=scalefactor_Snap/h_val
                            part_data_temp_FOF[field][str(ihalo)]=np.zeros((ihalo_Npart,3))
                        elif field=='Mass':
                            conversion_to_physical_fac=1/h_val
                            part_data_temp_FOF[field][str(ihalo)]=np.zeros(ihalo_Npart)
                        else:
                            conversion_to_physical_fac=1.0
                            part_data_temp_FOF[field][str(ihalo)]=np.zeros(ihalo_Npart)

                        for itype in parttypes:
                            itype_mask=np.where(ihalo_types==itype)
                            itype_partindices=ihalo_partindices[itype_mask]
                            part_data_temp_FOF[field][str(ihalo)][itype_mask]=PartData_Datasets_Snap_FULL[field][str(itype)][(itype_partindices,)]*conversion_to_physical_fac

                #SAVE TO FILE
                try:
                    ihalo_group=outfile_FOF.create_group(f'ihalo_{str(ihalo).zfill(6)}')
                except:
                    ihalo_group=outfile_FOF[f'ihalo_{str(ihalo).zfill(6)}']
                    datasets=list(ihalo_group.keys())
                    for dataset in datasets:
                        del ihalo_group[dataset]

                ihalo_Npart=part_data_temp_FOF["Npart"][str(ihalo)]
                ihalo_meancop=np.array([np.nanmean(part_data_temp_FOF["Coordinates"][str(ihalo)][:,i]) for i in range(3)])
                ihalo_mediancop=np.array([np.nanmedian(part_data_temp_FOF["Coordinates"][str(ihalo)][:,i]) for i in range(3)])
                ihalo_com=np.array([base_halo_data[snap]['Xc'][ihalo],base_halo_data[snap]['Yc'][ihalo],base_halo_data[snap]['Zc'][ihalo]])
                ihalo_cminpot=np.array([base_halo_data[snap]['Xcminpot'][ihalo],base_halo_data[snap]['Ycminpot'][ihalo],base_halo_data[snap]['Zcminpot'][ihalo]])
                ihalo_cmbp=np.array([base_halo_data[snap]['Xcmbp'][ihalo],base_halo_data[snap]['Ycmbp'][ihalo],base_halo_data[snap]['Zcmbp'][ihalo]])
                ihalo_r200crit=base_halo_data[snap]['R_200crit'][ihalo]
                ihalo_r200mean=base_halo_data[snap]['R_200mean'][ihalo]
                
                for field in fields_FOF:
                    if field=='Particle_IDs':
                        ihalo_group.require_dataset(field,data=part_data_temp_FOF[field][str(ihalo)],dtype=np.uint64,shape=np.shape(part_data_temp_FOF[field][str(ihalo)]))
                    elif 'Particle' in field:
                        ihalo_group.require_dataset(field,data=part_data_temp_FOF[field][str(ihalo)],dtype=np.int8,shape=np.shape(part_data_temp_FOF[field][str(ihalo)]))
                    else:
                        ihalo_group.require_dataset(field,data=part_data_temp_FOF[field][str(ihalo)],dtype=np.float32,shape=np.shape(part_data_temp_FOF[field][str(ihalo)]))

                ihalo_group.require_dataset('ihalo_meancop',data=ihalo_meancop,dtype=np.float32,shape=(1,3))
                ihalo_group.require_dataset('ihalo_mediancop',data=ihalo_mediancop,dtype=np.float32,shape=(1,3))
                ihalo_group.require_dataset('ihalo_com',data=ihalo_com,dtype=np.float32,shape=(1,3))
                ihalo_group.require_dataset('ihalo_cminpot',data=ihalo_cminpot,dtype=np.float32,shape=(1,3))
                ihalo_group.require_dataset('ihalo_cmbp',data=ihalo_cmbp,dtype=np.float32,shape=(1,3))
                ihalo_group.require_dataset('ihalo_r200crit',data=ihalo_r200crit,dtype=np.float32,shape=(1,))
                ihalo_group.require_dataset('ihalo_r200mean',data=ihalo_r200mean,dtype=np.float32,shape=(1,))

            outfile_FOF.close()
       
        # SO LISTS
        if isos:
            #factors
            h_val=base_halo_data[snap]['SimulationInfo']['h_val']
            scalefactor=base_halo_data[snap]['SimulationInfo']['ScaleFactor']
            dm_mass=base_halo_data[snap]['SimulationInfo']['Mass_DM_Physical']*h_val#comoving
            phystocom=h_val/scalefactor
            comtophys=1.0/phystocom

            #output file
            outpath_SO=outfolder+f'ParticleData_{runname}_{str(snap).zfill(3)}.hdf5'
            if os.path.exists(outpath_SO):
                outfile_SO=h5py.File(outpath_SO,'r+')
            else:
                outfile_SO=h5py.File(outpath_SO,'w')

            print('Entering main so loop to grab extra particle data ...')
            #initialise outputs
            part_data_temp_SO={}
            fields_SO=['Particle_IDs','Particle_Types','Npart']
            if add_partdata:
                fields_SO.extend(part_fields)
  
            for field in fields_SO:
                part_data_temp_SO[field]={}
            
            for iiso,iso in enumerate(so_index_list):
                if iso<10:
                    print(f'Processing iso {iso} at snap {snap} ({iiso/len(so_index_list)*100:.1f}% done)')
                elif iiso%100==0:
                    print(f'Processing iso {iso} at snap {snap} ({iiso/len(so_index_list)*100:.1f}% done)')

                iso_r200mean=base_halo_data[snap]['R_200mean'][iso]*phystocom
                iso_r200crit=base_halo_data[snap]['R_200crit'][iso]*phystocom
                iso_com=np.array([base_halo_data[snap]['Xc'][iso],base_halo_data[snap]['Yc'][iso],base_halo_data[snap]['Zc'][iso]])*phystocom
                iso_cminpot=np.array([base_halo_data[snap]['Xcminpot'][iso],base_halo_data[snap]['Ycminpot'][iso],base_halo_data[snap]['Zcminpot'][iso]])*phystocom
                iso_cmbp=np.array([base_halo_data[snap]['Xcmbp'][iso],base_halo_data[snap]['Ycmbp'][iso],base_halo_data[snap]['Zcmbp'][iso]])*phystocom

                #LOAD/SLICE EAGLE SNAPSHOT
                snapshot=read_eagle.EagleSnapshot(fname=base_halo_data[snap]['Part_FilePath'])
                snapshot.select_region(xmin=iso_com[0]-iso_r200mean,xmax=iso_com[0]+iso_r200mean,ymin=iso_com[1]-iso_r200mean,ymax=iso_com[1]+iso_r200mean,zmin=iso_com[2]-iso_r200mean,zmax=iso_com[2]+iso_r200mean)

                iso_IDs=[snapshot.read_dataset(itype,'ParticleIDs') for itype in [0,1,4,5]]
                iso_IDs_concatenated=np.concatenate(iso_IDs)
                iso_npart=[len(iso_IDs[iitype]) for iitype in range(len(iso_IDs))]
                iso_types=np.concatenate([np.ones(iso_npart[iitype])*itype for iitype,itype in enumerate([0,1,4,5])]).astype(int)

                part_data_temp_SO['Particle_IDs'][str(iso)]=iso_IDs_concatenated
                part_data_temp_SO['Particle_Types'][str(iso)]=iso_types
                part_data_temp_SO['Npart'][str(iso)]=int(len(iso_IDs_concatenated))

                if add_partdata:
                    for field in part_fields:
                        if field=='Velocity' or field=='Coordinates':
                            so_temp_partdata=np.concatenate([snapshot.read_dataset(itype,field) for itype in [0,1,4,5]])*comtophys
                        elif field=='Mass':
                            so_temp_partdata=[]
                            for itype in parttypes:
                                if not itype==1:
                                    so_temp_partdata.extend(snapshot.read_dataset(itype,field)*10**10)
                                else:
                                    so_temp_partdata.extend(np.ones(iso_npart[1])*dm_mass)
                            so_temp_partdata=np.array(so_temp_partdata)/h_val#convert to physical
                        part_data_temp_SO[field][str(iso)]=so_temp_partdata

                #SAVE TO FILE
                try:
                    iso_group=outfile_SO.create_group(f'iso_{str(iso).zfill(6)}')
                except:
                    iso_group=outfile_SO[f'iso_{str(iso).zfill(6)}']
                    datasets=list(iso_group.keys())
                    for dataset in datasets:
                        del iso_group[dataset]


                iso_Npart=part_data_temp_SO["Npart"][str(iso)]

                if iso_Npart>0:#if non-zero SO region
                    iso_meancop=np.array([np.nanmean(part_data_temp_SO["Coordinates"][str(iso)][:,i]) for i in range(3)])
                    iso_mediancop=np.array([np.nanmedian(part_data_temp_SO["Coordinates"][str(iso)][:,i]) for i in range(3)])
                    iso_r=(np.nanmax(part_data_temp_SO["Coordinates"][str(iso)][:,0])-np.nanmin(part_data_temp_SO["Coordinates"][str(iso)][:,0]))/2
                    if iso_r>1:
                        iso_r=(np.nanmax(part_data_temp_SO["Coordinates"][str(iso)][:,1])-np.nanmin(part_data_temp_SO["Coordinates"][str(iso)][:,1]))/2
                        if iso_r>1:
                            iso_r=(np.nanmax(part_data_temp_SO["Coordinates"][str(iso)][:,2])-np.nanmin(part_data_temp_SO["Coordinates"][str(iso)][:,2]))/2

                    for field in fields_SO:
                        if field=='Particle_IDs':
                            iso_group.require_dataset(field,data=part_data_temp_SO[field][str(iso)],dtype=np.uint64,shape=np.shape(part_data_temp_SO[field][str(iso)]))
                        elif 'Particle' in field:
                            iso_group.require_dataset(field,data=part_data_temp_SO[field][str(iso)],dtype=np.int8,shape=np.shape(part_data_temp_SO[field][str(iso)]))
                        else:
                            iso_group.require_dataset(field,data=part_data_temp_SO[field][str(iso)],dtype=np.float32,shape=np.shape(part_data_temp_SO[field][str(iso)]))


                else: 
                    iso_meancop=np.array([np.nan,np.nan,np.nan])
                    iso_mediancop=np.array([np.nan,np.nan,np.nan])
                    iso_r=np.nan
  
                    for field in fields_SO:
                        iso_group.require_dataset(field,data=np.nan,dtype=np.float16,shape=(1,))

                iso_group.require_dataset('iso_meancop',data=iso_meancop,dtype=np.float32,shape=(1,3))
                iso_group.require_dataset('iso_mediancop',data=iso_mediancop,dtype=np.float32,shape=(1,3))
                iso_group.require_dataset('iso_r',data=iso_r,dtype=np.float32,shape=(1,))
                iso_group.require_dataset('iso_com',data=iso_com*comtophys,dtype=np.float32,shape=(1,3))
                iso_group.require_dataset('iso_cminpot',data=iso_cminpot*comtophys,dtype=np.float32,shape=(1,3))
                iso_group.require_dataset('iso_cmbp',data=iso_cmbp*comtophys,dtype=np.float32,shape=(1,3))
                iso_group.require_dataset('iso_r200mean',data=iso_r200mean*comtophys,dtype=np.float32,shape=(1,))
                iso_group.require_dataset('iso_r200crit',data=iso_r200crit*comtophys,dtype=np.float32,shape=(1,))


            outfile_SO.close()

    return None

########################### MATCH FOF/SO PARTICLE DATA ###########################

