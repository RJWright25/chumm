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

# Preamble
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
from VRPythonTools import *

########################### CREATE BASE HALO DATA ###########################

def gen_base_halo_data(partdata_filelist,partdata_filetype,vr_filelist,vr_filetype,tf_filelist,add_descen=False,numsnaps=None,outname='',temporal_idval=10**12):
    
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
    
    if not numsnaps>0:
        numsnaps=len(partdata_filelist)

    # File lists
    part_list=partdata_filelist#particle data filepaths -- padded with None for snaps we don't have
    vr_list=vr_filelist#velociraptor data filepaths -- padded with None for snaps we don't have
    tf_list=tf_filelist#treefrog data filepaths -- padded with None for snaps we don't have

    # Get snapshot indices from number of particle data files 
    sim_snaps=list(range(numsnaps))
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
                    halo_data_all[isnap]['SimulationInfo']['Mass_DM_Physical']=0
                    halo_data_all[isnap]['SimulationInfo']['Mass_Gas_Physical']=0
                    
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
            halo_data_output

    if add_descen:
        print('Adding descendent information ...')
        isnap=-1
        for halo_data_snap in halo_data_output:
            if len(halo_data_snap)>5:
                isnap=isnap+1
                if not halo_tree[isnap]["Descen"]==[]:
                    halo_data_snap["Descen"]=halo_tree[isnap]["Descen"]
                    halo_data_snap["Merit"]=halo_tree[isnap]["Merit"]
                    print(f'Added descendants for snap {halo_data_snap["Snap"]}')
                else:
                    print(f'Could not add descendants for snap {halo_data_snap["Snap"]}')


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
    "n_bh",
    "n_gas",
    "n_star",
    "npart",
    "sigV",
    "sigV_gas",
    "tage_star",
    "M_bh",
    "M_star",
    "lambda_B",
    "T_gas",
    "cNFW",
    "Krot",
    "Efrac",
    "Efrac_gas",
    "Efrac_star",
    "Ekin",
    "Epot",
    "SFR_gas",
    "Zmet_gas",
    "q",
    "q_gas",
    "q_star",
    "s"
    ]

    if fields==None:
        fields=default_fields

    no_snaps=len(detailed_halo_data)
    snap_mask=[len(list(detailed_halo_data_snap.keys()))>5 for detailed_halo_data_snap in detailed_halo_data]

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

def find_progen_index(base_halo_data,index2,snap2,depth,return_all_depths=False): ### given halo index2 at snap 2, find progenitor index at snap1=snap2-depth
    
 
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
    index_depth=[]
    padding=np.sum([len(base_halo_data[isnap])<5 for isnap in range(len(base_halo_data))])
    index_idepth=index2
    for idepth in range(depth):
        current_ID=base_halo_data[snap2-idepth]["ID"][index_idepth]
        tail_ID=base_halo_data[snap2-idepth]["Tail"][index_idepth]
        index_idepth=np.where(base_halo_data[snap2-idepth-1]["ID"]==tail_ID)[0]
        if len(index_idepth)==0:
            index_idepth=np.nan
            index_depth.extend([np.nan]*(depth-idepth))
            break
        else:
            index_idepth=index_idepth[0]
            index_depth.append(index_idepth)
            if idepth==depth-1:
                if return_all_depths:
                    return index_depth
                else:
                    return index_idepth
    if return_all_depths:         
        return index_depth
    else:
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

