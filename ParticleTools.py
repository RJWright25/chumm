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

from GenPythonTools import *
from VRPythonTools import *
from STFTools import *
from AccretionTools import *
from pandas import DataFrame as df

#get IDs
def get_halo_particle_data(base_halo_data,snap2,ihalo,add_subparts_to_fofs=True):
    """
    dumps coordinates to file

    """
    outfolder='vis_data/coordinates/'
    fullpath=''
    for path in outfolder.split('/'):
        fullpath=fullpath+f'{path}/'
        if not os.path.exists(fullpath):
            os.mkdir(fullpath)

    outname_snap2=f'vis_data/coordinates/ihalo{str(ihalo).zfill(6)}_snap{str(snap2).zfill(3)}_current_xyz.dat'
    outname_snap1=f'vis_data/coordinates/ihalo{str(ihalo).zfill(6)}_snap{str(snap2).zfill(3)}_previous_xyz.dat'

    if os.path.exists(outname_snap2):
        proceed=bool(input("Data exists for this ihalo and snap. Overwrite?\n"))
    else:
        proceed=True

    if proceed:
        ihalo_s1=find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=1)
        snap1=snap2-1

        ihalo_snap2_particles=get_particle_lists(base_halo_data[snap2],halo_index_list=[ihalo],include_unbound=True,add_subparts_to_fofs=True)
        ihalo_snap2_particles_IDs=ihalo_snap2_particles["Particle_IDs"][0]
        ihalo_snap2_particles_Types=ihalo_snap2_particles["Particle_Types"][0]

        ihalo_snap1_particles=get_particle_lists(base_halo_data[snap1],halo_index_list=[ihalo_s1],include_unbound=True,add_subparts_to_fofs=True)
        ihalo_snap1_particles_IDs=ihalo_snap1_particles["Particle_IDs"][0]
        ihalo_snap1_particles_Types=ihalo_snap1_particles["Particle_Types"][0]
        
        if base_halo_data[snap2]['Part_FileType']=='EAGLE':
            parttypes=[0,1,4,5]
        else:
            parttypes=[0,1]
        
        print('Loading particle history data ...')
        PartHistories_Snap1_File=h5py.File(base_halo_data[snap1]['PartHist_FilePath'],'r')
        PartHistories_Snap1_IDs={str(itype):PartHistories_Snap1_File[f"PartType{itype}"]["ParticleIDs"] for itype in parttypes}
        PartHistories_Snap1_Indices={str(itype):PartHistories_Snap1_File[f"PartType{itype}"]["ParticleIndex"] for itype in parttypes}
        PartHistories_Snap2_File=h5py.File(base_halo_data[snap2]['PartHist_FilePath'],'r')
        PartHistories_Snap2_IDs={str(itype):PartHistories_Snap2_File[f"PartType{itype}"]["ParticleIDs"] for itype in parttypes}
        PartHistories_Snap2_Indices={str(itype):PartHistories_Snap2_File[f"PartType{itype}"]["ParticleIndex"] for itype in parttypes}

        print(f'Indexing {len(ihalo_snap1_particles_IDs)} particles at snap 1...')
        types_snap1,historyindices_snap1,partindices_snap1=get_particle_indices(base_halo_data,
                                                            SortedIDs=PartHistories_Snap1_IDs,
                                                            SortedIndices=PartHistories_Snap1_Indices,
                                                            PartIDs=ihalo_snap1_particles_IDs,
                                                            PartTypes=ihalo_snap1_particles_Types,
                                                            snap_taken=snap2,
                                                            snap_desired=snap1)
        print(f'Indexing {len(ihalo_snap2_particles_IDs)} particles at snap 2...')
        types_snap2,historyindices_snap2,partindices_snap2=get_particle_indices(base_halo_data,
                                                            SortedIDs=PartHistories_Snap2_IDs,
                                                            SortedIndices=PartHistories_Snap2_Indices,
                                                            PartIDs=ihalo_snap2_particles_IDs,
                                                            PartTypes=ihalo_snap2_particles_Types,
                                                            snap_taken=snap2,
                                                            snap_desired=snap2)

        print('Loading simulation data ...')
        if base_halo_data[snap2]['Part_FileType']=='EAGLE':
            EAGLE_boxsize=base_halo_data[snap2]['SimulationInfo']['BoxSize_Comoving']
            EAGLE_Snap1=read_eagle.EagleSnapshot(base_halo_data[snap1]['Part_FilePath'])
            EAGLE_Snap1.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
            EAGLE_Snap2=read_eagle.EagleSnapshot(base_halo_data[snap2]['Part_FilePath'])
            EAGLE_Snap2.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
            PartData_Coordinates_Snap1={str(itype):EAGLE_Snap1.read_dataset(itype,'Coordinates') for itype in parttypes}
            PartData_Coordinates_Snap2={str(itype):EAGLE_Snap2.read_dataset(itype,'Coordinates') for itype in parttypes}
        
        print('Extracting coordinates ...')
        ihalo_Coordinates_snap1=np.array([PartData_Coordinates_Snap1[str(ipart_type)][ipart_partdataindex] for ipart_type,ipart_partdataindex in zip(types_snap1,partindices_snap1)])
        ihalo_Coordinates_snap2=np.array([PartData_Coordinates_Snap2[str(ipart_type)][ipart_partdataindex] for ipart_type,ipart_partdataindex in zip(types_snap2,partindices_snap2)])

        dump_pickle(path=outname_snap2,data=ihalo_Coordinates_snap2)
        dump_pickle(path=outname_snap1,data=ihalo_Coordinates_snap1)

    else:
        ihalo_Coordinates_snap1=open_pickle(outname_snap1)
        ihalo_Coordinates_snap2=open_pickle(outname_snap2)

    return ihalo_Coordinates_snap1,ihalo_Coordinates_snap2
