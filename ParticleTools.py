
########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data Files ######################################################
########################################################################################################################################################################

#*** Preamble ***

import numpy as np
import h5py

########################### READ MASS DATA ###########################

def read_mass_table(run_directory,sim_type='SWIFT',snap_prefix="snap_",snap_lz=4):

    #return mass of PartType0, PartType1 particles in sim units
    temp_file=h5py.File(run_directory+snap_prefix+str(0).zfill(snap_lz)+".hdf5")

    if sim_type=='SWIFT':
        M0=temp_file['PartType0']['Masses'][0]
        M1=temp_file['PartType1']['Masses'][1]
        return np.array([M0,M1])

    if sim_type=='GADGET':
        M0=temp_file['Header'].attrs['MassTable'][0]
        M1=temp_file['Header'].attrs['MassTable'][1]
        return np.array([M0,M1])

