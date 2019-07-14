
########################################################################################################################################################################
############################################ 01/04/2019 Ruby Wright - Tools To Read Simulation Particle Data Files ######################################################
########################################################################################################################################################################

#*** Preamble ***

import numpy as np
import h5py
import read_eagle
from pandas import DataFrame as df
from astropy import units

def read_n_part(fname,sim_type):

    """
    read_n_part : function
	----------
    Return the number of particles in a SWIFT or GADGET simulation.
		
	Parameters
	----------
	fname : str 
        The file string of the snapshot file. 

    sim_type : string 
		Which type of simulation ("GADGET" OR "SWIFT").

	"""

    temp_file=h5py.File(fname)

    if sim_type=='SWIFT':
        n_0=len(temp_file['PartType0']['Masses'])
        n_1=len(temp_file['PartType1']['Masses'])
        npart=n_0+n_1
        return int(npart)

    elif sim_type=='GADGET':
        npart=int(np.sum(temp_file['Header'].attrs['NumPart_Total']))
        return int(npart)
    else:
        print('Please enter valid simulation string.')
        return []

########################### READ MASS TABLE ###########################

def read_mass_table(fname,sim_type):

    """
    read_mass_table : function
	----------
    Return the mass table of a SWIFT or GADGET simulation.
		
	Parameters
	----------
	run_directory : string 
		The directory in which the snapshot hdf5 snapshot files exist.

    fname : str 
        The file string of the snapshot file. 

    Returns
	-------
    np.array([M0,M1]) : np.ndarray
        np.ndarray of the masses of particle types 0 and 1 in order.
	
	"""

    #return mass of PartType0, PartType1 particles in sim units
    temp_file=h5py.File(fname)

    if sim_type=='SWIFT':
        M0=temp_file['PartType0']['Masses'][0]
        M1=temp_file['PartType1']['Masses'][1]
        return np.array([M0,M1])

    elif sim_type=='GADGET':
        M0=temp_file['Header'].attrs['MassTable'][0]
        M1=temp_file['Header'].attrs['MassTable'][1]
        return np.array([M0,M1])
    else:
        print('Please enter valid simulation string.')
        return []

########################### READ MASS DATA (EAGLE) ###########################

def read_mass_data_eagle(fname,extra_gas_props=[],verbose=True): 
    
    """

    get_mass_data_eagle : function
	----------

    Grabs the gas and dark matter masses (and any extra gas properties) for EAGLE particles in a given snapshot. 

    Parameters
	----------

    fname : str
        The base filename of the EAGLE snapshot file to be loaded. 

    Returns
	----------

    mass_table : list
        mass_table[0] : pd.DataFrame of gas particle IDs ['IDs'] and corresponding masses ['Mass'] (and any extra gas_props)
        mass_table[1] : float dark matter particle mass in sim units

    
    """

    snap = read_eagle.EagleSnapshot(fname)
    snap.select_region(xmin=0,xmax=snap.boxsize,ymin=0,ymax=snap.boxsize,zmin=0,zmax=snap.boxsize)

    if verbose:
        print ("# Total number of gas particles in snapshot = %d" % snap.numpart_total[0])
        print ("# Total number of DM particles in snapshot = %d" % snap.numpart_total[1])

    Gas_Props={}
    Gas_Props["IDs"]=snap.read_dataset(0,"ParticleIDs")
    Gas_Props["Mass"]=snap.read_dataset(0,"Mass")

    for prop in extra_gas_props:
        Gas_Props[prop]=snap.read_dataset(0,prop)

    Gas_Props=df(Gas_Props)
    
    fh5py=h5py.File(fname)
    h=fh5py["Header"].attrs.get("HubbleParam")
    a=fh5py["Header"].attrs.get("Time")
    dm_mass=fh5py["Header"].attrs.get("MassTable")[1]
    cgs=fh5py['PartType0/Mass'].attrs.get("CGSConversionFactor")
    aexp=fh5py['PartType0/Mass'].attrs.get("aexp-scale-exponent")
    hexp=fh5py['PartType0/Mass'].attrs.get("h-scale-exponent")

    Msun_cgs=np.float(units.M_sun.cgs.scale)
    DM_Mass=dm_mass*cgs*a**aexp*h**hexp/Msun_cgs

    DM_Props={}
    DM_Props['Mass']=DM_Mass

    mass_table=[Gas_Props,DM_Props]

    return mass_table


    