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

# Preamble
import os
import numpy as np
import h5py
import astropy.units as u
import read_eagle
import time

from GenPythonTools import *
from VRPythonTools import *
from STFTools import *
from ParticleTools import *
from pandas import DataFrame as df
from scipy import spatial


########################### GENERATE ACCRETION DATA: EAGLE (memory saving + SO compatibility) ###########################

def gen_accretion_data_eagle(base_halo_data,snap=None,halo_index_list=None,pre_depth=1,post_depth=1,vmax_facs_in=[-1],vmax_facs_out=[-1],r200_facs_in=[],r200_facs_out=[],write_partdata=False):
    
    """

    gen_accretion_data_eagle : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists from VELOCIraptor FOF outputs.
    This is specialised for EAGLE by using the read_eagle routine to slice snapshots and save memory.  

    ** note: particle histories, base_halo_data and halo particle data must have been generated as per gen_particle_history_serial (this file),
             gen_base_halo_data in STFTools.py and dump_structure_particle_data in STFTools.py

	Parameters
	----------
    base_halo_data : list of dictionaries
        The minimal halo data list of dictionaries previously generated ("B1" is sufficient)

    snap : int
        The index in the base_halo_data for which to calculate accretion rates (should be actual snap index)
        We will retrieve particle data based on the flags at this index
    
    halo_index_list : dict
        "iprocess": int
        "indices: list of int
        List of the halo indices for which to calculate accretion rates. If 'None',
        find for all halos in the base_halo_data dictionary at the desired snapshot. 

    pre_depth : int
        How many snaps to skip back to when comparing particle lists.
        Initial snap for calculation will be snap-pre_depth. 

    pre_depth : int
        How many snaps to skip back to when comparing particle lists.
        Initial snap (s1) for calculation will be s1=snap-pre_depth, and we will check particle histories at s1.

    vmax_facs_in : list of float
        List of the factors of vmax to cut inflow particles at. 

    vmax_facs_out : list of float.
        List of the factors of vmax to cut outflow particles at. 
        If empty, no outflow calculations are performed. 

    r200_facs_in : list of float. 
        List of the factors of r200_crit to calculate spherical accretion over. 
        If empty, no spherical accretion calculations are performed. 

    r200_facs_out : list of float. 
        List of the factors of r200_crit to calculate spherical outflow over. 
        If empty, no spherical outflow calculations are performed. 

    write_partdata : bool 
        Flag indicating whether to write accretion/outflow particle data to file (in halo groups).
        (In addition to integrated data)
    

	Returns
	----------
    FOF_AccretionData_snap{snap2}_pre{pre_depth}_post{post_depth}_px.hdf5: hdf5 file with datasets
        Header contains attributes:
            "snap1"
            "snap2"
            "snap3"
            "snap1_LookbackTime"
            "snap2_LookbackTime"
            "snap3_LookbackTime"
            "ave_LookbackTime" (snap 1 -> snap 2)
            "delta_LookbackTime" (snap 1 -> snap 2)
            "snap1_z"
            "snap2_z"
            "snap3_z"
            "ave_z (snap 1 -> snap 2)

        If particle data is output:
        Group "Particle":
            There is a group for each halo: ihalo_xxxxxx

            Each halo group with attributes:
            "snapx_com"
            "snapx_cminpot"
            "snapx_cmbp"
            "snapx_vmax"
            "snapx_v"
            "snapx_M_200crit"
            "snapx_R_200mean"
            "snapx_R_200crit"

            Each halo group will have a vast collection of particle data written for snaps 1, 2, 3. 
        
        Integrated Accretion/Outflow Data
        Group "Integrated":

            Inflow:
                For each particle type /PartTypeX/:
                    For each group definition:
                        'FOF-haloscale' : the inflow as calculated from new particles to the full FOF (only for field halos or halos with substructure).
                        'FOF-subhaloscale: the inflow as calculated from new particles to the relevant substructure (can be (i) the host halo of a FOF with substructure or (ii) a substructure halo).
                        'SO-r200_facx' : the inflow as calculated from new particles to a spherical region around the halo defined by r_200crit x factor.
                        
                        Note: particles 'new to' a halo are those which were not present in the relevant halo definition at snap 1, but were at snap 2.
                        (their type taken at snap 1, before inflow). 

                        For each vmax_facx: 
                            The mass of accretion to each halo is cut to particle satisfying certain, user defined inflow velocity cuts (factors of the halo's vmax).
                            
                            For each of the following particle histories:
                                'Total' : No selection based on particle history.
                                'Processed' : Only particles which have existed in a halo prior to accretion (snap 1).
                                'Unprocessed' : Only particles which have not existed in a halo prior to accretion (snap 1). 

                                We have the following datasets...
                                    [Stability]_GrossDelta[M/N]_In: The [mass(msun)/particle count] of selected inflow candidates of all origins. 
                                    [Stability]_FieldDelta[M/N]_In: The [mass(msun)/particle count] of selected inflow candidates from the field (at snap 1). 
                                    [Stability]_TransferDelta[M/N]_In: The [mass(msun)/particle count] of selected inflow candidates from other halos (at snap 1). 

                                    Where [Stability] can be either 'Stable' or 'All'.
                                        'Stable' requires the inflow candidates to remain in the halo at snap 3. 
                                        'All' has no further requirements on the inflow candidates at snap 3. 



            Outflow: 
                For each particle type /PartTypeX/:
                    For each group definition:
                        'FOF-haloscale' : the outflow as calculated from outgoing particles from the full FOF (only for field halos or halos with substructure).
                        'FOF-subhaloscale: the outflow as calculated from outgoing particles from the relevant substructure (can be (i) the host halo of a FOF with substructure or (ii) a substructure halo).
                        'SO-r200_facx' : the outflow as calculated from outgoing particles to a spherical region around the halo defined by r_200crit x factor. 
                        
                        Note: particles 'outgoing from' a halo are those which were present in the relevant halo definition at snap 1, but were not at snap 2.
                        (their type taken at snap 2, post outflow). 
                            
                        For each vmax_facx: 
                            The mass of outflow to each halo is cut to particle satisfying certain, user defined outflow velocity cuts (factors of the halo's vmax).
                            
                            For each of the following particle histories:
                                'Total' : No selection based on particle history.

                                We have the following datasets...
                                    [Stability]_GrossDelta[M/N]_In: The [mass(msun)/particle count] of selected outflow candidates, regardless of destination.

                                    Where [Stability] can be either 'Stable' or 'All'.
                                        'Stable' requires the outflow candidates to remain outside the halo at snap 3. 
                                        'All' has no further requirements on the outflow candidates at snap 3. 

    """
    
    t1_init=time.time()

    ##### Processing inputs #####
    # Processing the snap inputs
    snap1=snap-pre_depth
    snap2=snap
    snap3=snap+post_depth
    snaps=[snap1,snap2,snap3]
    
    # Processing the desired halo index list
    if halo_index_list==None:
        halo_index_list_snap2=list(range(len(base_halo_data[snap]["hostHaloID"])))#use all halos if not handed halo index list
        iprocess="x"
        num_processes=1
        test=True
    else:
        try:
            halo_index_list_snap2=halo_index_list["indices"] #extract index list from input dictionary
            iprocess=str(halo_index_list["iprocess"]).zfill(2) #the process for this index list (this is just used for the output file name)
            print(f'iprocess {iprocess} has {len(halo_index_list_snap2)} halo indices: {halo_index_list_snap2}')
            num_processes=halo_index_list["np"]
            test=halo_index_list["test"]
        except:
            print('Not parsed a valud halo index list. Exiting.')
            return None
    
    # Find the indices of halos at snap1 and snap3 (ordered by snap2 halo indices)
    halo_index_list_snap1=[find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=pre_depth) for ihalo in halo_index_list_snap2]
    halo_index_list_snap3=[find_descen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=post_depth) for ihalo in halo_index_list_snap2]

    # Determine whether we need to perform outflow calculationes
    if vmax_facs_out==[]:
        output_groups=['Inflow']
        outflow=False
    else:
        output_groups=['Inflow','Outflow']
        outflow=True

    # Factors of r200 to calculate SO accretion/outflow to
    r200_facs={'Inflow':r200_facs_in,'Outflow':r200_facs_out} 
    max_r200fac_in=np.nanmax(r200_facs_in)
    if outflow:
        max_r200fac_out=np.nanmax(r200_facs_out)

    # Add vmax factor of -1 to whatever the user input was
    vmax_facs_in=np.concatenate([[-1],vmax_facs_in])
    if not vmax_facs_out==[]:
        vmax_facs_out=np.concatenate([[-1],vmax_facs_out])
    vmax_facs={'Inflow':vmax_facs_in,'Outflow':vmax_facs_out} 

    # Define halo calculation types
    halo_defnames={}
    halo_defnames["Inflow"]=np.concatenate([['FOF-haloscale','FOF-subhaloscale'],['SO-r200_fac'+str(ir200_fac+1) for ir200_fac in range(len(r200_facs["Inflow"]))]])
    halo_defnames["Outflow"]=np.concatenate([['FOF-haloscale','FOF-subhaloscale'],['SO-r200_fac'+str(ir200_fac+1) for ir200_fac in range(len(r200_facs["Outflow"]))]])
    
    # Default options 
    ihalo_cube_rfac=5 #cube to grab EAGLE data from
    vel_conversion=978.462 #Mpc/Gyr to km/s
    use='cminpot' #which halo centre definition to use (from 'cminpot', 'com')
    compression='gzip'

    # Create log file and directories, initialising outputs
    if True:
        #Logs
        acc_log_dir=f"job_logs/acc_logs/"
        if not os.path.exists(acc_log_dir):
            os.mkdir(acc_log_dir)
        if test:
            run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_test/"
        else:
            run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}/"

        if not os.path.exists(run_log_dir):
            try:
                os.mkdir(run_log_dir)
            except:
                pass

        run_snap_log_dir=run_log_dir+f'snap_{str(snap).zfill(3)}/'

        if not os.path.exists(run_snap_log_dir):
            try:
                os.mkdir(run_snap_log_dir)
            except:
                pass
        if test:
            fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}_test.log"
            print(f'iprocess {iprocess} will save progress to log file: {fname_log}')

        else:
            fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}.log"

        if os.path.exists(fname_log):
            os.remove(fname_log)
        
        with open(fname_log,"a") as progress_file:
            progress_file.write('Initialising and loading in data ...\n')
        progress_file.close()
    
        # Initialising outputs
        if not os.path.exists('acc_data'):#create folder for outputs if doesn't already exist
            os.mkdir('acc_data')
        if test:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_test/'
        else:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}/'

        if not os.path.exists(calc_dir):#create folder for outputs if doesn't already exist
            try:
                os.mkdir(calc_dir)
            except:
                pass
        calc_snap_dir=calc_dir+f'snap_{str(snap2).zfill(3)}/'
        
        if not os.path.exists(calc_snap_dir):#create folder for outputs if doesn't already exist
            try:
                os.mkdir(calc_snap_dir)
            except:
                pass

        # Assigning snap
        if snap==None:
            snap=len(base_halo_data)-1#if not given snap, just use the last one

    # Create output file with metadata attributes
    run_outname=base_halo_data[snap]['outname']#extract output name (simulation name)
    outfile_name=calc_snap_dir+'AccretionData_pre'+str(pre_depth).zfill(2)+'_post'+str(post_depth).zfill(2)+'_snap'+str(snap).zfill(3)+'_p'+str(iprocess).zfill(3)+'.hdf5'
    
    # Remove existing output file if exists
    if not os.path.exists(outfile_name):
        print(f'Initialising output file at {outfile_name}...')
        output_hdf5=h5py.File(outfile_name,"w")
    else:
        print(f'Removing old file and initialising output file at {outfile_name}...')
        os.remove(outfile_name)
        output_hdf5=h5py.File(outfile_name,"w")

    # Make header for accretion data based on base halo data 
    if True:
        header_hdf5=output_hdf5.create_group("Header")
        lt_ave=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']+base_halo_data[snap2]['SimulationInfo']['LookbackTime'])/2
        z_ave=(base_halo_data[snap1]['SimulationInfo']['z']+base_halo_data[snap2]['SimulationInfo']['z'])/2
        dt=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']-base_halo_data[snap2]['SimulationInfo']['LookbackTime'])
        t1=base_halo_data[snap1]['SimulationInfo']['LookbackTime']
        t2=base_halo_data[snap2]['SimulationInfo']['LookbackTime']
        t3=base_halo_data[snap3]['SimulationInfo']['LookbackTime']
        z1=base_halo_data[snap1]['SimulationInfo']['z']
        z2=base_halo_data[snap2]['SimulationInfo']['z']
        z3=base_halo_data[snap3]['SimulationInfo']['z']
        header_hdf5.attrs.create('ave_LookbackTime',data=lt_ave,dtype=np.float16)
        header_hdf5.attrs.create('ave_z',data=z_ave,dtype=np.float16)
        header_hdf5.attrs.create('delta_LookbackTime',data=dt,dtype=np.float16)
        header_hdf5.attrs.create('snap1_LookbackTime',data=t1,dtype=np.float16)
        header_hdf5.attrs.create('snap2_LookbackTime',data=t2,dtype=np.float16)
        header_hdf5.attrs.create('snap3_LookbackTime',data=t3,dtype=np.float16)
        header_hdf5.attrs.create('snap1_z',data=z1,dtype=np.float16)
        header_hdf5.attrs.create('snap2_z',data=z2,dtype=np.float16)
        header_hdf5.attrs.create('snap3_z',data=z3,dtype=np.float16)
        header_hdf5.attrs.create('snap1',data=snap1,dtype=np.int16)
        header_hdf5.attrs.create('snap2',data=snap2,dtype=np.int16)
        header_hdf5.attrs.create('snap3',data=snap3,dtype=np.int16)
        header_hdf5.attrs.create('pre_depth',data=snap2-snap1,dtype=np.int16)
        header_hdf5.attrs.create('post_depth',data=snap3-snap2,dtype=np.int16)
        header_hdf5.attrs.create('outname',data=np.string_(base_halo_data[snap2]['outname']))
        header_hdf5.attrs.create('total_num_halos',data=base_halo_data[snap2]['Count'])
    
    # Standard particle type names from simulation
    PartNames=['Gas','DM','','','Star','BH']
    PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
    Mass_DM=base_halo_data[snap2]['SimulationInfo']['Mass_DM_Physical']
    Mass_Gas=base_halo_data[snap2]['SimulationInfo']['Mass_Gas_Physical']

    ##### Loading in Data #####
    #Load in FOF particle lists: snap 1, snap 2, snap 3
    FOF_Part_Data={}
    FOF_Part_Data[str(snap1)]=get_FOF_particle_lists(base_halo_data,snap1,halo_index_list=halo_index_list_snap1)
    FOF_Part_Data[str(snap2)]=get_FOF_particle_lists(base_halo_data,snap2,halo_index_list=halo_index_list_snap2)
    FOF_Part_Data[str(snap3)]=get_FOF_particle_lists(base_halo_data,snap3,halo_index_list=halo_index_list_snap3)

    for snap in [snap1,snap2,snap3]:
        del FOF_Part_Data[str(snap)]["Particle_Types"]
        del FOF_Part_Data[str(snap)]["Particle_Bound"]
        del FOF_Part_Data[str(snap)]["Npart"]
        del FOF_Part_Data[str(snap)]["Npart_unbound"]
        
    FOF_Part_Data_fields=list(FOF_Part_Data[str(snap1)].keys()) #Fields from FOF data

    #Particle data filepath
    hval=base_halo_data[snap1]['SimulationInfo']['h_val'];scalefactors={}
    scalefactors={str(snap):base_halo_data[snap]['SimulationInfo']['ScaleFactor'] for snap in snaps}
    Part_Data_FilePaths={str(snap):base_halo_data[snap]['Part_FilePath'] for snap in snaps}
    Part_Data_fields={str(snap1):['Coordinates','Velocity','Mass','ParticleIDs'],#The fields to be read initially from EAGLE cubes
                      str(snap2):['Coordinates','ParticleIDs'],
                      str(snap3):['Coordinates','ParticleIDs']}                                                   
    Part_Data_comtophys={str(snap):{'Coordinates':scalefactors[str(snap)]/hval, #Conversion factors for EAGLE cubes
                                    'Velocity':scalefactors[str(snap)]/hval,
                                    'Mass':10.0**10/hval,
                                    'ParticleIDs':1} for snap in snaps}
    
    #Load in particle histories: snap 1 (only need snap 1 to check origin of inflow particles - not checking destination of outflow particles)
    print(f'Retrieving & organising particle histories for snap = {snap1} ...')
    Part_Histories_Constant={str(0):False,str(1):False,str(4):False,str(5):False}
    Part_Histories_File_snap1=h5py.File("part_histories/PartHistory_"+str(snap1).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIDs'].value for parttype in PartTypes}
    Part_Histories_npart_snap1={str(parttype):len(Part_Histories_IDs_snap1[str(parttype)]) for parttype in PartTypes}
    Part_Histories_Index_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes} #Note: no .value
    Part_Histories_HostStructure_snap1={str(parttype):Part_Histories_File_snap1["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes} #Note: no .value
    Part_Histories_Processed_L1_snap1={}
    for parttype in PartTypes:
        if not Part_Histories_Constant[str(parttype)]:
            Part_Histories_Processed_L1_snap1[str(parttype)]=Part_Histories_File_snap1["PartType"+str(parttype)+'/Processed_L1'] #Note: no .value


    print()
    t2_init=time.time()
    print('*********************************************************')
    print(f'Done initialising in {(t2_init-t1_init):.2f} sec - entering main halo loop ...')
    print('*********************************************************')

    with open(fname_log,"a") as progress_file:
        progress_file.write(f'Done initialising in {(t2_init-t1_init):.2f} sec - entering main halo loop ...\n')
    progress_file.close()

    ##### Initialising outputs #####
    # Particle
    if write_partdata:
        #hdf5 group
        particle_output_hdf5=output_hdf5.create_group('Particle')
        
        #output dtypes
        output_fields_dtype={}
        output_fields_float32=["Mass","r_com","rabs_com","vrad_com","vtan_com"]
        for field in output_fields_float32:
            output_fields_dtype[field]=np.float32

        output_fields_int64=["ParticleIDs","Structure"]
        for field in output_fields_int64:
            output_fields_dtype[field]=np.int64
        
        output_fields_int8=["Processed","Particle_InFOF","Particle_Bound","Particle_InHost"]
        for field in output_fields_int8:
            output_fields_dtype[field]=np.int8 

    # Integrated (always written)
    num_halos_thisprocess=len(halo_index_list_snap2)
    integrated_output_hdf5=output_hdf5.create_group('Integrated')
    integrated_output_hdf5.create_dataset('ihalo_list',data=halo_index_list_snap2)

    #Defining which outputs for varying levels of detail
    output_processedgroups={'Detailed':['Total','Unprocessed','Processed'],
                            'Basic':['Total']}
    output_enddatasets={'Detailed':['Gross','Field','Transfer'],
                        'Basic':['Gross']}
    
    #Initialise output datasets with np.nans (len: total_num_halos)
    for output_group in output_groups:
        integrated_output_hdf5.create_group(output_group)
        for itype in PartTypes:
            itype_key=f'PartType{itype}'
            integrated_output_hdf5[output_group].create_group(itype_key)
            for ihalo_defname,halo_defname in enumerate(sorted(halo_defnames[output_group])):
                #Create group
                integrated_output_hdf5[output_group][itype_key].create_group(halo_defname)
                #Add attribute for SO variants
                if 'FOF' not in halo_defname: 
                    #Add attribute for R200_fac
                    ir200_fac=int(halo_defname.split('fac')[-1])-1
                    r200_fac=r200_facs[output_group][ir200_fac]
                    integrated_output_hdf5[output_group][itype_key][halo_defname].attrs.create('R200_fac',data=r200_fac)
                #Use detailed datasets for FOF inflow variants
                if 'FOF' in halo_defname and output_group=='Inflow': 
                    icalc_processedgroups=output_processedgroups['Detailed']
                    icalc_enddatasets=output_enddatasets['Detailed']
                else:
                    icalc_processedgroups=output_processedgroups['Basic']
                    icalc_enddatasets=output_enddatasets['Basic']
                #Now, for each Vmax cut
                for ivmax_fac, vmax_fac in enumerate(vmax_facs[output_group]):
                    #Create group
                    ivmax_key=f'vmax_fac{ivmax_fac+1}'
                    integrated_output_hdf5[output_group][itype_key][halo_defname].create_group(ivmax_key);integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
                    #Add attribute for vmax_fac
                    integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
                    #Initialise datasets with nans
                    for processedgroup in icalc_processedgroups:
                        integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key].create_group(processedgroup)
                        for dataset in icalc_enddatasets:
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'All_'+dataset+f'_DeltaM',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'All_'+dataset+f'_DeltaN',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'Stable_'+dataset+f'_DeltaM',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'Stable_'+dataset+f'_DeltaN',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)

    ####################################################################################################################################################################################
    ####################################################################################################################################################################################
    ########################################################################### MAIN HALO LOOP #########################################################################################
    ####################################################################################################################################################################################
    ####################################################################################################################################################################################

    for iihalo,ihalo_s2 in enumerate(halo_index_list_snap2):# for each halo (index at snap 2)
        
        # If needed, create group for this halo in output file
        if write_partdata:
            ihalo_hdf5=particle_output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))
            ihalo_hdf5.create_group('Metadata')
            if write_partdata:
                ihalo_hdf5.create_group('Inflow');ihalo_hdf5.create_group('Outflow')
                for itype in PartTypes:
                    ihalo_hdf5['Inflow'].create_group(f'PartType{itype}')
                    ihalo_hdf5['Outflow'].create_group(f'PartType{itype}')        
        
        # This catches any exceptions for a given halo and prevents the code from crashing 
        try:     
            # try:
            ########################################################################################################################################
            ###################################################### ihalo PRE-PROCESSING ############################################################
            ########################################################################################################################################
            t1_halo=time.time()
            
            # Find halo progenitor and descendants
            ihalo_indices={str(snap1):halo_index_list_snap1[iihalo],str(snap2):ihalo_s2,str(snap3):halo_index_list_snap3[iihalo]}
            
            # Record halo properties 
            ihalo_tracked=(ihalo_indices[str(snap1)]>-1 and ihalo_indices[str(snap3)]>-1)#track if have both progenitor and descendant
            ihalo_structuretype=base_halo_data[snap2]["Structuretype"][ihalo_indices[str(snap2)]]#structure type
            ihalo_numsubstruct=base_halo_data[snap2]["numSubStruct"][ihalo_indices[str(snap2)]]
            ihalo_hostHaloID=base_halo_data[snap2]["hostHaloID"][ihalo_indices[str(snap2)]]
            ihalo_sublevel=int(np.floor((ihalo_structuretype-0.01)/10))
            ihalo_recordsubaccretion=ihalo_numsubstruct>0 or ihalo_hostHaloID>0 #record substructure-scale inflow/outflow IF has substructure or is subhalo
            ihalo_recordfieldaccretion=ihalo_numsubstruct>0 or ihalo_hostHaloID<0 #record field-scale inflow/outflow IF has substructure or is field halo
            
            # Which scales to record for this halo [inflow and outflow]
            ihalo_scale_record={ihalo_halodef:True for ihalo_halodef in np.concatenate([['FOF-haloscale','FOF-subhaloscale'],[f'SO-r200_fac{ir200_fac+1}' for ir200_fac in range(len(r200_facs_in))],[f'SO-r200_fac{ir200_fac+1}' for ir200_fac in range(len(r200_facs_out))]])}
            if not ihalo_recordsubaccretion:
                ihalo_scale_record['FOF-subhaloscale']=False
            if not ihalo_recordfieldaccretion:
                ihalo_scale_record['FOF-haloscale']=False

            # Print progress to terminal and output file
            print();print('**********************************************')
            print('Halo index: ',ihalo_s2,f' - {ihalo_numsubstruct} substructures')
            print(f'Progenitor: {ihalo_indices[str(snap1)]} | Descendant: {ihalo_indices[str(snap3)]}')
            print('**********************************************');print()
            with open(fname_log,"a") as progress_file:
                progress_file.write(f' \n')
                progress_file.write(f'Starting with ihalo {ihalo_s2} ... \n')
            progress_file.close()
            
            # This catches any halos for which we can't find a progenitor/descendant 
            if ihalo_tracked:
                ### GRAB HALO METADATA ###
                ihalo_metadata={}
                for isnap,snap in enumerate(snaps):
                    ihalo_isnap=ihalo_indices[str(snap)]
                    if ihalo_isnap>=0:
                        ihalo_metadata[f'snap{isnap+1}_com']=np.array([base_halo_data[snap]['Xc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Yc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zc'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_cminpot']=np.array([base_halo_data[snap]['Xcminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Ycminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zcminpot'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_vcom']=np.array([base_halo_data[snap]['VXc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VYc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VZc'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_R_200crit']=base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_R_200mean']=base_halo_data[snap]['R_200mean'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_Mass_200crit']=base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*10**10
                        ihalo_metadata[f'snap{isnap+1}_vmax']=base_halo_data[snap]['Vmax'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_vesc_crit']=np.sqrt(2*base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*base_halo_data[snap]['SimulationInfo']['Gravity']/base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]])
                
                # Average some quantities
                ihalo_metadata['sublevel']=ihalo_sublevel
                ihalo_metadata['ave_R_200crit']=0.5*base_halo_data[snap1]['R_200crit'][ihalo_indices[str(snap1)]]+0.5*base_halo_data[snap2]['R_200crit'][ihalo_indices[str(snap2)]]
                ihalo_metadata['ave_vmax']=0.5*base_halo_data[snap1]['Vmax'][ihalo_indices[str(snap1)]]+0.5*base_halo_data[snap2]['Vmax'][ihalo_indices[str(snap2)]]

                # Write halo metadata to file (if desired)
                if write_partdata:
                    for ihalo_mdkey in list(ihalo_metadata.keys()): 
                        size=np.size(ihalo_metadata[ihalo_mdkey])
                        if size>1:
                            ihalo_hdf5['Metadata'].create_dataset(ihalo_mdkey,data=ihalo_metadata[ihalo_mdkey],dtype=np.float32,shape=(1,size))
                        else:
                            ihalo_hdf5['Metadata'].create_dataset(ihalo_mdkey,data=ihalo_metadata[ihalo_mdkey],dtype=np.float32)

                ### GET HALO DATA FROM VELOCIRAPTOR AND EAGLE ###
                # Grab the FOF particle data 
                ihalo_fof_particles={}
                for snap in snaps:
                    # Read the FOF data for this halo
                    ihalo_fof_particles[str(snap)]={field:FOF_Part_Data[str(snap)][field][str(ihalo_indices[str(snap)])] for field in FOF_Part_Data_fields}
                    # Add the sorted IDs/indices
                    ihalo_fof_particles[str(snap)]['SortedIndices']=np.argsort(ihalo_fof_particles[str(snap)]['Particle_IDs'])
                    ihalo_fof_particles[str(snap)]['SortedIDs']=ihalo_fof_particles[str(snap)]['Particle_IDs'][(ihalo_fof_particles[str(snap)]['SortedIndices'],)]
                # Add a set for snap 3 particle IDs (to check existence in FOF)
                ihalo_fof_particles[str(snap3)]['ParticleIDs_set']=set(ihalo_fof_particles[str(snap3)]['Particle_IDs'])

                # Grab/slice EAGLE datacubes
                print(f'Retrieving datacubes for ihalo {ihalo_s2} ...')
                #Cube parameters
                ihalo_com_physical={str(snap):np.array(ihalo_metadata[f'snap{isnap+1}_{use}']) for isnap,snap in enumerate(snaps)}
                ihalo_com_comoving={str(snap):np.array(ihalo_metadata[f'snap{isnap+1}_{use}'])/Part_Data_comtophys[str(snap)]['Coordinates'] for isnap,snap in enumerate(snaps)}
                ihalo_vcom_physical={str(snap):np.array(ihalo_metadata[f'snap{isnap+1}_vcom']) for isnap,snap in enumerate(snaps)}
                ihalo_cuberadius_physical={str(snap):ihalo_metadata[f'snap{isnap+1}_R_200mean']*ihalo_cube_rfac for snap in snaps}
                ihalo_cuberadius_comoving={str(snap):ihalo_metadata[f'snap{isnap+1}_R_200mean']/Part_Data_comtophys[str(snap)]['Coordinates']*ihalo_cube_rfac for isnap,snap in enumerate(snaps)}
                
                #Initialise cube outputs
                ihalo_cube_particles={str(snap):{field:[] for field in Part_Data_fields[str(snap)]} for snap in snaps}
                ihalo_cube_npart={str(snap):{} for snap in snaps}

                #Get cube outputs for each snap
                for snap in snaps:
                    ihalo_EAGLE_snap=read_eagle.EagleSnapshot(Part_Data_FilePaths[str(snap)])
                    ihalo_EAGLE_snap.select_region(xmin=ihalo_com_comoving[str(snap)][0][0]-ihalo_cuberadius_comoving[str(snap)],xmax=ihalo_com_comoving[str(snap)][0][0]+ihalo_cuberadius_comoving[str(snap)],
                                                ymin=ihalo_com_comoving[str(snap)][0][1]-ihalo_cuberadius_comoving[str(snap)],ymax=ihalo_com_comoving[str(snap)][0][1]+ihalo_cuberadius_comoving[str(snap)],
                                                zmin=ihalo_com_comoving[str(snap)][0][2]-ihalo_cuberadius_comoving[str(snap)],zmax=ihalo_com_comoving[str(snap)][0][2]+ihalo_cuberadius_comoving[str(snap)])
                    ihalo_EAGLE_types=[]
                    #Get data for each parttype and add to running ihalo_cube_particles
                    for itype in PartTypes:       
                        for ifield,field in enumerate(Part_Data_fields[str(snap)]):
                            #if dataset is not DM mass, read and convert 
                            if not (field=='Mass' and itype==1):
                                data=ihalo_EAGLE_snap.read_dataset(itype,field)*Part_Data_comtophys[str(snap)][field];ihalo_cube_npart[str(snap)][str(itype)]=len(data)     
                            #if dataset is DM mass, fill flat array with constant
                            else:
                                data=np.ones(ihalo_cube_npart[str(snap)]['1'])*Mass_DM
                            ihalo_cube_particles[str(snap)][field].extend(data)
                        ihalo_EAGLE_types.extend((np.ones(ihalo_cube_npart[str(snap)][str(itype)])*itype).astype(int))#record particle types
                    ihalo_cube_particles[str(snap)]['ParticleTypes']=np.array(ihalo_EAGLE_types)
                    #Convert to np.arrays
                    for field in Part_Data_fields[str(snap)]:
                        ihalo_cube_particles[str(snap)][field]=np.array(ihalo_cube_particles[str(snap)][field])
                    #Sort the cube particles by ID
                    ihalo_cube_particles[str(snap)]['SortedIndices']=np.argsort(ihalo_cube_particles[str(snap)]['ParticleIDs'])
                    ihalo_cube_particles[str(snap)]['SortedIDs']=ihalo_cube_particles[str(snap)]['ParticleIDs'][(ihalo_cube_particles[str(snap)]['SortedIndices'],)]
                
                print(f'Finished retrieving data from EAGLE and FOF for ihalo {ihalo_s2}')
                
                ########################################################################################################################################
                ############################################################ ihalo INFLOW ##############################################################
                ########################################################################################################################################

                ###### SELECT INFLOW CANDIDATES AS THOSE WITHIN R200crit OR the FOF envelope at snap 2 ######
                #############################################################################################

                #Find the mean r200 from snap 1 / snap 2
                ihalo_ave_R_200crit_physical=(ihalo_metadata['snap1_R_200crit']+ihalo_metadata['snap2_R_200crit'])/2
                #Find which particles are with in the largest SO region requested
                ihalo_cube_rcut_snap2=np.where(np.sqrt(np.sum(np.square(ihalo_cube_particles[str(snap2)]['Coordinates']-ihalo_com_physical[str(snap2)]),axis=1))<ihalo_ave_R_200crit_physical*max_r200fac_in)
                #Concatenate the IDs of the particles within r200 and the FOF
                ihalo_combined_inflow_candidate_IDs=np.concatenate([ihalo_fof_particles[str(snap2)]['Particle_IDs'],ihalo_cube_particles[str(snap2)]['ParticleIDs'][ihalo_cube_rcut_snap2]])
                #Remove duplicates and convert to np.array with long ints
                ihalo_combined_inflow_candidate_IDs_unique=np.array(np.unique(ihalo_combined_inflow_candidate_IDs),dtype=np.int64)
                #Count inflow candidates
                ihalo_combined_inflow_candidate_count=len(ihalo_combined_inflow_candidate_IDs_unique)

                ############################## GRAB DATA FOR INFLOW CANDIDATES ##############################
                #############################################################################################
                ihalo_combined_inflow_candidate_data={}

                # 1. OUTPUTS FROM DATACUBE: Coordinates, Velocity, Mass, Type 
                ihalo_combined_inflow_candidate_cubeindices={}
                print(f'Inflow candidates for ihalo {ihalo_s2}: n = {ihalo_combined_inflow_candidate_count}')
                for isnap,snap in enumerate(snaps):
                    #Find the indices of the IDs in the (sorted) datacube for this halo (will return nan if not in the cube) - outputs sorted cube index
                    ihalo_combined_inflow_candidate_IDindices_temp=binary_search(ihalo_combined_inflow_candidate_IDs_unique,sorted_list=ihalo_cube_particles[str(snap)]['SortedIDs'],check_entries=True)
                    #Use the indices from the sorted IDs above to extract the cube indices (will return nan if not in the cube) - outputs raw cube index
                    ihalo_combined_inflow_candidate_cubeindices[str(snap)]=mask_wnans(array=ihalo_cube_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_inflow_candidate_IDindices_temp)
                    #For each snap, grab detailed particle data
                    for field in np.concatenate([Part_Data_fields[str(snap)],['ParticleTypes']]):
                        ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_cube_particles[str(snap)][field],indices=ihalo_combined_inflow_candidate_cubeindices[str(snap)])  

                    #Derive other cubedata outputs
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com']=ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Coordinates']-ihalo_com_physical[str(snap)]
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_rabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com']),axis=1))

                    if isnap==0:
                        ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_runit_com']=np.divide(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com'],np.column_stack([ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_rabs_com']]*3))
                        ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_v_com']=ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Velocity']-ihalo_vcom_physical[str(snap)]
                        ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_vrad_com']=np.sum(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_runit_com']*ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_v_com'],axis=1)
                    
                    if not write_partdata:
                        del ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_r_com']
                        try:
                            del ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_runit_com']
                            del ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_v_com']
                        except:
                            pass

                # 2. OUTPUTS FROM FOF Data: InFOF, Bound
                ihalo_combined_inflow_candidate_fofindices={}
                for isnap,snap in enumerate(snaps):
                    #Find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
                    ihalo_combined_inflow_candidate_IDindices_temp=binary_search(ihalo_combined_inflow_candidate_IDs_unique,sorted_list=ihalo_fof_particles[str(snap)]['SortedIDs'],check_entries=True)
                    #Use the indices from the sorted IDs above to extract the fof indices (will return nan if not in the fof) - outputs index
                    ihalo_combined_inflow_candidate_fofindices[str(snap)]=mask_wnans(array=ihalo_fof_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_inflow_candidate_IDindices_temp)
                    #Use the fof indices to extract particle data, record which particles couldn't be found
                    ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']=np.isfinite(ihalo_combined_inflow_candidate_IDindices_temp)
                    ihalo_combined_inflow_candidate_fofdata_notinfofmask=np.where(np.logical_not(ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']))
                    ihalo_combined_inflow_candidate_fofdata_notinfofmask_count=len(ihalo_combined_inflow_candidate_fofdata_notinfofmask[0])
                    for field in ['Particle_InHost']:
                        ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_fof_particles[str(snap)][field],indices=ihalo_combined_inflow_candidate_fofindices[str(snap)])
                        ihalo_combined_inflow_candidate_data[f'snap{isnap+1}_{field}'][ihalo_combined_inflow_candidate_fofdata_notinfofmask]=np.zeros(ihalo_combined_inflow_candidate_fofdata_notinfofmask_count)

                # 3. OUTPUTS FROM HISTORIES: Processed, Structure (by particle type) -- just for snap 1
                ihalo_combined_inflow_candidate_histindices={}
                ihalo_combined_inflow_candidate_data['snap1_Structure']=np.zeros(ihalo_combined_inflow_candidate_count)
                ihalo_combined_inflow_candidate_data['snap1_Processed']=np.zeros(ihalo_combined_inflow_candidate_count)
                for itype in PartTypes:
                    ihalo_combined_inflow_candidate_typemask_snap1=np.where(ihalo_combined_inflow_candidate_data['snap1_ParticleTypes']==itype)
                    ihalo_combined_inflow_candidate_IDs_unique_itype=ihalo_combined_inflow_candidate_IDs_unique[ihalo_combined_inflow_candidate_typemask_snap1]
                    #Find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
                    ihalo_combined_inflow_candidate_histindices[str(itype)]=binary_search(ihalo_combined_inflow_candidate_IDs_unique_itype,sorted_list=Part_Histories_IDs_snap1[str(itype)],check_entries=False)
                    #Extract host structure and processing
                    ihalo_combined_inflow_candidate_data['snap1_Structure'][ihalo_combined_inflow_candidate_typemask_snap1]=mask_wnans(Part_Histories_HostStructure_snap1[str(itype)],ihalo_combined_inflow_candidate_histindices[str(itype)])
                    if not Part_Histories_Constant[str(itype)]:
                        ihalo_combined_inflow_candidate_data['snap1_Processed'][ihalo_combined_inflow_candidate_typemask_snap1]=mask_wnans(Part_Histories_Processed_L1_snap1[str(itype)],ihalo_combined_inflow_candidate_histindices[str(itype)])
                    else:
                        ihalo_combined_inflow_candidate_data['snap1_Processed'][ihalo_combined_inflow_candidate_typemask_snap1]=np.ones(len(ihalo_combined_inflow_candidate_IDs_unique_itype))+9

                ############################## SAVE DATA FOR INFLOW CANDIDATES ##############################
                #############################################################################################

                # Iterate through particle types
                for itype in PartTypes:
                    # Mask for particle types - note these are taken at snap 1 (before "entering" halo)
                    itype_key=f'PartType{itype}'
                    ihalo_itype_mask=np.where(ihalo_combined_inflow_candidate_data["snap1_ParticleTypes"]==itype)

                    ### PARTICLE OUTPUTS ###
                    ########################
                    if write_partdata:
                        ihalo_hdf5['Inflow'][itype_key].create_dataset('ParticleIDs',data=ihalo_combined_inflow_candidate_IDs_unique[ihalo_itype_mask],dtype=output_fields_dtype["ParticleIDs"],compression=compression)
                        ihalo_hdf5['Inflow'][itype_key].create_dataset('Mass',data=ihalo_combined_inflow_candidate_data['snap1_Mass'][ihalo_itype_mask],dtype=output_fields_dtype["Mass"],compression=compression)

                        #Rest of fields: snap 1
                        ihalo_snap1_inflow_outputs=["Structure","Processed","Particle_InFOF","Particle_InHost","r_com","rabs_com","vrad_com"]
                        for ihalo_snap1_inflow_output in ihalo_snap1_inflow_outputs:
                            ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap1_{ihalo_snap1_inflow_output}',data=ihalo_combined_inflow_candidate_data[f'snap1_{ihalo_snap1_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap1_inflow_output],compression=compression)
                        
                        #Rest of fields: snap 2
                        ihalo_snap2_inflow_outputs=["Particle_InFOF","Particle_InHost","r_com","rabs_com"]
                        for ihalo_snap2_inflow_output in ihalo_snap2_inflow_outputs:
                            ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap2_{ihalo_snap2_inflow_output}',data=ihalo_combined_inflow_candidate_data[f'snap2_{ihalo_snap2_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap2_inflow_output],compression=compression)
                        
                        #Rest of fields: snap 3
                        ihalo_snap3_inflow_outputs=["Particle_InFOF","Particle_InHost","rabs_com"]
                        for ihalo_snap3_inflow_output in ihalo_snap3_inflow_outputs:
                            ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap3_{ihalo_snap3_inflow_output}',data=ihalo_combined_inflow_candidate_data[f'snap3_{ihalo_snap3_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap3_inflow_output],compression=compression)
                    
                    ### INTEGRATED OUTPUTS ###
                    ########################## 
                    
                    ## GRAB MASSES
                    ihalo_itype_inflow_masses=ihalo_combined_inflow_candidate_data['snap1_Mass'][ihalo_itype_mask]
                    ihalo_itype_inflow_n=len(ihalo_itype_inflow_masses)
                    ## DEFINE MASKS
                    # Masks for halo inflow definitions
                    ihalo_itype_inflow_definition={'FOF-haloscale':np.logical_and(ihalo_combined_inflow_candidate_data["snap2_Particle_InFOF"][ihalo_itype_mask],np.logical_not(ihalo_combined_inflow_candidate_data["snap1_Particle_InFOF"][ihalo_itype_mask])),
                                                   'FOF-subhaloscale':np.logical_and(ihalo_combined_inflow_candidate_data["snap2_Particle_InHost"][ihalo_itype_mask],np.logical_not(ihalo_combined_inflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask]))}
                    for ir200_fac, r200_fac in enumerate(r200_facs["Inflow"]):
                        ir200_key=f'SO-r200_fac{ir200_fac+1}'
                        ihalo_itype_inflow_definition[ir200_key]=np.logical_and(ihalo_combined_inflow_candidate_data["snap2_rabs_com"][ihalo_itype_mask]<r200_fac*ihalo_metadata['ave_R_200crit'],ihalo_combined_inflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_metadata['ave_R_200crit'])
                    # Use the leys of above to recall halo types
                    ihalo_itype_halodefs=list(ihalo_itype_inflow_definition.keys())
                    # Masks for cuts on inflow velocity as per vmax_facs
                    ihalo_itype_inflow_vmax_masks={'vmax_fac'+str(ivmax_fac+1):-ihalo_combined_inflow_candidate_data[f'snap1_vrad_com'][ihalo_itype_mask]>vmax_fac*ihalo_metadata['ave_vmax']  for ivmax_fac,vmax_fac in enumerate(vmax_facs["Inflow"])}
                    ihalo_itype_inflow_vmax_masks['vmax_fac1']=np.ones(ihalo_itype_inflow_n)
                    # Masks for processing history of particles
                    ihalo_itype_inflow_processed_masks={'Unprocessed':ihalo_combined_inflow_candidate_data["snap1_Processed"][ihalo_itype_mask]<0.5,
                                                        'Processed':ihalo_combined_inflow_candidate_data["snap1_Processed"][ihalo_itype_mask]>0.5,
                                                        'Total': np.ones(ihalo_itype_inflow_n)}

                    # Masks for the origin of inflow particles
                    ihalo_itype_inflow_origin_masks={'Gross':np.ones(ihalo_itype_inflow_n),
                                                     'Field':ihalo_combined_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]==-1,
                                                     'Transfer':ihalo_combined_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]>0.0}
                    # Masks for stability of inflow particles
                    ihalo_itype_inflow_stability={}
                    ihalo_itype_inflow_stability={'FOF-haloscale':ihalo_combined_inflow_candidate_data["snap3_Particle_InFOF"][ihalo_itype_mask],
                                                  'FOF-subhaloscale':ihalo_combined_inflow_candidate_data["snap3_Particle_InHost"][ihalo_itype_mask]}
                    for ir200_fac, r200_fac in enumerate(r200_facs["Inflow"]):
                        ir200_key=f'SO-r200_fac{ir200_fac+1}'
                        ihalo_itype_inflow_stability[ir200_key]=ihalo_combined_inflow_candidate_data["snap3_rabs_com"][ihalo_itype_mask]<r200_fac*ihalo_metadata['ave_R_200crit']

                    ## ITERATE THROUGH THE ABOVE MASKS
                    # For each halo definition
                    for halo_defname in halo_defnames["Inflow"]:
                        # If to record data for this halo and halo definition
                        if ihalo_scale_record[halo_defname]:
                            idef_mask=ihalo_itype_inflow_definition[halo_defname]
                            stability_mask=ihalo_itype_inflow_stability[halo_defname]
                            # Use detailed datasets for FOF inflow variants
                            if 'FOF' in halo_defname: 
                                icalc_processedgroups=output_processedgroups['Detailed']
                                icalc_enddatasets=output_enddatasets['Detailed']
                            else:
                                icalc_processedgroups=output_processedgroups['Basic']
                                icalc_enddatasets=output_enddatasets['Basic']
                            # For each vmax cut
                            for ivmax_fac, vmax_fac in enumerate(vmax_facs["Inflow"]):
                                ivmax_key=f'vmax_fac{ivmax_fac+1}'
                                ivmax_mask=ihalo_itype_inflow_vmax_masks[ivmax_key]
                                # For each processed group
                                for processedgroup in icalc_processedgroups:
                                    iprocessed_mask=ihalo_itype_inflow_processed_masks[processedgroup]
                                    # For each dataset
                                    for dataset in icalc_enddatasets:
                                        idset_key=dataset
                                        # Masks to concatenate
                                        origin_mask=ihalo_itype_inflow_origin_masks[dataset]
                                        masks=[idef_mask,ivmax_mask,iprocessed_mask,origin_mask]
                                        running_mask=np.logical_and.reduce([idef_mask,ivmax_mask,iprocessed_mask,origin_mask])
                                        stable_running_mask=np.logical_and(running_mask,stability_mask)
                                        all_dset_where=np.where(running_mask)
                                        stable_dset_where=np.where(stable_running_mask)
                                        # Dump data to file
                                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaM'][iihalo]=np.float32(np.nansum(ihalo_itype_inflow_masses[all_dset_where]))
                                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaN'][iihalo]=np.float32(np.nansum(running_mask))
                                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaM'][iihalo]=np.float32(np.nansum(ihalo_itype_inflow_masses[stable_dset_where]))
                                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaN'][iihalo]=np.float32(np.nansum(stable_running_mask))
                    
               
                del ihalo_combined_inflow_candidate_data
                del ihalo_itype_inflow_definition
                del ihalo_itype_inflow_vmax_masks
                del ihalo_itype_inflow_processed_masks
                del ihalo_itype_inflow_origin_masks
                del ihalo_itype_inflow_stability

                ########################################################################################################################################
                ############################################################ ihalo OUTFLOW ##############################################################
                ########################################################################################################################################

                ###### SELECT OUTFLOW CANDIDATES AS THOSE WITHIN R200crit OR the FOF envelope at snap 1 ######   

                # Do outflow calcs if we have a non-empty outflow vmax list 
                if outflow:
                    #Find the mean r200 from snap 1 / snap 2
                    ihalo_ave_R_200crit_physical=(ihalo_metadata['snap1_R_200crit']+ihalo_metadata['snap2_R_200crit'])/2
                    #Find which particles are with in the max SO region requested 
                    ihalo_cube_rcut_snap1=np.where(np.sqrt(np.sum(np.square(ihalo_cube_particles[str(snap1)]['Coordinates']-ihalo_com_physical[str(snap1)]),axis=1))<ihalo_ave_R_200crit_physical*max_r200fac_out)
                    #Concatenate the IDs of the particles within r200 and the FOF
                    ihalo_combined_outflow_candidate_IDs=np.concatenate([ihalo_fof_particles[str(snap1)]['Particle_IDs'],ihalo_cube_particles[str(snap1)]['ParticleIDs'][ihalo_cube_rcut_snap1]])
                    #Remove duplicates and convert to np.array with long ints
                    ihalo_combined_outflow_candidate_IDs_unique=np.array(np.unique(ihalo_combined_outflow_candidate_IDs),dtype=np.int64)
                    #Count outflow candidates
                    ihalo_combined_outflow_candidate_count=len(ihalo_combined_outflow_candidate_IDs_unique)

                    ############################## GRAB DATA FOR OUTFLOW CANDIDATES ##############################
                    #############################################################################################
                    ihalo_combined_outflow_candidate_data={}

                    # 1. OUTPUTS FROM DATACUBE: Coordinates, Velocity, Mass, Type 
                    ihalo_combined_outflow_candidate_cubeindices={}
                    print(f'Outflow candidates for ihalo {ihalo_s2}: n = {ihalo_combined_outflow_candidate_count}')
                    for isnap,snap in enumerate(snaps):
                        #Find the indices of the IDs in the (sorted) datacube for this halo (will return nan if not in the cube) - outputs sorted cube index
                        ihalo_combined_outflow_candidate_IDindices_temp=binary_search(ihalo_combined_outflow_candidate_IDs_unique,sorted_list=ihalo_cube_particles[str(snap)]['SortedIDs'],check_entries=True)
                        #Use the indices from the sorted IDs above to extract the cube indices (will return nan if not in the cube) - outputs raw cube index
                        ihalo_combined_outflow_candidate_cubeindices[str(snap)]=mask_wnans(array=ihalo_cube_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_outflow_candidate_IDindices_temp)
                        #For each snap, grab detailed particle data
                        for field in np.concatenate([Part_Data_fields[str(snap)],['ParticleTypes']]):
                            ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_cube_particles[str(snap)][field],indices=ihalo_combined_outflow_candidate_cubeindices[str(snap)])
                        #Derive other cubedata outputs
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com']=ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Coordinates']-ihalo_com_physical[str(snap)]
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_rabs_com']=np.sqrt(np.sum(np.square(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com']),axis=1))
                        if isnap==0:
                            ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_runit_com']=np.divide(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com'],np.column_stack([ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_rabs_com']]*3))
                            ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_v_com']=ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Velocity']-ihalo_vcom_physical[str(snap)]
                            ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_vrad_com']=np.sum(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_runit_com']*ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_v_com'],axis=1)
                        if not write_partdata:
                            del ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_r_com']
                            try:
                                del ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_runit_com']
                                del ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_v_com']
                            except:
                                pass

                    # 2. OUTPUTS FROM FOF Data: InFOF, Bound
                    ihalo_combined_outflow_candidate_fofindices={}
                    for isnap,snap in enumerate(snaps):
                        #Find the indices of the IDs in the (sorted) fof IDs for this halo (will return nan if not in the fof) - outputs index
                        ihalo_combined_outflow_candidate_IDindices_temp=binary_search(ihalo_combined_outflow_candidate_IDs_unique,sorted_list=ihalo_fof_particles[str(snap)]['SortedIDs'],check_entries=True)
                        #Use the indices from the sorted IDs above to extract the fof indices (will return nan if not in the fof) - outputs index
                        ihalo_combined_outflow_candidate_fofindices[str(snap)]=mask_wnans(array=ihalo_fof_particles[str(snap)]['SortedIndices'],indices=ihalo_combined_outflow_candidate_IDindices_temp)
                        #Use the fof indices to extract particle data, record which particles couldn't be found
                        ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']=np.isfinite(ihalo_combined_outflow_candidate_IDindices_temp)
                        ihalo_combined_outflow_candidate_fofdata_notinfofmask=np.where(np.logical_not(ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']))
                        ihalo_combined_outflow_candidate_fofdata_notinfofmask_count=len(ihalo_combined_outflow_candidate_fofdata_notinfofmask[0])
                        for field in ['Particle_InHost']:
                            ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_{field}']=mask_wnans(array=ihalo_fof_particles[str(snap)][field],indices=ihalo_combined_outflow_candidate_fofindices[str(snap)])
                            ihalo_combined_outflow_candidate_data[f'snap{isnap+1}_{field}'][ihalo_combined_outflow_candidate_fofdata_notinfofmask]=np.zeros(ihalo_combined_outflow_candidate_fofdata_notinfofmask_count)
                    
                    ############################## SAVE DATA FOR OUTFLOW CANDIDATES ##############################
                    #############################################################################################
                    # Iterate through particle types
                    for itype in PartTypes:
                        # Mask for particle types - note these are taken at snap 2 (after "leaving" halo)
                        itype_key=f'PartType{itype}'
                        ihalo_itype_mask=np.where(ihalo_combined_outflow_candidate_data["snap2_ParticleTypes"]==itype)

                        ### PARTICLE OUTPUTS ###
                        ########################
                        if write_partdata:
                            ihalo_hdf5['Outflow'][itype_key].create_dataset('ParticleIDs',data=ihalo_combined_outflow_candidate_IDs_unique[ihalo_itype_mask],dtype=output_fields_dtype["ParticleIDs"],compression=compression)
                            ihalo_hdf5['Outflow'][itype_key].create_dataset('Mass',data=ihalo_combined_outflow_candidate_data['snap1_Mass'][ihalo_itype_mask],dtype=output_fields_dtype["Mass"],compression=compression)

                            #Rest of fields: snap 1
                            ihalo_snap1_outflow_outputs=["Particle_InFOF","Particle_InHost","r_com","rabs_com","vrad_com"]
                            for ihalo_snap1_outflow_output in ihalo_snap1_outflow_outputs:
                                ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap1_{ihalo_snap1_outflow_output}',data=ihalo_combined_outflow_candidate_data[f'snap1_{ihalo_snap1_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap1_outflow_output],compression=compression)
                            
                            #Rest of fields: snap 2
                            ihalo_snap2_outflow_outputs=["Particle_InFOF","Particle_InHost","r_com","rabs_com"]
                            for ihalo_snap2_outflow_output in ihalo_snap2_outflow_outputs:
                                ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap2_{ihalo_snap2_outflow_output}',data=ihalo_combined_outflow_candidate_data[f'snap2_{ihalo_snap2_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap2_outflow_output],compression=compression)
                            
                            #Rest of fields: snap 3
                            ihalo_snap3_outflow_outputs=["Particle_InFOF","Particle_InHost",'rabs_com']
                            for ihalo_snap3_outflow_output in ihalo_snap3_outflow_outputs:
                                ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap3_{ihalo_snap3_outflow_output}',data=ihalo_combined_outflow_candidate_data[f'snap3_{ihalo_snap3_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap3_outflow_output],compression=compression)
                        
                        ### INTEGRATED OUTPUTS ###
                        ##########################
                        
                        ## GRAB MASSES
                        ihalo_itype_outflow_masses=ihalo_combined_outflow_candidate_data['snap1_Mass'][ihalo_itype_mask]
                        
                        ## DEFINE MASKS
                        # Masks for halo outflow definitions
                        halo_itype_outflow_definition={'FOF-haloscale':np.logical_and(ihalo_combined_outflow_candidate_data["snap1_Particle_InFOF"][ihalo_itype_mask],np.logical_not(ihalo_combined_outflow_candidate_data["snap2_Particle_InFOF"][ihalo_itype_mask])),
                                                    'FOF-subhaloscale':np.logical_and(ihalo_combined_outflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask],np.logical_not(ihalo_combined_outflow_candidate_data["snap2_Particle_InHost"][ihalo_itype_mask]))}
                        for ir200_fac, r200_fac in enumerate(r200_facs["Outflow"]):
                            ir200_key=f'SO-r200_fac{ir200_fac+1}'
                            halo_itype_outflow_definition[ir200_key]=np.logical_and(ihalo_combined_outflow_candidate_data["snap1_rabs_com"][ihalo_itype_mask]<r200_fac*ihalo_metadata['ave_R_200crit'],ihalo_combined_outflow_candidate_data["snap2_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_metadata['ave_R_200crit'])
                        # Masks for cuts on outflow velocity as per vmax_facs
                        ihalo_itype_outflow_vmax_masks={'vmax_fac'+str(ivmax_fac+1):ihalo_combined_outflow_candidate_data[f'snap1_vrad_com'][ihalo_itype_mask]>vmax_fac*ihalo_metadata['ave_vmax']  for ivmax_fac,vmax_fac in enumerate(vmax_facs["Outflow"])}
                        # Masks for processing history of particles
                        ihalo_itype_outflow_processed_masks={'Total': np.ones(len(ihalo_itype_outflow_masses))}
                        # Masks for the destination of outflow particles
                        ihalo_itype_outflow_destination_masks={'Gross':np.ones(len(ihalo_itype_outflow_masses))}
                        # Masks for stability
                        ihalo_itype_outflow_stability={}
                        ihalo_itype_outflow_stability={'FOF-haloscale':np.logical_not(ihalo_combined_outflow_candidate_data["snap3_Particle_InFOF"][ihalo_itype_mask]),
                                                       'FOF-subhaloscale':np.logical_not(ihalo_combined_outflow_candidate_data["snap3_Particle_InHost"][ihalo_itype_mask])}
                        for ir200_fac, r200_fac in enumerate(r200_facs["Outflow"]):
                            ir200_key=f'SO-r200_fac{ir200_fac+1}'
                            ihalo_itype_outflow_stability[ir200_key]=ihalo_combined_outflow_candidate_data["snap3_rabs_com"][ihalo_itype_mask]>r200_fac*ihalo_metadata['ave_R_200crit']

                        ## ITERATE THROUGH THE ABOVE MASKS
                        # For each halo definition
                        for halo_defname in halo_defnames["Outflow"]:
                            # If to record data for this halo and halo definition
                            if ihalo_scale_record[halo_defname]:
                                idef_mask=halo_itype_outflow_definition[halo_defname]
                                stability_mask=ihalo_itype_outflow_stability[halo_defname]
                                icalc_processedgroups=output_processedgroups['Basic']
                                icalc_enddatasets=output_enddatasets['Basic']
                                # For each vmax fac
                                for ivmax_fac, vmax_fac in enumerate(vmax_facs["Outflow"]):
                                    ivmax_key=f'vmax_fac{ivmax_fac+1}'
                                    ivmax_mask=ihalo_itype_outflow_vmax_masks[ivmax_key]
                                    # For each processed group
                                    for processedgroup in icalc_processedgroups:
                                        iprocessed_mask=ihalo_itype_outflow_processed_masks[processedgroup]
                                        #For each dataset
                                        for dataset in icalc_enddatasets:
                                            idset_key=dataset
                                            # Masks to concatenate
                                            destination_mask=ihalo_itype_outflow_destination_masks[dataset]
                                            masks=[idef_mask,ivmax_mask,iprocessed_mask,origin_mask]
                                            running_mask=np.logical_and.reduce([idef_mask,ivmax_mask,iprocessed_mask,destination_mask])
                                            stable_running_mask=np.logical_and(running_mask,stability_mask)
                                            all_dset_where=np.where(running_mask)
                                            stable_dset_where=np.where(stable_running_mask)
                                            # Dump data to file
                                            integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaM'][iihalo]=np.float32(np.nansum(ihalo_itype_outflow_masses[all_dset_where]))
                                            integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaN'][iihalo]=np.float32(np.nansum(running_mask))
                                            integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaM'][iihalo]=np.float32(np.nansum(ihalo_itype_outflow_masses[stable_dset_where]))
                                            integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaN'][iihalo]=np.float32(np.nansum(stable_running_mask))

                    del ihalo_combined_outflow_candidate_data
                    del halo_itype_outflow_definition
                    del ihalo_itype_outflow_vmax_masks
                    del ihalo_itype_outflow_processed_masks
                    del ihalo_itype_outflow_destination_masks
                    del ihalo_itype_outflow_stability

                t2_halo=time.time()
                
                with open(fname_log,"a") as progress_file:
                    progress_file.write(f"Done with ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                    progress_file.write(f"[Took {t2_halo-t1_halo:.2f} sec]\n")
                    progress_file.write(f" \n")
                    progress_file.close()

            else:# Couldn't find the halo progenitor/descendant pair
                print(f'Skipping ihalo {ihalo_s2} - couldnt find progenitor/descendant pair')
                with open(fname_log,"a") as progress_file:
                    progress_file.write(f"Skipping ihalo {ihalo_s2} - no head/tail pair ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                    progress_file.write(f" \n")
                progress_file.close()
        
        # Some other error in the main halo loop
        except: 
            print(f'Skipping ihalo {ihalo_s2} - dont have the reason')
            with open(fname_log,"a") as progress_file:
                progress_file.write(f"Skipping ihalo {ihalo_s2} - unknown reason ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                progress_file.write(f" \n")
            progress_file.close()
            continue

    #Finished with output file
    output_hdf5.close()
    return None

########################### GENERATE ACCRETION DATA: ALL (fof-only) ###########################

def gen_accretion_data_fof(base_halo_data,snap=None,halo_index_list=None,pre_depth=1,post_depth=1,vmax_facs_in=[-1],vmax_facs_out=[-1],write_partdata=False):
    
    """

    gen_accretion_data_fof : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists from VELOCIraptor FOF outputs.

    ** note: particle histories, base_halo_data and halo particle data must have been generated as per gen_particle_history_serial (this file),
             gen_base_halo_data in STFTools.py and dump_structure_particle_data in STFTools.py

	Parameters
	----------
    base_halo_data : list of dictionaries
        The minimal halo data list of dictionaries previously generated ("B1" is sufficient)

    snap : int
        The index in the base_halo_data for which to calculate accretion rates (should be actual snap index)
        We will retrieve particle data based on the flags at this index
    
    halo_index_list : dict
        "iprocess": int
        "indices: list of int
        List of the halo indices for which to calculate accretion rates. If 'None',
        find for all halos in the base_halo_data dictionary at the desired snapshot. 

    pre_depth : int
        How many snaps to skip back to when comparing particle lists.
        Initial snap for calculation will be snap-pre_depth. 

    pre_depth : int
        How many snaps to skip back to when comparing particle lists.
        Initial snap (s1) for calculation will be s1=snap-pre_depth, and we will check particle histories at s1.

    vmax_facs_in : list of float
        List of the factors of vmax to cut inflow particles at. 

    vmax_facs_out : list of float.
        List of the factors of vmax to cut outflow particles at. 
        If empty, no outflow calculations are performed. 

    write_partdata : bool 
        Flag indicating whether to write accretion/outflow particle data to file (in halo groups).
        (In addition to integrated data)

	Returns
	----------
    FOF_AccretionData_snap{snap2}_pre{pre_depth}_post{post_depth}_px.hdf5: hdf5 file with datasets
        Header contains attributes:
            "snap1"
            "snap2"
            "snap3"
            "snap1_LookbackTime"
            "snap2_LookbackTime"
            "snap3_LookbackTime"
            "ave_LookbackTime" (snap 1 -> snap 2)
            "delta_LookbackTime" (snap 1 -> snap 2)
            "snap1_z"
            "snap2_z"
            "snap3_z"
            "ave_z (snap 1 -> snap 2)

        If particle data is output:
        Group "Particle":
            There is a group for each halo: ihalo_xxxxxx

            Each halo group with attributes:
            "snapx_com"
            "snapx_cminpot"
            "snapx_cmbp"
            "snapx_vmax"
            "snapx_v"
            "snapx_M_200crit"
            "snapx_R_200mean"
            "snapx_R_200crit"

            Each halo group will have a vast collection of particle data written for snaps 1, 2, 3. 
        
        Integrated Accretion/Outflow Data
        Group "Integrated":

            Inflow:
                For each particle type /PartTypeX/:
                    For each group definition:
                        'FOF-haloscale' : the inflow as calculated from new particles to the full FOF (only for field halos or halos with substructure).
                        'FOF-subhaloscale: the inflow as calculated from new particles to the relevant substructure (can be (i) the host halo of a FOF with substructure or (ii) a substructure halo).
                        
                        Note: particles 'new to' a halo are those which were not present in the relevant halo definition at snap 1, but were at snap 2.
                        (their type taken at snap 1, before inflow). 

                        For each vmax_facx: 
                            The mass of accretion to each halo is cut to particle satisfying certain, user defined inflow velocity cuts (factors of the halo's vmax).
                            
                            For each of the following particle histories:
                                'Total' : No selection based on particle history.
                                'Processed' : Only particles which have existed in a halo prior to accretion (snap 1).
                                'Unprocessed' : Only particles which have not existed in a halo prior to accretion (snap 1). 

                                We have the following datasets...
                                    [Stability]_GrossDelta[M/N]_In: The [mass(msun)/particle count] of selected inflow candidates of all origins. 
                                    [Stability]_FieldDelta[M/N]_In: The [mass(msun)/particle count] of selected inflow candidates from the field (at snap 1). 
                                    [Stability]_TransferDelta[M/N]_In: The [mass(msun)/particle count] of selected inflow candidates from other halos (at snap 1). 

                                    Where [Stability] can be either 'Stable' or 'All'.
                                        'Stable' requires the inflow candidates to remain in the halo at snap 3. 
                                        'All' has no further requirements on the inflow candidates at snap 3. 



            Outflow: 
                For each particle type /PartTypeX/:
                    For each group definition:
                        'FOF-haloscale' : the outflow as calculated from outgoing particles from the full FOF (only for field halos or halos with substructure).
                        'FOF-subhaloscale: the outflow as calculated from outgoing particles from the relevant substructure (can be (i) the host halo of a FOF with substructure or (ii) a substructure halo).
                        
                        Note: particles 'outgoing from' a halo are those which were present in the relevant halo definition at snap 1, but were not at snap 2.
                        (their type taken at snap 2, post outflow). 
                            
                        For each vmax_facx: 
                            The mass of outflow to each halo is cut to particle satisfying certain, user defined outflow velocity cuts (factors of the halo's vmax).
                            
                            For each of the following particle histories:
                                'Total' : No selection based on particle history.

                                We have the following datasets...
                                    [Stability]_GrossDelta[M/N]_In: The [mass(msun)/particle count] of selected outflow candidates, regardless of destination.

                                    Where [Stability] can be either 'Stable' or 'All'.
                                        'Stable' requires the outflow candidates to remain outside the halo at snap 3. 
                                        'All' has no further requirements on the outflow candidates at snap 3. 

    """
    
    t1_init=time.time()

    ##### Processing inputs #####
    # Processing the snap inputs
    snap1=snap-pre_depth
    snap2=snap
    snap3=snap+post_depth
    snaps=[snap1,snap2,snap3]
    
    # Processing the desired halo index list
    if halo_index_list==None:
        halo_index_list_snap2=list(range(len(base_halo_data[snap]["hostHaloID"])))#use all halos if not handed halo index list
        iprocess="x"
        num_processes=1
        test=True
    else:
        try:
            halo_index_list_snap2=halo_index_list["indices"] #extract index list from input dictionary
            iprocess=str(halo_index_list["iprocess"]).zfill(2) #the process for this index list (this is just used for the output file name)
            print(f'iprocess {iprocess} has {len(halo_index_list_snap2)} halo indices: {halo_index_list_snap2}')
            num_processes=halo_index_list["np"]
            test=halo_index_list["test"]
        except:
            print('Not parsed a valud halo index list. Exiting.')
            return None
    
    # Find the indices of halos at snap1 and snap3 (ordered by snap2 halo indices)
    halo_index_list_snap1=[find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=pre_depth) for ihalo in halo_index_list_snap2]
    halo_index_list_snap3=[find_descen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=post_depth) for ihalo in halo_index_list_snap2]

    # Check whether we are doing the vmax cuts based on the infow  list
    if vmax_facs_in==[]:
        vmax_cut=False
    else:
        vmax_cut=True

    # Determine whether we need to perform outflow calculations based on whether we are given outflow vmax cuts
    if vmax_facs_out==[]:
        output_groups=['Inflow']
        outflow=False
    else:
        output_groups=['Inflow','Outflow']
        outflow=True

    # Add vmax factor of -1 to whatever the user input was
    if vmax_cut:
        vmax_facs_in=np.concatenate([[-1],vmax_facs_in])
    if not vmax_facs_out==[]:
        vmax_facs_out=np.concatenate([[-1],vmax_facs_out])

    if vmax_cut:
        vmax_facs={'Inflow':vmax_facs_in,'Outflow':vmax_facs_out} 
    else:
        vmax_facs={'Inflow':[0],'Outflow':[0]}

    # Define halo calculation types
    halo_defnames={}
    halo_defnames["Inflow"]=['FOF-haloscale','FOF-subhaloscale']
    halo_defnames["Outflow"]=['FOF-haloscale','FOF-subhaloscale']
    
    # Default options 
    use='cminpot' #which halo centre definition to use (from 'cminpot', 'com')
    compression='gzip'

    # Create log file and directories, initialising outputs
    if True:
        #Logs
        acc_log_dir=f"job_logs/acc_logs/"
        if not os.path.exists(acc_log_dir):
            try:
                os.mkdir(acc_log_dir)
            except:
                pass
        if test:
            run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_FOFonly_test/"
        else:
            run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_FOFonly/"

        if not os.path.exists(run_log_dir):
            try:
                os.mkdir(run_log_dir)
            except:
                pass

        run_snap_log_dir=run_log_dir+f'snap_{str(snap).zfill(3)}/'

        if not os.path.exists(run_snap_log_dir):
            try:
                os.mkdir(run_snap_log_dir)
            except:
                pass
        if test:
            fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}_test.log"
            print(f'iprocess {iprocess} will save progress to log file: {fname_log}')

        else:
            fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}.log"

        if os.path.exists(fname_log):
            os.remove(fname_log)
        
        with open(fname_log,"a") as progress_file:
            progress_file.write('Initialising and loading in data ...\n')
        progress_file.close()
    
        # Initialising outputs
        if not os.path.exists('acc_data'):#create folder for outputs if doesn't already exist
            os.mkdir('acc_data')
        if test:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_FOFonly_test/'
        else:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_FOFonly/'

        if not os.path.exists(calc_dir):#create folder for outputs if doesn't already exist
            try:
                os.mkdir(calc_dir)
            except:
                pass
        calc_snap_dir=calc_dir+f'snap_{str(snap2).zfill(3)}/'
        
        if not os.path.exists(calc_snap_dir):#create folder for outputs if doesn't already exist
            try:
                os.mkdir(calc_snap_dir)
            except:
                pass

        # Assigning snap
        if snap==None:
            snap=len(base_halo_data)-1#if not given snap, just use the last one

    # Create output file with metadata attributes
    run_outname=base_halo_data[snap]['outname']#extract output name (simulation name)
    outfile_name=calc_snap_dir+'AccretionData_pre'+str(pre_depth).zfill(2)+'_post'+str(post_depth).zfill(2)+'_snap'+str(snap).zfill(3)+'_p'+str(iprocess).zfill(3)+'.hdf5'
    
    # Remove existing output file if exists
    if not os.path.exists(outfile_name):
        print(f'Initialising output file at {outfile_name}...')
        output_hdf5=h5py.File(outfile_name,"w")
    else:
        print(f'Removing old file and initialising output file at {outfile_name}...')
        os.remove(outfile_name)
        output_hdf5=h5py.File(outfile_name,"w")

    # Make header for accretion data based on base halo data 
    if True:
        header_hdf5=output_hdf5.create_group("Header")
        lt_ave=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']+base_halo_data[snap2]['SimulationInfo']['LookbackTime'])/2
        z_ave=(base_halo_data[snap1]['SimulationInfo']['z']+base_halo_data[snap2]['SimulationInfo']['z'])/2
        dt=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']-base_halo_data[snap2]['SimulationInfo']['LookbackTime'])
        t1=base_halo_data[snap1]['SimulationInfo']['LookbackTime']
        t2=base_halo_data[snap2]['SimulationInfo']['LookbackTime']
        t3=base_halo_data[snap3]['SimulationInfo']['LookbackTime']
        z1=base_halo_data[snap1]['SimulationInfo']['z']
        z2=base_halo_data[snap2]['SimulationInfo']['z']
        z3=base_halo_data[snap3]['SimulationInfo']['z']
        header_hdf5.attrs.create('ave_LookbackTime',data=lt_ave,dtype=np.float16)
        header_hdf5.attrs.create('ave_z',data=z_ave,dtype=np.float16)
        header_hdf5.attrs.create('delta_LookbackTime',data=dt,dtype=np.float16)
        header_hdf5.attrs.create('snap1_LookbackTime',data=t1,dtype=np.float16)
        header_hdf5.attrs.create('snap2_LookbackTime',data=t2,dtype=np.float16)
        header_hdf5.attrs.create('snap3_LookbackTime',data=t3,dtype=np.float16)
        header_hdf5.attrs.create('snap1_z',data=z1,dtype=np.float16)
        header_hdf5.attrs.create('snap2_z',data=z2,dtype=np.float16)
        header_hdf5.attrs.create('snap3_z',data=z3,dtype=np.float16)
        header_hdf5.attrs.create('snap1',data=snap1,dtype=np.int16)
        header_hdf5.attrs.create('snap2',data=snap2,dtype=np.int16)
        header_hdf5.attrs.create('snap3',data=snap3,dtype=np.int16)
        header_hdf5.attrs.create('pre_depth',data=snap2-snap1,dtype=np.int16)
        header_hdf5.attrs.create('post_depth',data=snap3-snap2,dtype=np.int16)
        header_hdf5.attrs.create('outname',data=np.string_(base_halo_data[snap2]['outname']))
        header_hdf5.attrs.create('total_num_halos',data=base_halo_data[snap2]['Count'])
    
    # Standard particle type names from simulation
    SimType=base_halo_data[snap2]['Part_FileType']
    

    ##### Loading in Data #####
    # Load in FOF particle lists: snap 1, snap 2, snap 3
    FOF_Part_Data={}
    FOF_Part_Data[str(snap1)]=get_FOF_particle_lists(base_halo_data,snap1,halo_index_list=halo_index_list_snap1)
    FOF_Part_Data[str(snap2)]=get_FOF_particle_lists(base_halo_data,snap2,halo_index_list=halo_index_list_snap2)
    FOF_Part_Data[str(snap3)]=get_FOF_particle_lists(base_halo_data,snap3,halo_index_list=halo_index_list_snap3)
    FOF_Part_Data_fields=list(FOF_Part_Data[str(snap1)].keys()) #Fields from FOF data

    # Particle data - LOAD HERE
    print('Retrieving & organising raw particle data ...')
    PartNames=['Gas','DM','','','Star','BH']
    hval=base_halo_data[snap1]['SimulationInfo']['h_val'];scalefactors={}
    scalefactors={str(snap):base_halo_data[snap]['SimulationInfo']['ScaleFactor'] for snap in snaps}
    Mass_DM=base_halo_data[snap2]['SimulationInfo']['Mass_DM_Physical']
    Mass_Gas=base_halo_data[snap2]['SimulationInfo']['Mass_Gas_Physical']
    BoxSize=base_halo_data[snap2]['SimulationInfo']['BoxSize_Comoving']

    #Conversion factors for particle data
    Part_Data_comtophys={str(snap):{'Coordinates':scalefactors[str(snap)]/hval, 
                                    'Velocity':scalefactors[str(snap)]/hval,
                                    'Mass':10.0**10/hval,
                                    'ParticleIDs':1} for snap in snaps}
    
    Part_Data_FilePaths={str(snap):base_halo_data[snap]['Part_FilePath'] for snap in snaps}
    #If EAGLE, read masses
    if SimType=='EAGLE':
        #Which fields do we need at each snap
        PartTypes_keys_all=list(h5py.File(base_halo_data[-1]['Part_FilePath'],'r').keys())
        PartTypes_keys=[key for key in PartTypes_keys_all if 'PartType'  in key]
        PartTypes=[int(key.split('PartType')[-1]) for key in PartTypes_keys]
        print(f'Using parttypes: {PartTypes}')

        Part_Data_fields={str(snap1):['Mass'],str(snap2):['Mass'],str(snap3):[]}
        if vmax_cut:
            for snap in [snap1,snap2]:
                Part_Data_fields[str(snap)].append('Coordinates')
                Part_Data_fields[str(snap)].append('Velocity')

    #If not EAGLE, assume constant masses
    else:
        #Which fields do we need at each snap
        PartTypes=[0,1] #Gas, DM, Stars, BH
        Part_Data_fields={str(snap1):['Mass'],str(snap2):['Mass'],str(snap3):[]}
        if vmax_cut:
            for snap in [snap1,snap2]:
                Part_Data_fields[str(snap)].append('Coordinates')
                Part_Data_fields[str(snap)].append('Velocity')
    
    Part_Data_Full={str(snap):{field:{} for field in Part_Data_fields[str(snap)]} for snap in snaps}

    for snap in snaps:
        if SimType=='EAGLE':
            EAGLE_snap=read_eagle.EagleSnapshot(Part_Data_FilePaths[str(snap)])
            EAGLE_snap.select_region(xmin=0,xmax=BoxSize,
                                     ymin=0,ymax=BoxSize,
                                     zmin=0,zmax=BoxSize)
            for field in Part_Data_fields[str(snap)]:
                if not field=='Mass':
                    for itype in PartTypes:
                        try:
                            Part_Data_Full[str(snap)][field][str(itype)]=EAGLE_snap.read_dataset(itype,field)*Part_Data_comtophys[str(snap)][field]
                        except:
                            Part_Data_Full[str(snap)][field][str(itype)]=np.array([])
                else:
                    for itype in [0,4,5]:
                        try:
                            Part_Data_Full[str(snap)][field][str(itype)]=EAGLE_snap.read_dataset(itype,field)*Part_Data_comtophys[str(snap)][field]
                        except:
                            Part_Data_Full[str(snap)][field][str(itype)]=np.array([])
                    
                    Part_Data_Full[str(snap)][field][str(1)]=Mass_DM*np.ones(len(Part_Data_Full[str(snap)]['Mass'][str(0)]))
        else:
            Part_Data_Full[str(snap)]['Mass']={}
            Mass_Constant={str(0):True,str(1):True}
            Part_Data_file=h5py.File(Part_Data_FilePaths[str(snap)],'r')
            npart=512**3
            for field in Part_Data_fields[str(snap)]:
                if not field=='Mass':
                    if field=='Velocities':
                        Part_Data_Full[str(snap)]['Velocity']={str(itype):Part_Data_file[f'PartType{itype}']['Velocities'].value*Part_Data_comtophys[str(snap)]['Velocity'] for itype in PartTypes}
                    else:
                        Part_Data_Full[str(snap)][field]={str(itype):Part_Data_file[f'PartType{itype}'][field].value*Part_Data_comtophys[str(snap)][field] for itype in PartTypes}                
            Part_Data_Full[str(snap)]['Mass'][str(0)]=np.ones(npart)*h5py.File(base_halo_data[snap]['Part_FilePath'],'r')['Header'].attrs['MassTable'][0]*Part_Data_comtophys[str(snap)]['Mass']
            Part_Data_Full[str(snap)]['Mass'][str(1)]=np.ones(npart)*h5py.File(base_halo_data[snap]['Part_FilePath'],'r')['Header'].attrs['MassTable'][1]*Part_Data_comtophys[str(snap)]['Mass']

    if not SimType=='EAGLE':
        if vmax_cut:
            Part_Data_fields={str(snap1):['Coordinates','Velocity','Mass'],str(snap2):['Coordinates','Velocity','Mass'],str(snap3):[]}
        else:
            Part_Data_fields={str(snap1):['Mass'],str(snap2):['Mass'],str(snap3):[]}


    print('Done retrieving & organising raw particle data ...')

    # Load in particle histories
    print(f'Retrieving & organising particle histories ...')
    Part_Histories_fields={str(snap1):["ParticleIDs",'ParticleIndex','HostStructure','Processed_L1'],str(snap2):["ParticleIDs",'ParticleIndex'],str(snap3):[]}
    Part_Histories_data={str(snap):{} for snap in snaps}
    Part_Histories_Constant={str(0):False,str(1):False,str(4):False,str(5):False}
    for snap in snaps[:-1]:
        Part_Histories_File_snap=h5py.File("part_histories/PartHistory_"+str(snap).zfill(3)+"_"+run_outname+".hdf5",'r')
        for field in Part_Histories_fields[str(snap)]:
            Part_Histories_data[str(snap)][field]={str(itype):Part_Histories_File_snap["PartType"+str(itype)+'/'+field].value for itype in PartTypes}

    print(f'Done retrieving & organising particle histories')

    print()
    t2_init=time.time()
    print('*********************************************************')
    print(f'Done initialising in {(t2_init-t1_init):.2f} sec - entering main halo loop ...')
    print('*********************************************************')

    with open(fname_log,"a") as progress_file:
        progress_file.write(f'Done initialising in {(t2_init-t1_init):.2f} sec - entering main halo loop ...\n')
    progress_file.close()

    ##### Initialising outputs #####
    # Particle
    if write_partdata:
        #hdf5 group
        particle_output_hdf5=output_hdf5.create_group('Particle')
        
        #output dtypes
        output_fields_dtype={}
        output_fields_float32=["Mass","r_com","rabs_com","vrad_com","vtan_com"]
        for field in output_fields_float32:
            output_fields_dtype[field]=np.float32

        output_fields_int64=["ParticleIDs","Structure"]
        for field in output_fields_int64:
            output_fields_dtype[field]=np.int64
        
        output_fields_int8=["Processed","Particle_InFOF","Particle_Bound","Particle_InHost",'ParticleTypes']
        for field in output_fields_int8:
            output_fields_dtype[field]=np.int8 

    # Integrated (always written)
    num_halos_thisprocess=len(halo_index_list_snap2)
    integrated_output_hdf5=output_hdf5.create_group('Integrated')
    integrated_output_hdf5.create_dataset('ihalo_list',data=halo_index_list_snap2)

    #Defining which outputs for varying levels of detail
    output_processedgroups={'Detailed':['Total','Unprocessed','Processed'],
                            'Basic':['Total']}
    output_enddatasets={'Detailed':['Gross','Field','Transfer'],
                        'Basic':['Gross']}
    
    #Initialise output datasets with np.nans (len: total_num_halos)
    for output_group in output_groups:
        integrated_output_hdf5.create_group(output_group)
        for itype in PartTypes:
            itype_key=f'PartType{itype}'
            integrated_output_hdf5[output_group].create_group(itype_key)
            for ihalo_defname,halo_defname in enumerate(sorted(halo_defnames[output_group])):
                #Create group
                integrated_output_hdf5[output_group][itype_key].create_group(halo_defname)
                #Use detailed datasets for inflow variants
                if output_group=='Inflow': 
                    icalc_processedgroups=output_processedgroups['Detailed']
                    icalc_enddatasets=output_enddatasets['Detailed']
                else:
                    icalc_processedgroups=output_processedgroups['Basic']
                    icalc_enddatasets=output_enddatasets['Basic']
                #Now, for each Vmax cut
                for ivmax_fac, vmax_fac in enumerate(vmax_facs[output_group]):
                    #Create group
                    ivmax_key=f'vmax_fac{ivmax_fac+1}'
                    integrated_output_hdf5[output_group][itype_key][halo_defname].create_group(ivmax_key)
                    #Add attribute for vmax_fac
                    integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
                    #Initialise datasets with nans
                    for processedgroup in icalc_processedgroups:
                        integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key].create_group(processedgroup)
                        for dataset in icalc_enddatasets:
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'All_'+dataset+f'_DeltaM',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'All_'+dataset+f'_DeltaN',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'Stable_'+dataset+f'_DeltaM',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'Stable_'+dataset+f'_DeltaN',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)

    integrated_output_hdf5['Inflow']['PartType0']['FOF-haloscale']['vmax_fac1']['Total'].create_dataset('All_Gross_fracSF',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
    integrated_output_hdf5['Inflow']['PartType0']['FOF-haloscale']['vmax_fac1']['Total'].create_dataset('All_Field_fracSF',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
    integrated_output_hdf5['Inflow']['PartType0']['FOF-haloscale']['vmax_fac1']['Total'].create_dataset('All_Transfer_fracSF',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)

    ####################################################################################################################################################################################
    ####################################################################################################################################################################################
    ########################################################################### MAIN HALO LOOP #########################################################################################
    ####################################################################################################################################################################################
    ####################################################################################################################################################################################

    for iihalo,ihalo_s2 in enumerate(halo_index_list_snap2):# for each halo (index at snap 2)
        
        # If needed, create group for this halo in output file
        if write_partdata:
            ihalo_hdf5=particle_output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))
            ihalo_hdf5.create_group('Metadata')
            if write_partdata:
                ihalo_hdf5.create_group('Inflow');ihalo_hdf5.create_group('Outflow')
                for itype in PartTypes:
                    ihalo_hdf5['Inflow'].create_group(f'PartType{itype}')
                    ihalo_hdf5['Outflow'].create_group(f'PartType{itype}')        
        
        # This catches any exceptions for a given halo and prevents the code from crashing 
        if True:     
            # try:
            ########################################################################################################################################
            ###################################################### ihalo PRE-PROCESSING ############################################################
            ########################################################################################################################################
            t1_halo=time.time()
            
            # Find halo progenitor and descendants
            ihalo_indices={str(snap1):halo_index_list_snap1[iihalo],str(snap2):ihalo_s2,str(snap3):halo_index_list_snap3[iihalo]}
            
            # Record halo properties 
            ihalo_tracked=(ihalo_indices[str(snap1)]>-1 and ihalo_indices[str(snap3)]>-1)#track if have both progenitor and descendant
            ihalo_structuretype=base_halo_data[snap2]["Structuretype"][ihalo_indices[str(snap2)]]#structure type
            ihalo_numsubstruct=base_halo_data[snap2]["numSubStruct"][ihalo_indices[str(snap2)]]
            ihalo_hostHaloID=base_halo_data[snap2]["hostHaloID"][ihalo_indices[str(snap2)]]
            ihalo_sublevel=int(np.floor((ihalo_structuretype-0.01)/10))
            ihalo_recordsubaccretion=ihalo_numsubstruct>0 or ihalo_hostHaloID>0 #record substructure-scale inflow/outflow IF has substructure or is subhalo
            ihalo_recordfieldaccretion=ihalo_numsubstruct>0 or ihalo_hostHaloID<0 #record field-scale inflow/outflow IF has substructure or is field halo
            
            # Which scales to record for this halo [inflow and outflow]
            ihalo_scale_record={ihalo_halodef:True for ihalo_halodef in ['FOF-haloscale','FOF-subhaloscale']}
            if not ihalo_recordsubaccretion:
                ihalo_scale_record['FOF-subhaloscale']=False
            if not ihalo_recordfieldaccretion:
                ihalo_scale_record['FOF-haloscale']=False

            # Print progress to terminal and output file
            print();print('**********************************************')
            print('Halo index: ',ihalo_s2,f' - {ihalo_numsubstruct} substructures')
            print(f'Progenitor: {ihalo_indices[str(snap1)]} | Descendant: {ihalo_indices[str(snap3)]}')
            print('**********************************************');print()
            with open(fname_log,"a") as progress_file:
                progress_file.write(f' \n')
                progress_file.write(f'Starting with ihalo {ihalo_s2} ... \n')
            progress_file.close()
            
            # This catches any halos for which we can't find a progenitor/descendant 
            nhalo_valid=np.nansum(base_halo_data[snap2]['Mass_200crit']>10**9.25/10**10)
            if ihalo_tracked and base_halo_data[snap2]['Mass_200crit'][ihalo_s2]>10**9.25/10**10:
                print(f'{iihalo/nhalo_valid*100:.1f} % done with valid halos')
                ### GRAB HALO METADATA ###
                ihalo_metadata={}
                for isnap,snap in enumerate(snaps):
                    ihalo_isnap=ihalo_indices[str(snap)]
                    if ihalo_isnap>=0:
                        ihalo_metadata[f'snap{isnap+1}_com']=np.array([base_halo_data[snap]['Xc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Yc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zc'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_cminpot']=np.array([base_halo_data[snap]['Xcminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Ycminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zcminpot'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_vcom']=np.array([base_halo_data[snap]['VXc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VYc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VZc'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_R_200crit']=base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_R_200mean']=base_halo_data[snap]['R_200mean'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_Mass_200crit']=base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*10**10
                        ihalo_metadata[f'snap{isnap+1}_vmax']=base_halo_data[snap]['Vmax'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_vesc_crit']=np.sqrt(2*base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*base_halo_data[snap]['SimulationInfo']['Gravity']/base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]])
                
                # Average some quantities
                ihalo_metadata['sublevel']=ihalo_sublevel
                ihalo_metadata['ave_R_200crit']=0.5*base_halo_data[snap1]['R_200crit'][ihalo_indices[str(snap1)]]+0.5*base_halo_data[snap2]['R_200crit'][ihalo_indices[str(snap2)]]
                ihalo_metadata['ave_vmax']=0.5*base_halo_data[snap1]['Vmax'][ihalo_indices[str(snap1)]]+0.5*base_halo_data[snap2]['Vmax'][ihalo_indices[str(snap2)]]
                
                #Cube parameters
                ihalo_com_physical={str(snap):np.array(ihalo_metadata[f'snap{isnap+1}_{use}']) for isnap,snap in enumerate(snaps)}
                ihalo_com_comoving={str(snap):np.array(ihalo_metadata[f'snap{isnap+1}_{use}'])/Part_Data_comtophys[str(snap)]['Coordinates'] for isnap,snap in enumerate(snaps)}
                ihalo_vcom_physical={str(snap):np.array(ihalo_metadata[f'snap{isnap+1}_vcom']) for isnap,snap in enumerate(snaps)}

                # Write halo metadata to file (if desired)
                if write_partdata:
                    for ihalo_mdkey in list(ihalo_metadata.keys()): 
                        size=np.size(ihalo_metadata[ihalo_mdkey])
                        if size>1:
                            ihalo_hdf5['Metadata'].create_dataset(ihalo_mdkey,data=ihalo_metadata[ihalo_mdkey],dtype=np.float32,shape=(1,size))
                        else:
                            ihalo_hdf5['Metadata'].create_dataset(ihalo_mdkey,data=ihalo_metadata[ihalo_mdkey],dtype=np.float32)

                ### GET HALO DATA FROM VELOCIRAPTOR AND SIM ###
                # Grab the FOF particle data 
                ihalo_fof_particles={}
                for snap in snaps:
                    # Read the FOF data for this halo
                    ihalo_fof_particles[str(snap)]={field:FOF_Part_Data[str(snap)][field][str(ihalo_indices[str(snap)])] for field in FOF_Part_Data_fields}
                    # Add the sorted IDs/indices
                    ihalo_fof_particles[str(snap)]['SortedIndices']=np.argsort(ihalo_fof_particles[str(snap)]['Particle_IDs'])
                    ihalo_fof_particles[str(snap)]['SortedIDs']=ihalo_fof_particles[str(snap)]['Particle_IDs'][(ihalo_fof_particles[str(snap)]['SortedIndices'],)]
                    # Add sets for particle IDs in FOF and in Host
                    ihalo_fof_particles[str(snap)]['ParticleIDs_InFOF']=ihalo_fof_particles[str(snap)]['Particle_IDs']
                    ihalo_fof_particles[str(snap)]['ParticleIDs_InHost']=ihalo_fof_particles[str(snap)]['Particle_IDs'][np.where(ihalo_fof_particles[str(snap)]['Particle_InHost']==1)]

                print(f'Finished retrieving data from FOF for ihalo {ihalo_s2}')
                
                ########################################################################################################################################
                ############################################################ ihalo INFLOW ##############################################################
                ########################################################################################################################################

                ###### SELECT INFLOW CANDIDATES AS THOSE WITHIN FOF envelope at snap 2 ######
                #############################################################################################

                #Get the particle data of the particles in the FOF
                ihalo_fof_inflow_candidate_data={str(snap):{field:ihalo_fof_particles[str(snap)][field] for field in FOF_Part_Data_fields} for snap in snaps}
                #Concatenate the IDs of the particles within r200 and the FOF
                ihalo_inflow_candidate_IDs=ihalo_fof_inflow_candidate_data[str(snap2)]['Particle_IDs']
                ihalo_inflow_candidate_Types=ihalo_fof_inflow_candidate_data[str(snap2)]['Particle_Types']
                #Count inflow candidates
                ihalo_inflow_candidate_count=len(ihalo_inflow_candidate_IDs)

                ############################## GRAB DATA FOR INFLOW CANDIDATES ##############################
                #############################################################################################
                ihalo_inflow_candidate_data={}
            
                #  1. OUTPUTS FROM FOF Data: InFOF, InHost
                for isnap,snap in enumerate(snaps):
                    ihalo_inflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']=np.in1d(ihalo_inflow_candidate_IDs,ihalo_fof_particles[str(snap)]['ParticleIDs_InFOF'])
                    ihalo_inflow_candidate_data[f'snap{isnap+1}_Particle_InHost']=np.in1d(ihalo_inflow_candidate_IDs,ihalo_fof_particles[str(snap)]['ParticleIDs_InHost'])

                # 2. OUTPUTS FROM SIM AND HISTORIES: Types,  Processed, Structure (and maybe: Coordinates, Velocity)
                # Grab particle indices from histories and sim
                for isnap,snap in enumerate(snaps[:-1]):
                    ihalo_isnap_inflow_candidate_parttypes,ihalo_isnap_inflow_candidate_historyindices,ihalo_isnap_inflow_candidate_partindices=get_particle_indices(base_halo_data,
                                                                                                   IDs_sorted=Part_Histories_data[str(snap)]['ParticleIDs'],
                                                                                                   indices_sorted=Part_Histories_data[str(snap)]['ParticleIndex'],
                                                                                                   IDs_taken=ihalo_inflow_candidate_IDs,
                                                                                                   types_taken=ihalo_inflow_candidate_Types,
                                                                                                   snap_taken=snap2,
                                                                                                   snap_desired=snap,
                                                                                                   return_partindices=True)
                    #Dump particle type
                    ihalo_inflow_candidate_data[f'snap{isnap+1}_ParticleTypes']=ihalo_isnap_inflow_candidate_parttypes
                    
                    #If first snap, dump structure/processing level
                    if snap==snap1:
                        ihalo_inflow_candidate_data[f'snap{isnap+1}_Structure']=np.array([Part_Histories_data[str(snap)]['HostStructure'][str(ipart_type)][ipart_histidx] for ipart_type,ipart_histidx in zip(ihalo_isnap_inflow_candidate_parttypes,ihalo_isnap_inflow_candidate_historyindices)])
                        ihalo_inflow_candidate_data[f'snap{isnap+1}_Processed']=np.zeros(len(ihalo_inflow_candidate_data[f'snap{isnap+1}_Structure']))+np.nan
                        
                        for iipart in range(ihalo_inflow_candidate_count):
                            ipart_type=ihalo_isnap_inflow_candidate_parttypes[iipart]
                            ipart_histidx=ihalo_isnap_inflow_candidate_historyindices[iipart]
                            ihalo_inflow_candidate_data[f'snap{isnap+1}_Processed'][iipart]=Part_Histories_data[str(snap)]['Processed_L1'][str(ipart_type)][ipart_histidx]


                    #Grab particle data
                    for field in Part_Data_fields[str(snap)]:
                        ex_point=Part_Data_Full[str(snap)][field][str(0)][0]
                        size=np.size(ex_point)
                        ihalo_inflow_candidate_data[f'snap{isnap+1}_{field}']=np.array(np.zeros((ihalo_inflow_candidate_count,size))+np.nan)
                        iipart=0
                        for ipart_type,ipart_partidx in zip(ihalo_isnap_inflow_candidate_parttypes,ihalo_isnap_inflow_candidate_partindices):
                            try:
                                ihalo_inflow_candidate_data[f'snap{isnap+1}_{field}'][iipart]=Part_Data_Full[str(snap)][field][str(ipart_type)][ipart_partidx]
                            except:
                                pass
                            iipart=iipart+1
                    #Derive other simulation outputs
                    if snap==snap1 or snap==snap2:
                        if vmax_cut:
                            ihalo_inflow_candidate_data[f'snap{isnap+1}_r_com']=ihalo_inflow_candidate_data[f'snap{isnap+1}_Coordinates']-ihalo_com_physical[str(snap)]
                            ihalo_inflow_candidate_data[f'snap{isnap+1}_rabs_com']=np.sqrt(np.sum(np.square(ihalo_inflow_candidate_data[f'snap{isnap+1}_r_com']),axis=1))
                            ihalo_inflow_candidate_data[f'snap{isnap+1}_runit_com']=np.divide(ihalo_inflow_candidate_data[f'snap{isnap+1}_r_com'],np.column_stack([ihalo_inflow_candidate_data[f'snap{isnap+1}_rabs_com']]*3))
                            ihalo_inflow_candidate_data[f'snap{isnap+1}_v_com']=ihalo_inflow_candidate_data[f'snap{isnap+1}_Velocity']-ihalo_vcom_physical[str(snap)]
                            ihalo_inflow_candidate_data[f'snap{isnap+1}_vrad_com']=np.sum(ihalo_inflow_candidate_data[f'snap{isnap+1}_runit_com']*ihalo_inflow_candidate_data[f'snap{isnap+1}_v_com'],axis=1)
                            if not write_partdata:
                                del ihalo_inflow_candidate_data[f'snap{isnap+1}_r_com']
                                del ihalo_inflow_candidate_data[f'snap{isnap+1}_runit_com']
                                del ihalo_inflow_candidate_data[f'snap{isnap+1}_v_com']

                ############################## SAVE DATA FOR INFLOW CANDIDATES ##############################
                #############################################################################################

                # Iterate through particle types
                for itype in PartTypes:
                    # Mask for particle types - note these are taken at snap 1 (before "entering" halo)
                    itype_key=f'PartType{itype}'
                    ihalo_itype_mask=np.where(ihalo_inflow_candidate_data["snap1_ParticleTypes"]==itype)

                    ### PARTICLE OUTPUTS ###
                    ########################
                    if write_partdata:
                        ihalo_hdf5['Inflow'][itype_key].create_dataset('ParticleIDs',data=ihalo_inflow_candidate_IDs[ihalo_itype_mask],dtype=output_fields_dtype["ParticleIDs"],compression=compression)
                        ihalo_hdf5['Inflow'][itype_key].create_dataset('Mass',data=ihalo_inflow_candidate_data['snap1_Mass'][ihalo_itype_mask],dtype=output_fields_dtype["Mass"],compression=compression)

                        #Rest of fields: snap 1
                        if vmax_cut:
                            ihalo_snap1_inflow_outputs=["Structure","Processed","Particle_InFOF","Particle_InHost","r_com","rabs_com","vrad_com"]
                        else:
                            ihalo_snap1_inflow_outputs=["Structure","Processed","Particle_InFOF","Particle_InHost"]

                        for ihalo_snap1_inflow_output in ihalo_snap1_inflow_outputs:
                            ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap1_{ihalo_snap1_inflow_output}',data=ihalo_inflow_candidate_data[f'snap1_{ihalo_snap1_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap1_inflow_output],compression=compression)
                        
                        #Rest of fields: snap 2
                        if vmax_cut:
                            ihalo_snap2_inflow_outputs=["Particle_InFOF","Particle_InHost",'ParticleTypes',"r_com","rabs_com","vrad_com"]
                        else:
                            ihalo_snap2_inflow_outputs=["Particle_InFOF","Particle_InHost",'ParticleTypes']

                        for ihalo_snap2_inflow_output in ihalo_snap2_inflow_outputs:
                            ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap2_{ihalo_snap2_inflow_output}',data=ihalo_inflow_candidate_data[f'snap2_{ihalo_snap2_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap2_inflow_output],compression=compression)
                        
                        #Rest of fields: snap 3
                        ihalo_snap3_inflow_outputs=["Particle_InFOF","Particle_InHost"]
                        for ihalo_snap3_inflow_output in ihalo_snap3_inflow_outputs:
                            ihalo_hdf5['Inflow'][itype_key].create_dataset(f'snap3_{ihalo_snap3_inflow_output}',data=ihalo_inflow_candidate_data[f'snap3_{ihalo_snap3_inflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap3_inflow_output],compression=compression)
                    
                    ### INTEGRATED OUTPUTS ###
                    ########################## 

                    ## GRAB MASSES
                    ihalo_itype_inflow_masses=ihalo_inflow_candidate_data['snap1_Mass'][ihalo_itype_mask]
                    ihalo_itype_inflow_finaltypes=ihalo_inflow_candidate_data['snap2_ParticleTypes'][ihalo_itype_mask]

                    ## DEFINE MASKS
                    # Masks for halo inflow definitions
                    ihalo_itype_inflow_definition={'FOF-haloscale':np.logical_and(ihalo_inflow_candidate_data["snap2_Particle_InFOF"][ihalo_itype_mask],np.logical_not(ihalo_inflow_candidate_data["snap1_Particle_InFOF"][ihalo_itype_mask])),
                                                   'FOF-subhaloscale':np.logical_and(ihalo_inflow_candidate_data["snap2_Particle_InHost"][ihalo_itype_mask],np.logical_not(ihalo_inflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask]))}
                   
                    # Use the leys of above to recall halo types
                    ihalo_itype_halodefs=list(ihalo_itype_inflow_definition.keys())
                    
                    # Masks for cuts on inflow velocity as per vmax_facs
                    if vmax_cut:
                        ihalo_itype_inflow_vmax_masks={'vmax_fac'+str(ivmax_fac+1):-ihalo_inflow_candidate_data[f'snap1_vrad_com'][ihalo_itype_mask]>vmax_fac*ihalo_metadata['ave_vmax']  for ivmax_fac,vmax_fac in enumerate(vmax_facs["Inflow"])}
                    else:
                        ihalo_itype_inflow_vmax_masks={'vmax_fac1':np.ones(len(ihalo_inflow_candidate_data[f'snap1_Processed'][ihalo_itype_mask]))}

                    # Masks for processing history of particles
                    ihalo_itype_inflow_processed_masks={'Unprocessed':ihalo_inflow_candidate_data["snap1_Processed"][ihalo_itype_mask]==0.0,
                                                        'Processed':ihalo_inflow_candidate_data["snap1_Processed"][ihalo_itype_mask]>0.0,
                                                        'Total': np.isfinite(ihalo_inflow_candidate_data["snap1_Processed"][ihalo_itype_mask])}
                    # Masks for the origin of inflow particles
                    ihalo_itype_inflow_origin_masks={'Gross':np.isfinite(ihalo_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]),
                                                     'Field':ihalo_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]==-1,
                                                     'Transfer':ihalo_inflow_candidate_data["snap1_Structure"][ihalo_itype_mask]>0}
                    # Masks for stability of inflow particles
                    ihalo_itype_inflow_stability={}
                    ihalo_itype_inflow_stability={'FOF-haloscale':ihalo_inflow_candidate_data["snap3_Particle_InFOF"][ihalo_itype_mask],
                                                  'FOF-subhaloscale':ihalo_inflow_candidate_data["snap3_Particle_InHost"][ihalo_itype_mask]}
                   

                    ###### quickly save star formation frac for gas particles


                    ## ITERATE THROUGH THE ABOVE MASKS
                    # For each halo definition
                    for halo_defname in halo_defnames["Inflow"]:
                        # If to record data for this halo and halo definition
                        if ihalo_scale_record[halo_defname]:
                            idef_mask=ihalo_itype_inflow_definition[halo_defname]
                            stability_mask=ihalo_itype_inflow_stability[halo_defname]
                            icalc_processedgroups=output_processedgroups['Detailed']
                            icalc_enddatasets=output_enddatasets['Detailed']
                            # For each vmax cut
                            for ivmax_fac, vmax_fac in enumerate(vmax_facs["Inflow"]):
                                ivmax_key=f'vmax_fac{ivmax_fac+1}'
                                ivmax_mask=ihalo_itype_inflow_vmax_masks[ivmax_key]

                                # For each processed group
                                for processedgroup in icalc_processedgroups:
                                    iprocessed_mask=ihalo_itype_inflow_processed_masks[processedgroup]
                                    # For each dataset
                                    for dataset in icalc_enddatasets:
                                        idset_key=dataset
                                        # Masks to concatenate
                                        origin_mask=ihalo_itype_inflow_origin_masks[dataset]
                                        masks=[idef_mask,ivmax_mask,iprocessed_mask,origin_mask]
                                        running_mask=np.logical_and.reduce([idef_mask,ivmax_mask ,iprocessed_mask,origin_mask])
                                        stable_running_mask=np.logical_and(running_mask,stability_mask)
                                        all_dset_where=np.where(running_mask)
                                        stable_dset_where=np.where(stable_running_mask)

                                        # Dump data to file
                                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaM'][iihalo]=np.float32(np.nansum(ihalo_itype_inflow_masses[all_dset_where]))            
                                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaN'][iihalo]=np.float32(np.nansum(running_mask))
                                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaM'][iihalo]=np.float32(np.nansum(ihalo_itype_inflow_masses[stable_dset_where]))
                                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaN'][iihalo]=np.float32(np.nansum(stable_running_mask))
                                        if itype_key=='PartType0' and halo_defname=='FOF-haloscale' and ivmax_key=='vmax_fac1' and processedgroup=='Total':
                                            integrated_output_hdf5['Inflow']['PartType0']['FOF-haloscale']['vmax_fac1']['Total']['All_'+dataset+'_fracSF'][iihalo]=np.nansum(ihalo_itype_inflow_finaltypes[all_dset_where]==4)/len(ihalo_itype_inflow_finaltypes[all_dset_where])
                                            print(f'Fraction of accreted gas transformed to stars: {np.nansum(ihalo_itype_inflow_finaltypes[all_dset_where]==4)/len(ihalo_itype_inflow_finaltypes[all_dset_where])*100:.2e}%')
                        
                
                ########################################################################################################################################
                ############################################################ ihalo OUTFLOW #############################################################
                ########################################################################################################################################
                del ihalo_inflow_candidate_data

                if outflow:
                    ###### SELECT OUTFLOW CANDIDATES AS THOSE WITHIN FOF envelope at snap 1 ######
                    ###############################################################################

                    #Get the particle data of the particles in the FOF
                    ihalo_fof_outflow_candidate_data={str(snap):{field:ihalo_fof_particles[str(snap)][field] for field in FOF_Part_Data_fields} for snap in snaps}
                    #Concatenate the IDs of the particles within r200 and the FOF
                    ihalo_outflow_candidate_IDs=ihalo_fof_outflow_candidate_data[str(snap1)]['Particle_IDs']
                    ihalo_outflow_candidate_Types=ihalo_fof_outflow_candidate_data[str(snap1)]['Particle_Types']
                    #Count outflow candidates
                    ihalo_outflow_candidate_count=len(ihalo_outflow_candidate_IDs)

                    ############################## GRAB DATA FOR INFLOW CANDIDATES ##############################
                    #############################################################################################
                    ihalo_outflow_candidate_data={}
                
                    #  1. OUTPUTS FROM FOF Data: InFOF, InHost
                    for isnap,snap in enumerate(snaps):
                        ihalo_outflow_candidate_data[f'snap{isnap+1}_Particle_InFOF']=np.in1d(ihalo_outflow_candidate_IDs,ihalo_fof_particles[str(snap)]['ParticleIDs_InFOF'])
                        ihalo_outflow_candidate_data[f'snap{isnap+1}_Particle_InHost']=np.in1d(ihalo_outflow_candidate_IDs,ihalo_fof_particles[str(snap)]['ParticleIDs_InHost'])

                    # 2. OUTPUTS FROM SIM AND HISTORIES: Types, Coordinates, Velocity, Processed, Structure
                    # Grab particle indices from histories and sim
                    for isnap,snap in enumerate(snaps):
                        ihalo_isnap_outflow_candidate_parttypes,ihalo_isnap_outflow_candidate_historyindices,ihalo_isnap_outflow_candidate_partindices=get_particle_indices(base_halo_data,
                                                                                                    IDs_sorted=Part_Histories_data[str(snap)]['ParticleIDs'],
                                                                                                    indices_sorted=Part_Histories_data[str(snap)]['ParticleIndex'],
                                                                                                    IDs_taken=ihalo_outflow_candidate_IDs,
                                                                                                    types_taken=ihalo_outflow_candidate_Types,
                                                                                                    snap_taken=snap1,
                                                                                                    snap_desired=snap,
                                                                                                    return_partindices=True)
                        #Dump particle type
                        ihalo_outflow_candidate_data[f'snap{isnap+1}_ParticleTypes']=ihalo_isnap_outflow_candidate_parttypes

                        #Grab particle data
                        for field in Part_Data_fields[str(snap)]:
                            ex_point=Part_Data_Full[str(snap)][field][str(0)][0]
                            size=np.size(ex_point)
                            ihalo_outflow_candidate_data[f'snap{isnap+1}_{field}']=np.array(np.zeros((ihalo_outflow_candidate_count,size))+np.nan)
                            iipart=0
                            for ipart_type,ipart_partidx in zip(ihalo_isnap_outflow_candidate_parttypes,ihalo_isnap_outflow_candidate_partindices):
                                try:
                                    ihalo_outflow_candidate_data[f'snap{isnap+1}_{field}'][iipart]=Part_Data_Full[str(snap)][field][str(ipart_type)][ipart_partidx]
                                except:
                                    pass
                                iipart=iipart+1
                        #Derive other simulation outputs
                        if snap==snap1 or snap==snap2:
                            if vmax_cut:
                                ihalo_outflow_candidate_data[f'snap{isnap+1}_r_com']=ihalo_outflow_candidate_data[f'snap{isnap+1}_Coordinates']-ihalo_com_physical[str(snap)]
                                ihalo_outflow_candidate_data[f'snap{isnap+1}_rabs_com']=np.sqrt(np.sum(np.square(ihalo_outflow_candidate_data[f'snap{isnap+1}_r_com']),axis=1))
                                ihalo_outflow_candidate_data[f'snap{isnap+1}_runit_com']=np.divide(ihalo_outflow_candidate_data[f'snap{isnap+1}_r_com'],np.column_stack([ihalo_outflow_candidate_data[f'snap{isnap+1}_rabs_com']]*3))
                                ihalo_outflow_candidate_data[f'snap{isnap+1}_v_com']=ihalo_outflow_candidate_data[f'snap{isnap+1}_Velocity']-ihalo_vcom_physical[str(snap)]
                                ihalo_outflow_candidate_data[f'snap{isnap+1}_vrad_com']=np.sum(ihalo_outflow_candidate_data[f'snap{isnap+1}_runit_com']*ihalo_outflow_candidate_data[f'snap{isnap+1}_v_com'],axis=1)
                                
                                if not write_partdata:
                                    del ihalo_outflow_candidate_data[f'snap{isnap+1}_r_com']
                                    del ihalo_outflow_candidate_data[f'snap{isnap+1}_runit_com']
                                    del ihalo_outflow_candidate_data[f'snap{isnap+1}_v_com']
                                                    
                    ############################## SAVE DATA FOR OUTFLOW CANDIDATES ##############################
                    #############################################################################################

                    # Iterate through particle types
                    for itype in PartTypes:
                        # Mask for particle types - note these are taken at snap 2 (after "leaving" halo)
                        itype_key=f'PartType{itype}'
                        ihalo_itype_mask=np.where(ihalo_outflow_candidate_data["snap2_ParticleTypes"]==itype)

                        ### PARTICLE OUTPUTS ###
                        ########################
                        if write_partdata:
                            ihalo_hdf5['Outflow'][itype_key].create_dataset('ParticleIDs',data=ihalo_outflow_candidate_IDs[ihalo_itype_mask],dtype=output_fields_dtype["ParticleIDs"],compression=compression)
                            ihalo_hdf5['Outflow'][itype_key].create_dataset('Mass',data=ihalo_outflow_candidate_data['snap2_Mass'][ihalo_itype_mask],dtype=output_fields_dtype["Mass"],compression=compression)

                            #Rest of fields: snap 1
                            if vmax_cut:
                                ihalo_snap1_outflow_outputs=["Particle_InFOF","Particle_InHost","r_com","rabs_com","vrad_com"]
                            else:
                                ihalo_snap1_outflow_outputs=["Particle_InFOF","Particle_InHost"]

                            for ihalo_snap1_outflow_output in ihalo_snap1_outflow_outputs:
                                ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap1_{ihalo_snap1_outflow_output}',data=ihalo_outflow_candidate_data[f'snap1_{ihalo_snap1_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap1_inflow_output],compression=compression)
                            
                            #Rest of fields: snap 2
                            if vmax_cut:
                                ihalo_snap2_outflow_outputs=["Particle_InFOF","Particle_InHost","r_com","rabs_com","vrad_com"]
                            else:
                                ihalo_snap2_outflow_outputs=["Particle_InFOF","Particle_InHost"]

                            for ihalo_snap2_outflow_output in ihalo_snap2_outflow_outputs:
                                ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap2_{ihalo_snap2_outflow_output}',data=ihalo_outflow_candidate_data[f'snap2_{ihalo_snap2_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap2_inflow_output],compression=compression)
                            
                            #Rest of fields: snap 3
                            ihalo_snap3_outflow_outputs=["Particle_InFOF","Particle_InHost"]
                            for ihalo_snap3_outflow_output in ihalo_snap3_outflow_outputs:
                                ihalo_hdf5['Outflow'][itype_key].create_dataset(f'snap3_{ihalo_snap3_outflow_output}',data=ihalo_outflow_candidate_data[f'snap3_{ihalo_snap3_outflow_output}'][ihalo_itype_mask],dtype=output_fields_dtype[ihalo_snap3_inflow_output],compression=compression)
                        
                        ### INTEGRATED OUTPUTS ###
                        ########################## 
                        
                        ## GRAB MASSES
                        ihalo_itype_outflow_masses=ihalo_outflow_candidate_data['snap2_Mass'][ihalo_itype_mask]

                        ## DEFINE MASKS
                        # Masks for halo outflow definitions
                        halo_itype_outflow_definition={'FOF-haloscale':np.logical_and(ihalo_outflow_candidate_data["snap1_Particle_InFOF"][ihalo_itype_mask],np.logical_not(ihalo_outflow_candidate_data["snap2_Particle_InFOF"][ihalo_itype_mask])),
                                                    'FOF-subhaloscale':np.logical_and(ihalo_outflow_candidate_data["snap1_Particle_InHost"][ihalo_itype_mask],np.logical_not(ihalo_outflow_candidate_data["snap2_Particle_InHost"][ihalo_itype_mask]))}
                        
                        # Masks for cuts on outflow velocity as per vmax_facs
                        if vmax_cut:
                            ihalo_itype_outflow_vmax_masks={'vmax_fac'+str(ivmax_fac+1):ihalo_outflow_candidate_data[f'snap1_vrad_com'][ihalo_itype_mask]>vmax_fac*ihalo_metadata['ave_vmax']  for ivmax_fac,vmax_fac in enumerate(vmax_facs["Outflow"])}
                        else:
                            ihalo_itype_outflow_vmax_masks={'vmax_fac_1':np.ones(len(ihalo_itype_outflow_masses))}

                        # Masks for processing history of particles
                        ihalo_itype_outflow_processed_masks={'Total': np.ones(len(ihalo_itype_outflow_masses))}
                        # Masks for the destination of outflow particles
                        ihalo_itype_outflow_destination_masks={'Gross':np.ones(len(ihalo_itype_outflow_masses))}
                        # Masks for stability
                        ihalo_itype_outflow_stability={}
                        ihalo_itype_outflow_stability={'FOF-haloscale':np.logical_not(ihalo_outflow_candidate_data["snap3_Particle_InFOF"][ihalo_itype_mask]),
                                                        'FOF-subhaloscale':np.logical_not(ihalo_outflow_candidate_data["snap3_Particle_InHost"][ihalo_itype_mask])}
                    
                        ## ITERATE THROUGH THE ABOVE MASKS
                        # For each halo definition
                        for halo_defname in halo_defnames["Outflow"]:
                            # If to record data for this halo and halo definition
                            if ihalo_scale_record[halo_defname]:
                                idef_mask=halo_itype_outflow_definition[halo_defname]
                                stability_mask=ihalo_itype_outflow_stability[halo_defname]
                                icalc_processedgroups=output_processedgroups['Basic']
                                icalc_enddatasets=output_enddatasets['Basic']
                                # For each vmax fac
                                for ivmax_fac, vmax_fac in enumerate(vmax_facs["Outflow"]):
                                    ivmax_key=f'vmax_fac{ivmax_fac+1}'
                                    ivmax_mask=ihalo_itype_outflow_vmax_masks[ivmax_key]
                                    # For each processed group
                                    for processedgroup in icalc_processedgroups:
                                        iprocessed_mask=ihalo_itype_outflow_processed_masks[processedgroup]
                                        #For each dataset
                                        for dataset in icalc_enddatasets:
                                            idset_key=dataset
                                            # Masks to concatenate
                                            destination_mask=ihalo_itype_outflow_destination_masks[dataset]
                                            masks=[idef_mask,ivmax_mask,iprocessed_mask,origin_mask]
                                            running_mask=np.logical_and.reduce([idef_mask,ivmax_mask,iprocessed_mask,destination_mask])
                                            stable_running_mask=np.logical_and(running_mask,stability_mask)
                                            all_dset_where=np.where(running_mask)
                                            stable_dset_where=np.where(stable_running_mask)
                                            # Dump data to file
                                            integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaM'][iihalo]=np.float32(np.nansum(ihalo_itype_outflow_masses[all_dset_where]))
                                            integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_{idset_key}_DeltaN'][iihalo]=np.float32(np.nansum(running_mask))
                                            integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaM'][iihalo]=np.float32(np.nansum(ihalo_itype_outflow_masses[stable_dset_where]))
                                            integrated_output_hdf5['Outflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'Stable_{idset_key}_DeltaN'][iihalo]=np.float32(np.nansum(stable_running_mask))
                
                t2_halo=time.time()
                        
                with open(fname_log,"a") as progress_file:
                    progress_file.write(f"Done with ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                    progress_file.write(f"[Took {t2_halo-t1_halo:.2f} sec]\n")
                    progress_file.write(f" \n")
                    progress_file.close()

            else:# Couldn't find the halo progenitor/descendant pair
                print(f'Skipping ihalo {ihalo_s2} - couldnt find progenitor/descendant pair')
                with open(fname_log,"a") as progress_file:
                    progress_file.write(f"Skipping ihalo {ihalo_s2} - no head/tail pair ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                    progress_file.write(f" \n")
                progress_file.close()
        
        # Some other error in the main halo loop
        else: 
            print(f'Skipping ihalo {ihalo_s2} - dont have the reason')
            with open(fname_log,"a") as progress_file:
                progress_file.write(f"Skipping ihalo {ihalo_s2} - unknown reason ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                progress_file.write(f" \n")
            progress_file.close()
            continue

    #Finished with output file
    output_hdf5.close()
    return None

########################### GENERATE ACCRETION DATA: ALL (r200) ###########################

def gen_accretion_data_r200(base_halo_data,snap=None,halo_index_list=None,pre_depth=1,post_depth=1,vmax_facs_in=[-1],vmax_facs_out=[-1],r200_facs_in=[],r200_facs_out=[],write_partdata=False):
    
    t1_init=time.time()
    t1_io=time.time()

    ##### Processing inputs #####
    # Processing the snap inputs
    snap1=snap-pre_depth
    snap2=snap
    snap3=snap+post_depth
    snaps=[snap1,snap2]
    
    # Processing the desired halo index list
    if halo_index_list==None:
        halo_index_list_snap2=list(range(len(base_halo_data[snap]["hostHaloID"])))#use all halos if not handed halo index list
        iprocess="x"
        num_processes=1
        test=True
    else:
        try:
            halo_index_list_snap2=halo_index_list["indices"] #extract index list from input dictionary
            iprocess=str(halo_index_list["iprocess"]).zfill(2) #the process for this index list (this is just used for the output file name)
            print(f'iprocess {iprocess} has {len(halo_index_list_snap2)} halo indices: {halo_index_list_snap2}')
            num_processes=halo_index_list["np"]
            test=halo_index_list["test"]
        except:
            print('Not parsed a valud halo index list. Exiting.')
            return None
    
    # Find the indices of halos at snap1 and snap3 (ordered by snap2 halo indices)
    halo_index_list_snap1=[find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=pre_depth) for ihalo in halo_index_list_snap2]
    halo_index_list_snap3=[find_descen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=post_depth) for ihalo in halo_index_list_snap2]

    # Check whether we are doing the vmax cuts based on the infow  list
    if vmax_facs_in==[]:
        vmax_cut=False
    else:
        vmax_cut=True

    # Determine whether we need to perform outflow calculations based on whether we are given outflow vmax cuts
    output_groups=['Inflow']

    # Add vmax factor of -1 to whatever the user input was
    vmax_facs_in=np.concatenate([[-1],vmax_facs_in])

    if vmax_cut:
        vmax_facs={'Inflow':vmax_facs_in} 
    else:
        vmax_facs={'Inflow':[0]}

    # Define halo calculation types
    halo_defnames={}
    halo_defnames["Inflow"]=[f'r200_fac{ifac+1}' for ifac in range(len(r200_facs_in))]
    
    # Default options 
    use='cminpot' #which halo centre definition to use (from 'cminpot', 'com')
    compression='gzip'

    # Create log file and directories, initialising outputs
    if True:
        #Logs
        acc_log_dir=f"job_logs/acc_logs/"
        if not os.path.exists(acc_log_dir):
            try:
                os.mkdir(acc_log_dir)
            except:
                pass
        if test:
            run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_R200only_test/"
        else:
            run_log_dir=f"job_logs/acc_logs/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_R200only/"

        if not os.path.exists(run_log_dir):
            try:
                os.mkdir(run_log_dir)
            except:
                pass

        run_snap_log_dir=run_log_dir+f'snap_{str(snap).zfill(3)}/'

        if not os.path.exists(run_snap_log_dir):
            try:
                os.mkdir(run_snap_log_dir)
            except:
                pass
        if test:
            fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}_test.log"
            print(f'iprocess {iprocess} will save progress to log file: {fname_log}')

        else:
            fname_log=run_snap_log_dir+f"progress_p{str(iprocess).zfill(3)}_n{str(len(halo_index_list_snap2)).zfill(6)}.log"

        if os.path.exists(fname_log):
            os.remove(fname_log)
        
        with open(fname_log,"a") as progress_file:
            progress_file.write('Initialising and loading in data ...\n')
        progress_file.close()
    
        # Initialising outputs
        if not os.path.exists('acc_data'):#create folder for outputs if doesn't already exist
            os.mkdir('acc_data')
        if test:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_R200only_test/'
        else:
            calc_dir=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_np{str(num_processes).zfill(2)}_R200only/'

        if not os.path.exists(calc_dir):#create folder for outputs if doesn't already exist
            try:
                os.mkdir(calc_dir)
            except:
                pass
        calc_snap_dir=calc_dir+f'snap_{str(snap2).zfill(3)}/'
        
        if not os.path.exists(calc_snap_dir):#create folder for outputs if doesn't already exist
            try:
                os.mkdir(calc_snap_dir)
            except:
                pass

        # Assigning snap
        if snap==None:
            snap=len(base_halo_data)-1#if not given snap, just use the last one

    # Create output file with metadata attributes
    run_outname=base_halo_data[snap]['outname']#extract output name (simulation name)
    outfile_name=calc_snap_dir+'AccretionData_pre'+str(pre_depth).zfill(2)+'_post'+str(post_depth).zfill(2)+'_snap'+str(snap).zfill(3)+'_p'+str(iprocess).zfill(3)+'.hdf5'
    
    # Remove existing output file if exists
    if not os.path.exists(outfile_name):
        print(f'Initialising output file at {outfile_name}...')
        output_hdf5=h5py.File(outfile_name,"w")
    else:
        print(f'Removing old file and initialising output file at {outfile_name}...')
        os.remove(outfile_name)
        output_hdf5=h5py.File(outfile_name,"w")

    # Make header for accretion data based on base halo data 
    if True:
        header_hdf5=output_hdf5.create_group("Header")
        lt_ave=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']+base_halo_data[snap2]['SimulationInfo']['LookbackTime'])/2
        z_ave=(base_halo_data[snap1]['SimulationInfo']['z']+base_halo_data[snap2]['SimulationInfo']['z'])/2
        dt=(base_halo_data[snap1]['SimulationInfo']['LookbackTime']-base_halo_data[snap2]['SimulationInfo']['LookbackTime'])
        t1=base_halo_data[snap1]['SimulationInfo']['LookbackTime']
        t2=base_halo_data[snap2]['SimulationInfo']['LookbackTime']
        t3=base_halo_data[snap3]['SimulationInfo']['LookbackTime']
        z1=base_halo_data[snap1]['SimulationInfo']['z']
        z2=base_halo_data[snap2]['SimulationInfo']['z']
        z3=base_halo_data[snap3]['SimulationInfo']['z']
        header_hdf5.attrs.create('ave_LookbackTime',data=lt_ave,dtype=np.float16)
        header_hdf5.attrs.create('ave_z',data=z_ave,dtype=np.float16)
        header_hdf5.attrs.create('delta_LookbackTime',data=dt,dtype=np.float16)
        header_hdf5.attrs.create('snap1_LookbackTime',data=t1,dtype=np.float16)
        header_hdf5.attrs.create('snap2_LookbackTime',data=t2,dtype=np.float16)
        header_hdf5.attrs.create('snap3_LookbackTime',data=t3,dtype=np.float16)
        header_hdf5.attrs.create('snap1_z',data=z1,dtype=np.float16)
        header_hdf5.attrs.create('snap2_z',data=z2,dtype=np.float16)
        header_hdf5.attrs.create('snap3_z',data=z3,dtype=np.float16)
        header_hdf5.attrs.create('snap1',data=snap1,dtype=np.int16)
        header_hdf5.attrs.create('snap2',data=snap2,dtype=np.int16)
        header_hdf5.attrs.create('snap3',data=snap3,dtype=np.int16)
        header_hdf5.attrs.create('pre_depth',data=snap2-snap1,dtype=np.int16)
        header_hdf5.attrs.create('post_depth',data=snap3-snap2,dtype=np.int16)
        header_hdf5.attrs.create('outname',data=np.string_(base_halo_data[snap2]['outname']))
        header_hdf5.attrs.create('total_num_halos',data=base_halo_data[snap2]['Count'])
    
    # Standard particle type names from simulation
    SimType=base_halo_data[snap2]['Part_FileType']
    

    ##### Loading in Data #####

    ####Create master data structure if needed - particle coordinates/types/masses indexed identically

    # Particle data - LOAD HERE
    print('Retrieving & organising raw particle data ...')
    PartNames=['Gas','DM','','','Star','BH']
    hval=base_halo_data[snap1]['SimulationInfo']['h_val'];scalefactors={}
    scalefactors={str(snap):base_halo_data[snap]['SimulationInfo']['ScaleFactor'] for snap in snaps}
    Mass_DM=base_halo_data[snap2]['SimulationInfo']['Mass_DM_Physical']
    Mass_Gas=base_halo_data[snap2]['SimulationInfo']['Mass_Gas_Physical']
    BoxSize=base_halo_data[snap2]['SimulationInfo']['BoxSize_Comoving']

    #Conversion factors for particle data
    Part_Data_comtophys={str(snap):{'Coordinates':scalefactors[str(snap)]/hval, 
                                    'Velocity':scalefactors[str(snap)]/hval,
                                    'Mass':10.0**10/hval,
                                    'ParticleIDs':1} for snap in snaps}
    
    Part_Data_FilePaths={str(snap):base_halo_data[snap]['Part_FilePath'] for snap in snaps}

    #If EAGLE, read masses
    if SimType=='EAGLE':
        #Which fields do we need at each snap
        PartTypes_keys_all=list(h5py.File(base_halo_data[-1]['Part_FilePath'],'r').keys())
        PartTypes_keys=[key for key in PartTypes_keys_all if 'PartType'  in key]
        PartTypes=[int(key.split('PartType')[-1]) for key in PartTypes_keys]
        print(f'Using parttypes: {PartTypes}')

        Part_Data_fields={str(snap1):['ParticleIDs','Mass','Coordinates'],str(snap2):['Coordinates']}
        if vmax_cut:
            for snap in [snap1,snap2]:
                Part_Data_fields[str(snap)].append('Velocity')

    #If not EAGLE, assume constant masses
    else:
        #Which fields do we need at each snap
        PartTypes=[0,1] #Gas, DM, Stars, BH
        Part_Data_fields= {str(snap1):['ParticleIDs','Coordinates','Velocities'],str(snap2):['ParticleIDs','Coordinates','Velocities']}
    

    # Check for existence of master array
    Part_Master_fname=f'acc_data/pre{str(pre_depth).zfill(2)}_post{str(post_depth).zfill(2)}_snap{str(snap2).zfill(3)}_masterarray.dat'

    if os.path.exists(Part_Master_fname):
        print(f'Found existing master particle array at {Part_Master_fname} - loading ...')
        Part_Master_Array=open_pickle(Part_Master_fname)

    else:
        print(f'No existing master particle array at {Part_Master_fname} - proceeding ...')
        Part_Data_Full={str(snap):{field:{} for field in Part_Data_fields[str(snap)]} for snap in snaps}
        for snap in snaps:
            if SimType=='EAGLE':
                EAGLE_snap=read_eagle.EagleSnapshot(Part_Data_FilePaths[str(snap)])
                EAGLE_snap.select_region(xmin=0,xmax=BoxSize,
                                        ymin=0,ymax=BoxSize,
                                        zmin=0,zmax=BoxSize)
                for field in Part_Data_fields[str(snap)]:
                    if not field=='Mass':
                        for itype in PartTypes:
                            try:
                                Part_Data_Full[str(snap)][field][str(itype)]=EAGLE_snap.read_dataset(itype,field)*Part_Data_comtophys[str(snap)][field]
                            except:
                                Part_Data_Full[str(snap)][field][str(itype)]=np.array([])
                    else:
                        for itype in [0,4,5]:
                            try:
                                Part_Data_Full[str(snap)][field][str(itype)]=EAGLE_snap.read_dataset(itype,field)*Part_Data_comtophys[str(snap)][field]
                            except:
                                Part_Data_Full[str(snap)][field][str(itype)]=np.array([])
                        
                        Part_Data_Full[str(snap)][field][str(1)]=Mass_DM*np.ones(len(Part_Data_Full[str(snap)]['Mass'][str(0)]))
            else:
                Mass_Constant={str(0):True,str(1):True}
                Part_Data_file=h5py.File(Part_Data_FilePaths[str(snap)],'r')
                for field in Part_Data_fields[str(snap)]:
                    if field=='Velocities':
                        Part_Data_Full[str(snap)]['Velocity']={str(itype):Part_Data_file[f'PartType{itype}']['Velocities'].value*Part_Data_comtophys[str(snap)]['Velocity'] for itype in PartTypes}
                    else:
                        Part_Data_Full[str(snap)][field]={str(itype):Part_Data_file[f'PartType{itype}'][field].value*Part_Data_comtophys[str(snap)][field] for itype in PartTypes}
                
                Part_Data_Full[str(snap)]['Mass'][str(0)]=h5py.File(base_halo_data[snap]['Part_FilePath'],'r')['Header'].attrs['MassTable'][0]*Part_Data_comtophys[str(snap)]['Mass']
                Part_Data_Full[str(snap)]['Mass'][str(1)]=h5py.File(base_halo_data[snap]['Part_FilePath'],'r')['Header'].attrs['MassTable'][1]*Part_Data_comtophys[str(snap)]['Mass']

        if not SimType=='EAGLE':
            Part_Data_fields={str(snap1):['Coordinates','Velocity','Mass'],str(snap2):['Coordinates','Velocity','Mass'],str(snap3):[]}

        print('Done retrieving & organising raw particle data ...')


        # Particle histories - LOAD HERE
        print(f'Retrieving & organising particle histories ...')
        # Part_Histories_fields={str(snap1):["ParticleIDs",'ParticleIndex','HostStructure','Processed_L1'],str(snap2):["ParticleIDs",'ParticleIndex'],str(snap3):["ParticleIDs",'ParticleIndex']}
        Part_Histories_fields={str(snap1):["ParticleIDs",'ParticleIndex'],str(snap2):["ParticleIDs",'ParticleIndex']}
        Part_Histories_data={str(snap):{} for snap in snaps}
        Part_Histories_Constant={str(0):False,str(1):False,str(4):False,str(5):False}
        for snap in snaps:
            Part_Histories_File_snap=h5py.File("part_histories/PartHistory_"+str(snap).zfill(3)+"_"+run_outname+".hdf5",'r')
            for field in Part_Histories_fields[str(snap)]:
                Part_Histories_data[str(snap)][field]={str(itype):Part_Histories_File_snap["PartType"+str(itype)+'/'+field].value for itype in PartTypes}
        print(f'Done retrieving & organising particle histories')
        t2_io=time.time()

        print(f'I/O took {t2_io-t1_io:.1f} sec')

        # Master particle array
        print('Starting with organising master particle array ...')

        t1_array=time.time()
        print('Counting particles')
        npart_tot=0
        for itype in PartTypes:
            npart_tot=npart_tot+len(Part_Data_Full[str(snap1)]['ParticleIDs'][str(itype)])
        print(f'Total number of particles: {npart_tot}')

        snap1_IDs=np.zeros((npart_tot,),dtype=np.int64)
        snap1_Types=np.zeros((npart_tot,),dtype=np.int8)
        snap1_Mass=np.zeros((npart_tot,),dtype=np.float32)
        snap1_Coordinates=np.zeros((npart_tot,3),dtype=np.float32)
        snap2_Coordinates=np.zeros((npart_tot,3),dtype=np.float32)+np.nan
        
        # Grabbing snap 1 IDs and Types
        print('Recording snap1 IDs, Types & Coordinates')
        npart_running=0
        for itype in PartTypes:
            print(f'At itype {itype} ...')
            npart_itype=len(Part_Data_Full[str(snap1)]['ParticleIDs'][str(itype)])
            snap1_IDs[npart_running:npart_running+npart_itype]=Part_Data_Full[str(snap1)]['ParticleIDs'][str(itype)]
            snap1_Types[npart_running:npart_running+npart_itype]=np.ones(npart_itype,dtype=np.int8)*itype
            if not itype==1:
                snap1_Mass[npart_running:npart_running+npart_itype]=Part_Data_Full[str(snap1)]['Mass'][str(itype)]
            else:
                snap1_Mass[npart_running:npart_running+npart_itype]=np.ones((npart_itype,),dtype=np.float32)*Mass_DM
            snap1_Coordinates[npart_running:npart_running+npart_itype]=Part_Data_Full[str(snap1)]['Coordinates'][str(itype)]
            npart_running+=npart_itype
        
        del Part_Data_Full[str(snap1)]
        
        # Grabbing snap 2 Types and Coordinates
        print('Finding snap2 Types ...')
        snap2_Types,snap2_histidx,snap2_idx=get_particle_indices(base_halo_data=base_halo_data,IDs_taken=snap1_IDs,IDs_sorted=Part_Histories_data[str(snap2)]['ParticleIDs'],indices_sorted=Part_Histories_data[str(snap2)]['ParticleIndex'],types_taken=snap1_Types,snap_taken=snap1,snap_desired=snap2,return_partindices=True,verbose=True)
        print('Done indexing, now extracting snap 2 Coordinates ...')
        for iipart,(ipart_ID,ipart_Type,ipart_partidx) in enumerate(zip(snap1_IDs,snap2_Types,snap2_idx)):
            if iipart%10000==0:
                print(f'{iipart/npart_tot*100:.1f}% done extracting Coordinates')
            if ipart_Type>=0:
                snap2_Coordinates[iipart]=Part_Data_Full[str(snap2)]['Coordinates'][str(ipart_Type)][ipart_partidx]

        del Part_Data_Full
        del Part_Histories_data

        #### Create KDTrees for Coordinates 
        print('Generating KDtrees for coordinates')
        t1_tree=time.time()
        snap1_tree=spatial.cKDTree(snap1_Coordinates)
        snap2_tree=spatial.cKDTree(snap2_Coordinates)
        t2_tree=time.time()
        print(f'Finished KDtrees for coordinates in {t2_tree-t1_tree:.1f} sec')

        # Collate into final master data structure
        Part_Master_Array={'ParticleIDs':snap1_IDs,'snap1_Types':snap1_Types,'snap2_Types':snap2_Types,'Mass':snap1_Mass,'snap1_KDtree':snap1_tree,'snap2_KDtree':snap2_tree}
        try:
            dump_pickle(path=Part_Master_fname,data=Part_Master_Array)
        except:
            try:
                os.remove(Part_Master_fname)
            except:
                pass

        t2_array=time.time()
        print(f'Generated master particle array in {t2-t1:.1f} sec')

    t2_init=time.time()
    print()
    print('*********************************************************')
    print(f'Done initialising in {(t2_init-t1_init):.2f} sec - entering main halo loop ...')
    print('*********************************************************')

    with open(fname_log,"a") as progress_file:
        progress_file.write(f'Done initialising in {(t2_init-t1_init):.2f} sec - entering main halo loop ...\n')
    progress_file.close()

    ##### Initialising outputs #####
    # Particle
    if write_partdata:
        #hdf5 group
        particle_output_hdf5=output_hdf5.create_group('Particle')
        
        #output dtypes
        output_fields_dtype={}
        output_fields_float32=["Mass","r_com","rabs_com","vrad_com","vtan_com"]
        for field in output_fields_float32:
            output_fields_dtype[field]=np.float32

        output_fields_int64=["ParticleIDs"]
        for field in output_fields_int64:
            output_fields_dtype[field]=np.int64
        
        # output_fields_int8=["Processed","Particle_InFOF","Particle_Bound","Particle_InHost"]
        # for field in output_fields_int8:
        #     output_fields_dtype[field]=np.int8 

    # Integrated (always written)
    num_halos_thisprocess=len(halo_index_list_snap2)
    integrated_output_hdf5=output_hdf5.create_group('Integrated')
    integrated_output_hdf5.create_dataset('ihalo_list',data=halo_index_list_snap2)

    #Defining which outputs for varying levels of detail
    output_processedgroups={'Basic':['Total']}
    output_enddatasets={'Basic':['Gross']}
    
    #Initialise output datasets with np.nans (len: total_num_halos)
    for output_group in output_groups:
        integrated_output_hdf5.create_group(output_group)
        for itype in PartTypes:
            itype_key=f'PartType{itype}'
            integrated_output_hdf5[output_group].create_group(itype_key)
            for ihalo_defname,halo_defname in enumerate(sorted(halo_defnames[output_group])):
                #Create group 
                integrated_output_hdf5[output_group][itype_key].create_group(halo_defname)
                
                #Create attribute for r200 factor
                r200_fac=r200_facs_in[ihalo_defname]
                integrated_output_hdf5[output_group][itype_key][halo_defname].attrs.create('r200_fac',data=r200_fac)

                #Use detailed datasets for inflow variants
                if output_group=='Inflow': 
                    icalc_processedgroups=output_processedgroups['Basic']
                    icalc_enddatasets=output_enddatasets['Basic']
                else:
                    icalc_processedgroups=output_processedgroups['Basic']
                    icalc_enddatasets=output_enddatasets['Basic']

                #Now, for each Vmax cut
                for ivmax_fac, vmax_fac in enumerate(vmax_facs[output_group]):
                    #Create group
                    ivmax_key=f'vmax_fac{ivmax_fac+1}'
                    integrated_output_hdf5[output_group][itype_key][halo_defname].create_group(ivmax_key)
                    #Add attribute for vmax_fac
                    integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key].attrs.create('vmax_fac',data=vmax_fac)
                    #Initialise datasets with nans
                    for processedgroup in icalc_processedgroups:
                        integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key].create_group(processedgroup)
                        for dataset in icalc_enddatasets:
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'All_'+dataset+f'_DeltaM',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)
                            integrated_output_hdf5[output_group][itype_key][halo_defname][ivmax_key][processedgroup].create_dataset(f'All_'+dataset+f'_DeltaN',data=np.zeros(num_halos_thisprocess)+np.nan,dtype=np.float32)


    for iihalo,ihalo_s2 in enumerate(halo_index_list_snap2):# for each halo (index at snap 2)
        
        # If needed, create group for this halo in output file
        if write_partdata:
            ihalo_hdf5=particle_output_hdf5.create_group('ihalo_'+str(ihalo_s2).zfill(6))
            ihalo_hdf5.create_group('Metadata')
            if write_partdata:
                ihalo_hdf5.create_group('Inflow')
                for itype in PartTypes:
                    ihalo_hdf5['Inflow'].create_group(f'PartType{itype}')
        

        # This catches any exceptions for a given halo and prevents the code from crashing 
        if True:     
            # try:
            ########################################################################################################################################
            ###################################################### ihalo PRE-PROCESSING ############################################################
            ########################################################################################################################################
            t1_halo=time.time()
            
            # Find halo progenitor and descendants
            ihalo_indices={str(snap1):halo_index_list_snap1[iihalo],str(snap2):ihalo_s2,str(snap3):halo_index_list_snap3[iihalo]}
            
            # Record halo properties 
            ihalo_tracked=(ihalo_indices[str(snap1)]>-1 and ihalo_indices[str(snap3)]>-1)#track if have both progenitor and descendant
            ihalo_structuretype=base_halo_data[snap2]["Structuretype"][ihalo_indices[str(snap2)]]#structure type
            ihalo_numsubstruct=base_halo_data[snap2]["numSubStruct"][ihalo_indices[str(snap2)]]
            ihalo_hostHaloID=base_halo_data[snap2]["hostHaloID"][ihalo_indices[str(snap2)]]
            ihalo_sublevel=int(np.floor((ihalo_structuretype-0.01)/10))

            # Print progress to terminal and output file
            print();print('**********************************************')
            print('Halo index: ',ihalo_s2,f" - {ihalo_numsubstruct} substructures, mass {base_halo_data[snap2]['Mass_200crit'][ihalo_s2]:.2e}")
            print(f'Progenitor: {ihalo_indices[str(snap1)]} | Descendant: {ihalo_indices[str(snap3)]}')
            print('**********************************************');print()
            with open(fname_log,"a") as progress_file:
                progress_file.write(f' \n')
                progress_file.write(f'Starting with ihalo {ihalo_s2} ... \n')
            progress_file.close()
            
            # This catches any halos for which we can't find a progenitor/descendant 
            if ihalo_tracked and base_halo_data[snap2]['Mass_200crit'][ihalo_s2]>10**9.0/10**10:
                ### GRAB HALO METADATA ###
                ihalo_metadata={}
                for isnap,snap in enumerate(snaps):
                    ihalo_isnap=ihalo_indices[str(snap)]
                    if ihalo_isnap>=0:
                        ihalo_metadata[f'snap{isnap+1}_com']=np.array([base_halo_data[snap]['Xc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Yc'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zc'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_cminpot']=np.array([base_halo_data[snap]['Xcminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Ycminpot'][ihalo_indices[str(snap)]],base_halo_data[snap]['Zcminpot'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_vcom']=np.array([base_halo_data[snap]['VXc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VYc'][ihalo_indices[str(snap)]],base_halo_data[snap]['VZc'][ihalo_indices[str(snap)]]],ndmin=2)
                        ihalo_metadata[f'snap{isnap+1}_R_200crit']=base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_R_200mean']=base_halo_data[snap]['R_200mean'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_Mass_200crit']=base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*10**10
                        ihalo_metadata[f'snap{isnap+1}_vmax']=base_halo_data[snap]['Vmax'][ihalo_indices[str(snap)]]
                        ihalo_metadata[f'snap{isnap+1}_vesc_crit']=np.sqrt(2*base_halo_data[snap]['Mass_200crit'][ihalo_indices[str(snap)]]*base_halo_data[snap]['SimulationInfo']['Gravity']/base_halo_data[snap]['R_200crit'][ihalo_indices[str(snap)]])
                
                # Average some quantities
                ihalo_metadata['sublevel']=ihalo_sublevel
                ihalo_metadata['ave_R_200crit']=0.5*base_halo_data[snap1]['R_200crit'][ihalo_indices[str(snap1)]]+0.5*base_halo_data[snap2]['R_200crit'][ihalo_indices[str(snap2)]]
                ihalo_metadata['ave_vmax']=0.5*base_halo_data[snap1]['Vmax'][ihalo_indices[str(snap1)]]+0.5*base_halo_data[snap2]['Vmax'][ihalo_indices[str(snap2)]]
                
                ########################################################################################################################################
                ############################################################ ihalo INFLOW ##############################################################
                ########################################################################################################################################
                
                ihalo_r200_inflow_type={}
                ihalo_r200_inflow_mass={}
                
                for ir200_fac,r200_fac in enumerate(r200_facs_in):
                    ihalo_snap1_within_r200=Part_Master_Array['snap1_KDtree'].query_ball_point(x=ihalo_metadata[f'snap1_cminpot'],r=r200_fac*ihalo_metadata['ave_R_200crit'])[0]
                    ihalo_snap2_within_r200=Part_Master_Array['snap2_KDtree'].query_ball_point(x=ihalo_metadata[f'snap2_cminpot'],r=r200_fac*ihalo_metadata['ave_R_200crit'])[0]
                    
                    ihalo_r200_inflow_idxs=(np.compress(np.in1d(ihalo_snap2_within_r200,ihalo_snap1_within_r200,invert=True,assume_unique=True),ihalo_snap2_within_r200),)
                    ihalo_r200_inflow_type[f'r200_fac{ir200_fac+1}']=Part_Master_Array['snap1_Types'][ihalo_r200_inflow_idxs]
                    ihalo_r200_inflow_mass[f'r200_fac{ir200_fac+1}']=Part_Master_Array['Mass'][ihalo_r200_inflow_idxs]
        
                # ITERATE THROUGH THE ABOVE MASKS
                # For each halo definition
                for halo_defname in halo_defnames["Inflow"]:
                    ihalo_ir200_masses=ihalo_r200_inflow_mass[halo_defname]
                    ihalo_ir200_types=ihalo_r200_inflow_type[halo_defname]

                    for itype in PartTypes:
                        itype_key=f'PartType{itype}'
                        itype_mask=np.where(ihalo_ir200_types==itype)
                        ivmax_key='vmax_fac1'
                        processedgroup='Total'

                        # Dump data to file
                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_Gross_DeltaM'][iihalo]=np.float32(np.nansum(ihalo_ir200_masses[itype_mask]))            
                        integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_Gross_DeltaN'][iihalo]=np.float32(len(ihalo_ir200_masses[itype_mask]))
                      
                        print(f'All delta M:', integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_Gross_DeltaM'][iihalo])
                        print(f'All delta N:', integrated_output_hdf5['Inflow'][itype_key][halo_defname][ivmax_key][processedgroup][f'All_Gross_DeltaN'][iihalo])


                t2_halo=time.time()
                        
                with open(fname_log,"a") as progress_file:
                    progress_file.write(f"Done with ihalo {ihalo_s2} ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                    progress_file.write(f"[Took {t2_halo-t1_halo:.2f} sec]\n")
                    progress_file.write(f" \n")
                    progress_file.close()

            else:# Couldn't find the halo progenitor/descendant pair
                print(f'Skipping ihalo {ihalo_s2} - couldnt find progenitor/descendant pair or halo has mass < 10^9Msun')
                with open(fname_log,"a") as progress_file:
                    progress_file.write(f"Skipping ihalo {ihalo_s2} - no head/tail pair ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                    progress_file.write(f" \n")
                progress_file.close()
        
        # Some other error in the main halo loop
        else: 
            print(f'Skipping ihalo {ihalo_s2} - dont have the reason')
            with open(fname_log,"a") as progress_file:
                progress_file.write(f"Skipping ihalo {ihalo_s2} - unknown reason ({iihalo+1} out of {num_halos_thisprocess} for this process - {(iihalo+1)/num_halos_thisprocess*100:.2f}% done)\n")
                progress_file.write(f" \n")
            progress_file.close()
            continue

    #Finished with output file
    output_hdf5.close()
    return None

########################### COLLATE DETAILED ACCRETION DATA ###########################

def postprocess_accretion_data_serial(base_halo_data,path=None):
    """

    postprocess_accretion_data_serial : function
	----------

    Collate the integrated accretion data from above into one file.

	Parameters
	----------
    base_halo_data : list of dictionaries
        The minimal halo data list of dictionaries previously generated ("B1" is sufficient)

    path : str
        The directory in which the accretion data exists.   

    """

    if not path.endswith('/'):
        path=path+'/'

    snapname=path.split('/')[-2]
    snap=int(snapname.split('_')[-1])

    # Read in the hdf5 data structure of outputs
    allfnames=os.listdir(path)
    accfnames=[path+fname for fname in allfnames if ('AccretionData' in fname and 'All' not in fname)]
    integrated_datasets_list=np.array(hdf5_struct(accfnames[-1]))
    print(f'Total num datasets: {len(integrated_datasets_list)}')

    # Initialise outputs
    outname=accfnames[-1][:-10]+'_All.hdf5'
    if os.path.exists(outname):
        os.system(f"rm -rf {outname}")
    outfile=h5py.File(outname,'w')

    # Carry over header
    print('Carring over header ...')
    outfile.create_group('Header')
    header_keys=list(h5py.File(accfnames[-1])['Header'].attrs)
    for header_key in header_keys:
        outfile['Header'].attrs.create(header_key,data=h5py.File(accfnames[-1])['Header'].attrs[header_key])

    # Initialise output datasets
    print('Initialising output datasets ...')
    t1_init=time.time()
    total_num_halos=len(base_halo_data[snap]["ID"])
    for integrated_dataset in integrated_datasets_list:
        groups=integrated_dataset.split('/')[1:-1]
        running_group=''
        for igroup,group in enumerate(groups):
            if igroup==0:
                try:
                    outfile.create_group(group)
                except:
                    pass
            else:
                try:
                    outfile[running_group].create_group(group)
                    group_attrs=list(h5py.File(accfnames[-1])[running_group].attrs)
                    for attr in group_attrs:
                        outfile[running_group].attrs.create(attr,data=h5py.File(accfnames[-1])[running_group].attrs[attr])
                except:
                    pass

            running_group=running_group+'/'+group
        outfile[running_group].create_dataset(integrated_dataset,data=np.zeros(total_num_halos)+np.nan,dtype=np.float32)
    t2_init=time.time()
    print(f'Done initialising datasets in {t2_init-t1_init:.2f} sec')

    # Copy over datasets to correct indices
    print('Copying over datasets ...')
    t1_dsets=time.time()
    collated_datasets={dataset:np.zeros(total_num_halos)+np.nan for dataset in integrated_datasets_list}

    for accfname in accfnames:
        accfile=h5py.File(accfname,'r')
        accfile_ihalo_list=accfile['Integrated']['ihalo_list'].value.astype(int)
        for integrated_dataset in integrated_datasets_list:
            accfile_dset_val=accfile[integrated_dataset].value
            collated_datasets[integrated_dataset][(accfile_ihalo_list,)]=accfile_dset_val

    for integrated_dataset in integrated_datasets_list:
        outfile[integrated_dataset][:]=collated_datasets[integrated_dataset]

    t2_dsets=time.time()
    print(f'Done copying over datasets in {t2_dsets-t1_dsets:.2f} sec')

########################### ADD PARTICLE DATA TO ACCRETION DATA ###########################

def add_particle_data_serial(path=None,fileidx=[],fullhalo=False,mcut=10**10):

    """

    add_particle_data_serial : function
	----------

    Add particle data to detailed outputs.

	Parameters
	----------
    base_halo_data : list of dictionaries
        The minimal halo data list of dictionaries previously generated ("B1" is sufficient)

    path : str
        The file to process.   

    """

    eagle_keys={'0':
                ['Coordinates',
                'Metallicity',
                'Temperature',
                'MaximumTemperature',
                'Density',
                'Velocity'],
                '1':
                []}

    size_keys= {'Coordinates':3,
                'Metallicity':1,
                'Temperature':1,
                'MaximumTemperature':1,
                'Density':1,
                'StarFormationRate':1,
                'Velocity':3,
                'ParticleIDs':1}
    

    # Load in metadata
    accfiles_all=list_dir(path)
    accfiles_paths=sorted([accfile_path for accfile_path in accfiles_all if ('All' not in accfile_path and 'recyc' not in accfile_path)])
    accfiles_thisworker=[accfiles_paths[fileiidx] for fileiidx in fileidx]
    accfile_ex=h5py.File(accfiles_thisworker[0],'r+')
    snap1=accfile_ex['Header'].attrs['snap1']
    snap2=accfile_ex['Header'].attrs['snap2']
    parttypes=[0]

    # Load in base halo data
    print('Loading halo data')
    base_folder=path.split('acc_data')[0]
    base_folder_contents=list_dir(base_folder)
    base_halo_data_path=[path for path in base_folder_contents if 'B4' in path][0]
    base_halo_data=open_pickle(base_halo_data_path)
    boxsize=base_halo_data[snap1]['SimulationInfo']['BoxSize_Comoving']

    #patch for testing  (particle data paths)
    # base_halo_data[snap2]['Part_FilePath']=base_folder+'sim_data/snapshots/snapshot_027_z000p101/snap_027_z000p101.0.hdf5'
    # base_halo_data[snap1]['Part_FilePath']=base_folder+'sim_data/snapshots/snapshot_026_z000p183/snap_026_z000p183.0.hdf5'

    # Load in particle histories
    print('Loading particle histories')
    try:
        parthistory_files={str(snap):h5py.File(base_folder+base_halo_data[snap]['PartHist_FilePath'],'r+') for snap in [snap1,snap2]}
        partindices={str(snap):{str(itype):parthistory_files[str(snap)][f'PartType{itype}']['ParticleIndex'].value for itype in parttypes} for snap in [snap1,snap2]}
        partids={str(snap):{str(itype):parthistory_files[str(snap)][f'PartType{itype}']['ParticleIDs'].value for itype in parttypes} for snap in [snap1,snap2]}
    except:
        print('Waiting for files to be available...')
        for itry in range(3600):
            try:
                time.sleep(1)
                parthistory_files={str(snap):h5py.File(base_folder+base_halo_data[snap]['PartHist_FilePath'],'r+') for snap in [snap1,snap2]}
                partindices={str(snap):{str(itype):parthistory_files[str(snap)][f'PartType{itype}']['ParticleIndex'].value for itype in parttypes} for snap in [snap1,snap2]}
                partids={str(snap):{str(itype):parthistory_files[str(snap)][f'PartType{itype}']['ParticleIDs'].value for itype in parttypes} for snap in [snap1,snap2]}
                if True:
                    print(f'Files available after {itry} sec')
                    break
            except:
                continue
    
    for snap in [snap1,snap2]:
        parthistory_files[str(snap)].close()

    # # Load in EAGLE data
    partdata_files={str(snap):read_eagle.EagleSnapshot(base_halo_data[snap]['Part_FilePath']) for snap in [snap1,snap2]}
    for snap in [snap1,snap2]:
        partdata_files[str(snap)].select_region(xmin=-0.1,xmax=boxsize+0.1,ymin=-0.1,ymax=boxsize+0.1,zmin=-0.1,zmax=boxsize+0.1)
        
    print('Loading particle data')
    partdata={str(snap):{str(itype):{key:partdata_files[str(snap)].read_dataset(itype,key) for key in eagle_keys[str(itype)]} for itype in parttypes} for snap in [snap1,snap2]}

    for ifile,accfile_path in enumerate(accfiles_thisworker):
        accfile=h5py.File(accfile_path,'r+')
        ihalo_list=sorted(accfile['Integrated']['ihalo_list'])
        for ihalo in ihalo_list:
            if base_halo_data[snap2]['Mass_FOF'][ihalo]>mcut:
                print(f'Processing ihalo {ihalo}')
                for itype in parttypes:
                    try:
                        ncandidates=len(accfile['Particle'][f'ihalo_{str(ihalo).zfill(6)}']['Inflow'][f'PartType{itype}']["ParticleIDs"])
                    except:
                        continue

                    for key in eagle_keys[str(itype)]:
                        try:
                            del accfile['Particle'][f'ihalo_{str(ihalo).zfill(6)}']['Inflow'][f'PartType{itype}'][f'snap1_{key}']
                            del accfile['Particle'][f'ihalo_{str(ihalo).zfill(6)}']['Inflow'][f'PartType{itype}'][f'snap2_{key}']
                        except:
                            pass
                    
                    snap1_inFOF=accfile['Particle'][f'ihalo_{str(ihalo).zfill(6)}']['Inflow'][f'PartType{itype}']['snap1_Particle_InFOF'].value
                    snap2_inFOF=accfile['Particle'][f'ihalo_{str(ihalo).zfill(6)}']['Inflow'][f'PartType{itype}']['snap2_Particle_InFOF'].value

                    if fullhalo:
                        accreted_mask=np.where(np.ones(ncandidates))
                    else:
                        accreted_mask=np.where(np.logical_and(snap2_inFOF,np.logical_not(snap1_inFOF)))
                    accreted_IDs=accfile['Particle'][f'ihalo_{str(ihalo).zfill(6)}']['Inflow'][f'PartType{itype}']['ParticleIDs'].value[accreted_mask]

                    #find previous types, particle indices
                    print('Fidning particle indices')
                    t1=time.time()
                    snap2_types,snap2_histidx,snap2_partidx=get_particle_indices(base_halo_data,accreted_IDs,IDs_sorted=partids[str(snap2)],indices_sorted=partindices[str(snap2)],types_taken=np.ones(len(accreted_IDs))*itype,snap_taken=snap1,snap_desired=snap2,return_partindices=True,verbose=False)
                    snap1_types,snap1_histidx,snap1_partidx=get_particle_indices(base_halo_data,accreted_IDs,IDs_sorted=partids[str(snap1)],indices_sorted=partindices[str(snap1)],types_taken=np.ones(len(accreted_IDs))*itype,snap_taken=snap1,snap_desired=snap1,return_partindices=True,verbose=False)
                    t2=time.time()
                    print(f'Found particle indices in {t2-t1} sec')

                    print('Extracting particle data')
                    t1=time.time()
                    validmask_candidates=np.where(snap2_types==itype)
                
                    for key in eagle_keys[str(itype)]:
                        if size_keys[key]==1:
                            snap1_dset=np.zeros(ncandidates)+np.nan
                            snap2_dset=np.zeros(ncandidates)+np.nan
                        else:
                            snap1_dset=np.zeros(shape=(ncandidates,size_keys[key]))+np.nan
                            snap2_dset=np.zeros(shape=(ncandidates,size_keys[key]))+np.nan
                        snap1_dset[accreted_mask]=partdata[str(snap1)][str(itype)][key][(snap1_partidx,)]
                        snap2_dset[accreted_mask]=partdata[str(snap2)][str(itype)][key][(snap2_partidx,)]
                        
                        accfile['Particle'][f'ihalo_{str(ihalo).zfill(6)}']['Inflow'][f'PartType{itype}'].create_dataset(f'snap1_{key}',data=snap1_dset)
                        accfile['Particle'][f'ihalo_{str(ihalo).zfill(6)}']['Inflow'][f'PartType{itype}'].create_dataset(f'snap2_{key}',data=snap2_dset)
                        # accfile['Particle'][f'ihalo_{str(ihalo).zfill(6)}']['Inflow'][f'PartType{itype}'][f'snap2_{key}'][accreted_mask][validmask_candidates]=partdata[str(snap2)][str(itype)][key][(snap2_partidx,)][validmask_candidates]
                    t2=time.time()
                    print(f'Found particle data in {t2-t1} sec')

                    print(f'Done with ihalo {ihalo} for itype {itype}')
            else:
                continue
        accfile.close()

########################### ADD LAST STRUCTURE DATA TO RECYCLED DATA ###########################

def add_recycling_data_serial(path=None,mcut=10**10):
    
    # Load in metadata
    accfiles_all=list_dir(path)
    accfiles_paths=sorted([accfile_path for accfile_path in accfiles_all if ('All' not in accfile_path and 'recyc' not in accfile_path)])
    accfiles=accfiles_paths
    accfile_ex=h5py.File(accfiles[0],'r+')
    snap2=accfile_ex['Header'].attrs['snap2']
    snap1=int(np.nanmax([10,snap2-10]))
    PartTypes=[0,1]

    # Load in base halo data
    print('Loading halo data')
    base_folder=path.split('acc_data')[0]
    base_folder_contents=list_dir(base_folder)
    base_halo_data_path=[path for path in base_folder_contents if 'B2' in path][0]
    base_halo_data=open_pickle(base_halo_data_path)

    # Load in particle histories - data
    print(f'Retrieving & organising particle histories ...')
    snaps=list(range(snap1,snap2+1))
    run_outname=base_halo_data[snaps[-1]]['outname']
    run=run_outname
    log=f'job_logs/{run_outname}_Recycling-snap{str(snap2).zfill(3)}.log'

    # Load in particle histories
    Part_Histories_fields=["ParticleIDs",'HostStructure']
    Part_Histories_data={str(snap):{} for snap in snaps}
    for snap in snaps[::-1]:
        print(f'Loading part histories for snap {snap}...')
        with open(log,"a") as logfile:
            logfile.write(f' \n')
            logfile.write(f'Loading part histories for snap {snap}...')
        logfile.close()
        Part_Histories_File_snap=h5py.File(base_halo_data[snap]['PartHist_FilePath'],'r')
        for field in Part_Histories_fields:
            Part_Histories_data[str(snap)][field]={str(itype):Part_Histories_File_snap["PartType"+str(itype)+'/'+field].value for itype in PartTypes}

    # Here we go
    snap_master=snaps[-1]
    nhalo=len(base_halo_data[snap_master]['Mass_200crit'])
    ihalos_valid=np.where(np.logical_and(base_halo_data[snap_master]['Mass_200crit']>mcut/10**10,base_halo_data[snap_master]['hostHaloID']<0))[0]

    frac_mp={str(itype):np.zeros(nhalo)+np.nan for itype in PartTypes}
    frac_nmp={str(itype):np.zeros(nhalo)+np.nan for itype in PartTypes}
    frac_transfer={str(itype):np.zeros(nhalo)+np.nan for itype in PartTypes}
    frac_lost={str(itype):np.zeros(nhalo)+np.nan for itype in PartTypes}
    n_recyc={str(itype):np.zeros(nhalo)+np.nan for itype in PartTypes}

    t1=time.time()
    acc_dir=path
    acc_files_all=np.sort(list_dir(acc_dir))
    acc_files=[acc_file for acc_file in acc_files_all if ('All' not in acc_file and '.hdf5' in acc_file)]

    with open(log,"a") as logfile:
        logfile.write(f' \n')
        logfile.write(f'Starting with halo loop ... \n')
    logfile.close()

    iihalo=0
    for ifile,acc_file in enumerate(acc_files):
        acc_file_hdf5=h5py.File(acc_file,'r+')
        acc_file_ihalos=acc_file_hdf5['Integrated']['ihalo_list'].value
        for ifile_ihalo,ihalo in enumerate(acc_file_ihalos):
            if ihalo in ihalos_valid:
                if iihalo%1==0 or ifile_ihalo==0:
                    print(f'[on file {ifile+1}/{len(acc_files)}]: {iihalo/len(ihalos_valid)*100:.2f}% done in total')
                    with open(log,"a") as logfile:
                        logfile.write(f' \n')
                        logfile.write(f'[on file {ifile+1}/{len(acc_files)}]: {iihalo/len(ihalos_valid)*100:.2f}% done in total\n')
                    logfile.close()
                iihalo=iihalo+1
                ihalo_key='ihalo_'+str(ihalo).zfill(6)
                ihalo_ID=base_halo_data[snap_master]['ID'][ihalo]
                try:
                    #main progens & subhalos
                    ihalo_mainprogens=find_progen_index(base_halo_data=base_halo_data,index2=ihalo,snap2=snap_master,depth=len(snaps)-1,return_all_depths=True)
                    ihalo_mainprogen_IDs=[base_halo_data[isnap]['ID'][ihalo_progen] for isnap,ihalo_progen in zip(snaps[:-1][::-1],ihalo_mainprogens)]
                    ihalo_submainprogens=[np.where(ihalo_mainprogen_ID==base_halo_data[isnap]['hostHaloID'])[0] for isnap,ihalo_mainprogen_ID in zip(snaps[:-1][::-1],ihalo_mainprogen_IDs)]
                    ihalo_submainprogen_IDs=[[base_halo_data[snap]['ID'][ihalo_isnap_subprogen] for ihalo_isnap_subprogen in  ihalo_isnap_subprogens] for snap,ihalo_isnap_subprogens in zip(snaps[:-1][::-1],ihalo_submainprogens)]

                    #all progens & subhalos
                    ihalo_allprogens=find_progen_index_tree(base_halo_data=base_halo_data,index2=ihalo,snap2=snap_master,snap1=snaps[0])
                    # print(ihalo_allprogens)
                    ihalo_allprogen_IDs=[ihalo_allprogens[str(snap)] for snap in snaps[:-1][::-1]]
                    ihalo_suballprogen_IDs=[[] for snap in snaps[:-1][::-1]]
                    for isnap, (snap,ihalo_isnap_allprogen_IDs) in enumerate(zip(snaps[:-1][::-1],ihalo_allprogen_IDs)):
                        ihalo_isnap_suballprogen_IDs=[]
                        for ihalo_isnap_iprogen in ihalo_isnap_allprogen_IDs:
                            ihalo_isnap_iprogen_subs=np.where(base_halo_data[snap]['hostHaloID']==ihalo_isnap_iprogen)
                            ihalo_isnap_iprogen_sub_IDs=base_halo_data[snap]['ID'][ihalo_isnap_iprogen_subs]
                            ihalo_isnap_suballprogen_IDs.extend(ihalo_isnap_iprogen_sub_IDs)
                        ihalo_suballprogen_IDs[isnap]=ihalo_isnap_suballprogen_IDs


                except:
                    print(f'had to skip ihalo {ihalo}')
                    continue
                
                #Retrieve particle histories
                ihalo_recycledIDs_history={str(snap):{str(iitype):{} for iitype in PartTypes} for snap in snaps}
                for itype in PartTypes:
                    # print(acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}'].keys())
                    try:
                        snap2_inFOF=acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["snap2_Particle_InFOF"].value
                        snap1_inFOF=acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["snap1_Particle_InFOF"].value
                        snap1_Processed=acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["snap1_Processed"].value
                        snap1_Structure=acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["snap1_Structure"].value
                        npart=len(snap2_inFOF)
                    except:
                        print(f'had to skip ihalo {ihalo} for itype {itype}')
                        continue

                    try:
                        del acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["last_Structure"]
                        acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}'].create_dataset('last_Structure',data=np.zeros(npart)+np.nan)
                    except:
                        acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}'].create_dataset('last_Structure',data=np.zeros(npart)+np.nan)



                    accreted=np.logical_and(snap2_inFOF,np.logical_not(snap1_inFOF))
                    recycled=np.logical_and(snap1_Processed>0,snap1_Structure==-1)
                    ihalo_idmask=np.where(np.logical_and(accreted,recycled))
                    ihalo_recycledIDs=acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["ParticleIDs"].value[ihalo_idmask]
                    ihalo_recycledmasses=acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["Mass"].value[ihalo_idmask]

                    
                    for snap in snaps[:-1][::-1]:
                        ihalo_recycledIDs_history[str(snap)][str(itype)]['itype'],ihalo_recycledIDs_history[str(snap)][str(itype)]['ihist'],x=get_particle_indices(base_halo_data=base_halo_data,
                                                                                                                                        IDs_taken=ihalo_recycledIDs,
                                                                                                                                        IDs_sorted=Part_Histories_data[str(snap)]['ParticleIDs'],
                                                                                                                                        indices_sorted={},
                                                                                                                                        types_taken=np.ones(len(ihalo_recycledIDs))*itype,
                                                                                                                                        snap_taken=snap_master,
                                                                                                                                        snap_desired=snap,
                                                                                                                                        return_partindices=False,
                                                                                                                                        verbose=False)
                    ihalo_itype_recycled_npart=len(ihalo_recycledIDs)
                    ihalo_itype_laststructure=np.zeros(ihalo_itype_recycled_npart)+np.nan
                    for ipart in list(range(ihalo_itype_recycled_npart)):
                        try:
                            ipart_candidx=ihalo_idmask[0][ipart]
                        except:
                            ipart_candidx=-1
                        for isnap,snap in enumerate(snaps[:-1][::-1]):
                            ihalo_isnap_mainprogen=ihalo_mainprogen_IDs[isnap]
                            ihalo_isnap_submainprogens=ihalo_submainprogen_IDs[isnap]

                            ihalo_isnap_allprogens=ihalo_allprogen_IDs[isnap]
                            ihalo_isnap_suballprogens=ihalo_suballprogen_IDs[isnap]


                            ipart_isnap_itype_type=ihalo_recycledIDs_history[str(snap)][str(itype)]['itype'][ipart]
                            ipart_isnap_itype_histidx=ihalo_recycledIDs_history[str(snap)][str(itype)]['ihist'][ipart]
                            ipart_isnap_itype_host=Part_Histories_data[str(snap)]['HostStructure'][str(ipart_isnap_itype_type)][ipart_isnap_itype_histidx]

                            if ipart_isnap_itype_host>0:
                                if ipart_isnap_itype_host==ihalo_isnap_mainprogen:
                                    ihalo_itype_laststructure[ipart]=0.0
                                    acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["last_Structure"][ipart_candidx]=0.0
                                elif ipart_isnap_itype_host in ihalo_isnap_submainprogens:
                                    ihalo_itype_laststructure[ipart]=0.5
                                    acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["last_Structure"][ipart_candidx]=0.5
                                elif ipart_isnap_itype_host in ihalo_isnap_allprogens:
                                    ihalo_itype_laststructure[ipart]=1.0
                                    acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["last_Structure"][ipart_candidx]=1.0
                                elif ipart_isnap_itype_host in ihalo_isnap_suballprogens:
                                    ihalo_itype_laststructure[ipart]=1.5
                                    acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["last_Structure"][ipart_candidx]=1.5
                                else:
                                    ihalo_itype_laststructure[ipart]=2.0
                                    acc_file_hdf5['Particle'][ihalo_key]['Inflow'][f'PartType{itype}']["last_Structure"][ipart_candidx]=2.0
                                break
                            else:
                                continue
                    
                    mp_mask=np.where(ihalo_itype_laststructure<0.1)
                    nmp_mask=np.where(np.logical_and(ihalo_itype_laststructure>0.6,ihalo_itype_laststructure<1.6))
                    transfer_mask=np.where(ihalo_itype_laststructure>1.6)
                    lost_mask=np.where(np.logical_not(np.isfinite(ihalo_itype_laststructure)))
                    
                    # print(f'breakdown for ihalo {ihalo}, itype {itype}' )
                    if ihalo_itype_recycled_npart>0:
                        if not itype==1:
                            frac_mp[str(itype)][ihalo]=np.sum(ihalo_recycledmasses[mp_mask])/np.sum(ihalo_recycledmasses)
                            frac_nmp[str(itype)][ihalo]=np.sum(ihalo_recycledmasses[nmp_mask])/np.sum(ihalo_recycledmasses)
                            frac_transfer[str(itype)][ihalo]=np.sum(ihalo_recycledmasses[transfer_mask])/np.sum(ihalo_recycledmasses)
                            frac_lost[str(itype)][ihalo]=np.sum(ihalo_recycledmasses[lost_mask])/np.sum(ihalo_recycledmasses)
                            n_recyc[str(itype)][ihalo]=ihalo_itype_recycled_npart
                        else:
                            frac_mp[str(itype)][ihalo]=np.size(mp_mask[0])/ihalo_itype_recycled_npart
                            frac_nmp[str(itype)][ihalo]=np.size(nmp_mask[0])/ihalo_itype_recycled_npart
                            frac_transfer[str(itype)][ihalo]=np.size(transfer_mask[0])/ihalo_itype_recycled_npart
                            frac_lost[str(itype)][ihalo]=np.size(lost_mask[0])/ihalo_itype_recycled_npart
                            n_recyc[str(itype)][ihalo]=ihalo_itype_recycled_npart
                    else:
                        continue
                    # #0=mp,1=sat,2=else
                    # last_structure[ipart]
            # else:
            #     continue

    print(f'Saving to {acc_dir}recycling_breakdown.dat ...')
    frac_all={'mp':frac_mp,'nmp':frac_nmp,'transfer':frac_transfer,'lost':frac_lost,'n':n_recyc}
    dump_pickle(data=frac_all,path=acc_dir+'recycling_breakdown.dat')
    print('Done saving to file')
    t2=time.time()
    print()
    print(f'Took {t2-t1:.1f} sec for {len(ihalos_valid)} halos')


def gen_averaged_accretion_data(base_halo_data,path=None):
    
    # Load in metadata
    accfiles_all=list_dir(path)
    accfiles_paths=sorted([accfile_path for accfile_path in accfiles_all if ('All' not in accfile_path and 'recyc' not in accfile_path and 'ave' not in accfile_path)])
    accfiles=accfiles_paths
    accfile_ex=h5py.File(accfiles[0],'r+')
    for ex_iihalo in range(100):
        accfile_ex_ihalo_key=list(accfile_ex['Particle'].keys())[ex_iihalo]
        property_keys_all=list(accfile_ex['Particle'][accfile_ex_ihalo_key]['Inflow']['PartType0'].keys())
        if 'snap1_Metallicity' in property_keys_all:
            break
    property_keys=[property_key for property_key in property_keys_all if ('Structure' not in property_key and 'structure' not in property_key and 'FOF' not in property_key and 'Host' not in property_key and 'Mass' not in property_key and 'ID' not in property_key and 'Processed' not in property_key and 'Types' not in property_key and 'Velocity' not in property_key)]
    property_keys_forfile=flatten([property_keys,['snap1_rcom','snap1_rcmbp','snap2_rcom','snap2_rcmbp','snap1_ffhist','snap2_ffhist']])

    snap2=accfile_ex['Header'].attrs['snap2']
    snap1=accfile_ex['Header'].attrs['snap1']
    snap2_comtophys=base_halo_data[snap2]['SimulationInfo']['ScaleFactor']/base_halo_data[snap2]['SimulationInfo']['h_val']
    snap1_comtophys=base_halo_data[snap1]['SimulationInfo']['ScaleFactor']/base_halo_data[snap1]['SimulationInfo']['h_val']

    origins=['snap1_halo','snap2_halo','Accreted','First-infall','Recycled','Transfer','Merger']
    averages=['Means','Medians','Min','Max']
    print(property_keys)

    nhalos=len(base_halo_data[snap2]['Mass_FOF'])
    output_props={origin:{key:{average:np.zeros(nhalos)+np.nan for average in averages} for key in property_keys_forfile} for origin in origins}
    
    nhist=25
    hist_cuts_dex=[0.5,1,2,3,4]

    for origin in origins:
        for ihist_cut,hist_cut in enumerate(hist_cuts_dex):
            output_props[origin]['snap1_ffhist'][f'acc_cut_{str(ihist_cut).zfill(2)}']=np.zeros(nhalos)+np.nan
            output_props[origin]['snap2_ffhist'][f'acc_cut_{str(ihist_cut).zfill(2)}']=np.zeros(nhalos)+np.nan
            output_props[origin]['snap1_ffhist'][f'halo_cut_{str(ihist_cut).zfill(2)}']=np.zeros(nhalos)+np.nan
            output_props[origin]['snap2_ffhist'][f'halo_cut_{str(ihist_cut).zfill(2)}']=np.zeros(nhalos)+np.nan

    for ifile,accfile_path in enumerate(accfiles_paths):
        print(f'On file {ifile+1}/{len(accfiles_paths)} (for snap {snap2}) ...')
        accfile=h5py.File(accfile_path,'r')
        accfile_ihalo_keys=list(accfile['Particle'].keys())
        nhalo_file=len(accfile_ihalo_keys)

        for iihalo,ihalo_key in enumerate(accfile_ihalo_keys):
            ihalo=int(ihalo_key.split('_')[-1])
            if iihalo<10 or iihalo%100==0:
                print(f'{(iihalo+1)/nhalo_file*100:.2f}% done with file {ifile+1}')
            try:
                ihalo_snap1=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap1_Particle_InFOF'].value
                ihalo_snap2=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap2_Particle_InFOF'].value
                ihalo_accreted=np.logical_and(np.logical_not(ihalo_snap1),ihalo_snap2)
                ihalo_origin={}
                ihalo_origin['snap1_halo']=ihalo_snap1;ihalo_origin['snap2_halo']=ihalo_snap2;ihalo_origin['Accreted']=ihalo_accreted

                ihalo_processed=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap1_Processed'].value
                ihalo_structure=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap1_Structure'].value
                ihalo_laststructure=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['last_Structure'].value
                
                ihalo_origin['First-infall']=np.logical_and(ihalo_processed==0,ihalo_accreted)
                ihalo_origin['Recycled']=np.logical_and(ihalo_accreted,ihalo_laststructure<1.8)
                ihalo_origin['Transfer']=np.logical_and(ihalo_accreted,ihalo_laststructure==2)
                ihalo_origin['Merger']=np.logical_and(ihalo_accreted,ihalo_structure>0)

                ihalo_progen=find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=snap2-snap1)
                ihalo_snap2_com=np.array([base_halo_data[snap2]['Xc'][ihalo],base_halo_data[snap2]['Yc'][ihalo],base_halo_data[snap2]['Zc'][ihalo]],ndmin=2)
                ihalo_snap1_com=np.array([base_halo_data[snap1]['Xc'][ihalo_progen],base_halo_data[snap1]['Yc'][ihalo_progen],base_halo_data[snap1]['Zc'][ihalo_progen]],ndmin=2)
                ihalo_snap2_cmbp=np.array([base_halo_data[snap2]['Xcmbp'][ihalo],base_halo_data[snap2]['Ycmbp'][ihalo],base_halo_data[snap2]['Zcmbp'][ihalo]],ndmin=2)
                ihalo_snap1_cmbp=np.array([base_halo_data[snap1]['Xcmbp'][ihalo_progen],base_halo_data[snap1]['Ycmbp'][ihalo_progen],base_halo_data[snap1]['Zcmbp'][ihalo_progen]],ndmin=2)
                
                ihalo_r200_ave=(base_halo_data[snap2]['R_200crit'][ihalo]+base_halo_data[snap1]['R_200crit'][ihalo_progen])/2
                accreted_mask=np.where(ihalo_origin['Accreted'])
                snap1_halomask=np.where(ihalo_origin['snap1_halo'])
        
                ihalo_snap1_accretedcomxyz=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap1_Coordinates'].value[accreted_mask]*snap1_comtophys-ihalo_snap1_com
                ihalo_snap2_accretedcomxyz=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap2_Coordinates'].value[accreted_mask]*snap2_comtophys-ihalo_snap2_com
                ihalo_snap1_halocomxyz=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap1_Coordinates'].value[snap1_halomask]*snap1_comtophys-ihalo_snap1_com
                ihalo_snap2_halocomxyz=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap2_Coordinates'].value[snap1_halomask]*snap2_comtophys-ihalo_snap2_com
                
                ihalo_snap1_accretedcomxyzhist,foo=np.histogramdd(ihalo_snap1_accretedcomxyz,bins=[nhist]*3,range=[(-ihalo_r200_ave*1.5,ihalo_r200_ave*1.5)]*3,density=True)
                ihalo_snap2_accretedcomxyzhist,foo=np.histogramdd(ihalo_snap2_accretedcomxyz,bins=[nhist]*3,range=[(-ihalo_r200_ave*1.5,ihalo_r200_ave*1.5)]*3,density=True)
                ihalo_snap1_halocomxyzhist,foo=np.histogramdd(ihalo_snap1_halocomxyz,bins=[nhist]*3,range=[(-ihalo_r200_ave*1.5,ihalo_r200_ave*1.5)]*3,density=True)
                ihalo_snap2_halocomxyzhist,foo=np.histogramdd(ihalo_snap2_halocomxyz,bins=[nhist]*3,range=[(-ihalo_r200_ave*1.5,ihalo_r200_ave*1.5)]*3,density=True)

            except:
                if base_halo_data[snap2]['Mass_FOF'][ihalo]>10**10:
                    print(f'Skipping ihalo {ihalo}')
                continue
        
            for origin in origins:
                mask=np.where(ihalo_origin[origin])
                accreted_mask=np.where(ihalo_origin['Accreted'])
                for property_key in property_keys:
                    if not 'Coordinates' in property_key:
                        try:
                            ihalo_origin_prop=accfile['Particle'][ihalo_key]['Inflow']['PartType0'][property_key].value[mask]
                        except:
                            if base_halo_data[snap2]['Mass_FOF'][ihalo]>10**10:
                                print(f'Skipping {property_key} for ihalo {ihalo}')
                            continue
                        output_props[origin][property_key]['Means'][ihalo]=np.nanmean(ihalo_origin_prop)
                        output_props[origin][property_key]['Medians'][ihalo]=np.nanmedian(ihalo_origin_prop)
                        try:
                            output_props[origin][property_key]['Max'][ihalo]=np.nanmax(ihalo_origin_prop)
                            output_props[origin][property_key]['Min'][ihalo]=np.nanmin(ihalo_origin_prop)
                        except:
                            pass
                    else:
                        try:
                            ihalo_origin_prop=accfile['Particle'][ihalo_key]['Inflow']['PartType0'][property_key].value[mask]
                        except:
                            if base_halo_data[snap2]['Mass_FOF'][ihalo]>10**10:
                                print(f'Skipping {property_key} for ihalo {ihalo}')
                            continue
                        if 'snap1' in property_key:
                            ihalo_rcom_coords=np.sqrt(np.sum(np.square(ihalo_origin_prop*snap1_comtophys-ihalo_snap1_com),axis=1))
                            ihalo_rcmbp_coords=np.sqrt(np.sum(np.square(ihalo_origin_prop*snap1_comtophys-ihalo_snap1_cmbp),axis=1))
                            output_props[origin]['snap1_rcom']['Means'][ihalo]=np.nanmean(ihalo_rcom_coords)
                            output_props[origin]['snap1_rcom']['Medians'][ihalo]=np.nanmedian(ihalo_rcom_coords)
                            output_props[origin]['snap1_rcmbp']['Means'][ihalo]=np.nanmean(ihalo_rcmbp_coords)
                            output_props[origin]['snap1_rcmbp']['Medians'][ihalo]=np.nanmedian(ihalo_rcmbp_coords)
                        else:
                            ihalo_rcom_coords=np.sqrt(np.sum(np.square(ihalo_origin_prop*snap2_comtophys-ihalo_snap2_com),axis=1))
                            ihalo_rcmbp_coords=np.sqrt(np.sum(np.square(ihalo_origin_prop*snap2_comtophys-ihalo_snap2_cmbp),axis=1))
                            output_props[origin]['snap2_rcom']['Means'][ihalo]=np.nanmean(ihalo_rcom_coords)
                            output_props[origin]['snap2_rcom']['Medians'][ihalo]=np.nanmedian(ihalo_rcom_coords)
                            output_props[origin]['snap2_rcmbp']['Means'][ihalo]=np.nanmean(ihalo_rcmbp_coords)
                            output_props[origin]['snap2_rcmbp']['Medians'][ihalo]=np.nanmedian(ihalo_rcmbp_coords)
                
                #Filling factor calcs
                # print(f'Filling factor for ihalo {ihalo} origin {origin}')
                try:
                    ihalo_snap1_comxyz=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap1_Coordinates'].value[mask]*snap1_comtophys-ihalo_snap1_com
                    ihalo_snap2_comxyz=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap2_Coordinates'].value[mask]*snap2_comtophys-ihalo_snap2_com
                    ihalo_snap1_comxyz=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap1_Coordinates'].value[mask]*snap1_comtophys-ihalo_snap1_com
                    ihalo_snap2_comxyz=accfile['Particle'][ihalo_key]['Inflow']['PartType0']['snap2_Coordinates'].value[mask]*snap2_comtophys-ihalo_snap2_com
                except:
                    continue

                ihalo_snap1_comxyz_hist,foo=np.histogramdd(ihalo_snap1_comxyz,bins=nhist,range=[(-ihalo_r200_ave*1.5,ihalo_r200_ave*1.5)]*3,density=True)
                ihalo_snap2_comxyz_hist,foo=np.histogramdd(ihalo_snap2_comxyz,bins=nhist,range=[(-ihalo_r200_ave*1.5,ihalo_r200_ave*1.5)]*3,density=True)
                
                ihalo_snap1_comxyz_hist_accdelta=np.absolute((ihalo_snap1_comxyz_hist-ihalo_snap1_accretedcomxyzhist)/np.nanstd(ihalo_snap1_accretedcomxyzhist))
                ihalo_snap2_comxyz_hist_accdelta=np.absolute((ihalo_snap2_comxyz_hist-ihalo_snap2_accretedcomxyzhist)/np.nanstd(ihalo_snap2_accretedcomxyzhist))
                ihalo_snap1_comxyz_hist_halodelta=np.absolute((ihalo_snap1_comxyz_hist-ihalo_snap1_halocomxyzhist)/np.nanstd(ihalo_snap1_halocomxyzhist))
                ihalo_snap2_comxyz_hist_halodelta=np.absolute((ihalo_snap2_comxyz_hist-ihalo_snap2_halocomxyzhist)/np.nanstd(ihalo_snap2_halocomxyzhist))
                
                if 'Accreted' not in origin and 'halo' not in origin:
                    for ihist_cut,hist_cut in enumerate(hist_cuts_dex):
                        output_props[origin]['snap1_ffhist'][f'acc_cut_{str(ihist_cut).zfill(2)}'][ihalo]=np.sum(ihalo_snap1_comxyz_hist_accdelta>hist_cut)/nhist**3
                        output_props[origin]['snap2_ffhist'][f'acc_cut_{str(ihist_cut).zfill(2)}'][ihalo]=np.sum(ihalo_snap2_comxyz_hist_accdelta>hist_cut)/nhist**3
                        output_props[origin]['snap1_ffhist'][f'halo_cut_{str(ihist_cut).zfill(2)}'][ihalo]=np.sum(ihalo_snap1_comxyz_hist_halodelta>hist_cut)/nhist**3
                        output_props[origin]['snap2_ffhist'][f'halo_cut_{str(ihist_cut).zfill(2)}'][ihalo]=np.sum(ihalo_snap2_comxyz_hist_halodelta>hist_cut)/nhist**3

                

    dump_pickle(path=path+'aveprops.dat',data=output_props)




   

