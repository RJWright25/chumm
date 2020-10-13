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
                                                                                                                                                                  
                                                                                                                                                                  
# ProcessingTools.py - Python routines to read accretion data and post-process to generate catalogues.                     
# Author: RUBY WRIGHT 

# Preamble
import os
import sys
import numpy as np
import h5py
import time

sys.path.append('/home/rwright/Software/read_eagle/build/lib/python3.7/site-packages/')

from GenPythonTools import *
from VRPythonTools import *
from STFTools import *
from ParticleTools import *
from AccretionTools import *


########################### GENERATE ACCRETION DATA CATALOGUE ###########################

def gen_base_accretion_catalogue(path='',recycling=False):
    
    run=os.getcwd().split('/')[-1]
    run_short=run.split('EAGLE_')[-1][:3]+'-'+run.split('-')[-1]
    files=list_dir('.')
    if path.endswith('/'):
        path=path[:-1]
    outname=run_short+'_AccData.dat'

    snaps=list(range(12,28))

    testfile=h5py.File(path+f'/snap_{str(snaps[-1]).zfill(3)}/AccretionData_pre01_post01_snap{str(snaps[-1]).zfill(3)}_All.hdf5','r')
    parttype_keys=list(testfile['Integrated']['Inflow'].keys())
    parttypes=[int(parttype_key.split('Type')[-1]) for parttype_key in parttype_keys]
    run_barparttypes=[parttype for parttype in parttypes if parttype in [0,4,5]]
    vmax_key='vmax_fac1'
    m_dm=9.71*10**6

    run_rawaccdata={}
    run_recycdata={}
    run_intaccdata={}

    #initialise raw data file objects
    run_rawaccdata={}
    run_recycdata={}
    snaps_recorded=[]
    for snap in snaps:
        try:
            run_rawaccdata[str(snap)]=h5py.File(f'{path}/snap_{str(snap).zfill(3)}/AccretionData_pre01_post01_snap{str(snap).zfill(3)}_All.hdf5','r')
            if recycling:
                run_recycdata[str(snap)]=open_pickle(f'{path}/snap_{str(snap).zfill(3)}/recycling_breakdown.dat')
            snaps_recorded.append(snap)
        except:
            print(f'Skipping snap {snap} - couldnt find integrated accretion data')
            continue
    snaps=snaps_recorded
    # load and categorise accretion data
    run_intaccdata={snap:{itype:{} for itype in parttypes} for snap in snaps}
    for snap in snaps:
        print(f'Generating base accretion catalogue for snap {snap}')
        for itype in parttypes:
            #total acc
            if itype in run_barparttypes:
                run_intaccdata[snap][itype]['Total']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Total'][f'All_Gross_DeltaM'].value
                #pristine acc
                run_intaccdata[snap][itype]['First-infall']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Unprocessed'][f'All_Gross_DeltaM'].value
                #merger acc
                run_intaccdata[snap][itype]['Merger']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Total'][f'All_Transfer_DeltaM'].value
                #pre-processed acc
                run_intaccdata[snap][itype]['Pre-processed']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Processed'][f'All_Field_DeltaM'].value

                #recycled & transfer
                if recycling and itype in [0,1]:
                    totproc_frac=run_recycdata[str(snap)]['mp'][str(itype)]+run_recycdata[str(snap)]['nmp'][str(itype)]+run_recycdata[str(snap)]['transfer'][str(itype)]
                    recyc_frac=(run_recycdata[str(snap)]['mp'][str(itype)]+run_recycdata[str(snap)]['nmp'][str(itype)])/totproc_frac
                    transfer_frac=(run_recycdata[str(snap)]['transfer'][str(itype)])/totproc_frac
                    run_intaccdata[snap][itype]['Recycled']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Processed'][f'All_Field_DeltaM']*recyc_frac
                    run_intaccdata[snap][itype]['Transfer']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Processed'][f'All_Field_DeltaM']*transfer_frac
            else:
                run_intaccdata[snap][itype]['Total']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Total'][f'All_Gross_DeltaN'].value*m_dm
                #pristine acc
                run_intaccdata[snap][itype]['First-infall']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Unprocessed'][f'All_Gross_DeltaN'].value*m_dm
                #merger acc
                run_intaccdata[snap][itype]['Merger']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Total'][f'All_Transfer_DeltaN'].value*m_dm
                #pre-processed acc
                run_intaccdata[snap][itype]['Pre-processed']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Processed'][f'All_Field_DeltaN'].value*m_dm
                #recycled & transfer
                if recycling:
                    totproc_frac=run_recycdata[str(snap)]['mp'][str(itype)]+run_recycdata[str(snap)]['nmp'][str(itype)]+run_recycdata[str(snap)]['transfer'][str(itype)]
                    recyc_frac=(run_recycdata[str(snap)]['mp'][str(itype)]+run_recycdata[str(snap)]['nmp'][str(itype)])/totproc_frac
                    transfer_frac=(run_recycdata[str(snap)]['transfer'][str(itype)])/totproc_frac

                    run_intaccdata[snap][itype]['Recycled']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Processed'][f'All_Field_DeltaN'].value*m_dm*recyc_frac
                    run_intaccdata[snap][itype]['Transfer']=run_rawaccdata[str(snap)]["Integrated"]['Inflow'][f"PartType{itype}"]['FOF-haloscale'][vmax_key]['Processed'][f'All_Field_DeltaN'].value*m_dm*transfer_frac 

        dump_pickle(data=run_intaccdata,path=outname)


# ########################### FIND AVERAGED ACCRETION PROPERTIES ###########################

def append_accretion_catalogue(path=''):
    
    """

    append_accretion_catalogue : function
	----------

    Uses physical properties of gas particles from add_particle_data_serial, and adds accretion properties.
    This is done on a halo-to-halo basis, and for each inflow mode individually. 

	Parameters
	----------

    path : str
        The directory with files to process.   

    Returns 
    ----------

    [run]-AccData.dat: pickled data-structure (at path).
        Has keys for each subset of particles:
            'First-infall': first-infall accreted gas particles
            'Pre-processed': recycled accreted gas particles
            'Merger': merger origin accreted gas particles
            'Hot': merger origin accreted gas particles
            'Cold': merger origin accreted gas particles
                    
    """


    run=os.getcwd().split('/')[-1]
    run_short=run.split('EAGLE_')[-1][:3]+'-'+run.split('-')[-1]
    files=list_dir('.')
    if path.endswith('/'):
        path=path[:-1]
    outname=run_short+'_AccData.dat'

    print('Loading halo & accretion data ...')
    halodata_file=[fname for fname in files if ('B4' in fname and '.tar' not in fname)][0]
    halodata=open_pickle(halodata_file)
    accdata=open_pickle(outname)
    print('Done loading halo & accretion data ...')

    snaps=list(accdata.keys())
    print(snaps)

    # filling factor parameters
    nhist_azimuth=16
    nhist_elevation=8
    nhist_r=1
    rhist_fac=10
    phi_bins=gen_bins(-np.pi,np.pi,n=16)
    theta_bins=gen_bins(-np.pi/2,np.pi/2,n=8)
    binned_solidangle=np.zeros((1,16,8))
    for itheta in range(len(theta_bins['mid'])):
        theta_lo=theta_bins['edges'][itheta]
        theta_hi=theta_bins['edges'][itheta+1]
        theta_mid=theta_bins['mid'][itheta]
        delta_theta=theta_hi-theta_lo
        for iphi in range(len(phi_bins['mid'])):
            delta_phi=phi_bins['width'][0]
            omega=np.cos(theta_mid)*delta_theta*delta_phi
            binned_solidangle[0,iphi,itheta]=omega
        binned_solidangle_frac=binned_solidangle/(4*np.pi)


    for snap in snaps:
        valid_ihalo=np.where(halodata[snap]['Mass_FOF']>10**10)[0]
        nhalo=len(halodata[snap]['Mass_FOF'])
        accdata_filepaths=list_dir(path+f'/snap_{str(snap).zfill(3)}')
        accdata_filepaths_truncated=[path for path in accdata_filepaths if '.hdf5' in path and 'All' not in path]
        keys=list(accdata[snap][0].keys())
        snap2_comovingfac=halodata[snap]['SimulationInfo']['ScaleFactor']/halodata[snap]['SimulationInfo']['h_val']
        snap2=int(snap)
        snap1=snap2-1
        snap2_comtophys=halodata[snap2]['SimulationInfo']['ScaleFactor']/halodata[snap2]['SimulationInfo']['h_val']
        snap1_comtophys=halodata[snap1]['SimulationInfo']['ScaleFactor']/halodata[snap1]['SimulationInfo']['h_val']

        origins_fromcat=list(accdata[snap][0].keys())
        if 'Recycled' in origins_fromcat:
            origins=['Total','First-infall','Pre-processed','Merger','Recycled','Transfer']
        else:
            origins=['Total','First-infall','Pre-processed','Merger']

        origins.extend(['Hot','Cold'])

        for key in origins_fromcat:
            if key not in origins:
                del accdata[snap][0][key]

        for origin in origins:
            accdata[snap][0][origin+'_Metals']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_f0p05']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_f0p10']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_f0p15']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_f0p25']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_f0p50']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_f1p00']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_fhot_s2']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_fhot_s1']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_avetemp_s1']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_avetemp_s2']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_ffill_s1']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_ffill_s2']=np.zeros(nhalo)+np.nan
            accdata[snap][0][origin+'_nacc']=np.zeros(nhalo)+np.nan

        snap1_fac=halodata[snap-1]['SimulationInfo']['ScaleFactor']/halodata[snap-1]['SimulationInfo']['h_val']
        iihalo=-1
        for ihalo in valid_ihalo:
            found=False
            try:
                ihalo_cmbp=np.array([halodata[snap]["Xcmbp"][ihalo],halodata[snap]["Ycmbp"][ihalo],halodata[snap]["Zcmbp"][ihalo]],ndmin=2)
                ihalo_r200=halodata[snap]["R_200crit"][ihalo]
            except:
                continue
            iihalo+=1
            if iihalo%10==0:
                print(f'{iihalo/len(valid_ihalo)*100:.1f} % done with snap {snap}')
            for itry,accdata_filepath in enumerate(accdata_filepaths_truncated):
                accdata_file=h5py.File(accdata_filepath,'r+')
                try:
                    ihalo_group=accdata_file['Particle'][f'ihalo_{str(ihalo).zfill(6)}']
                    found=True
                    break
                    # print(f'Found halo {ihalo} in file {accdata_filepath}')
                except:
                    accdata_file.close()
                    if itry==3:
                        found=False
                    # print(f'Couldnt find halo {ihalo} in file {accdata_filepath}')
            if not found:
                print(f'Couldnt get data for ihalo {ihalo}')
                accdata_file.close()
                continue
                
            #halo data
            ihalo_progen=find_progen_index(halodata,index2=ihalo,snap2=snap2,depth=snap2-snap1)
            if not ihalo_progen>=0:
                print(f'Couldnt find ihalo {ihalo} progenitor')
                continue
                
            ihalo_snap2_com=np.array([halodata[snap2]['Xc'][ihalo],halodata[snap2]['Yc'][ihalo],halodata[snap2]['Zc'][ihalo]],ndmin=2)
            ihalo_snap1_com=np.array([halodata[snap1]['Xc'][ihalo_progen],halodata[snap1]['Yc'][ihalo_progen],halodata[snap1]['Zc'][ihalo_progen]],ndmin=2)
            ihalo_snap2_cmbp=np.array([halodata[snap2]['Xcmbp'][ihalo],halodata[snap2]['Ycmbp'][ihalo],halodata[snap2]['Zcmbp'][ihalo]],ndmin=2)
            ihalo_snap1_cmbp=np.array([halodata[snap1]['Xcmbp'][ihalo_progen],halodata[snap1]['Ycmbp'][ihalo_progen],halodata[snap1]['Zcmbp'][ihalo_progen]],ndmin=2)
            ihalo_r200_ave=(halodata[snap2]['R_200crit'][ihalo]+halodata[snap1]['R_200crit'][ihalo_progen])/2

            masks={}
            try:
                accreted=np.logical_and(ihalo_group['Inflow']['PartType0']['snap2_Particle_InFOF'].value,np.logical_not(ihalo_group['Inflow']['PartType0']['snap1_Particle_InFOF'].value))
                snap1_halo=ihalo_group['Inflow']['PartType0']['snap1_Particle_InFOF'].value
                masks['Total']=np.where(accreted)
                masks['First-infall']=np.where(np.logical_and(accreted,ihalo_group['Inflow']['PartType0']['snap1_Processed'].value==0))
                masks['Merger']=np.where(np.logical_and.reduce([accreted,ihalo_group['Inflow']['PartType0']['snap1_Structure'].value>0]))
                masks['Pre-processed']=np.where(np.logical_and.reduce([ihalo_group['Inflow']['PartType0']['snap1_Structure'].value==-1,ihalo_group['Inflow']['PartType0']['snap1_Processed'].value>0]))
                masks['Hot']=np.where(np.logical_and(accreted,ihalo_group['Inflow']['PartType0']['snap2_Temperature'].value>=10**5.5))
                masks['Cold']=np.where(np.logical_and(accreted,ihalo_group['Inflow']['PartType0']['snap2_Temperature'].value<10**5.5))
            except:
                accdata_file.close()
                continue
            
            try:
                masses=ihalo_group['Inflow']['PartType0']['Mass'].value.flatten()
                mets=ihalo_group['Inflow']['PartType0']['snap1_Metallicity'].value
                snap2_radii=np.sqrt(np.sum(np.square(ihalo_group['Inflow']['PartType0']['snap2_Coordinates'].value*snap2_comovingfac-ihalo_cmbp),axis=1))
                temp_post=ihalo_group['Inflow']['PartType0']['snap2_Temperature'].value
                temp_pre=ihalo_group['Inflow']['PartType0']['snap1_Temperature'].value
            except:
                print('Couldnt get temp')
                accdata_file.close()
                continue


            for origin in origins:
                origin_masses=masses[masks[origin]]
                origin_mets=mets[masks[origin]]
                metmass=np.nansum(origin_masses*origin_mets)
                accdata[snap][0][origin+'_Metals'][ihalo]=metmass

                origin_finalradii=snap2_radii[masks[origin]]
                origin_temp=temp_post[masks[origin]]
                origin_temp_s1=temp_pre[masks[origin]]

                mask_f0p05=np.where(origin_finalradii<0.05*ihalo_r200)
                mask_f0p10=np.where(origin_finalradii<0.10*ihalo_r200)
                mask_f0p15=np.where(origin_finalradii<0.15*ihalo_r200)
                mask_f0p25=np.where(origin_finalradii<0.25*ihalo_r200)
                mask_f0p50=np.where(origin_finalradii<0.50*ihalo_r200)
                mask_f1p00=np.where(origin_finalradii<ihalo_r200)
                mask_hot_s1=np.where(origin_temp_s1>10**5.5)
                mask_hot_s2=np.where(origin_temp>10**5.5)

                accdata[snap][0][origin+'_f0p05'][ihalo]=np.nansum(origin_masses[mask_f0p05])/np.nansum(origin_masses)
                accdata[snap][0][origin+'_f0p10'][ihalo]=np.nansum(origin_masses[mask_f0p10])/np.nansum(origin_masses)
                accdata[snap][0][origin+'_f0p15'][ihalo]=np.nansum(origin_masses[mask_f0p15])/np.nansum(origin_masses)
                accdata[snap][0][origin+'_f0p25'][ihalo]=np.nansum(origin_masses[mask_f0p25])/np.nansum(origin_masses)
                accdata[snap][0][origin+'_f0p50'][ihalo]=np.nansum(origin_masses[mask_f0p50])/np.nansum(origin_masses)
                accdata[snap][0][origin+'_f1p00'][ihalo]=np.nansum(origin_masses[mask_f1p00])/np.nansum(origin_masses)
                accdata[snap][0][origin+'_fhot_s2'][ihalo]=np.nansum(origin_masses[mask_hot_s2])/np.nansum(origin_masses)
                accdata[snap][0][origin+'_fhot_s1'][ihalo]=np.nansum(origin_masses[mask_hot_s1])/np.nansum(origin_masses)
                accdata[snap][0][origin+'_avetemp_s1'][ihalo]=np.nansum(origin_temp_s1*origin_masses)/np.nansum(origin_masses)
                accdata[snap][0][origin+'_avetemp_s2'][ihalo]=np.nansum(origin_temp*origin_masses)/np.nansum(origin_masses)
                accdata[snap][0][origin+'_nacc'][ihalo]=len(origin_temp)


            ## filling factors
            for origin in origins:
                mask=masks[origin]
                try:
                    ihalo_snap1_comxyz=cart_to_sph(ihalo_group['Inflow']['PartType0']['snap1_Coordinates'].value[mask]*snap1_comtophys-ihalo_snap1_cmbp)
                    ihalo_snap2_comxyz=cart_to_sph(ihalo_group['Inflow']['PartType0']['snap2_Coordinates'].value[mask]*snap2_comtophys-ihalo_snap2_cmbp)
                    ihalo_snap1_comxyz_hist,foo=np.histogramdd(ihalo_snap1_comxyz,bins=[nhist_r,nhist_azimuth,nhist_elevation],range=[(0,ihalo_r200_ave*rhist_fac),(-np.pi,np.pi),(-np.pi/2,np.pi/2)],density=False)
                    ihalo_snap2_comxyz_hist,foo=np.histogramdd(ihalo_snap2_comxyz,bins=[nhist_r,nhist_azimuth,nhist_elevation],range=[(0,ihalo_r200_ave*rhist_fac),(-np.pi,np.pi),(-np.pi/2,np.pi/2)],density=False)

                    #snap 1
                    npart_acc=np.sum(ihalo_snap1_comxyz_hist)
                    expectedpercell=npart_acc*binned_solidangle_frac
                    occupied_cells=np.where(ihalo_snap1_comxyz_hist>0.1*expectedpercell)
                    occupied_angle=np.sum(binned_solidangle[occupied_cells])
                    accdata[snap][0][origin+'_ffill_s1']=occupied_angle/(4*np.pi)

                    #snap 2
                    npart_acc=np.sum(ihalo_snap2_comxyz_hist)
                    expectedpercell=npart_acc*binned_solidangle_frac
                    occupied_cells=np.where(ihalo_snap2_comxyz_hist>0.1*expectedpercell)
                    occupied_angle=np.sum(binned_solidangle[occupied_cells])
                    accdata[snap][0][origin+'_ffill_s2']=occupied_angle/(4*np.pi)
            
                except:
                    print(f'No coordinates for ihalo {ihalo}')
                    raise

    dump_pickle(path=outname,data=accdata)

