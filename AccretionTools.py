########################### CREATE PARTICLE HISTORIES ###########################
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

#  python tools 
from VRPythonTools import *
from STFTools import *
from RW_GenPythonTools import *

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
#         snaps=list(range(len(base_halo_data)))

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
    
#     Part_Names=['gas','DM','stars','BH']
#     if base_halo_data[valid_snaps[0]]['Part_FileType']=='EAGLE':
#         PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
#     else:
#         PartTypes=[0,1] #Gas, DM

#     isnap=0
#     # Iterate through snapshots and flip switches as required
#     for snap in valid_snaps:
#         if verbose:
#             print(f'Processing for snap = {snap}')
    
#         #load new snap data
#         if base_halo_data[snap]['Part_FileType']=='EAGLE': 
#             EAGLE_boxsize=base_halo_data[snap]['SimulationInfo']['BoxSize_Comoving']
#             EAGLE_Snap=read_eagle.EagleSnapshot(base_halo_data[snap]['Part_FilePath'])
#             if verbose:
#                 print('Reading & slicing new EAGLE snap data ...')
#             t1=time.time()
#             EAGLE_Snap.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
#             Particle_IDs_FRESH=[EAGLE_Snap.read_dataset(itype,"ParticleIDs") for itype in PartTypes]
#             t2=time.time()
#             if verbose:
#                 print(f'Finished loading new EAGLE snap data in {t2-t1} sec')

#         else:
#             h5py_Snap=h5py.File(base_halo_data[snap]['Part_FilePath'])
#             Particle_IDs_FRESH=[h5py_Snap['PartType'+str(itype)+'/ParticleIDs'] for itype in PartTypes]

#         N_Particles_FRESH=[len(Particle_IDs_FRESH[itype]) for itype in range(len(Particle_IDs_FRESH))]

#         # If needed, initialise data
#         if isnap==0:
#             #initialise: columns: 0: ID, 1: F1, 2: F2 (of length n_particles; all flags are 0)
#             Processed_Flags_FRESH=[df(np.column_stack((np.sort(Particle_IDs_FRESH[itype]),np.zeros(N_Particles_FRESH[itype]),np.zeros(N_Particles_FRESH[itype]),np.zeros(N_Particles_FRESH[itype]))),columns=['ParticleID','Processed_L1','Processed_L2','ParticleIndex'],dtype=int) for itype in range(len(PartTypes))]

#         # Carry over old flags and index particle IDs
#         t1=time.time()
#         for itype in [2,3,0,1]:#per type array
#             if verbose:
#                 print('Carrying over flags for ',Part_Names[itype],' from previous snap')

#             if itype==0: #if Gas
#                 if verbose:
#                     print(f'Cleaning particle list for {Part_Names[itype]} at snap = {snap}')
#                     print('Finding transformed gas particles ...')
#                 t1=time.time()
#                 #check the new ID list, find the IDs which have disappeared
#                 t11=time.time()
#                 Particle_IDs_REMOVED_IDs=np.concatenate([Particle_IDs_NEW_STARS_IDs,Particle_IDs_NEW_BH_IDs])
#                 print('Number of new Gas+Bh particles this snap: ',len(Particle_IDs_REMOVED_IDs))
#                 print('Delta length of fresh/old id list: ',len(Processed_Flags_FRESH[itype]['ParticleID'])-N_Particles_FRESH[itype])
#                 Particle_IDs_REMOVED_GAS_indices=np.searchsorted(np.array(Processed_Flags_FRESH[itype]['ParticleID']),Particle_IDs_REMOVED_IDs)
#                 Processed_Flags_FRESH[itype]=Processed_Flags_FRESH[itype].drop(index=Particle_IDs_REMOVED_GAS_indices)
#                 t12=time.time()
#                 print(f'This bit took {t12-t11}')

#                 #check the ID list is now the same length as the previous snap
#                 if len(Particle_IDs_FRESH[itype])==len(Processed_Flags_FRESH[itype]['ParticleID']):
#                     Processed_Flags_FRESH[itype]['ParticleIndex']=np.argsort(Particle_IDs_FRESH[itype])
#                     t2=time.time()
#                     if verbose:
#                         print(f'Successfully carried flags and indexed IDs for {Part_Names[itype]} at snap = {snap} in {t2-t1} sec')
#                 else:
#                     print("Couldn't coerce new particle indices with old ones")
#                     return []

#             if itype==1: #if DM
#                 if verbose:
#                     print(f'Cleaning particle list for {Part_Names[itype]} at snap = {snap}')
#                 t1=time.time()
#                 #check the ID list is the same length as the previous snap
#                 if len(Particle_IDs_FRESH[itype])==len(Processed_Flags_FRESH[itype]['ParticleID']):
#                     #flags are carried over (structure still ordered by ID), now updating the index information
#                     Processed_Flags_FRESH[itype]['ParticleIndex']=np.argsort(Particle_IDs_FRESH[itype])
#                     t2=time.time()
#                     if verbose:
#                         print(f'Successfully carried flags and indexed IDs for {Part_Names[itype]} at snap = {snap} in {t2-t1} sec')
#                 else:
#                     print("Couldn't coerce new particle indices with old ones")
#                     return []

#             if itype==2: #if STARS
#                 if verbose:
#                     print(f'Cleaning particle list for {Part_Names[itype]} at snap = {snap}')
#                     print('Finding new star particles ...')

#                 t1=time.time()
#                 Particle_IDs_NEW_STARS_mask=np.in1d(Particle_IDs_FRESH[itype],Processed_Flags_FRESH[itype]['ParticleID'],invert=True)#star IDs that we didn't have last snap (should be from gas)
#                 Particle_IDs_NEW_STARS_IDs=np.compress(Particle_IDs_NEW_STARS_mask,Particle_IDs_FRESH[itype])

#                 istar=0
#                 GAS_index_PREV=[]
#                 transfer_L1_flag=[]
#                 transfer_L2_flag=[]

#                 for NEW_STAR_ID in Particle_IDs_NEW_STARS_IDs:
#                     #for each new star particle, find its past gas properties
#                     if istar%10000==0:
#                         if verbose:
#                             print(f'{istar/len(Particle_IDs_NEW_STARS_IDs)*100}% done finding new star particle gas history')

#                     index_would_be=np.searchsorted(Processed_Flags_FRESH[0]['ParticleID'],NEW_STAR_ID)
#                     gasID_atthatindex=int(Processed_Flags_FRESH[0]['ParticleID'][index_would_be])

#                     transfer_L1_flag.append(int(Processed_Flags_FRESH[0]['Processed_L1'].iloc[index_would_be]))
#                     transfer_L2_flag.append(int(Processed_Flags_FRESH[0]['Processed_L2'].iloc[index_would_be]))
#                     istar=istar+1
                
#                 Processed_Flags_FRESH[itype]=Processed_Flags_FRESH[itype].append(df(np.column_stack((Particle_IDs_NEW_STARS_IDs,transfer_L1_flag,transfer_L2_flag,np.zeros(len(Particle_IDs_NEW_STARS_IDs)))),columns=['ParticleID','Processed_L1','Processed_L2','ParticleIndex'],dtype=int))
#                 Processed_Flags_FRESH[itype]=Processed_Flags_FRESH[itype].sort_values(['ParticleID'])
                
#                 #check the ID list is the same length as the previous snap
#                 if len(Particle_IDs_FRESH[itype])==len(Processed_Flags_FRESH[itype]['ParticleID']):
#                     #flags are carried over (structure still ordered by ID), now updating the index information
#                     Processed_Flags_FRESH[itype]['ParticleIndex']=np.argsort(Particle_IDs_FRESH[itype])
#                     t2=time.time()
#                     if verbose:
#                         print(f'Successfully carried flags and indexed IDs for {Part_Names[itype]} at snap = {snap} in {t2-t1} sec')
#                 else:
#                     print("Couldn't coerce new particle indices with old ones")
#                     return []

#             if itype==3: #if BH
#                 if verbose:
#                     print(f'Cleaning particle list for {Part_Names[itype]} at snap = {snap}')
#                     print('Finding new BH particles ...')

#                 t1=time.time()
#                 Particle_IDs_NEW_BH_mask=np.in1d(Particle_IDs_FRESH[itype],Processed_Flags_FRESH[itype]['ParticleID'],invert=True)#BH IDs that we didn't have last snap (should be from gas)
#                 Particle_IDs_NEW_BH_IDs=np.compress(Particle_IDs_NEW_BH_mask,Particle_IDs_FRESH[itype])


#         t2=time.time()
#         print(f'Finished carrying over old data in {t2-t1} sec')

#         # Find new particles in halos and flip the required switches
#         temp_subhalo_indices=list(np.where(base_halo_data[snap]['hostHaloID']>0)[0])

#         print('Retrieving and organising particles in structure...')
#         #recall previous data

#         t1=time.time()
#         Halo_Particle_Lists=get_particle_lists(base_halo_data[snap],include_unbound=True,add_subparts_to_fofs=False)
#         L1_Processed_Particles_FRESH=np.column_stack((np.concatenate(Halo_Particle_Lists['Particle_IDs']),np.concatenate(Halo_Particle_Lists['Particle_Types'])))
#         L2_Processed_Particles_FRESH_IDs=np.concatenate([Halo_Particle_Lists['Particle_IDs'][temp_subhalo_index] for temp_subhalo_index in temp_subhalo_indices])
#         L2_Processed_Particles_FRESH_Types=np.concatenate([Halo_Particle_Lists['Particle_Types'][temp_subhalo_index] for temp_subhalo_index in temp_subhalo_indices])
#         L2_Processed_Particles_FRESH=np.column_stack((L2_Processed_Particles_FRESH_IDs,L2_Processed_Particles_FRESH_Types))
#         t2=time.time()
#         print(f'Finished finding particles in structure in {t2-t1} sec')


        
        
#         isnap=isnap+1

#     return Processed_Flags_FRESH

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
        

def gen_particle_history_serial(base_halo_data,snaps=[],verbose=1):

    """

    gen_particle_history_serial : function
	----------

    Generate and save particle history data from velociraptor property and particle files.

	Parameters
	----------
    base_halo_data : list of dictionaries
        The halo data list of dictionaries previously generated (by gen_base_halo_data). Should contain the type of particle file we'll be reading. 

    snaps : list of ints
        The list of absolute snaps (corresponding to index in base_halo_data) for which we will add 
        particles in halos or subhalos (and save accordingly). The running lists will build on the previous snap. 

	Returns
	----------
    PartHistory_xxx-outname.hdf5 : hdf5 file with datasets

        '/PartTypeX/PartID'
        '/PartTypeX/PartIndex'
        '/PartTypeX/HostStructure'

	"""

    # Will save to file at: part_histories/PartTypeX_History_xxx-outname.dat
    # Snaps
    if snaps==[]:
        snaps=list(range(len(base_halo_data)))

    try:
        valid_snaps=[len(base_halo_data[snap].keys())>3 for snap in snaps] #which indices of snaps are valid
        valid_snaps=np.compress(valid_snaps,snaps)
        outname=base_halo_data[valid_snaps[0]]['outname']

    except:
        print("Couldn't validate snaps")
        return []

    # if the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    if not os.path.isdir("part_histories"):
        os.mkdir("part_histories")
    
    PartNames=['gas','DM','','','star','BH']

    if base_halo_data[valid_snaps[0]]['Part_FileType']=='EAGLE':
        PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
    else:
        PartTypes=[0,1] #Gas, DM

    isnap=0
    # Iterate through snapshots and flip switches as required
    for snap in valid_snaps:
        outfile=h5py.File("part_histories/PartHistory_"+str(snap).zfill(3)+"_"+outname+".hdf5",'w')

        #Load the EAGLE data for this snapshot
        EAGLE_boxsize=base_halo_data[snap]['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap=read_eagle.EagleSnapshot(base_halo_data[snap]['Part_FilePath'])
        EAGLE_Snap.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)

        Particle_History_Flags=dict()

        #Load the Halo particle lists for this snapshot for each particle type
        t1=time.time()
        snap_Halo_Particle_Lists=get_particle_lists(base_halo_data[snap],include_unbound=True,add_subparts_to_fofs=False)
        
        n_halos=len(snap_Halo_Particle_Lists["Particle_IDs"])
        n_halo_particles=[len(snap_Halo_Particle_Lists["Particle_IDs"][ihalo]) for ihalo in range(n_halos)]
        temp_subhalo_indices=np.where(base_halo_data[snap]["hostHaloID"]>0)[0]
        
        fieldhalo_Particle_hosts=np.concatenate([np.ones(n_halo_particles[ihalo])*base_halo_data[snap]["hostHaloID"][ihalo] for ihalo in range(n_halos)])
        subhalo_Particle_hosts=np.concatenate([np.ones(n_halo_particles[ihalo])*base_halo_data[snap]["hostHaloID"][ihalo] for ihalo in temp_subhalo_indices])
        print(fieldhalo_Particle_hosts[5000:5200])        
        #fieldhalo==l1, subhalo==l2
        fieldhalo_Particles=df({'ParticleIDs':np.concatenate(snap_Halo_Particle_Lists['Particle_IDs']),'ParticleTypes':np.concatenate(snap_Halo_Particle_Lists['Particle_Types'])},dtype=int).sort_values(["ParticleIDs"])
        subhalo_Particles=df({'ParticleIDs':np.concatenate([snap_Halo_Particle_Lists['Particle_IDs'][temp_subhalo_index] for temp_subhalo_index in temp_subhalo_indices]),'ParticleTypes':np.concatenate([snap_Halo_Particle_Lists['Particle_Types'][temp_subhalo_index] for temp_subhalo_index in temp_subhalo_indices])},dtype=int).sort_values(["ParticleIDs"])
            
        fieldhalo_Particles_bytype={str(itype):np.array(fieldhalo_Particles["ParticleIDs"].loc[fieldhalo_Particles["ParticleTypes"]==itype]) for itype in PartTypes}
        subhalo_Particles_bytype={str(itype):np.array(subhalo_Particles["ParticleIDs"].loc[subhalo_Particles["ParticleTypes"]==itype]) for itype in PartTypes}

       
        
        t2=time.time()
        print(f"Loaded, concatenated and sorted halo particle lists in {t2-t1} sec")
        print(f"There are {np.sum([len(fieldhalo_Particles_bytype[str(itype)]) for itype in PartTypes])} particles in structure (L1), and {np.sum([len(subhalo_Particles_bytype[str(itype)]) for itype in PartTypes])} particles in substructure (L2)")

        # map IDs to indices from EAGLE DATA and initialise array
        
        for itype in PartTypes:
            
            t1=time.time()
            #load new snap data
            if base_halo_data[snap]['Part_FileType']=='EAGLE': 
                Particle_IDs_Unsorted_itype=EAGLE_Snap.read_dataset(itype,"ParticleIDs")
                N_Particles_itype=len(Particle_IDs_Unsorted_itype)
            else:
                h5py_Snap=h5py.File(base_halo_data[snap]['Part_FilePath'])
                Particle_IDs_Unsorted_itype=h5py_Snap['PartType'+str(itype)+'/ParticleIDs']
                N_Particles_itype=len(Particle_IDs_Unsorted_itype)

            #initialise flag data structure with mapped IDs
            print(f"Mapping IDs to indices for all {PartNames[itype]} particles at snap {snap} ...")
            Particle_History_Flags[str(itype)]={"ParticleIDs_Sorted":np.sort(Particle_IDs_Unsorted_itype),"ParticleIndex_Original":np.argsort(Particle_IDs_Unsorted_itype),"HostStructure":np.zeros(N_Particles_itype)}
            t2=time.time()
            print(f"Mapped IDs to indices for all {PartNames[itype]} particles at snap {snap} in {t2-t1} sec")
            
            #flip switches of new particles
            print("Flipping L1&L2 switches ...")
            t1=time.time()
            ipart_switch=0
            subhalo_Particles_bytype_SET=set(subhalo_Particles_bytype[str(itype)])

            for temp_ID_L1 in fieldhalo_Particles_bytype[str(itype)]:
    
                ipart_switch=ipart_switch+1
                if ipart_switch%10000==0:
                    print(ipart_switch/len(fieldhalo_Particles_bytype[str(itype)])*100,f'% done flipping L1&L2 switches for {PartNames[itype]} particles')

                sorted_index_temp_ID_L1=binary_search_2(element=temp_ID_L1,sorted_array=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"])
                Particle_History_Flags[str(itype)]["HostStructure"][sorted_index_temp_ID_L1]=1

                if temp_ID_L1 in subhalo_Particles_bytype_SET:
                    Particle_History_Flags[str(itype)]["HostStructure"][sorted_index_temp_ID_L1]=2

            t2=time.time()
            print(f"Flipped L1&L2 switches in {t2-t1} sec for {PartNames[itype]} particles")

        print(f'Dumping data to file')
        t1=time.time()
        if len(base_halo_data[snap]["hostHaloID"])<65000:
            dtype_for_host='uint16'
        else:
            dtype_for_host='uint32'

        for itype in PartTypes:
            dset_write=outfile.create_dataset(f'/PartType{itype}/ParticleIDs',dtype='int64',compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"])
            dset_write=outfile.create_dataset(f'/PartType{itype}/ParticleIndex',dtype='int32',compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIndex_Original"])
            dset_write=outfile.create_dataset(f'/PartType{itype}/HostStructure',dtype=dtype_for_host,compression='gzip',data=Particle_History_Flags[str(itype)]["HostStructure"])
        outfile.close()
        t2=time.time()
        print(f'Dumped {PartNames[itype]} data to file in {t2-t1} sec')

        isnap+=1

    return Particle_History_Flags

