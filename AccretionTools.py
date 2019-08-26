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
from GenPythonTools import *


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
        

def gen_particle_history_serial(base_halo_data,snaps=[],test_run=False,verbose=1):

    """

    gen_particle_history_serial : function
	----------

    Generate and save particle history data from velociraptor property and particle files.

	Parameters
	----------
    base_halo_data : list of dictionaries
        The halo data list of dictionaries previously generated (by gen_base_halo_data). Should contain the type of particle file we'll be reading. 

    test_run : bool
        Flag for whether we want to 

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
        run_outname=base_halo_data[valid_snaps[0]]['outname']

    except:
        print("Couldn't validate snaps")
        return []

    # if the directory with particle histories doesn't exist yet, make it (where we have run the python script)
    
    PartNames=['gas','DM','','','star','BH']

    if base_halo_data[valid_snaps[0]]['Part_FileType']=='EAGLE':
        PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
        SimType='EAGLE'
    else:
        PartTypes=[0,1] #Gas, DM
        SimType='OtherHydro'

    isnap=0
    # Iterate through snapshots and flip switches as required
    for snap in valid_snaps:
        if test_run:
            if not os.path.isdir("part_histories_test"):
                os.mkdir("part_histories_test")
            outfile_name="part_histories_test/PartHistory_"+str(snap).zfill(3)+"_"+run_outname+".hdf5"
            if os.path.exists(outfile_name):
                os.remove(outfile_name)
            outfile=h5py.File(outfile_name,'w')
        else:
            if not os.path.isdir("part_histories"):
                os.mkdir("part_histories")
            outfile_name="part_histories/PartHistory_"+str(snap).zfill(3)+"_"+run_outname+".hdf5"
            if os.path.exists(outfile_name):
                os.remove(outfile_name)
            outfile=h5py.File(outfile_name,'w')

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
        temp_field_indices=np.where(base_halo_data[snap]["hostHaloID"]<0)[0]
        allhalo_Particle_hosts=np.concatenate([np.ones(n_halo_particles[ihalo],dtype='uint32')*ihalo for ihalo in range(n_halos)])
        subhalo_Particle_hosts=np.concatenate([np.ones(n_halo_particles[ihalo],dtype='uint32')*ihalo for ihalo in temp_subhalo_indices])
        
        #fieldhalo==l1, subhalo==l2
        fieldhalo_Particles=df({'ParticleIDs':np.concatenate(snap_Halo_Particle_Lists['Particle_IDs']),'ParticleTypes':np.concatenate(snap_Halo_Particle_Lists['Particle_Types']),"HostHaloIndex":allhalo_Particle_hosts},dtype=int).sort_values(["ParticleIDs"])
        subhalo_Particles=df({'ParticleIDs':np.concatenate([snap_Halo_Particle_Lists['Particle_IDs'][temp_subhalo_index] for temp_subhalo_index in temp_subhalo_indices]),'ParticleTypes':np.concatenate([snap_Halo_Particle_Lists['Particle_Types'][temp_subhalo_index] for temp_subhalo_index in temp_subhalo_indices]),"HostHaloIndex":subhalo_Particle_hosts},dtype=int).sort_values(["ParticleIDs"])
        fieldhalo_Particles_bytype={str(itype):np.array(fieldhalo_Particles[["ParticleIDs","HostHaloIndex"]].loc[fieldhalo_Particles["ParticleTypes"]==itype]) for itype in PartTypes}
        subhalo_Particles_bytype={str(itype):np.array(subhalo_Particles[["ParticleIDs","HostHaloIndex"]].loc[subhalo_Particles["ParticleTypes"]==itype]) for itype in PartTypes}

        print(len())
        n_fieldhalo_particles=np.sum([len(fieldhalo_Particles_bytype[str(itype)][:,0]) for itype in PartTypes])
        n_subhalo_particles=np.sum([len(subhalo_Particles_bytype[str(itype)][:,0]) for itype in PartTypes])
        t2=time.time()
        print(f"Loaded, concatenated and sorted halo particle lists for snap {snap} in {t2-t1} sec")
        print(f"There are {n_fieldhalo_particles} particles in structure (L1), and {n_subhalo_particles} particles in substructure (L2) at snap {snap}")

        # map IDs to indices from EAGLE DATA and initialise array
        
        for itype in PartTypes:
            
            t1=time.time()
            #load new snap data
            if SimType=='EAGLE': 
                Particle_IDs_Unsorted_itype=EAGLE_Snap.read_dataset(itype,"ParticleIDs")
                N_Particles_itype=len(Particle_IDs_Unsorted_itype)
            else:
                h5py_Snap=h5py.File(base_halo_data[snap]['Part_FilePath'])
                Particle_IDs_Unsorted_itype=h5py_Snap['PartType'+str(itype)+'/ParticleIDs']
                N_Particles_itype=len(Particle_IDs_Unsorted_itype)
            

            #initialise flag data structure with mapped IDs
            print(f"Mapping IDs to indices for all {PartNames[itype]} particles at snap {snap} ...")
            Particle_History_Flags[str(itype)]={"ParticleIDs_Sorted":np.sort(Particle_IDs_Unsorted_itype),"ParticleIndex_Original":np.argsort(Particle_IDs_Unsorted_itype),"HostHaloIndex":np.ones(N_Particles_itype,dtype='int32')*(-1)}
            t2=time.time()
            print(f"Mapped IDs to indices for all {PartNames[itype]} particles at snap {snap} in {t2-t1} sec")
            
            #flip switches of new particles
            print("Adding host indices ...")
            t1=time.time()
            ipart_switch=0
            subhalo_Particles_bytype_SET=set(subhalo_Particles_bytype[str(itype)][:,0])

            for field_particle_ID_and_host in fieldhalo_Particles_bytype[str(itype)]:
                field_particle_ID=field_particle_ID_and_host[0]
                field_particle_HostHalo=field_particle_ID_and_host[1]

                if ipart_switch%100000==0:
                    print(ipart_switch/len(fieldhalo_Particles_bytype[str(itype)])*100,f'% done adding host halos for {PartNames[itype]} particles')

                sorted_index_temp_ID=binary_search_1(element=field_particle_ID,sorted_array=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"])[0]
                Particle_History_Flags[str(itype)]["HostHaloIndex"][sorted_index_temp_ID]=int(field_particle_HostHalo)
                ipart_switch=ipart_switch+1

            t2=time.time()
            print(f"Added host halos in {t2-t1} sec for {PartNames[itype]} particles")

        print(f'Dumping data to file')
        t1=time.time()

        if len(base_halo_data[snap]["hostHaloID"])<65000:
            dtype_for_host='uint16'
        else:
            dtype_for_host='uint32'

        for itype in PartTypes:
            dset_write=outfile.create_dataset(f'/PartType{itype}/ParticleIDs',dtype='int64',compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIDs_Sorted"])
            dset_write=outfile.create_dataset(f'/PartType{itype}/ParticleIndex',dtype='int32',compression='gzip',data=Particle_History_Flags[str(itype)]["ParticleIndex_Original"])
            dset_write=outfile.create_dataset(f'/PartType{itype}/HostStructure',dtype=dtype_for_host,compression='gzip',data=Particle_History_Flags[str(itype)]["HostHaloIndex"])
        
        outfile.close()
        t2=time.time()

        print(f'Dumped snap {snap} data to file in {t2-t1} sec')

        isnap+=1

    return Particle_History_Flags


def gen_accretion_data_serial(base_halo_data,snap=None,test_run=False,halo_index_list=None,snap_gap=1,fidelity_gap=1,verbose=1):
    
    """

    gen_accretion_data_serial : function
	----------

    Generate and save accretion rates for each particle type by comparing particle lists with appropriate kwargs. 

    ** note: particle histories and base_halo_data must have been created as per gen_particle_history_serial (this file)
             and gen_base_halo_data in STFTools.py

	Parameters
	----------
    base_halo_data : list of dictionaries
        The minimal halo data list of dictionaries previously generated ("B1" is sufficient)

    snap : int
        The index in the base_halo_data for which to calculate accretion rates (should be actual snap index)
        We will retrieve particle data based on the flags at this index
    
    halo_index_list : list
        List of the halo indices for which to calculate accretion rates. If 'None',
        find for all halos in the base_halo_data dictionary at the desired snapshot. 

    snap_gap : int
        How many snaps to skip back to when comparing particle lists.
        Initial snap for calculation will be snap-snap_gap. 

    snap_gap : int
        How many snaps to skip back to when comparing particle lists.
        Initial snap (s1) for calculation will be s1=snap-snap_gap, and we will check particle histories at s1-1. 

	Returns
	----------
    
    AccretionData_snap{snap2}_sg{snap_gap}_fg{fidelity_gap}_ihalo_xxxxxx_xxxxxx_outname.hdf5: hdf5 file with datasets

        '/PartTypeX/ihalo_xxxxxx/ParticleID': ParticleID (in particle data for given type) of all accreted particles (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/Fidelity': Whether this particle stayed at the given fidelity gap (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/PreviousHost': Which structure was this particle host to (-1 if not in any fof object) (length: n_new_particles)
        '/PartTypeX/ihalo_xxxxxx/TotalDeltaN': Total gross particle growth (length: 1)
        '/PartTypeX/ihalo_xxxxxx/UnprocessedDeltaN': Unprocessed particle growth (length: 1)
        '/PartTypeX/ihalo_xxxxxx/TotalDeltaM': Total gross mass growth in physical Msun (length: 1)
        '/PartTypeX/ihalo_xxxxxx/UnprocessedDeltaM': Unprocessed mass growth in physical Msun (length: 1)

        Where there will be n_halos ihalo datasets. 

        '/Header': Contains attributes: "t1","t2","dt","z_ave","lt_ave"

    
    """
    #Initialising halo index list
    if halo_index_list==None:
        halo_index_list_snap2=list(range(len(base_halo_data[snap]["hostHaloID"])))#use all halos if not handed halo index list
    else:
        halo_index_list_snap2=halo_index_list

    #Assigning snap
    if snap==None:
        snap=len(base_halo_data)-1#if not given snap, just use the last one
    
    snap1=snap-snap_gap
    snap2=snap
    snap3=snap+fidelity_gap

    halo_index_list_snap1=[find_progen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=snap_gap) for ihalo in halo_index_list_snap2]
    halo_index_list_snap3=[find_descen_index(base_halo_data,index2=ihalo,snap2=snap2,depth=fidelity_gap) for ihalo in halo_index_list_snap2]

    print(np.column_stack((halo_index_list_snap1,halo_index_list_snap2,halo_index_list_snap3)))

    #Initialising outputs
    run_outname=base_halo_data[snap]['outname']
    if test_run:
        if not os.path.exists('acc_data_test'):
            os.mkdir('acc_data_test')
        outfile_name='acc_data_test/AccretionData_snap'+str(snap).zfill(3)+'_sg'+str(snap_gap)+'_fg'+str(fidelity_gap)+'_ihalo'+str(halo_index_list[0]).zfill(6)+"_"+str(halo_index_list[-1]).zfill(6)+"_"+run_outname+'_test.hdf5'
    else:
        if not os.path.exists('acc_data'):
            os.mkdir('acc_data')
        outfile_name='acc_data/AccretionData_snap'+str(snap).zfill(3)+'_sg'+str(snap_gap)+'_fg'+str(fidelity_gap)+'_ihalo'+str(halo_index_list[0]).zfill(6)+"_"+str(halo_index_list[-1]).zfill(6)+"_"+run_outname+'.hdf5'
    
    part_filetype=base_halo_data[snap]['Part_FileType']

    # Particle types from sim type
    PartNames=['gas','DM','','','star','BH']
    if part_filetype=='EAGLE':
        PartTypes=[0,1,4,5] #Gas, DM, Stars, BH
        SimType='EAGLE'
    else:
        PartTypes=[0,1] #Gas, DM
        SimType='OtherHydro'

    if part_filetype=='EAGLE':
        print('Reading in EAGLE snapshot data ...')
        EAGLE_boxsize=base_halo_data[snap]['SimulationInfo']['BoxSize_Comoving']
        EAGLE_Snap_1=read_eagle.EagleSnapshot(base_halo_data[snap1]['Part_FilePath'])
        EAGLE_Snap_1.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)
        EAGLE_Snap_2=read_eagle.EagleSnapshot(base_halo_data[snap2]['Part_FilePath'])
        EAGLE_Snap_2.select_region(xmin=0,xmax=EAGLE_boxsize,ymin=0,ymax=EAGLE_boxsize,zmin=0,zmax=EAGLE_boxsize)

        snap_1_masses=dict()
        snap_2_masses=dict()

        for itype in PartTypes:
            if not itype==1:#everything except DM
                snap_1_masses[str(itype)]=EAGLE_Snap_1.read_dataset(itype,"Mass")*10**10
                snap_2_masses[str(itype)]=EAGLE_Snap_2.read_dataset(itype,"Mass")*10**10
            else:#DM
                hdf5file=h5py.File(base_halo_data[snap1]['Part_FilePath'])
                dm_mass=hdf5file['Header'].attrs['MassTable'][1]*10**10
                snap_1_masses[str(itype)]=dm_mass
                snap_2_masses[str(itype)]=dm_mass
        print('Done reading in EAGLE snapshot data')
       
    else:#assuming constant mass (convert to physical!)
        hdf5file=h5py.File(base_halo_data[snap1]['Part_FilePath'])
        snap_1_masses=dict()
        snap_2_masses=dict()
        masses_0=hdf5file["Header"].attrs["MassTable"][0]
        masses_1=hdf5file["Header"].attrs["MassTable"][1]
        snap_1_masses[str(0)]=masses_0
        snap_1_masses[str(1)]=masses_1


    #Load in particle histories
    print(f'Retrieving & organising particle histories for snap = {snap1} ...')
    Part_Histories_File_snap1=h5py.File("part_histories/PartHistory_"+str(snap1).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap1=[Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes]
    Part_Histories_Index_snap1=[Part_Histories_File_snap1["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes]
    Part_Histories_HostStructure_snap1=[Part_Histories_File_snap1["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes]
    print(f'Done retrieving & organising particle histories for snap = {snap1}')

    print(f'Retrieving & organising particle histories for snap = {snap2} ...')
    Part_Histories_File_snap2=h5py.File("part_histories/PartHistory_"+str(snap2).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap2=[Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes]
    Part_Histories_Index_snap2=[Part_Histories_File_snap2["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes]
    Part_Histories_HostStructure_snap2=[Part_Histories_File_snap2["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes]
    print(f'Done retrieving & organising particle histories for snap = {snap2}')

    print(f'Retrieving & organising particle histories for snap = {snap3} ...')
    Part_Histories_File_snap3=h5py.File("part_histories/PartHistory_"+str(snap3).zfill(3)+"_"+run_outname+".hdf5",'r')
    Part_Histories_IDs_snap3=[Part_Histories_File_snap3["PartType"+str(parttype)+'/ParticleIDs'] for parttype in PartTypes]
    Part_Histories_Index_snap3=[Part_Histories_File_snap3["PartType"+str(parttype)+'/ParticleIndex'] for parttype in PartTypes]
    Part_Histories_HostStructure_snap3=[Part_Histories_File_snap3["PartType"+str(parttype)+'/HostStructure'] for parttype in PartTypes]
    print(f'Done retrieving & organising particle histories for snap = {snap3}')


    #Load in particle lists from VR
    print('Retrieving VR halo particle lists ...')
    snap_1_halo_particles=get_particle_lists(base_halo_data[snap1],halo_index_list=halo_index_list_snap1,include_unbound=True,add_subparts_to_fofs=True)
    snap_2_halo_particles=get_particle_lists(base_halo_data[snap2],halo_index_list=halo_index_list_snap2,include_unbound=True,add_subparts_to_fofs=True)
    snap_3_halo_particles=get_particle_lists(base_halo_data[snap3],halo_index_list=halo_index_list_snap3,include_unbound=True,add_subparts_to_fofs=True)
    # print(snap_1_halo_particles)
    print('Done loading VR halo particle lists')

    count=0    
    subhalos=set(np.where(base_halo_data[snap]['hostHaloID']>0)[0])
    fieldhalos=set(np.where(base_halo_data[snap]['hostHaloID']>0)[0])

    for iihalo,ihalo_s2 in enumerate(halo_index_list):# for each halo at snap 2
        subhalo=int(base_halo_data[snap]['hostHaloID'][ihalo_s2]>0)#flag as to whether this is a subhalo(True) or a field halo(False)
        processed_flag=subhalo+1#1 if field halo, 2 if subhalo
        ihalo_s1=halo_index_list_snap1[iihalo]
        ihalo_s3=halo_index_list_snap3[iihalo]
        print('Halo index: ',ihalo_s2)
        print(f'Progenitor: {ihalo_s1}, descendent: {ihalo_s3}')
        if ihalo_s1>0 and ihalo_s3>0:# if we found both the progenitor and the descendent 
            count=count+1
            print(ihalo_s2)
            print(snap_1_halo_particles['Particle_IDs'])
            snap1_IDs_temp=snap_1_halo_particles['Particle_IDs'][ihalo_s2]
            snap1_Types_temp=snap_1_halo_particles['Particle_Types'][ihalo_s2]
            snap2_IDs_temp=snap_2_halo_particles['Particle_IDs'][ihalo_s2]
            snap2_Types_temp=snap_2_halo_particles['Particle_Types'][ihalo_s2]
            snap3_IDs_temp=snap_3_halo_particles['Particle_IDs'][ihalo_s2]
            snap3_Types_temp=snap_3_halo_particles['Particle_Types'][ihalo_s2]

            #returns mask for s2 of particles which were not in s1
            print(f"Finding new particles to ihalo {ihalo_s2} ...")
            new_particle_IDs_mask_snap2=np.in1d(snap2_IDs_temp,snap1_IDs_temp,invert=True)

            #returns mask for s1 of particles which are in s1 but not s2          
            # lost_particle_IDs_mask_snap1=np.in1d(snap1_IDs_temp,snap2_IDs_temp,invert=True)

            for iitype,itype in enumerate(PartTypes):
                
                print(f"Compressing for new particles of type {itype} ...")

                new_particle_mask_itype=np.logical_and(new_particle_IDs_mask_snap2,snap2_Types_temp==itype)
                new_particle_IDs_itype_snap2=np.compress(new_particle_mask_itype,snap2_IDs_temp)
                # lost_particle_mask_itype=np.logical_and(lost_particle_IDs_mask_snap1,snap1_Types_temp==itype)
                # lost_particle_IDs_itype_snap1=np.compress(lost_particle_mask_itype,snap1_IDs_temp)

                print(f"Finding index of accreted particles in halo {ihalo_s2} of type {itype}: n = {len(new_particle_IDs_itype_snap2)}")
                if itype==1:#DM:
                    new_particle_IDs_itype_snap2_historyindex=np.searchsorted(a=Part_Histories_IDs_snap2[iitype],v=new_particle_IDs_itype_snap2)
                    new_particle_IDs_itype_snap1_historyindex=np.searchsorted(a=Part_Histories_IDs_snap1[iitype],v=new_particle_IDs_itype_snap2)
                    #particle_masses
                    new_particle_masses=np.ones(len(new_particle_IDs_itype_snap2))*snap_2_masses[str(itype)]
                    #pre-processed
                    previous_hostIDs=[Part_Histories_HostStructure_snap1[history_index] for history_index in new_particle_IDs_itype_snap1_historyindex]


            
                elif itype==0:#Gas
                    new_particle_IDs_itype_snap2_historyindex=binary_search_1(sorted_array=Part_Histories_IDs_snap2[iitype],elements=new_particle_IDs_itype_snap2)
                    new_particle_IDs_itype_snap1_historyindex=binary_search_1(sorted_array=Part_Histories_IDs_snap1[iitype],elements=new_particle_IDs_itype_snap2)
                    #particle_masses
                    new_particle_masses=[snap_2_masses[""]]

        else:
            #### return nan accretion rate

            pass






     