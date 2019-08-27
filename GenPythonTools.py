
import numpy as np
import pickle as pickle
from bisect import bisect_left

########################### INDEX LISTS GENERATOR FOR MP ###########################

def gen_indices_mp(index_list,n_processes):
    """

    gen_halo_indices_mp : function
	----------

    Generate list of lists of desired indices divided amongst a given amount of processes.

	Parameters
	----------
    index_list : list or int
        If list, a list of integer halo indices to divide.
        If int, a list of integer halo indices up to the int is generated.

    n_processes : int
        Number of processes (likely number of cores) to distribute halo indices across. 

    Returns
	----------
    output_index_lists : list of lists
        The resulting halo index lists for each process. 

    """
    # Create halo index list from integer or provided list
    if type(index_list)==int:
        index_list=list(range(index_list))
    else:
        index_list=list(index_list)

    n_indices=len(index_list)
    num_rem=n_indices%n_processes
    n_indices_per_process=int(n_indices/n_processes)

    #initialising loop variables
    last_index=0
    i_index_lists=[]
    output_index_lists=[]

    #loop for each process to generate halo index lists
    for iprocess in range(n_processes):
        if num_rem==0: #if there's an exact multiple of halos as cpu cores then distribute evenly
            indices_temp=list(range(iprocess*n_indices_per_process,(iprocess+1)*n_indices_per_process))
            i_index_lists.append(indices_temp)
            index_list_temp=[index_list[index_temp] for index_temp in indices_temp]
            output_index_lists.append(index_list_temp)

        else: #otherwise split halos evenly except last process
            if iprocess<num_rem:
                indices_temp=list(range(last_index,last_index+n_indices_per_process+1))
                i_index_lists.append(indices_temp)
                last_index=indices_temp[-1]+1
                index_list_temp=[index_list[index_temp] for index_temp in indices_temp]
                output_index_lists.append(index_list_temp)

            else:
                indices_temp=list(range(last_index,last_index+n_halos_per_process))
                i_index_lists.append(indices_temp)
                last_index=indices_temp[-1]+1
                index_list_temp=[index_list[index_temp] for index_temp in indices_temp]
                output_index_lists.append(index_list_temp)

    return output_index_lists


def open_pickle(path):
    with open(path,'rb') as picklefile:
        pickledata=pickle.load(picklefile)
        picklefile.close()

    return pickledata

def dump_pickle(data,path):
    with open(path,'wb') as picklefile:
        pickle.dump(data,picklefile)
        picklefile.close()
    return data

def binary_search_1(elements,sorted_array):
    expected_indices=np.searchsorted(sorted_array,elements)
    expected_indices_checked=[]
    for ielement,expected_index in enumerate(expected_indices):
        element_at_expected_index=sorted_array[expected_index]
        if element_at_expected_index==elements[ielement]:
            expected_indices_checked.append(expected_index)
        else:
            expected_indices_checked.append(np.nan)
    return expected_indices_checked

def binary_search_2(element,sorted_array, lo=0, hi=None):   # can't use a to specify default for hi
    hi = hi if hi is not None else len(sorted_array) # hi defaults to len(a)   
    expected_index = bisect_left(sorted_array,element,lo,hi)          # find insertion position
    element_at_expected_index=sorted_array[expected_index]
    if element_at_expected_index==element:
        return expected_index
    else:
        return np.nan





