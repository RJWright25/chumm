
import numpy as np
import pickle as pickle
from bisect import bisect_left

########################### INDEX LISTS GENERATOR FOR MP ###########################

def gen_mp_indices(indices,n):
    """

    gen_mp_indices : function
	----------

    Generate list of lists of desired indices divided amongst a given amount of processes.

	Parameters
	----------
    indices : list or int
        If list, a list of integer halo indices to divide.
        If int, a list of integer halo indices up to the int is generated.

    n : int
        Number of processes (likely number of cores) to distribute halo indices across. 

    Returns
	----------
    output_index_lists : list of dict
        The resulting halo index lists for each process. 
        e.g. gen_mp_indices(indices=[0,1,2,3,4,5,6,7],n=4)
        will return [{"iprocess":0,"indices":[0,4]},{"iprocess":1,"indices":[1,5]} ... etc]

    """
    # Create halo index list from integer or provided list
    if type(indices)==int:
        indices=list(range(indices))
    else:
        indices=list(indices)

    n_indices=len(indices)
    n_rem=n_indices%n
    n_indices_per_process=np.floor(n_indices/n)

    


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





