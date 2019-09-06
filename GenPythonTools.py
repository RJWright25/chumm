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
                                                                                                                                                                  
                                                                                                                                                                  
# GenPythonTools.py - Miscellaneous python tools for use in the rest of the package. 
# Author: RUBY WRIGHT 

# PREAMBLE
import numpy as np
import pickle as pickle
from bisect import bisect_left

def flatten(listoflists):
    output=[]
    for sublist in listoflists:
        if not (type(sublist)==list or type(sublist)==np.ndarray):
            sublist=np.nan
            output.extend([np.nan])
        else:
            output.extend(list(sublist))
    return output


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
    n_indices_per_process=np.floor(n_indices/n)*np.ones(n)

    for irem in range(n_rem):
        n_indices_per_process[irem]=int(n_indices_per_process[irem]+1)
    n_indices_per_process=np.array(n_indices_per_process).astype(int)

    output=[]
    for iprocess in range(n):
        n_indices_iprocess=n_indices_per_process[iprocess]
        iprocess_indices=[]
        for iprocess_iindex in range(n_indices_iprocess):
            iprocess_index=iprocess_iindex*n+iprocess
            iprocess_indices.append(indices[iprocess_index])
        output_iprocess={"iprocess":iprocess,"indices":list(iprocess_indices)}
        output.append(output_iprocess)

    return output

def open_pickle(path):
    """

    open_pickle : function
	----------

    Open a (binary) pickle file at the specified path, close file, return result.

	Parameters
	----------
    path : str
        Path to the desired pickle file. 


    Returns
	----------
    output : pickled object


    """

    with open(path,'rb') as picklefile:
        pickledata=pickle.load(picklefile)
        picklefile.close()

    return pickledata

def dump_pickle(data,path):
    """

    dump_pickle : function
	----------

    Dump data to a (binary) pickle file at the specified path, close file.

	Parameters
	----------
    data : any type
        The object to pickle. 

    path : str
        Path to the desired pickle file. 


    Returns
	----------
    None

    Creates a file containing the pickled object at path. 

    """

    with open(path,'wb') as picklefile:
        pickle.dump(data,picklefile)
        picklefile.close()
    return data

def binary_search_1(elements,sorted_array):
    """

    binary_search_1 : function
	----------

    Search a sorted array for the desired elements and return their indices 
    (if the elements are at their expected position) - uses np.searchsorted

	Parameters
	----------
    elements : list or list-like
        The elements to search for in sorted_array.  

    sorted_array : list list-like
        The array in which to search for the elements. Must be sorted in ascending order. 


    Returns
	----------
    indices : list

    A list of the indices (or np.nan if element not found) of each element
    in sorted array (in order of each element in elements)

    """

    expected_indices=np.searchsorted(sorted_array,elements)
    expected_indices_checked=[]
    for ielement,expected_index in enumerate(expected_indices):
        element_at_expected_index=sorted_array[expected_index]
        if element_at_expected_index==elements[ielement]:
            expected_indices_checked.append(expected_index)
        else:
            expected_indices_checked.append(np.nan)
    return expected_indices_checked

def binary_search_2(element,sorted_array, lo=0, hi=None):   
    """

    binary_search_2 : function
	----------

    Search a sorted array for the desired element and return its index 
    (if the element is at its expected position) - uses bisect package

	Parameters
	----------
    elements : list or list-like
        The elements to search for in sorted_array.  

    sorted_array : list list-like
        The array in which to search for the elements. Must be sorted in ascending order. 


    Returns
	----------
    index : list

    The index (or np.nan if element not found) of each element
    in sorted array (in order of each element in elements)
    
    """

    hi = hi if hi is not None else len(sorted_array) # hi defaults to len(a)   
    expected_index = bisect_left(sorted_array,element,lo,hi)         # find insertion position
    try:
        element_at_expected_index=sorted_array[expected_index]
        if element_at_expected_index==element:
            return expected_index
        else:
            return np.nan
    except:
        return np.nan

