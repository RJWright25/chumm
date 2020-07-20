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

# Preamble
import numpy as np
import pickle as pickle
import os
import subprocess
from bisect import bisect_left

def flatten(listoflists):
    """
    flatten : function
    ------------------
    Flatten a simple list of lists into a 1d list.

    Parameters
    ----------
    listoflists : list of lists
        List of lists to flatten.
    """

    output=[]
    for sublist in listoflists:
        if not (type(sublist)==list or type(sublist)==np.ndarray):
            output.append(np.nan)
        else:
            output.extend(list(sublist))
    return output

def gen_mp_indices(indices,n,test=False):
    """

    gen_mp_indices : function
	-------------------------

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
        output_iprocess={"iprocess":iprocess,"indices":list(iprocess_indices),"np":n,"test":test}
        output.append(output_iprocess)


    return output

def open_pickle(path):
    """

    open_pickle : function
	----------------------

    Open a (binary) pickle file at the specified path, close file, return result.

	Parameters
	----------
    path : str
        Path to the desired pickle file. 


    Returns
	----------
    output : data structure of desired pickled object

    """

    with open(path,'rb') as picklefile:
        pickledata=pickle.load(picklefile)
        picklefile.close()

    return pickledata

def dump_pickle(data,path):
    """

    dump_pickle : function
	----------------------

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
        pickle.dump(data,picklefile,protocol=4)
        picklefile.close()
    return data

def binary_search(items,sorted_list,algorithm=None,check_entries=False,verbose=False):
    """

    binary_search : function
	------------------------

    Search a sorted array for the desired elements and return their expected indices in the sorted list.
    Will return np.nan if check_entries is True and the element at the expected index is not the desired item. 

	Parameters
	----------
    items : list or list-like
        The elements to search for in sorted_array.  

    sorted_list : list list-like
        The array in which to search for the elements. Must be sorted in ascending order. 

    algorithm : int 
        0: np.searchsorted
        1: bisect in list comprehension

    check_entries : bool
        Ensure the entries in the sorted list are indeed the desired entries.


    Returns
	----------
    indices : list

    A list of the indices (or np.nan if element not found) of each element
    in sorted array (in order of each element in elements)

    """
    hi=len(sorted_list)
    n_items=len(items)
    if n_items==0:
        return []

    if algorithm==None:
        if n_items>250:
            a1=True
        else:
            a1=False
    elif algorithm==1:
        a1=True
    else:
        a1=False

    if a1:
        indices=list(np.searchsorted(a=sorted_list,v=items))

    else:
        indices=[bisect_left(sorted_list,item,lo=0,hi=hi) for item in items]

    if check_entries:
        items_at_calculated_indices=[]
        for supposed_index in indices:
            try:
                item_at_calculated_index=sorted_list[supposed_index]
            except:
                item_at_calculated_index=np.nan
            items_at_calculated_indices.append(item_at_calculated_index)

        incorrect_indices=np.where(items_at_calculated_indices!=items)[0]
        count=len(incorrect_indices)
        
        for incorrect_index in incorrect_indices:
            indices[incorrect_index]=np.nan
        # try:
        #     print(f'{100-count/len(indices)*100:.2f}% of entries were correct')
        # except:
        #     print("Couldn't print error indices")


    return indices

def rank_list(items):
    """

    rank_list : function
	--------------------

    Take a list and return a list with the corresponding ranking of the element in the list. 

    e.g. items=[0,1,2,3,4,5]
        ranks=[6,5,4,3,2,1] - 1 is the highest

	Parameters
	----------
    items : list or list-like
        The elements to rank.  

    Returns
	----------
    ranks : np.ndarray

    A list of the rank of each element in items. 

    """

    items=np.array(items)
    ranks=[]
    for item in items:
        num_elements_greater=int(np.sum(item<items))+1
        ranks.append(num_elements_greater)
    ranks=np.array(ranks)
    return ranks

def list_dir(path,only_outer=False):

    """

    list_dir : function
	-------------------

    List the contents of a directory with its path.


	Parameters
	----------
    path : str
        The path in which to list files.

    only_outer : bool
        Whether to only return the file objects in the current directory (if True, will not enter sub-directories).

    Returns
	----------
    dir_list : np.ndarray

    A list of tfiles in the directory. 

    """
    if not only_outer:
        stdout=subprocess.Popen(f'find {path} -type f',shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT, universal_newlines=True).stdout
        dir_list=[str(item)[:-1] for item in stdout]
    else:
        stdout=subprocess.Popen(f'find {path} -maxdepth 1',shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT, universal_newlines=True).stdout
        dir_list=[str(item)[:-1] for item in stdout]
    return dir_list

def mask_wnans(array,indices):
    """

    mask_wnans : function
	-------------------

    Return the desired elements of an list or list-like object based on indices, which may include some nans. 

	Parameters
	----------
    array : 1-d list or np.array
        The array to extract certain elements from.

    
    indices : 1-d list or np.array
        The desired elements to extract from array (may include nans).

    Returns
	----------
    output_array : np.ndarray

        The output - only including desired array elements. Indices requested from a nan will return a nan. 

    """

    indices=np.array(indices)
    array=np.array(array)
    array_shape=np.shape(array)
    if len(array_shape)>1:
        entry_size=array_shape[1]
        output_shape=(len(indices),entry_size)
    else:
        entry_size=0
        output_shape=(len(indices),)

    output_array=np.zeros(output_shape)+np.nan
    valid_indices=np.where(np.isfinite(indices))
    try:
        output_array[valid_indices]=array[(indices[valid_indices].astype(int),)]
    except:
        for valid_index in valid_indices[0]:
            try:
                output_array[valid_index]=array[indices[valid_index]]
            except:
                pass
    return output_array

def list_to_string(items,delimiter=','):
    """

    list_to_string : function
	-------------------

    Concetenate a list into a string object, with optional delimiter.

	Parameters
	----------
    items : 1-d list or np.array
        The array/list to convert to a string. 

    delimiter : str
        Optional delimiter to add between list elements.  

    Returns
	----------
    output : string

        The delimited string.

    """

    output=''
    for iitem,item in enumerate(items):
        if iitem==0:
            idelim=''
        else:
            idelim=delimiter
        if type(item)==float:
            output=output+idelim+f'{item:.3f}'
        else:
            output=output+idelim+str(item)
    if output=='':
        output='None'
    return output

def cart_to_sph(xyz):
    """

    cart_to_sph : function
	-------------------

    Convert cartesian coordinates (x, y, z) to spherical coordinates (r, azimuth, elevation)

	Parameters
	----------
    xyz : array or array-like
        2d array of cartesian coordinates.  

    Returns
	----------
    ptsnew : np.ndarray

        2d array of corresponding cartesian coordinates (r: [0,inf], azimuth: [-pi,pi], elevation: [-pi/2,pi/2]).
        
    """


    #returns r, azimuth (-pi,pi), elevation (-pi/2,pi/2)
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    # ptsnew[:,5] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,5] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,4] = np.arctan2(xyz[:,1], xyz[:,0]) #azimuth
    return ptsnew[:,3:]
