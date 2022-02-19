import numpy as np
from collections import Counter
import math
from typing import Iterable


# -----------------------------------------------
#   ARRAY MANIPULATION
# -----------------------------------------------
def is_unique(arr,rate=False):
    """
    Return a boolean value for whether all elements in the array is unique, or as a rate
    """
    if rate:
        return len(np.unique(arr))/len(arr)
    else:
        return len(np.unique(arr)) == len(arr)

def overlap(*arrs):
    """
    Return the set of overlapped elements, i.e. the intersection, in the form of an array
    """
    overlap_set = set(arrs[0])
    for arr in arrs[1:]:
        overlap_set = overlap_set.intersection(set(arr))
    return np.array(list(overlap_set))

def union(*arrs):
    """
    Return the union set of all arrays, in the form of an array
    """
    union_set = set()
    for arr in arrs:
        union_set = union_set.union(set(arr))
    return np.array(list(union_set))

def difference(*arrs,how="outer"):
    """
    Return the difference elements in the form of an array
    Parameters:
        - how = `"outer"`: difference between the union set with the overlap set, i.e., elements that do not appear in all sets
        - how = `"left"`: differece between the left (or first) set with every other set
        - how = `"right"`: differece between the right (or last) set with every other set

    """
    difference_sets = {
        "outer": set(union(*arrs)) - set(overlap(*arrs)),
        "left" : set(arrs[0]) - set(overlap(*arrs)),
        "right": set(arrs[-1]) - set(overlap(*arrs))
    }
    return np.array(list(difference_sets[how]))

def label_counts(arr,labels=None):
    """ 
    @Description: Return a dict of label_counts and labels of a list-like object
    @Parameters:
        - arr: list-like object (LLO)
        - labels: labels to be included in the counting, even those not in the LLO. Order-sensitive
    """
    arr_ = np.array(arr).ravel() # Omitted dtype=object
    labels = np.unique(arr_) if isNone(labels) else np.array(labels)
    return {'counts':np.array([Counter(arr_)[lab] for lab in labels]),'labels':labels,'num_classes':len(labels)}

def as_1d_array(arr):
    """
    Ensure the object is a 1D array. 
    
    Typically used in a `for` loop when the object is not guaranteed to be an `Iterable`
    """

    if isinstance(arr,Iterable):
        if isinstance(arr,str) or len(np.shape(arr)) == 0:
            return np.array([arr])
        else:
            return np.ravel(arr)
    else:
        return np.array([arr])

# -----------------------------------------------
#   DATA TYPES MANIPULATION
# -----------------------------------------------

def dict_subset(dict_obj, keys):
    """
    Return a subset of the dict object based on the keys
    """
    return {key:dict_obj[key] for key in list(keys)}


def enc_str_fr_np(arr,sep=',',br="[|]"):
    """
    Encode an array as a string
    """
    if isNone(sep): sep = ""
    if isNone(br):
        return f"{sep.join(np.array(arr).astype(str))}"
    else:
        return f"{br[0]}{sep.join(np.array(arr).astype(str))}{br[-1]}"

# def np_fr_str(string,dtype=int):
#     return np.array(list(string)).astype(dtype)

def dec_np_fr_str(string,dtype=int,sep=',',br="[|]"):
    """
    Decode an array from a string
    """
    if isNone(sep):
        strings = string.strip(br)
    else: 
        strings = string.strip(br).split(sep)
    return np.array([dtype(n) for n in strings])

# -----------------------------------------------
#   LOGIC FUNCTIONS
# -----------------------------------------------
def isNone(var):
    """
    @Description: Check if a value is None. The typical boolean expression `var == None` may give rise to error when var is a list/array
    """
    return isinstance(var,type(None))

def default_value(var,default_val,default_trigger=None):
    """
    @Description: Set a value if variable is default value or another default trigger
    """
    if isNone(default_trigger):
        return default_val if isNone(var) else var
    else:
        return default_val if (type(var) != type(default_trigger)) or (var == default_trigger) else var

def converse(var,choices):
    assert len(choices)==2, "The converse of more than 2 choices is ambiguous"
    return choices[0] if var == choices[1] else choices[1]


# -----------------------------------------------
#   MATH FUNCTIONS
# -----------------------------------------------
def clamp(val,lower=0,upper=1,default_nan=0):
    """
    @Description: Clamp a numerical value between lower and upper. 
    """
    return max(lower,min(val,upper))


# -----------------------------------------------
#   STRING FORMATTING
# -----------------------------------------------

def fmt_time(seconds):
    """
    Convert time in seconds to a string representation of the format hh:mm:ss
    """
    seconds = int(seconds)
    hours = math.floor(seconds / 3600)
    minutes = math.floor((seconds % 3600) / 60)
    odd_secs = seconds % 60
    if hours < 10: 
        hh = f'0{hours}' 
    else: 
        hh = f'{hours}'
    if minutes < 10 : 
        mm = f'0{minutes}'
    else : 
        mm = f'{minutes}'
    if odd_secs < 10 : 
        ss = f'0{int(odd_secs)}'
    else: 
        ss = f'{int(odd_secs)}'
    return f'{hh}:{mm}:{ss}'
