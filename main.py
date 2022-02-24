import numpy as np
import pandas as pd
from collections import Counter
import math
from typing import Iterable
import sys
import time
import random
import re

# -----------------------------------------------
#   DICT LIKE DUMMY OBJECT
# -----------------------------------------------
class DictObj():
    """
    @Description: Object that serves as a namespace for related attributes within other classes
    @Example:
        cpa = CPA1()\n
        cpa.var1 = DictObj()\n
        cpa.var1.inputs = [1,2,3]\n
        print(cpa.var1.inputs)\n
        >> [1, 2, 3]\n
        cpa.var2 = DictObj({'a':1,'b':2})\n
        print(cpa.var2.a,cpa.var2.b)\n
        >> 1 2

    """
    def __init__(self,__name__="Dict Object",**kwargs):
        super(DictObj,self)
        self.__name__ = __name__
        self.dict_ = {}
        self.update(**kwargs)

    def update(self,**kwargs):
        if kwargs:
            self.dict_.update(kwargs)
            for key,value in kwargs.items():
                self.__setattr__(key,value)
    
    def dict(self,keys=None):
        """
        Return the dictionary for all or a subset of attributes
        """
        if isNone(keys):
            return self.dict_
        else:
            return dict_subset(self.dict_,keys)

    def keys(self):
        """
        Return the keys of attributes
        """
        return self.dict().keys()

    def values(self,keys=None):
        """
        Return the values for a subset of attributes
        """
        return self.dict(keys).values()

    def to_frame(self,keys=None,index=None):
        """
        Return the DataFrame for all or a subset of attributes
        """
        index = isNone(index,then=[0])
        return pd.DataFrame(self.dict(keys),index=index)

    def __str__(self):
        return f'{self.__name__} {self.dict_}'

# -----------------------------------------------
#   TIMER FOR PROCESSES/TASKS
# -----------------------------------------------

class ProcessTimer:
    def __init__(self) -> None:
        self.start_ = {}
        self.prev_ = {}
        self.curr_ = {}
        self.NEW_ID = 0;

    def start(self,job_id=None):
        job_id = self.NEW_ID if job_id is None else job_id
        self.start_[job_id] = time.time()
        self.curr_[job_id] = self.start_[job_id]
        self.prev_[job_id] = -1
        self.NEW_ID += 1

    def record(self,job_id=0):
        self.prev_[job_id] = self.curr_[job_id]
        self.curr_[job_id] = time.time()

    def execute(self,func,job_id=-1,return_val=True,**func_args):
        """
        Record the time for executing a function.

        Return 
        ---------
        Return the time of execution followed by the function followed by the return value(s)
        """
        self.start(job_id)
        val = func(**func_args)
        self.record(job_id)
        if return_val:
            return self.time_elapsed(job_id),val
        else:
            return self.time_elapsed(job_id)

    def step_elapsed(self,job_id=0):
        return -1 if (job_id not in self.prev_.keys() or job_id not in self.curr_.keys()) else self.curr_[job_id] - self.prev_[job_id]

    def time_elapsed(self,job_id=0):
        return -1 if (job_id not in self.start_.keys() or job_id not in self.curr_.keys()) else self.curr_[job_id] - self.start_[job_id]

# -----------------------------------------------
#   LOGIC FUNCTIONS
# -----------------------------------------------
def isNone(var,then=None,els=None):
    """
    @Description: Check if a value is None. The typical boolean expression `if var == None` may give rise to error when var is a list/array.

    When `then` != `None` and/or `else_` != `None` 
    - return `then` if `var` == `None` 
    - return `els` if if `var` != `None` 

    """
    is_None = isinstance(var,type(None))
    then_return = not isinstance(then,type(None))
    else_return = not isinstance(els,type(None))
    # then != None -> return then or else or original value
    # then == None -> return True or False
    if then_return:
        if is_None:
            return then
        elif else_return:
            return els
        else:
            return var
    else:
        return is_None

def converse(var,choices):
    assert len(choices)==2, "The converse of more than 2 choices is ambiguous"
    return choices[0] if var == choices[1] else choices[1]


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

    @Return: Dict entries
        - dict[`'counts'`]: Class counts in the order specified by `'labels'`
        - dict[`'labels'`]: The order in which the class counts are presented
        - dict[`'num_classes'`]: Number of classes in the array
    """
    arr_copy = as_1d_array(arr)
    if isNone(labels):
        labels = np.sort(np.unique(arr_copy))
    else:
        labels = np.sort(as_1d_array(labels))
    counts = np.array([Counter(arr_copy)[lab] for lab in labels])

    lab_cnt_obj = DictObj("Label Counts",
        counts = counts,
        labels = labels,
        total_count = np.sum(counts),
        num_classes = len(labels),
    )
    return lab_cnt_obj

def as_1d_array(arr,dtype=None):
    """
    Ensure the object is a 1D array. 
    
    Typically used in a `for` loop when the object is not guaranteed to be an `Iterable`
    """
    is_nd_iter = isinstance(arr,Iterable) and len(np.shape(arr)) > 0 and not isinstance(arr,str)

    if is_nd_iter:
        arr_1d = np.ravel(arr)
        if isNone(dtype): 
            dtype = arr_1d.dtype
        return arr_1d.astype(dtype)
    else:
        if isNone(dtype): 
            dtype = type(arr)
        return np.array([arr]).astype(dtype)

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

def dec_np_fr_str2(string,dtype=None,levels=1,sep="\s+",br="[|]| |\n|\t",nest_pattern="\]\s+\[|\]\["):
    """
    Split the string representation of a nested list into an n-d array
    """
    if levels <= 1:
        return np.array(re.split(sep,string.strip(br)),dtype=dtype)
    else: # Verified to work for levels == [1,2]
        return np.array([dec_np_fr_str2(s,dtype,levels-1,sep,br) for s in re.split(nest_pattern,string)])

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


# -----------------------------------------------
#   MOCK DATA & FUNCTIONS
# -----------------------------------------------
def random_task(total=30,rate=0.1):
    low = 1
    t = ProcessTimer()

    sys.stdout.write(f"Executing a mock job for total epochs = {total} at rate = {rate}\n")
    sys.stdout.flush()
    t.start()
    for epoch in range(total):
        time.sleep(0.1)
        result = random.random()        
        t.record()
        if result < low:
            low = result    
        sys.stdout.write(f"\r> epoch: {epoch+1}/{total} -- current: {result:.3f} -- best: {low:.3f} -- time elapsed: {fmt_time(t.time_elapsed())}")
        sys.stdout.flush()

# random_task(20)

def random_dataframe(low=2,high=None,size=None,random_state=None):
    np.random.seed(random_state)
    rd = np.random.randint(low,high,size)
    return pd.DataFrame(rd)