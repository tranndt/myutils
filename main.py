import numpy as np
import pandas as pd
from collections import Counter
import math
from typing import Iterable
import sys
import time
import random
import re
import inspect
import os
import torch
from os import walk

# -----------------------------------------------
#   DICT LIKE DUMMY OBJECT
# -----------------------------------------------
class PseudoObject():
    """
    Utility object class
    """
    def __init__(self,base={},__name__=None,**kwargs):
        super(PseudoObject,self)
        super().__setattr__("__name__",isNone(__name__,then=classname(self)))
        super().__setattr__("__dict__",{})
        self.update(**base)
        self.update(**kwargs)

    def update(self, base={},**kwargs):
        if base:
            self.__dict__.update(base)
            for key,value in base.items():
                super().__setattr__(str(key),value)   
        if kwargs:
            self.__dict__.update(kwargs)
            for key,value in kwargs.items():
                super().__setattr__(str(key),value)
        return self

    def dict(self,keys=None):
        """
        Return the dictionary for all or a subset of attributes
        """
        if isNone(keys):
            return self.__dict__
        else:
            return dict_subset(self.__dict__,keys)

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

    def to_frame(self,keys=None,index=None,**kwargs):
        """
        Return the DataFrame for all or a subset of attributes
        """
        index = isNone(index,then=[0])
        return pd.DataFrame(self.dict(keys),index=index,**kwargs)

    def to_series(self,keys=None,**kwargs):
        """
        Return the DataFrame for all or a subset of attributes
        """
        return pd.Series(self.dict(keys),**kwargs)

    def __str__(self):
        return f'{self.__name__} {self.__dict__}'

    def __getattribute__(self,__name):
        try:
            return super().__getattribute__(__name)
        except:
            return None

    def __setattr__(self,__name,__value):
        super().__setattr__(__name,__value)
        self.__dict__.update({__name : __value})

    def __repr__(self):
        return f"{self.__name__} {self.dict()}"


def set_attributes(obj,**kwargs):
    if kwargs:
        for attr in kwargs.keys():
            obj.__setattr__(attr,kwargs[attr])
    return obj
    
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

class TaskTimer:
    def __init__(self,id=None):
        self.id = id
        self.start()

    def start(self):
        self.start_= time.time()
        self.curr_ = self.start_
        self.prev_ = self.start_

    def step(self):
        self.prev_ = self.curr_
        self.curr_ = time.time()

    def execute(self,f,*args,**kwargs):
        """
        Record the time for executing a function.

        Return 
        ---------
        Return the time of execution followed by the function followed by the return value(s)
        """
        self.start()
        ret_val = f(*args,**kwargs)
        self.step()
        time_exec = self.total_time()
        return time_exec,ret_val


    def step_time(self):
        return self.curr_ - self.prev_

    def total_time(self):
        return self.curr_ - self.start_

    def fmt(self,seconds):
        """
        Convert time in seconds to a string representation of the format hh:mm:ss
        """
        format_digits = lambda x: f'0{int(x)}' if x < 10 else f'{int(x)}'
        hh = format_digits(seconds // 3600)
        mm = format_digits(seconds % 3600 // 60)
        ss = format_digits(seconds % 60)
        return f'{hh}:{mm}:{ss}'

# -----------------------------------------------
#   LOGIC FUNCTIONS
# -----------------------------------------------
def dir_to_file(filename):
    if "/" in filename:
        directory = filename[:len(filename) - filename[::-1].index("/")-1]
    else:
        directory = "."
    return directory

def makedir_to_file(write_to):
    directory = dir_to_file(write_to)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return directory


def isNone(var,then=None,els=None):
    """
    @Description: Check if a value is None. The typical boolean expression `if var == None` may give rise to error when var is a list/array.

    When `then` != `None` and/or `else_` != `None` 
    - return `then` if `var` == `None` 
    - return `els` if if `var` != `None` 

    """

    is_None = isinstance(var,type(None))
    then_is_None = isinstance(then,type(None))
    else_is_None = isinstance(els,type(None))

    if then_is_None and else_is_None:
        # isNone(var=None,then=None,els=None) -> True
        # isNone(var="Not None",then=None,els=None) -> False
        return is_None
    elif not then_is_None and is_None:
        # isNone(var=None,then="then outcome",els="else outcome") -> "then outcome"
        return then
    elif not else_is_None and not is_None:
        # isNone(var="Not None","then outcome",els="else outcome") -> "else outcome"
        return els
    else:
        # isNone(var=None,then=None,els="else outcome") -> None
        # isNone(var="Not None",then="then outcome",els=None) -> "Not None"
        return var

def converse(var,choices):
    assert len(choices)==2, "The converse of more than 2 choices is ambiguous"
    return choices[0] if var == choices[1] else choices[1]

def switch(arg=None,cases={},default=None):
    """
    Implement a switch-case function

    `cases` keys can either be a value or a boolean condition. When `arg` is not found as a key in cases or no condition holds true, defer to the `default` value 

    Ex:     
    ```
        a = 2
        switch( arg = a,
                cases = {
                    1: "Is 1",
                    2: "Is 2",
                    3: "Is 3"},
                default="Unknown")
        >> 'Is 2'
    """
    if True not in cases.keys():
        cases.update({True:default})

    if arg in cases.keys():
        return cases[arg]
    else:
        return cases[True]


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
    arr_copy = ravel(arr)
    if isNone(labels):
        labels = np.sort(np.unique(arr_copy))
    else:
        labels = np.sort(ravel(labels))
    counts = np.array([Counter(arr_copy)[lab] for lab in labels])

    lab_cnt_obj = PseudoObject(__name__ = "Label Counts",
        counts = counts,
        labels = labels,
        total_count = np.sum(counts),
        num_classes = len(labels),
    )
    return lab_cnt_obj

def ravel_list(arr,except_=()):
    res = as_iterable(arr,list)
    is_non_string_iterable = lambda x: isinstance(x,Iterable) and not isinstance(x,(str,*as_iterable(except_)))
    while True in apply(res,is_non_string_iterable): # While there exists a child of Iterable type and not a str
        temp = []
        for a in as_iterable(res):
            temp += as_iterable(a,list) # Converting a to a list makes concatenation easier
        res = temp
    return res

def ravel(arr,dtype=None,except_=()):
    """
    Ravel an Iterable of any depth

    Example
    --------
    ```
    arr = [[-1,0],[1,2,[3,[4,5,[6,[7],[8,[9]]]]],10]]
    ravel(arr)
    >> array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    arr = 1
    ravel(arr)
    >> array([1])

    arr = ["a",["string"],"is also", ["an",["iterable"]]]
    ravel(arr)
    >> array(['a', 'string', 'is also', 'an', 'iterable'], dtype='<U8')
    ```
    """
    res = ravel_list(arr,except_=except_)
    return np.array(res,dtype)

def get_array_iloc(arr,arr_true=None):
    """
    Get the indexes where a value exist in an array, compared to the indexes of a ground truth array if supplied
    """
    if isNone(arr_true):
        return np.where(arr)[0]
    else:
        enc,dec = create_mappings(arr_true)
        return pd.Series(enc).loc[arr].values

def as_iterable(a,astype=None,force=False):
    """
    Convert an object into a specific iterable type. 
    
    When the object is not an iterable or when `force`, add 1 list dimension around the object to make it an iterable 
    """
    if isinstance(a,Iterable) and not isinstance(a,str) and not force:
        try:
            if isinstance(a,torch.Tensor):
                a = a.detach().numpy()
            return astype(a)
        except:
            return a
    else:
        try:
            return astype([a])
        except:
            return [a]

# def apply(array,func=None,*args,**kwargs):
#     """
#     Apply a function over all elements of an array/iterable
#     """
#     if isinstance(array,Iterable) and not isNone(func):
#         try:
#             return type(array)([func(a,*args,**kwargs) for a in array])
#         except:
#             return [func(a,*args,**kwargs) for a in array]
#     else: 
#         return array
        
def cross_permutate(arr1,arr2,astype=tuple):
    return [as_iterable((*as_iterable(a1),a2),astype) for a1 in as_iterable(arr1) for a2 in as_iterable(arr2)]

def cross_permutations(*arrs, astype=tuple):
    res = arrs[0]
    for arr in arrs[1:]:
        res = cross_permutate(res,arr,astype)
    return res

def params_permutations(*arrs,keys=None):
    keys = isNone(keys,then=range(len(arrs)))
    return [dict(zip(keys,vals)) for vals in cross_permutations(*arrs)]


# -----------------------------------------------
#   DATA TYPES MANIPULATION
# -----------------------------------------------
def dtype(array):
    return np.array(array).dtype.__str__()

def as_dtype(array):
    _dtype_ = dtype(array) if not isinstance(array,str) else array
    return lambda x: np.array(x,dtype=_dtype_)


def classname(a):
    return f"{type(a).__name__}"

def dict_subset(dict_obj, keys):
    """
    Return a subset of the dict object based on the keys
    """
    subset = dict()
    for key in as_iterable(keys):
        if key in dict_obj.keys():
            subset.update({key:dict_obj[key]})
        else:
            subset.update({key:None})
    return subset

def dec_dict_fr_str(string):
    import json
    return json.loads(str(string).replace("'","\""))


def create_mappings(labels,targets=None):
    if isNone(targets):
        targets = list(range(len(labels)))
    encode_mappings = dict(zip(labels,targets))
    decode_mappings = dict(zip(targets,labels))
    return encode_mappings,decode_mappings

def mappings_transform(labels,mappings):
    return [mappings[inp] for inp in ravel(labels)]

# 

def enc_str_fr_np(arr,sep=',',br="[|]"):
    """
    Encode an array as a string
    """
    if isNone(sep): 
        sep = ""
    if isNone(br):
        return f"{sep.join(np.array(arr).astype(str))}"
    else:
        return f"{br[0]}{sep.join(np.array(arr).astype(str))}{br[-1]}"


def dec_np_fr_str(string,dtype=int,sep=',',br="[|]"):
    """
    Decode an array from a string
    """
    if isNone(sep):
        strings = str(string).strip(br)
    else: 
        strings = str(string).strip(br).split(sep)
    return np.array([dtype(n) for n in strings])

def dec_np_fr_str2(string,dtype=None,levels=1,sep="\s+",br="[|]| |\n|\t",nest_pattern="\]\s+\[|\]\["):
    """
    Split the string representation of a nested list into an n-d array
    """
    if levels <= 1:
        try:
            ret =  np.array(re.split(sep,string.strip(br)))
            if sep == "":
                return ret[1:-1].astype(dtype)
            else:
                return ret.astype(dtype)
        except:
            try:
                return dtype(string)
            except:
                return string
    else: # Verified to work for levels == [1,2]
        return np.array([dec_np_fr_str2(s,dtype,levels-1,sep,br) for s in re.split(nest_pattern,str(string))])

# -----------------------------------------------
#   MATH FUNCTIONS
# -----------------------------------------------
def clamp(val,lower=0,upper=1,default_nan=0):
    """
    @Description: Clamp a numerical value between lower and upper. 
    """
    return max(lower,min(val,upper))

def sorting_order(array,descending=True):
    list(zip(sorted(array,reverse=descending)))
    sorted_element_indexes = dict(zip(sorted(array,reverse=descending),list(range(len(array)))))
    return np.array([sorted_element_indexes[a] for a in array])

def density(df,normalized=True):
    if isinstance(df,pd.DataFrame):
        dens = df.notna().sum().sum()
        if normalized:
            dens = dens/df.size
    elif isinstance(df,pd.Series):
        dens = df.notna().sum()
        if normalized:
            dens = dens/len(df)
    else:
        dens = sum(np.array(df) != None)
        if normalized:
            dens = dens/len(df)
    return dens

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

def get_args(func):
    return inspect.getfullargspec(func).args


def write_file(string,write_to=None,mode='w'):
    """
    Write or append string to a file. Create the new path if necessary
    """
    directory = write_to[:len(write_to) - write_to[::-1].index("/")]
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if mode == 'a':
        string = "\n"+string
    with open(file=write_to,mode=mode) as f:
        f.write(string)
        f.close()

def read_file(filename):
    try:
        with open(file=filename,mode='r') as f:
            string = f.read()
            f.close()
        return string
    except:
        return None


# -----------------------------------------------
#   V1.1 Updates
# -----------------------------------------------
# def apply(array,func,*args,**kwargs):
#     """
#     Apply a function to all elements. Elements where the function can not be applied to (i.e., due to an exception) will be ignored from the output array
#     """
#     arr = as_iterable(array)
#     idxes = range(len(arr))
#     res = []
#     for i,v in zip(idxes,arr):
#         try:
#             res.append(func(i,v,*args,**kwargs))
#         except:
#             try:
#                 res.append(func(v,*args,**kwargs))
#             except:
#                 continue
#     return np.array(res)

# def all_which(array,condition,*args,**kwargs):
#     """
#     Return all elements that satisfy the condition
#     """
#     arr = as_iterable(array)
#     idxes = range(len(arr))
#     res = []
#     for i,v in zip(idxes,arr):
#         try:
#             if condition(i,v,*args,**kwargs):
#                 res.append(v)
#         except:
#             try:
#                 if condition(v,*args,**kwargs):
#                     res.append(v)
#             except:
#                 continue
#     return np.array(res)

# def which(array,condition,*args,**kwargs):
#     """
#     Return the first element that satisfy the condition
#     """
#     try:
#         return all_which(array,condition,*args,**kwargs)[0]
#     except:
#         return None


# # def all_where(array,condition,*args,**kwargs):
# #     """
# #     Return the all indexes that satisfy the condition
# #     """
# #     arr = as_iterable(array)
# #     idxes = range(len(arr))
# #     idx = []
# #     for i,v in zip(idxes,arr):
# #         try:
# #             if condition(i,v,*args,**kwargs):
# #                 idx.append(i)
# #         except:
# #             try:
# #                 if condition(v,*args,**kwargs):
# #                     idx.append(i)
# #             except:
# #                 continue
# #     return np.array(idx)

# def all_where(array,condition=lambda x:True,*args,**kwargs):
#     """
#     Return the all indexes that satisfy the condition

#     Update: 
#         - By default return the entire index of array
#         - If array or condition is an iterable of booleans, return indexes where values == `True`
#         - If condition is a value (`int`,`float` or `None`) then look for such values in the array
#         - Simplified the try-except expression using `tryf()`
#     """

#     arr = as_iterable(array)
#     idxes = range(len(arr))
#     idx = []
#     if isinstance(array,Iterable) and dtype(array) == "bool":
#         idx = np.where(array)[0]
#     elif isinstance(array,Iterable) and dtype(condition) == "bool":
#         idx = np.where(condition)[0]
#     elif type_of(condition).startswith(("int","float","None")):
#         temp = condition
#         condition = lambda x: x == temp
#         for i,v in zip(idxes,arr):
#             if tryf(condition,i,v,*args,**kwargs) and condition(i,v,*args,**kwargs):
#                 idx.append(i)
#             elif tryf(condition,v,*args,**kwargs) and condition(v,*args,**kwargs):
#                 idx.append(i)

#     return np.array(idx)

# def where(array,condition,*args,**kwargs):
#     """
#     Return the first index that satisfy the condition
#     """
#     try:
#         return all_where(array,condition,*args,**kwargs)[0]
#     except:
#         return None

# def at(array,iloc=None,loc=None):
#     """
#     Pandas-like array accessor method for generic iterable class. 

#     Array elements can be accessed using `iloc` as indexes or `loc` as a boolean array indexer

#     Example:
#     ---------
#     ```
#     array = range(10)
#     at(array,loc=[True,False,True,True,True,True])
#     >> array([0, 2, 3, 4, 5])

#     at(array,[3,5,3,12])
#     >> array([3,5,3])
#     ```
#     """
#     arr  = as_iterable(array)
#     loc  = isNone(loc, then = [True]*len(arr))
#     iloc = isNone(iloc,then = np.where(loc)[0]) 
#     res  = apply(iloc, lambda v: arr[v])
#     return np.array(res)

def iter_pairs(array):
    """
    Return the list of adjacent pairs of elements in the iterable
    """
    return apply(array, lambda i,a:(array[i],array[i+1]))


def sample(array,k=None,replacement=False,seed=None,**kwargs):
    """
    Sample `k` elements from an iterable object, with or without replacement
    """
    import random
    random.seed(seed)
    if isNone(k):
        res = array
    elif replacement:
        res = random.choices(as_iterable(array,list),k=k,**kwargs)
    else:
        res = random.sample(as_iterable(array,list),k)
    return np.array(res)



# def all(arr,condition,*args,**kwargs):
#     """
#     Whether all elements in the array satisfy the condition
#     """
#     return np.all(apply(arr,condition,*args,**kwargs))

# def any(arr,condition,*args,**kwargs):
#     """
#     Whether any element in the array satisfy the condition. 
    
#     To retrieve which elements satisfy the condition, use `which()` or `all_which()`
#     """
#     return np.any(apply(arr,condition,*args,**kwargs))


def num_range(num,step=1):
    """
    Return the range a number belongs in, whose two edges are multiples of `step`. The result is lower bound inclusive and upper bound exclusive
    """
    div = math.floor(num/step)
    return np.array([step*div,step*(div+1)])

def round_up(num,step=1):
    """
    Round a number to the nearest higher multiple of `step`
    """
    return num_range(num,step)[1]

def round_down(num,step=1):
    """
    Round a number to the nearest lower multiple of `step`
    """
    return num_range(num,step)[0]

def random_strings(length,n=1,alpha=None,replacement=False):
    """
    Create `n` random strings of length `length` from an alphabet `alpha`, with/without replacement

    Example:
    --------
    ```
    random_strings(3,9,alpha="123")

    >> array(['112', '333', '131', '132', '123', '113', '323', '313', '111'],dtype='<U3')
    ```
    """
    if isNone(alpha):
        alpha = list("abcdefghijklmnopqrstuvwxyz")
    elif isinstance(alpha,str):
        alpha = list(alpha)
    else:
        alpha = apply(alpha,str)

    res = []
    while len(res) < n:
        string = "".join(sample(alpha,length,True))
        if replacement or string not in res or len(alpha)**length < n:
            res.append(string)
    return np.array(res)


def tryf(f,*args,**kwargs):
    try:
        f(*args,**kwargs)
        return True
    except:
        return False

def tryf_return(els=None,f=None,*args,**kwargs):
    return f(*args,**kwargs) if tryf(f,*args,**kwargs) else els

def tryf_catch(f,*args,**kwargs):
    try:
        return f(*args,**kwargs)
    except Exception as e:
        return e

def type_of(a):
    return type(a).__name__.__str__()


def whether(a,value_or_condition=lambda x: True,*args,**kwargs):
    is_array_of_dtype = lambda x,dty: isinstance(x,Iterable) and dtype(x).startswith(dty)
    is_function       = lambda x: type_of(x).startswith("function")
    is_value          = lambda x: type_of(x).startswith(("int","float","None"))

    if is_array_of_dtype(a,"bool"):
        res = np.array(a)
    elif is_value(value_or_condition):
        res = np.array(a) == value_or_condition
    elif is_function(value_or_condition):
        res = apply(a,value_or_condition,*args,**kwargs)
    else:
        res = np.array(a) != None
    return res


def where(a,value_or_condition =lambda x: True,*args,**kwargs):
    is_array_of_dtype = lambda x,dty: isinstance(x,Iterable) and dtype(x).startswith(dty)
    is_function       = lambda x: type_of(x).startswith("function")
    is_value          = lambda x: type_of(x).startswith(("int","float","None"))

    if is_array_of_dtype(a,"bool"):
        res = np.array(a)

    elif is_array_of_dtype(value_or_condition,"bool"):
        res = np.where(value_or_condition)[0]

    elif is_value(value_or_condition):
        res = np.where(np.array(a) == value_or_condition)[0]

    elif is_function(value_or_condition):
        res = np.where(apply(a,value_or_condition,*args,**kwargs))[0]

    else:
        res = np.where(np.array(a) != None)[0]

    return res

def at(array,loc_or_iloc=lambda x: True):
    is_array_of_dtype = lambda x,dty: isinstance(x,Iterable) and dtype(x).startswith(dty)
    is_function       = lambda x: type_of(x).startswith("function")
    is_value          = lambda x: type_of(x).startswith(("int","float","None"))

    if is_function(loc_or_iloc) or is_value(loc_or_iloc):
        res = np.array(array)[where(array,loc_or_iloc)]

    elif is_array_of_dtype(loc_or_iloc,"bool"):
        res = np.array(array)[where(loc_or_iloc)]

    elif is_array_of_dtype(loc_or_iloc,"int"):
        res = np.array(array)[loc_or_iloc]

    else:
        res = np.array(array)
    return res

def any(a,value_or_condition =lambda x: True,*args,**kwargs):
    return np.any(whether(a,value_or_condition,*args,**kwargs))

def all(a,value_or_condition =lambda x: True,*args,**kwargs):
    return np.all(whether(a,value_or_condition,*args,**kwargs))


def apply(array,func,*args,**kwargs):
    """
    Apply a function to all elements. Elements where the function can not be applied to (i.e., due to an exception) will be ignored from the output array
    """
    arr = as_iterable(array)
    idxes = range(len(arr))
    res = []
    for i,v in zip(idxes,arr):
        
        if tryf(func,i,v,*args,**kwargs):
            res.append(func(i,v,*args,**kwargs))

        elif tryf(func,v,*args,**kwargs):
            res.append(func(v,*args,**kwargs))

    return np.array(res)


def printif(c=True,*args,**kwargs):
    if c:
        print(*args,**kwargs)

def flushif(c=True,*args,**kwargs):
    if c:
        sys.stdout.write(*args,**kwargs)
        sys.stdout.flush()

def func(x):
    return x*x

def get_files(path,ext=None,full_path=False):
    all_files_raw = sorted(next(walk(path), (None, None, []))[2])
    all_files = []
    for f in all_files_raw:
        if full_path and (not ext or f.endswith(ext)):
            all_files.append(f"{path}/{f}")
        elif (not ext or f.endswith(ext)):
            all_files.append(f)
    return all_files

def get_folders(path,full_path=False):
    all_folders_raw = sorted(next(walk(path), (None, [], None))[1])
    all_folders = []
    for f in all_folders_raw:
        if full_path and os.path.isdir(f"{path}/{f}"):
            all_folders.append(f"{path}/{f}")
        elif os.path.isdir(f"{path}/{f}"):
            all_folders.append(f)
    return all_folders
