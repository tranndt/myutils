from typing import Iterable
import numpy as np
import pandas as pd
from .utils import *

#----------------------------------------
#   EXTRACT DATA FRAME INFORMATION
#----------------------------------------

def summary(df:pd.DataFrame|pd.Series) -> pd.DataFrame:
    """
    Summary table for each columns in the dataframe
    """
    df_sum = pd.DataFrame(index=df.columns,columns=["dtypes","length","unique","samples","mode","range","mean","std","fill"])
    df_sum.index.name = "columns"
    for c in df.columns:
        df_c_notna = df[c][df[c].notna()]
        df_sum.loc[c,"dtypes"] = df[c].dtypes
        df_sum.loc[c,"samples"] = (list(df[c].value_counts(ascending=False).index.values[:5]))
        df_sum.loc[c,"length"] = len(df[c])
        df_sum.loc[c,"unique"] = len(df[c].unique())
        df_sum.loc[c,"mode"] = df_c_notna.mode()[0]
        df_sum.loc[c,"fill"] = np.round(len(df_c_notna)/len(df[c]),2)
        if isinstance(df[c].dtype,(type(np.dtype("float64")),type(np.dtype("int64")))):
            df_sum.loc[c,"range"] = ([df_c_notna.min(),df_c_notna.max()])
            df_sum.loc[c,"mean"] = df_c_notna.mean().round(2)
            df_sum.loc[c,"std"] = df_c_notna.std().round(2)
    return df_sum

def compare(df1 : pd.DataFrame|pd.Series, df2: pd.DataFrame|pd.Series,numeric: bool=False) -> pd.DataFrame:
    """
    Compare the 2 DataFrames along the same indexes and columns. 
        - For numeric columns, return the difference. 
        - For object columns, return whether the values match
    """
    overlapped_index = overlap(df1.index,df2.index)
    overlapped_columns = overlap(df1.columns,df2.columns)
    index = df1.index if set(df1.index) == set(overlapped_index) else overlapped_index
    columns = df1.columns if set(df1.columns) == set(overlapped_columns) else overlapped_columns
    df1_comp = df1.loc[index,columns]
    df2_comp = df2.loc[index,columns]
    df_comp = pd.DataFrame(index=index,columns=columns)

    if numeric:
        numeric_cols = []
        for c in columns:
            if is_np_numeric(df1[c]) and is_np_numeric(df2[c]): 
                numeric_cols.append(c)
                df_comp.loc[index,c] = (df1_comp.loc[index,c] - df2_comp.loc[index,c])
        return df_comp.loc[index,numeric_cols]

    else:
        for c in columns:
            df_comp.loc[index,c] = (df1_comp.loc[index,c] == df2_comp.loc[index,c]) | ((df1_comp.loc[index,c] != df2_comp.loc[index,c]) 
                                                                                        & (df1_comp.loc[index,c].isna() == df2_comp.loc[index,c].isna())
                                                                                        & df1_comp.loc[index,c].isna() 
                                                                                        & df2_comp.loc[index,c].isna())
        return df_comp.loc[index,columns]

#----------------------------------------
# DATA FRAME MANIPULATION & TRANSFORMATION
#----------------------------------------

def filter(df: pd.DataFrame|pd.Series, condition: pd.DataFrame|pd.Series, filter_columns: bool=False) -> pd.DataFrame|pd.Series:
    """
    @Description: Filter a `DataFrame` object based on a match over one or multiple columns. Results can be filtered by rows or both rows and columns.
    @Parameters:
        - condition: boolean Series, typically a DataFrame expression involving one or more conditions. E.g., `df['A'] == 1` or `df[['B','C']] == [2,3]`
        - filter_columns: When filter_columns is False, only filter by rows, otherwise filter by both rows and columns
    """
    df_ = pd.DataFrame(df)
    filter_ = pd.DataFrame(condition)
    # Ensure that condition works over multiple columns matching
    match_all_columns = (pd.DataFrame(condition).sum(axis=1) == len(filter_.columns.values))
    condition =  match_all_columns if len(filter_.columns.values) >= 1 else condition 

    # Select columns to keep and rows to display based on condition
    columns_filt = filter_.columns.values if filter_columns else df_.columns.values
    index_filt = df_.loc[:,columns_filt][condition].dropna().index    
    return df_.loc[index_filt,columns_filt]

def per_class_sample(inputs: pd.DataFrame|pd.Series, targets: pd.DataFrame|pd.Series, 
                        sampling_dist: str|int|float|Iterable='min',random_state: int=None) -> tuple[pd.DataFrame|pd.Series, pd.DataFrame|pd.Series]:
    """
    ### Description: 
    Sample inputs based on a distribution of target labels. By default will attempt to sample all classes equally according to the least populous class.

    ### Parameters:
    - sampling_dist:
        - If sampling_dist is `None`: Sample all classes .
        - If sampling_dist is `min`: Attempt to sample all classes equally according to the least populous class.
        - If sampling_dist is type `int`: Attempt to sample all classes up to a maximum of label_dists
        - If sampling_dist is type `float` (within (0,1)): Attempt to sample all classes each with the proportion of label_dists
        - If sampling_dist is type `list`: Attempt to sample classes based on the distribution specified.
            - If a class distribution is `None`, all members of that class is sampled
            - If a class distribution is a fraction (within (0,1)), it will be understood as the class proportion
            - If a class distribution is an integer (>=1), it will be understood as class counts
    """
    # Convert sampling_dist into list of distribution if not already is
    counts,labels,num_labels = label_counts(targets).values()
    if isNone(sampling_dist):
        sampling_dist = counts
    if isinstance(sampling_dist,str) & sampling_dist == 'min':
        sampling_dist = min(counts)
    if isinstance(sampling_dist,(int,float)):
        sampling_dist = np.full(shape = labels.shape,fill_value = sampling_dist)

    sampled_index = pd.Index([])
    for labels_i,counts_i, sampling_dist_i in zip(labels,counts,sampling_dist):
        # Convert distribution values to actual class counts
        if isNone(sampling_dist_i): 
            dist_i = int(counts_i)
        elif 0 <= sampling_dist_i and sampling_dist_i < 1:
            dist_i = int(counts_i * sampling_dist_i)
        else:
            dist_i = int(clamp(sampling_dist_i,0,counts_i))
        # Obtain samples with the labels_i   
        sampled_targets_i = filter(targets,targets==labels_i).sample(dist_i,random_state=random_state)
        sampled_index = sampled_index.append(sampled_targets_i.index)

    return inputs.loc[sampled_index], targets.loc[sampled_index]


def match(df_in: pd.DataFrame|pd.Series, oper: str ,values: Iterable[int],strict: bool=False) -> pd.DataFrame|pd.Series:
    """
    Apply a comparison operation to a list of values and return all results that matches all elements in the list.

    In strict mode, only keep rows that satisfy all matching requirements 
    """
    mult_result = (df_in != df_in) if oper == "==" else (df_in == df_in)
    for val in np.array(values):
        mult_result = {
            "<=": mult_result & (df_in <= val), "<"  : mult_result & (df_in < val),
            ">=": mult_result & (df_in >= val), ">"  : mult_result & (df_in > val),
            "==": mult_result | (df_in == val), "!=" : mult_result & (df_in != val),
        } [oper]
    df_out = df_in[mult_result].dropna(how={True:"any",False:"all"}[strict])
    return df_out

#----------------------------------------
#   DATA TYPES
#----------------------------------------

def is_np_dtypes(series: pd.Series ,dtypes : str|tuple|Iterable) -> bool:
    """
    Check if a series is of one or multiple numpy dtypes
    - For `int` dtype, use `"int64"`
    - For `float` dtype, use  `"float64"`
    - For `object` dtype, use  `"O"`
    """
    return isinstance(series.dtypes,tuple([type(np.dtype(typ)) for typ in dtypes]))

def is_np_numeric(series: pd.Series) -> bool:
    """
    Check if a series is of numeric numpy dtypes
    """
    return is_np_dtypes(series,("float64","int64"))





