from typing import Iterable, Tuple
import numpy as np
import pandas as pd
import os
from .main import *

#----------------------------------------
#   READ AND WRITE DATAFRAMES
#----------------------------------------

def write_dataframe(df,write_to=None,**kwargs):
    """
    Write the dataframe according to the extension. Create the new path if necessary
    """
    if not isNone(write_to) and isinstance(write_to,str):
        directory = write_to[:len(write_to) - write_to[::-1].index("/")]
        if not os.path.isdir(directory):
            os.makedirs(directory)
        ext = write_to.split(".")[-1] # get the extension
        if ext == 'xlsx':
            df.to_excel(write_to,**kwargs)
        elif ext in ['tsv','tab']:
            df.to_csv(write_to,sep='\t',**kwargs)
        else:
            df.to_csv(write_to,**kwargs)
    return df

def read_dataframe(filename,index=None,unnamed_col=False,**kwargs):
    ext = filename.split(".")[-1] # get the extension
    if ext == 'xlsx':
        df = pd.read_excel(filename,**kwargs)
    elif ext in ['tsv','tab']:
        df = pd.read_csv(filename,sep='\t',**kwargs)
    else:
        df = pd.read_csv(filename,**kwargs)

    # Set index column if not None
    if not isNone(index) and index in df.columns:
        df = df.set_index(index)

    # Drop the typical 'Unnamed: 0' from the 
    UNNAMED = "Unnamed: 0"
    if UNNAMED in df.columns:
        if unnamed_col == True:
            pass
        elif unnamed_col == False:
            df = df.drop(columns=[UNNAMED])
        elif isinstance(unnamed_col,(str,int,float)):
            if unnamed_col in df.columns:
                unnamed_col = f"index: {unnamed_col}"
            df = df.rename(columns={UNNAMED:unnamed_col}) 

    return df

#----------------------------------------
#   EXTRACT DATA FRAME INFORMATION
#----------------------------------------

def summary(df:pd.DataFrame,nan_values=None) -> pd.DataFrame:
    """
    Summary table for each columns in the dataframe
    """
    df_sum = pd.DataFrame(index=df.columns,columns=["dtypes","length","unique","samples","mode","range","mean","std","fill"])
    df_sum.index.name = "columns"
    for c in df.columns:
        df_sum.loc[c,"dtypes"] = df[c].dtypes
        df_sum.loc[c,"samples"] = (list(df[c].value_counts(ascending=False,dropna=False).index.values[:5]))
        df_sum.loc[c,"length"] = len(df[c])
        df_sum.loc[c,"unique"] = len(df[c].unique())
        df_sum.loc[c,"mode"] = [] if len(df[c].mode()) == 0 else as_1d_array(df[c].mode())[0]

        df_c_notna = df[c][df[c].notna()]
        df_sum.loc[c,"fill"] = np.round(len(df_c_notna)/len(df[c]),2)
        if isinstance(df[c].dtype,(type(np.dtype("float64")),type(np.dtype("int64")))):
            df_sum.loc[c,"range"] = np.round([df_c_notna.min(),df_c_notna.max()],4)
            df_sum.loc[c,"mean"] = np.round(df_c_notna.mean(),4)
            df_sum.loc[c,"std"] = np.round(df_c_notna.std(),4)
    return df_sum

def compare(df1 : pd.DataFrame or pd.Series, df2: pd.DataFrame or pd.Series,numeric: bool=False) -> pd.DataFrame or pd.Series:
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
            if is_numeric(df1[c]) and is_numeric(df2[c]): 
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

def filter(df: pd.DataFrame or pd.Series, condition: pd.DataFrame or pd.Series, filter_columns: bool=False) -> pd.DataFrame or pd.Series:
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

def per_class_sample(inputs: pd.DataFrame, targets: pd.DataFrame or pd.Series, 
                        sampling_dist: str or int or float or Iterable='min',random_state: int=None) -> Tuple[pd.DataFrame or pd.Series, pd.DataFrame or pd.Series]:
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
    counts,labels = label_counts(targets).values(['counts','labels'])
    if isNone(sampling_dist):
        sampling_dist = counts
    if isinstance(sampling_dist,str) & sampling_dist == 'min':
        sampling_dist = min(counts)
    if isinstance(sampling_dist,(int,float,np.int64,np.float64)):
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


def match(df_in: pd.DataFrame or pd.Series, oper: str ,values: Iterable[int],strict: bool=False) -> pd.DataFrame or pd.Series:
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

def aggregate(df,identifiers,key,columns=None,action=None):    
    """
    Aggregate a subset of columns in a DataFrame along a key column
    
    action == [`sum`,`mean`,`mode`,`median`,`count`]
    """
    df_indexed = df.set_index(identifiers).sort_index()
    indexes = df_indexed.index.drop_duplicates()
    columns = isNone(columns,then = df_indexed.columns.drop(key))

    if isNone(action) or action == "count":
        df_aggregated = df.value_counts(identifiers).sort_index().reset_index(name="count")
    else:
        column_names = [f"{action}_{c}" for c in columns]
        df_aggregated = pd.DataFrame(index=indexes,columns=column_names)
        for index in indexes:
            if action == "mean":
                df_aggregated.loc[index,column_names] = df_indexed.loc[index,columns].mean().values
            elif action == "mode":
                df_aggregated.loc[index,column_names] = df_indexed.loc[index,columns].mode().values
            elif action == "median":
                df_aggregated.loc[index,column_names] = df_indexed.loc[index,columns].median().values
            elif action == "sum":
                df_aggregated.loc[index,column_names] = df_indexed.loc[index,columns].sum().values           
        df_aggregated = df_aggregated.reset_index()
    return df_aggregated

#----------------------------------------
#   DATA TYPES
#----------------------------------------

def is_dtypes(df: pd.DataFrame or pd.Series ,dtypes : str or Iterable) -> bool:
    """
    Check if a series is of one or multiple numpy dtypes
    - For `int` dtype, use `"int64"`
    - For `float` dtype, use  `"float64"`
    - For `object` dtype, use  `"O"`
    - For `boolean` dtype, use `"bool"`
    """

    if isinstance(df, pd.Series):
        return isinstance(df.dtypes,tuple([type(np.dtype(typ)) for typ in as_1d_array(dtypes)]))
    elif isinstance(df, Iterable):
        return isinstance(np.array(df).dtype,tuple([type(np.dtype(typ)) for typ in as_1d_array(dtypes)]))
    elif isinstance(df, pd.DataFrame):
        all_dtypes = True
        for c in df.columns:
            all_dtypes &= is_dtypes(df[c],dtypes) # Shallow recursion so should not affect performance
        return all_dtypes
    else:
        return False

def is_numeric(df: pd.DataFrame or pd.Series or Iterable) -> bool:
    """
    Check if a series is of numeric numpy dtypes
    """
    return is_dtypes(df,("float64","int64"))


def is_bool(df: pd.DataFrame or pd.Series or Iterable,binary_allowed: bool=False) -> bool:
    """
    Check if a DataFrame, Series or Iterable is of dtypes boolean.  
    """
    # First check the dtype information about the object if available
    is_dtype_bool = is_dtypes(df,"bool") 

    # In case the input dtype is not labeled as bool, or represented as a binary matrix instead
    can_check_count = binary_allowed or not is_numeric(df)  
    count_satisfied = False
    if isinstance(df,pd.DataFrame):
        count_satisfied = ((df == True)|(df == False)).sum().sum() == df.size
    elif isinstance(df,pd.Series):
        count_satisfied = ((df == True)|(df == False)).sum() == df.size
    elif isinstance(df,Iterable):
        arr = np.array(df,copy=True)
        np.place(arr,(arr == True)|(arr == False),True)
        count_satisfied = arr.sum() == len(arr)
        
    return is_dtype_bool or (can_check_count and count_satisfied) 

#----------------------------------------
#   FEATURES
#----------------------------------------


def features_density(df,sort=False):
    if isinstance(df,pd.DataFrame):
        density = summary(df)["fill"]
    elif isinstance(df,pd.Series):
        density = df.notna().replace({True:1,False:0})
    if sort:
        return density.sort_values(ascending=False)
    else:
        return density

def features_density_by_threshold(df,threshold=0.5):
    df_dens = features_density(df)
    if threshold > 1:
        threshold = threshold/len(df)
    return df_dens[df_dens >= threshold].sort_values(ascending=False)

def features_density_by_rank(df,top=None):
    df_dens = features_density(df)
    if isNone(top):
        top = len(df_dens)
    if isinstance(top,float):
        top = int(min(len(df_dens),len(df_dens)*top))
    return df_dens.sort_values(ascending=False)[:top]

def rows_density(df,sort=False):
    if isinstance(df,pd.DataFrame):
        density = as_binary_frame(df).sum(axis=1) / len(df.columns)
    elif isinstance(df,pd.Series):
        density = df.notna().replace({True:1,False:0})
    if sort:
        return density.sort_values(ascending=False)
    else:
        return density
    # return as_binary_frame(df).sum(axis=1) / len(df.columns)

def rows_density_by_threshold(df,threshold=0.5):
    df_dens = rows_density(df)
    if threshold > 1:
        threshold = threshold/len(df.columns)
    return df_dens[df_dens >= threshold].sort_values(ascending=False)

def rows_density_by_rank(df,top=None):
    df_dens = rows_density(df)
    if isNone(top):
        top = len(df_dens)
    if isinstance(top,float):
        top = int(min(len(df_dens),len(df_dens)*top))
    return df_dens.sort_values(ascending=False)[:top]

def as_binary_frame(X,bool=False):
    if bool:
        return X.notna()
    else:
        return X.notna().replace({True:1,False:0})

def as_binary_matrix(X,bool=False):
    return as_binary_frame(X,bool).values


#----------------------------------------
#   CLUSTERING
#----------------------------------------

def as_cluster_indexes(cluster_labels,index=None):
    cluster_idxes = []
    for i in np.sort(np.unique(cluster_labels)):
        cluster_iloc = np.where(cluster_labels==i)[0]
        if isNone(index):
            cluster_idxes.append(cluster_iloc)
        else:
            cluster_idxes.append(np.array(index)[cluster_iloc])
    return cluster_idxes

def cluster_counts(cluster_idxes_or_labels,y,drop_total=False):
    """
    Find the class make-up of each cluster
    """
    if isinstance(cluster_idxes_or_labels[0],(pd.Int64Index,np.ndarray)): # if is_indexes
        cluster_idxes = cluster_idxes_or_labels
    else:
        cluster_idxes = as_cluster_indexes(cluster_idxes_or_labels)

    classes, class_counts = label_counts(y).values(['labels','counts'])
    # Calculate the class distribution for each cluster
    results = pd.DataFrame()
    for clust_idx_i in cluster_idxes:
        labels_i,counts_i,total_i = label_counts(y[clust_idx_i],
                                                labels=classes).values(['labels','counts','total_count'])
        cluser_i_cnt = DictObj(
            **dict(zip(as_1d_array(labels_i,str),counts_i)),
            cluster_total = total_i
        )
        results = pd.concat([results,cluser_i_cnt.to_frame()],ignore_index=True)

    # Add the class counts info
    class_total = DictObj(
        **dict(zip(as_1d_array(classes,str),class_counts)),
        cluster_total = np.sum(class_counts)
    ).to_frame(index=["class_total"])
    results = pd.concat([results,class_total])

    # Finally, format the result
    results.index.name = "cluster"
    if drop_total:
        return results.drop(index="class_total",columns="cluster_total")
    else:
        return results

def bimatrix_cluster(X,clusterer,**kwargs):
    """
    Cluster the input based on features' (X) existence rather than their actual content.
    """
    # Fit the clusterer to the binary matrix of X
    clst = clusterer(**kwargs).fit(as_binary_matrix(X))

    # Retrieve the indexes and counts from the clustering result
    cluster_idxes = as_cluster_indexes(clst.labels_)
    return cluster_idxes #, cluster_cnts

def load_clusters(cluster_idxes,X=None,y=None,is_iloc=True):
    results = []
    if not isNone(X):
        if is_iloc:
            X_cl = [X.iloc[cl_i] for cl_i in cluster_idxes]
        else:
            X_cl = [X.loc[cl_i] for cl_i in cluster_idxes]
        results.append(X_cl)

    if not isNone(y):
        if is_iloc:
            y_cl = [y.iloc[cl_i] for cl_i in cluster_idxes]
        else:
            y_cl = [y.loc[cl_i] for cl_i in cluster_idxes]
        results.append(y_cl)

    if not isNone(X) and not isNone(y):
        return results
    else:
        return results[0]



def features_density_summary(df):
    feat_dens_sum = pd.Series(dtype=float)
    for t in [0.25,0.5,0.75]:
        feat_dens_sum.loc[f"d(top {int(t*100)}%)"] = features_density_by_rank(df,top=t).mean()
    for t in [0.25,0.5,0.75]:
        feat_dens_sum.loc[f"n(top {int(t*100)}%)"] = len(features_density_by_rank(df,top=t))
    for d in [0.75,0.5,0.25]:
        feat_dens_sum.loc[f"d(d > {d})"] = features_density_by_threshold(df,d).mean()
    for d in [0.75,0.5,0.25]:
        feat_dens_sum.loc[f"n(d > {d})"] = len(features_density_by_threshold(df,d))
    feat_dens_sum.loc[f"overall"] = features_density(df).mean()
    return feat_dens_sum

# All our data 
def cluster_mse_table(cluster_idxes,X,y):
    """
    Table of SSE for each cluster
    """
    X_cl = load_clusters(cluster_idxes,X=X)
    sse_tabl = pd.DataFrame()
    for X_i in X_cl:
        X_bin = as_binary_frame(X)
        X_i_mean = features_density(X_i) # Our average as the ground truth
        sse_i = ((X_bin - X_i_mean)**2).mean(axis=1) # Calculate the SSE for all data
        sse_tabl = pd.concat([sse_tabl,sse_i],axis=1,ignore_index=True)
    sse_tabl['y'] = y

    sse_tabl.index.name = 'index'
    sse_tabl.columns.name = 'cluster'
    return sse_tabl

def cluster_likeness_table(cluster_idxes,X,y,average=True,TN=False):
    """
    Table for total likeness of each row with a cluster. The bigger the value the better

    `average`: the likeness is averaged over each cluster
    """
    feat_in_common_tbl = feat_in_common_frame(X,normalized=True,TN=TN)
    cl_likeness_tbl = pd.DataFrame()
    for cl_i in cluster_idxes:
        if average:
            cl_i_likeness_res = feat_in_common_tbl.loc[:,cl_i].mean(axis=1)
        else:
            cl_i_likeness_res = feat_in_common_tbl.loc[:,cl_i].sum(axis=1)
        cl_likeness_tbl = pd.concat([cl_likeness_tbl,cl_i_likeness_res],axis=1,ignore_index=True)
    cl_likeness_tbl['y'] = y

    cl_likeness_tbl.index.name = 'index'
    cl_likeness_tbl.columns.name = 'cluster'

    return cl_likeness_tbl


def cluster_reindex(cluster_idxes,X,y,quota=10,criteria=cluster_mse_table,ascending=True,**kwargs):
    crit_table = criteria(cluster_idxes,X,y,**kwargs)
    classes,num_classes = label_counts(y).values(['labels','num_classes'])

    cluster_idxes_new = []
    for i in range(len(cluster_idxes)):
        cl_i = cluster_idxes[i]
        y_i = y.iloc[cl_i]
        class_counts_i = label_counts(y_i,labels=classes).counts

        cl_idx_new = pd.Index(cl_i)
        for j in classes:
            index1 = y.index.drop(y_i.index)    # Set of all cases not in this cluster
            index2 = crit_table[crit_table['y'] == j].index   # Set of cases that is of class j
            diff_quota = max(0,quota-class_counts_i[j])
            crit_table_ij = crit_table.loc[overlap(index1,index2),i].sort_values(ascending=ascending)[:diff_quota]
            cl_idx_new = cl_idx_new.append(crit_table_ij.index)
        cluster_idxes_new.append(cl_idx_new.values)
    return cluster_idxes_new

def cluster_cofreq_matrix(cluster_idxes_or_labels):
    """
    Adjacency matrix indicating the frequency with which each pair of elements is grouped into the same cluster.
    """
    if isinstance(cluster_idxes_or_labels[0],(pd.Int64Index,np.ndarray)): # if is_indexes
        cluster_idxes = cluster_idxes_or_labels
    else:
        cluster_idxes = as_cluster_indexes(cluster_idxes_or_labels)

    # dim_size = np.sum([len(cl_i) for cl_i in cluster_idxes])
    # dim_size = label_counts(cluster_idxes).num_classes
    
    dim_size = max(as_1d_array(label_counts(cluster_idxes).labels))+1
    sim_matrix = np.zeros(shape=(dim_size,dim_size))
    for cl_i in cluster_idxes:
        for j in range(len(cl_i)):
            idx_ij = cl_i[j]
            for k in range(j,len(cl_i)):
                idx_ik = cl_i[k]
                sim_matrix[idx_ij,idx_ik] += 1
                if idx_ij != idx_ik: # Avoid adding the same cell twice
                    sim_matrix[idx_ik,idx_ij] += 1

    return sim_matrix

def bimatrix_cluster_cv(X,cluster_args,clusterer_cv,return_cofreq_matrix=False,**kwargs):
    cofreq_matrix_cv = np.zeros(shape=(len(X),len(X)))
    for clusterer,args in cluster_args:
        cluster_idxes_i = bimatrix_cluster(X,clusterer,**args)
        cofreq_matrix_i = cluster_cofreq_matrix(cluster_idxes_i)
        cofreq_matrix_cv += cofreq_matrix_i

    # Similar to bimatrix_cluster but now we fit the sim_matrix using the clusterer_cv
    clst = clusterer_cv(**kwargs).fit(cofreq_matrix_cv)
    cluster_idxes_cv = as_cluster_indexes(clst.labels_)
    if return_cofreq_matrix:
        return cluster_idxes_cv,cofreq_matrix_cv
    else:
        return cluster_idxes_cv

def feat_in_common_matrix(df,normalized=True,TN=False):
    """
    When TN = `False`, only calculate the TP counts of column matches between 2 elements. Else calculate both the TP and TN 

    When normalized = `True`: calculate the matches averaged over the number of features 
    """
    df_bin = as_binary_matrix(df)
    nrows,ncols = df_bin.shape
    feat_in_common_mat = []
    for row_i in df_bin:
        if not TN:
            feat_in_common_mat.append(((df_bin==row_i)&(row_i==1)).sum(axis=1))
        else:
            feat_in_common_mat.append((df_bin==row_i).sum(axis=1))
    if normalized:
        return np.array(feat_in_common_mat) / ncols
    else:
        return np.array(feat_in_common_mat)

def feat_in_common_frame(df,normalized=True,TN=False,sum=False):
    df = pd.DataFrame(feat_in_common_matrix(df,normalized,TN))
    if sum:
        df['sum'] = df.sum(axis=1)
    return df
