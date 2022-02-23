import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import *
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.cluster import *
import matplotlib.gridspec as gridspec
from .main import *
from .pandas_ext import *


def as_binary_frame(X,bool=False):
    if bool:
        return X.notna()
    else:
        return X.notna().replace({True:1,False:0})

def as_binary_matrix(X,bool=False):
    return as_binary_frame(X,bool).values

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
    return df_dens[df_dens >= threshold].sort_values(ascending=False)

def rows_density_by_rank(df,top=None):
    df_dens = rows_density(df)
    if isNone(top):
        top = len(df_dens)
    if isinstance(top,float):
        top = int(min(len(df_dens),len(df_dens)*top))
    return df_dens.sort_values(ascending=False)[:top]

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

def grid_shape(num_plots,size):
    MAX_WIDTH = 30
    ncols = MAX_WIDTH//size
    nrows = ceil(num_plots/ncols)
    return nrows,ncols

def grid_plot(plot_func,plot_args,plot_size=(8,6),gr_shape=None,title=None,**kwargs):
    sizeX,sizeY = plot_size
    nrows,ncols = isNone(gr_shape,
        then = grid_shape(len(plot_args),sizeX)
    )
    fig = plt.figure(figsize=(sizeX*ncols,sizeY*nrows))
    spec = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig)
    for i in range(len(plot_args)):
        r = i//ncols
        c = i%ncols
        ax = fig.add_subplot(spec[r, c])
        plot_func(ax=ax,**plot_args[i],**kwargs)

    if not isNone(title):
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_bimatrix(data,sort=False,cmap="Greys",ax=None,xlabel=None,ylabel=None,title=None,**kwargs):
    if isinstance(data,np.ndarray):
        data = pd.DataFrame(data)
    if sort:
        index   = rows_density_by_rank(data).index if sort=="rows" and sort!="columns" else data.index
        columns = features_density_by_rank(data).index if sort=="columns" and sort!="rows" else data.columns
        data = data.loc[index,columns]
    # if 'title' in kwargs.keys():
    # plt.title(kwargs['title'])
    ax = sns.heatmap(as_binary_matrix(data),cmap=cmap,**kwargs)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)


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

# def binary_likeness_matrix(df,normalized=True,FP=False):
#     df_bin = as_binary_matrix(df)
#     nrows,ncols = df_bin.shape
#     bi_likeness_mat = []
#     for row_i in df_bin:
#         if FP:
#             bi_likeness_mat.append(((df_bin==row_i)|(row_i==0)).sum(axis=1))
#         else:
#             bi_likeness_mat.append((df_bin==row_i).sum(axis=1))
#     if normalized:
#         return np.array(bi_likeness_mat) / ncols
#     else:
#         return np.array(bi_likeness_mat)

# def binary_likeness_frame(df,normalized=True,FP=False):
#     return pd.DataFrame(binary_likeness_matrix(df,normalized,FP))

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