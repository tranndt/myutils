import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import *
import matplotlib.gridspec as gridspec
from .main import *
from .pandas_ext import *

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
