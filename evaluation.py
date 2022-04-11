import pandas as pd 
import numpy as np
import torch    
import seaborn as sns
import torch.nn as nn
from .main import *
from .pandas_tools import *
from sklearn.metrics import *
from sklearn.model_selection import *


# -----------------------------------------------
#   TESTING MODELS
# -----------------------------------------------
def enc_as_str(arr):
    return enc_str_fr_np(arr,sep=None,br=None)

def dec_as_np(string):
    return dec_np_fr_str(string,sep=None,br=None)

def test_iteration(        
        name    = None,
        random_state = None,
        X       = None,
        y       = None,
        X_train = None,
        y_train = None,
        X_test  = None,
        y_test  = None,
        **kwargs
    ):

    return PseudoObject(
        name    = name,
        random_state = random_state,
        X       = X,
        y       = y,
        X_train = X_train,
        y_train = y_train,
        X_test  = X_test,
        y_test  = y_test,
        **kwargs
    )

def test_iteration_results(
        y_true      =None,
        y_pred      =None,
        dataset     =None,
        random_state=None,
        model       =None,
        args        =None,
        **kwargs
    ):

    return PseudoObject(
        y_true      =y_true,
        y_pred      =y_pred,
        dataset     =dataset,
        random_state=random_state,
        model       =model,
        args        =args,
        **kwargs
    )

def test_ilocs(
        dataset     = None,
        random_state= None,
        labels      = None,
        total_cnt   = None,
        train_cnt   = None,
        test_cnt    = None,
        train_rows_iloc = None,
        test_rows_iloc  = None,
        train_cols_iloc = None,
        test_cols_iloc  = None,
        **kwargs
    ):

    return PseudoObject(
        dataset     = dataset,
        random_state= random_state,
        labels      = labels,
        total_cnt   = total_cnt,
        train_cnt   = train_cnt,
        test_cnt    = test_cnt,
        train_rows_iloc = train_rows_iloc,
        test_rows_iloc  = test_rows_iloc,
        train_cols_iloc = train_cols_iloc,
        test_cols_iloc  = test_cols_iloc,
        **kwargs
    )


def run_test_iteration(test_iter, model_iter, return_trained_model=False):
    # Retrieve the dataset
    X_train,X_test,y_train,y_test,random_state,name = test_iter.values(['X_train','X_test','y_train','y_test','random_state','name'])

    # Initialize our model
    model_init, args = model_iter
    if 'random_state' in get_args(model_init.__init__): # check if model accepts random_state keyword
        model = model_init(random_state=random_state,**args)
    else:
        model = model_init(**args)

    # Execute our test iteration and record the time elapsed
    time_elapsed,model = ProcessTimer().execute(
        job_id=0,         
        return_val=True,
        func=model.fit, 
        X=X_train, 
        y=y_train,
    )
    y_pred = model.predict(X_test)
    y_true = y_test

    # Save our test result
    test_iter_result = test_iteration_results(
        dataset = name,
        random_state = random_state,
        model = type(model).__name__,
        args = str(args),
        y_true = enc_as_str(ravel(y_true,dtype=int)), # Encode the results as str instead of array
        y_pred = enc_as_str(ravel(y_pred,dtype=int)), 
        time_elapsed = np.round(time_elapsed,4),
    )

    if return_trained_model:
        return test_iter_result, model
    else:
        return test_iter_result


def run_test_batch(dataset_obj,random_states,fsl_samples,model_batch,input_flag=0,write_to=None,save_test_ilocs=True):
    test_batch_result = pd.DataFrame()
    test_ilocs = pd.DataFrame()
    for random_state in random_states:
        # Test each model+kwargs combination and record the result 
        for model_iter in model_batch:
            for n_samples in fsl_samples: #ravel() to prevent bugs
                test_iter = create_test_iteration(dataset_obj,n_samples,random_state,input_flag)
                test_iter_result = run_test_iteration(test_iter,model_iter)
                test_batch_result = pd.concat([test_batch_result,test_iter_result.to_frame(dtype="object")],ignore_index=True)  # Append result to DataFrame, converted to "object" first to avoid turning string into int
                write_dataframe(test_batch_result,write_to)

                test_iloc = create_test_iloc(test_iter)
                test_ilocs = pd.concat([test_ilocs,test_iloc.to_frame(dtype="object")],ignore_index=True)

    write_dataframe(test_batch_result,write_to)
    if save_test_ilocs:
        write_dataframe(test_ilocs.drop_duplicates(),f"{dir_to_file(write_to)}/test_iloc_table.csv")

    return test_batch_result


def create_test_iloc(test_iter):
    labels,counts = label_counts(test_iter.y).values(["labels","counts"])  
    test_iloc = test_ilocs(
        dataset = test_iter.name,
        random_state= test_iter.random_state,
        labels = enc_str_fr_np(labels),
        total_cnt = enc_str_fr_np(counts),
        train_cnt = enc_str_fr_np(label_counts(test_iter.y_train,labels).counts),
        test_cnt = enc_str_fr_np(label_counts(test_iter.y_test,labels).counts),
        train_rows_iloc = enc_str_fr_np(get_array_iloc(test_iter.y_train.index,test_iter.y.index),sep=" ",br=None),
        test_rows_iloc = enc_str_fr_np(get_array_iloc(test_iter.y_test.index,test_iter.y.index),sep=" ",br=None),
        train_cols_iloc = enc_str_fr_np(get_array_iloc(test_iter.X_train.columns,test_iter.X.columns),sep=" ",br=None),
        test_cols_iloc = enc_str_fr_np(get_array_iloc(test_iter.X_test.columns,test_iter.X.columns),sep=" ",br=None),
    )
    return test_iloc


def load_test_iloc(test_iloc):
    return test_ilocs(
        dataset = test_iloc.loc['dataset'],
        random_state= test_iloc.loc['random_state'],
        labels = dec_np_fr_str2(test_iloc.loc['labels'],sep=",",dtype=int),
        total_cnt = dec_np_fr_str2(test_iloc.loc['total_cnt'],sep=",",dtype=int),
        train_cnt = dec_np_fr_str2(test_iloc.loc['train_cnt'],sep=",",dtype=int),
        test_cnt = dec_np_fr_str2(test_iloc.loc['test_cnt'],sep=",",dtype=int),
        train_rows_iloc = dec_np_fr_str2(test_iloc.loc['train_rows_iloc'],sep=" ",br=None,dtype=int),
        test_rows_iloc = dec_np_fr_str2(test_iloc.loc['test_rows_iloc'],sep=" ",br=None,dtype=int),
        train_cols_iloc = dec_np_fr_str2(test_iloc.loc['train_cols_iloc'],sep=" ",br=None,dtype=int),
        test_cols_iloc = dec_np_fr_str2(test_iloc.loc['test_cols_iloc'],sep=" ",br=None,dtype=int),
    )

# -----------------------------------------------
#   SCORING MODELS
# -----------------------------------------------
def test_iteration_score(y_true,y_pred,pos_label=1,average="auto"):
    # Calculating the scores
    true_counts,true_labels,true_n_class = label_counts(y_true).values(['counts','labels','num_classes'])
    pred_counts = label_counts(y_pred,labels=true_labels).counts # Coerce the counts to be in the same order as the true counts
    mcm = multilabel_confusion_matrix(y_true,y_pred,labels=true_labels)
    corr_counts = np.array([mcm[i,-1,-1] for i in range(true_n_class)])

    if true_n_class == 2:
        average = "binary" if average == "auto" else average
        conf_matrix = confusion_matrix(y_true,y_pred,labels=true_labels).ravel()
    else:
        average = "macro" if average == "auto" else average
        conf_matrix = confusion_matrix(y_true,y_pred,labels=true_labels)

    precision,recall,fscore,support = precision_recall_fscore_support(y_true,y_pred,pos_label=pos_label,average=average,zero_division=0)
    accuracy = accuracy_score(y_true,y_pred)

    # Save the score
    # test_iter_score = test_iter_result.drop(['y_true','y_pred']).copy()
    test_iter_score = pd.Series()
    test_iter_score.loc['test_labels'] = true_labels
    test_iter_score.loc['test_counts'] = true_counts
    test_iter_score.loc['pred_counts'] = pred_counts
    test_iter_score.loc['corr_counts'] = corr_counts
    test_iter_score.loc['conf_matrix'] = conf_matrix
    test_iter_score.loc['accuracy'] = np.round(accuracy,4)
    test_iter_score.loc['precision'] = np.round(precision,4)
    test_iter_score.loc['recall'] = np.round(recall,4)
    test_iter_score.loc['fscore'] = np.round(fscore,4)
    test_iter_score.loc['average'] = average

    # Added later
    tpr = corr_counts/true_counts
    f1_binary = f1_score(y_true,y_pred,pos_label=pos_label,average="binary") if true_n_class == 2 else None
    f1_micro = f1_score(y_true,y_pred,pos_label=pos_label,average="micro")
    f1_macro = f1_score(y_true,y_pred,pos_label=pos_label,average="macro")
    f1_weighted = f1_score(y_true,y_pred,pos_label=pos_label,average="weighted")
    test_iter_score.loc['f1_binary'] = np.round(f1_binary,4) if not isNone(f1_binary) else f1_binary
    test_iter_score.loc['f1_micro'] = np.round(f1_micro,4)
    test_iter_score.loc['f1_macro'] = np.round(f1_macro,4)
    test_iter_score.loc['f1_weighted'] = np.round(f1_weighted,4)
    for i in true_labels:
        test_iter_score.loc[f'tpr[{i}]'] = tpr[i]

    return test_iter_score

def test_iteration_score_from_result(test_iter_result,pos_label=1,average="auto"):
    if isinstance(test_iter_result,PseudoObject):
        test_iter_result = test_iter_result.to_series()
    
    y_true = dec_np_fr_str2(test_iter_result['y_true'],sep="",dtype=int)
    y_pred = dec_np_fr_str2(test_iter_result['y_pred'],sep="",dtype=int)

    test_iter_result = test_iter_result.drop(['y_true','y_pred']).copy()
    test_iter_score = test_iteration_score(y_true,y_pred,pos_label=pos_label,average=average)
    return pd.concat([test_iter_result,test_iter_score],axis=0)


def test_batch_score_from_result(test_batch_result,write_to=None,**kwargs):
    if isinstance(test_batch_result,str):
        test_batch_result = read_dataframe(test_batch_result,dtype="object") # Avoid converting y_true and y_pred to integers, which would remove leading 0s

    test_batch_score_ = pd.DataFrame()
    for i in range(len(test_batch_result)):
        test_iter_result = test_batch_result.iloc[i]
        test_iter_score = test_iteration_score_from_result(test_iter_result,**kwargs).to_frame().T
        test_batch_score_ = pd.concat([test_batch_score_,test_iter_score],ignore_index=True)
    write_dataframe(test_batch_score_,write_to)
    return test_batch_score_

def read_test_iter_scores(test_iter_score_str):
    test_iter_score = test_iter_score_str.copy()
    test_iter_score['test_labels'] = dec_np_fr_str2(test_iter_score_str['test_labels'],dtype=int)
    test_iter_score['test_counts'] = dec_np_fr_str2(test_iter_score_str['test_counts'],dtype=int)
    test_iter_score['pred_counts'] = dec_np_fr_str2(test_iter_score_str['pred_counts'],dtype=int)
    test_iter_score['corr_counts'] = dec_np_fr_str2(test_iter_score_str['corr_counts'],dtype=int)
    test_iter_score['conf_matrix'] = dec_np_fr_str2(test_iter_score_str['conf_matrix'],dtype=int,levels=2)
    test_iter_score['precision'] = dec_np_fr_str2(test_iter_score_str['precision'],dtype=float)
    test_iter_score['recall'] = dec_np_fr_str2(test_iter_score_str['recall'],dtype=float)
    test_iter_score['fscore'] = dec_np_fr_str2(test_iter_score_str['fscore'],dtype=float)
    return test_iter_score

def read_scores_file(filename,**kwargs):
    df = read_dataframe(filename,**kwargs)
    test_batch_scores = pd.DataFrame()
    for i in df.index:
        test_iter_score = read_test_iter_scores(df.loc[i]).to_frame().T
        test_batch_scores = pd.concat([test_batch_scores,test_iter_score],axis=0)
    return test_batch_scores


def read_results_file(filename):
    return read_dataframe(filename)


def plot_scores(title=None,**kwargs):
    ax = sns.lineplot(**kwargs)
    ax.set(title=title)
    ax.grid(True,axis='y')
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,-0.8))

def create_datasets(directory,**kwargs):
    for key in kwargs.keys():
        if key == "description":
            write_file(kwargs[key],f"{directory}/description.txt")
        else:
            write_dataframe(kwargs[key],f"{directory}/{key}.csv")

def load_datasets(directory,keys=["data","X","y","metadata","description","X_train","X_test","y_train","y_test","test_iloc_table"],**kwargs):
    dataset_obj = PseudoObject(name=directory)
    for key in keys:
        if key == "description":
            filename = f"{directory}/{key}.txt"
            if os.path.isfile(filename):
                dataset_obj.update(**{key:read_file(filename)})
        else:
            filename = f"{directory}/{key}.csv"
            if os.path.isfile(filename):
                dataset_obj.update(**{key:read_dataframe(filename,**kwargs)})
    return dataset_obj