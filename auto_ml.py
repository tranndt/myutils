
from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from .main import *
from sklearn.model_selection import *
from sklearn.decomposition import *
from sklearn.metrics import *
from sklearn.exceptions import *
import warnings

from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.svm import *
from sklearn.gaussian_process import *
from sklearn.naive_bayes import *
from sklearn.neural_network import *
from sklearn.kernel_ridge import *

# ----------------------------------------------
# Encoders
#----------------------------------------------

class CategoricalEncoder:
    def __init__(self,target_labels=None):
        super().__init__()
        self.target_labels = target_labels
    
    def fit(self,array):
        array_labels = pd.unique(array)
        self.target_labels = isNone(self.target_labels,then=range(len(array_labels)))
        self.encode_mappings = dict(zip(array_labels,self.target_labels))
        self.decode_mappings = dict(zip(self.target_labels,array_labels))
        return self

    def transform(self,array,dtype=object):
        return np.array([self.encode_mappings[a] for a in as_iterable(array)],dtype=dtype)

    def inv_transform(self,array,dtype=object):
        return np.array([self.decode_mappings[a] for a in as_iterable(array)],dtype=dtype)

    def fit_transform(self,array,dtype=object):
        return self.fit(array).transform(array,dtype)

    def mappings(self):
        return self.encode_mappings, self.decode_mappings


class NullEncoder:
    def __init__(self,fillna=None,unique=None):
        self.fillna = fillna
        self.unique = unique
    
    def fit(self,array):
        arr = pd.Series(array)
        if tryf(self.fillna,arr):
            self.nan_value = self.fillna(arr)  
        else:
            self.nan_value = self.fillna
        # Ensure that the nan value is unique (if required) by incrementing by a specified value
        while self.nan_value in np.array(array) and self.unique:
            self.nan_value += self.unique
        return self

    def transform(self,array):
        return pd.Series(array).fillna(self.nan_value).values

    def fit_transform(self,array):
        return self.fit(array).transform(array)

    def inv_transform(self,array):
        return pd.Series(array).replace({self.nan_value: None}).values

    def mappings(self):
        return {None:self.nan_value},{self.nan_value:None}


class AutoDataPrep:
    def __init__(self,max_categories=10,
                fillna_cont=np.mean, unique_cont=None,
                fillna_cat="$null$",unique_cat=None):
        self.max_categories = max_categories
        self.fillna_cont = fillna_cont
        self.unique_cont = unique_cont
        self.fillna_cat = fillna_cat
        self.unique_cat = unique_cat
        self.mappings_ = dict()

    def fit_transform(self,X):
        X_prep = X.copy()
        for c in X.columns:
            is_categorical = lambda x: dtype(x)=="object" or len(pd.unique(x)) <= self.max_categories # or not try_f(as_dtype("float"),x)
            equals = lambda a,b: np.all(a==b)
            if is_categorical(X[c]):
                encoders    = [ NullEncoder(fillna=self.fillna_cat,unique=self.unique_cat),
                                CategoricalEncoder()]
                X0 = encoders[0].fit_transform(X[c])
                X1 = encoders[1].fit_transform(X0,dtype=int)
                X_prep[c]   = X1
                mappings = encoders[1].mappings()[0]
                if self.fillna_cat in mappings:
                    mappings.update({None:mappings.pop(self.fillna_cat)})
            else:
                encoders = [NullEncoder(fillna=self.fillna_cont,unique=self.unique_cont)]
                X0 = encoders[0].fit_transform(X[c])
                X_prep[c] = X0
                if equals(X_prep[c],X[c]):
                    mappings = None
                else:
                    mappings = encoders[0].mappings()[0]
            self.mappings_[c] = mappings
        return X_prep

    def mappings(self):
        return self.mappings_


def classification_scores(y_true,y_pred,pos_label=1,average="auto",catch=False):
    # Calculating the scores
    f = lambda *args,**kwargs: tryf_catch(*args,**kwargs) if catch else tryf_return(np.nan,*args,**kwargs)
    true_counts = f(lambda x: label_counts(x).counts, y_true)
    true_labels = f(lambda x: label_counts(x).labels, y_true)
    true_n_class = f(lambda x: label_counts(x).num_classes, y_true)
    pred_counts = f(lambda x: label_counts(x,labels=true_labels).counts,y_pred) # Coerce the counts to be in the same order as the true counts
    
    mcm = f(multilabel_confusion_matrix,y_true,y_pred,labels=true_labels)
    corr_counts = f(lambda m: np.array([m[i,-1,-1] for i in range(true_n_class)]), mcm)
    # tpr = f(lambda corr,true: corr/true,corr_counts,true_counts)

    if true_n_class == 2:
        average = "binary" if average == "auto" else average
        conf_matrix = f(lambda yt,yp: confusion_matrix(yt,yp,labels=true_labels).ravel(),y_true,y_pred)
    else:
        average = "macro" if average == "auto" else average
        conf_matrix = f(lambda yt,yp: confusion_matrix(yt,yp,labels=true_labels),y_true,y_pred)

    precision = f(precision_score,y_true,y_pred,pos_label=pos_label,average=average,zero_division=0)
    recall = f(recall_score,y_true,y_pred,pos_label=pos_label,average=average,zero_division=0)
    accuracy = f(accuracy_score,y_true,y_pred)
    balanced_accuracy = f(balanced_accuracy_score,y_true,y_pred)
    roc_auc = f(roc_auc_score,y_true,y_pred)
    f1_binary = f(f1_score,y_true,y_pred,pos_label=pos_label,average="binary")
    f1_micro =  f(f1_score,y_true,y_pred,pos_label=pos_label,average="micro")
    f1_macro =  f(f1_score,y_true,y_pred,pos_label=pos_label,average="macro")
    f1_weighted = f(f1_score,y_true,y_pred,pos_label=pos_label,average="weighted")

    # Save the score
    scores = dict()
    scores['test_labels'] = true_labels
    scores['test_counts'] = true_counts
    scores['pred_counts'] = pred_counts
    scores['corr_counts'] = corr_counts
    scores['confusion_matrix'] = conf_matrix
    scores['f1_binary'] = f1_binary
    scores['f1_micro'] = f1_micro
    scores['f1_macro'] = f1_macro
    scores['f1_weighted'] = f1_weighted
    if isinstance(true_n_class,int):
        for l,i in zip(true_labels,range(true_n_class)):
            scores[f'tpr[{l}]'] = f(lambda corr,true,x: (corr/true)[x],corr_counts,true_counts,i)
    scores['accuracy'] = accuracy
    scores['balanced_accuracy'] = balanced_accuracy
    scores[f'precision_{average}'] = precision
    scores[f'recall_{average}'] = recall
    scores['auc'] = roc_auc
    return scores

def regression_scores(y_true,y_pred):
    test_counts = len(y_true)
    test_quartiles = np.round(np.quantile(y_true,[0,0.25,0.5,0.75,1]),2)
    pred_quartiles = np.round(np.quantile(y_pred,[0,0.25,0.5,0.75,1]),2)

    spread_counts = lambda y_p: np.array([len(y_p[y_p < y_true.min()]),*np.histogram(y_p,test_quartiles)[0],len(y_p[y_p > y_true.max()])])
    test_spread = spread_counts(y_true)
    pred_spread = spread_counts(y_pred)

    explained_variance_ = tryf_return (None,explained_variance_score,y_true,y_pred)
    max_error_ = tryf_return (None,max_error,y_true,y_pred)
    neg_mean_absolute_error_ = tryf_return (None,mean_absolute_error,y_true,y_pred)
    neg_mean_squared_error_ = tryf_return (None,mean_squared_error,y_true,y_pred,squared=True)
    neg_root_mean_squared_error_ = tryf_return (None,mean_squared_error,y_true,y_pred,squared=False)
    neg_mean_squared_log_error_ = tryf_return (None,mean_squared_log_error,y_true,y_pred)
    neg_median_absolute_error_ = tryf_return (None,median_absolute_error,y_true,y_pred) 
    r2_ = tryf_return (None,r2_score,y_true,y_pred)
    neg_mean_poisson_deviance_ = tryf_return (None,mean_poisson_deviance,y_true,y_pred)
    neg_mean_gamma_deviance_ = tryf_return (None,mean_gamma_deviance,y_true,y_pred)
    neg_mean_absolute_percentage_error_ = tryf_return (None,mean_absolute_percentage_error,y_true,y_pred)

    scores = dict()
    scores['test_counts'] = test_counts
    scores['test_quartiles'] = test_quartiles
    scores['pred_quartiles'] = pred_quartiles
    scores['test_spread'] = test_spread
    scores['pred_spread'] = pred_spread
    scores['R2'] = r2_
    scores['explained_variance'] = explained_variance_
    scores['max_error'] = max_error_
    scores['MAE']   = neg_mean_absolute_error_
    scores['MedAE'] = neg_median_absolute_error_
    scores['MSE']   = neg_mean_squared_error_
    scores['RMSE']  = neg_root_mean_squared_error_
    scores['MSLE']  = neg_mean_squared_log_error_
    scores['MAPE']  = neg_mean_absolute_percentage_error_
    scores['mean_poisson_deviance'] = neg_mean_poisson_deviance_
    scores['mean_gamma_deviance']   = neg_mean_gamma_deviance_
    return scores

    
class AutoClassifier:    
    def __init__(self,models=None,drop=None,verbose=True,ignore_warnings=False):
        """
        - `models`: 
            - `"all"`: All models
            - `"auto"`: Common classification models
                - LogisticRegression([penalty, ...]) Logistic Regression (aka logit, MaxEnt) classifier.
                - SGDClassifier([loss, penalty, ...]) Linear classifiers (SVM, logistic regression, etc.) with SGD training.
                - SVC(*[, C, kernel, degree, gamma, ...]) C-Support Vector Classification.
                - DecisionTreeClassifier(*[, criterion, ...]) A decision tree classifier.
                - GradientBoostingClassifier(*[, ...]) Gradient Boosting for classification.
                - RandomForestClassifier([...]) A random forest classifier.
                - GaussianProcessClassifier([...]) Gaussian process classification (GPC) based on Laplace approximation.
                - GaussianNB(*[, priors, ...]) Gaussian Naive Bayes (GaussianNB).
                - MLPClassifier -- Multi-layer Perceptron classifier.

            - `"linear"`: 
                - LogisticRegression([penalty, ...]) Logistic Regression (aka logit, MaxEnt) classifier.
                - LogisticRegressionCV(*[, Cs, ...]) Logistic Regression CV (aka logit, MaxEnt) classifier.
                - PassiveAggressiveClassifier(*) Passive Aggressive Classifier.
                - Perceptron(*[, penalty, alpha, ...]) Linear perceptron classifier.
                - RidgeClassifier([alpha, ...]) Classifier using Ridge regression.
                - RidgeClassifierCV([alphas, ...]) Ridge classifier with built-in cross-validation.
                - SGDClassifier([loss, penalty, ...]) Linear classifiers (SVM, logistic regression, etc.) with SGD training.
                - SGDOneClassSVM([nu, ...]) Solves linear One-Class SVM using Stochastic Gradient Descent.

            - `"svm"`:
                - LinearSVC([penalty, loss, dual, tol, C, ...]) Linear Support Vector Classification.
                - NuSVC(*[, nu, kernel, degree, gamma, ...]) Nu-Support Vector Classification.
                - SVC(*[, C, kernel, degree, gamma, ...]) C-Support Vector Classification.

            - `"neighbors"` 
                - KNeighborsClassifier([...]) Classifier implementing the k-nearest neighbors vote.

            - `"tree"`
                - DecisionTreeClassifier(*[, criterion, ...]) A decision tree classifier.
                - ExtraTreeClassifier(*[, criterion, ...]) An extremely randomized tree classifier.

            - `"ensemble"`
                - AdaBoostClassifier([...]) An AdaBoost classifier.
                - ExtraTreesClassifier([...]) An extra-trees classifier.
                - GradientBoostingClassifier(*[, ...]) Gradient Boosting for classification.
                - RandomForestClassifier([...]) A random forest classifier.

            - `"gaussian_process"`
                - GaussianProcessClassifier([...]) Gaussian process classification (GPC) based on Laplace approximation.

            - `"naive_bayes"`
                - BernoulliNB(*[, alpha, ...]) Naive Bayes classifier for multivariate Bernoulli models.
                - ComplementNB(*[, alpha, ...]) The Complement Naive Bayes classifier described in Rennie et al. (2003).
                - GaussianNB(*[, priors, ...]) Gaussian Naive Bayes (GaussianNB).
                - MultinomialNB(*[, alpha, ...]) Naive Bayes classifier for multinomial models.

            - `"neural_network"`
                - MLPClassifier -- Multi-layer Perceptron classifier.

            - Reference:
                - [sklearn.neighbors](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
                - [sklearn.gaussian_process](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process)
                - [sklearn.neural_network](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network)
                - [sklearn.svm](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
                - [sklearn.tree](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)
                - [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
                - [sklearn.naive_bayes](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)
                - [sklearn.ensemble](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
        """
        self.DEFAULT_MODELS = {
        #tree
            'DecisionTreeClassifier': ('tree',DecisionTreeClassifier()),
            'ExtraTreeClassifier': ('tree',ExtraTreeClassifier()),
            # linear
            'LogisticRegression': ('linear',LogisticRegression()),
            'LogisticRegressionCV': ('linear',LogisticRegressionCV()),
            'PassiveAggressiveClassifier': ('linear',PassiveAggressiveClassifier()),
            'Perceptron': ('linear',Perceptron()),
            'RidgeClassifier': ('linear',RidgeClassifier()),
            'RidgeClassifierCV': ('linear',RidgeClassifierCV()),
            'SGDClassifier': ('linear',SGDClassifier()),
            # svm
            'LinearSVC': ('svm',LinearSVC()),
            'NuSVC': ('svm',NuSVC()),
            'SVC': ('svm',SVC()),
            # neighbors
            'KNeighborsClassifier': ('neighbors',KNeighborsClassifier()),
            # ensemble
            'AdaBoostClassifier': ('ensemble',AdaBoostClassifier()),
            'ExtraTreesClassifier': ('ensemble',ExtraTreesClassifier()),
            'GradientBoostingClassifier': ('ensemble',GradientBoostingClassifier()),
            'RandomForestClassifier': ('ensemble',RandomForestClassifier()),
            # gaussian_process
            # 'GaussianProcessClassifier': ('gaussian_process',GaussianProcessClassifier()),
            # naive_bayes
            'BernoulliNB': ('naive_bayes',BernoulliNB()),
            'ComplementNB': ('naive_bayes',ComplementNB()),
            'GaussianNB': ('naive_bayes',GaussianNB()),
            'MultinomialNB': ('naive_bayes',MultinomialNB()),
            # neural_network
            'MLPClassifier': ('neural_network',MLPClassifier())
        }
        self.verbose = verbose
        self.__init_models__(models)
        self.__filter_warnings__(ignore_warnings)
        pass

    def __filter_warnings__(self,ignore_warnings):
        if ignore_warnings is not False:
            if ignore_warnings is True:
                warnings_ = [
                    ConvergenceWarning,
                    DataConversionWarning,
                    DataDimensionalityWarning,
                    EfficiencyWarning,
                    UndefinedMetricWarning,
                    RuntimeWarning,
                    FutureWarning,
                    np.VisibleDeprecationWarning
                ]
            for w in as_iterable(warnings_):
                tryf_return (None,warnings.filterwarnings,"ignore", category=w)

    def __init_models__(self, models):
        get_models_         = lambda x: at(list(self.DEFAULT_MODELS.keys()),whether(self.DEFAULT_MODELS.values(),lambda v: v[0] == x))
        get_group          = lambda x: tryf_return("other", lambda m: self.DEFAULT_MODELS[m][0],x)
        is_module_wo_name   = lambda x: type_of(x).startswith("ABCMeta")
        is_module_w_name    = lambda x: isinstance(x,Iterable) and type_of(x[1]).startswith("ABCMeta")
        is_model_name_key   = lambda x: isinstance(x,str) and x in self.DEFAULT_MODELS.keys()
        is_group_key       = lambda x: isinstance(x,str) and x in [v[0] for v in self.DEFAULT_MODELS.values()]

        models_ = []
        if isNone(models):
            models_ = list(self.DEFAULT_MODELS.values())
            
        else:
            for option in as_iterable(models):
                if is_group_key(option):
                    for m in get_models_(option):
                        models += self.DEFAULT_MODELS[m]

                elif is_model_name_key(option):
                    models += self.DEFAULT_MODELS[option]

                elif is_module_wo_name(option):
                    models.append((get_group(option),option))

                elif is_module_w_name(option):
                    models.append(option)

        self.models = models_


    def trained_models(self):
        return [m for n,m in self.models]

    def model_names(self):
        return [type_of(m) for n,m in self.models]

    def model_groups(self):
        return [n for n,m in self.models]

    def fit(self,X,y):
        for m in self.trained_models():
            flushif(self.verbose,f"\n> Fitting {type_of(m)}")
            t = TaskTimer()
            total_time, m = t.execute(
                tryf_return,m,m.fit,X,y
            )
            flushif(self.verbose,f"\r>> Fitted {type_of(m)} in {t.fmt(total_time)}")
        return self

    def predict(self,X):
        predictions = []
        for m in self.trained_models():
            predictions_i = tryf_catch(m.predict,X)
            predictions.append(predictions_i)
        predictions = np.array(predictions,dtype=object)
        return predictions

    def fit_predict(self,X,y,X_test):
        return self.fit(X,y).predict(X_test)
        
    def score(self,X,y,catch=False):
        y_pred = self.predict(X)
        scores = pd.DataFrame()
        for (n,m),y_i in zip(self.models,y_pred):
            iter_score = classification_scores(y,y_i,catch=catch)
            iter_score["model"] = type_of(m)
            iter_score["params"] = tryf_return(str({}),str(m.get_params))
            iter_score["group"] = n
            scores = pd.concat([scores,pd.DataFrame([iter_score])],ignore_index=True)
        scores = scores.set_index(["model","params","group"])
        return scores.reset_index(["model","params","group"])

    def cross_validate(self,X,y,cv=3,catch=False):
        if isinstance(cv,int):
            cv = range(cv)
        scores = pd.DataFrame()
        for i in as_iterable(cv):
            if tryf(train_test_split,X,y,stratify=y,random_state=i):
                X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=i)
            else:
                X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=i)
            self.fit(X_train,y_train)
            scores_i = self.score(X_test,y_test,catch)
            scores_i["seed"] = i
            scores = pd.concat([scores,scores_i],axis=0,ignore_index=True)
        return scores

class AutoRegressor(AutoClassifier):
    def __init__(self,models=None,verbose=True,ignore_warnings=False):
        """
        - `models`
            - `svm`:
                - LinearSVR(*[, epsilon, tol, C, loss, ...]) Linear Support Vector Regression.
                - NuSVR(*[, nu, C, kernel, degree, gamma, ...]) Nu Support Vector Regression.
                - SVR(*[, kernel, degree, gamma, coef0, ...]) Epsilon-Support Vector Regression.

            - `neighbors` 
                - KNeighborsRegressor([n_neighbors, ...]) Regression based on k-nearest neighbors.

            - `tree`
                - DecisionTreeRegressor(*[, criterion, ...]) A decision tree regressor.
                - ExtraTreeRegressor(*[, criterion, ...]) An extremely randomized tree regressor.  

            - `ensemble`: Combine the predictions of several base estimators in order to improve generalizability / robustness. -- In averaging methods, reduce variance by building several estimators independently and then averaging their predictions (Bagging methods, Forests of randomized trees, …) -- In boosting methods, base estimators are built sequentially and one tries to reduce the bias of the combined estimator. (AdaBoost, Gradient Tree Boosting, …)
                - AdaBoostRegressor([base_estimator, ...]) An AdaBoost regressor.
                - BaggingRegressor([base_estimator, ...]) A Bagging regressor.
                - ExtraTreesRegressor([n_estimators, ...]) An extra-trees regressor.
                - GradientBoostingRegressor(*[, ...]) Gradient Boosting for regression.
                - RandomForestRegressor([...]) A random forest regressor.
                - StackingRegressor(estimators[, ...]) Stack of estimators with a final regressor.
                - VotingRegressor(estimators, *[, ...]) Prediction voting regressor for unfitted estimators.
                - HistGradientBoostingRegressor([...]) Histogram-based Gradient Boosting Regression Tree.

            - `gaussian_process`
                - GaussianProcessRegressor([...]) Gaussian process regression (GPR).

            - `neural_network`
                - MLPRegressor -- Multi-layer Perceptron regressor.

            - `linear`: Classical linear regressors
                - LinearRegression(*[, ...]) Ordinary least squares Linear Regression.
                - Ridge([alpha, fit_intercept, ...]) Linear least squares with l2 regularization.
                - RidgeCV([alphas, ...]) Ridge regression with built-in cross-validation.
                - SGDRegressor([loss, penalty, ...]) Linear model fitted by minimizing a regularized empirical loss with SGD.

            - `linear-variable-selection`: (Linear) Regressors with variable selection. The following estimators have built-in variable selection fitting procedures, but any estimator using a L1 or elastic-net penalty also performs variable selection: typically SGDRegressor or SGDClassifier with an appropriate penalty.
                - ElasticNet([alpha, l1_ratio, ...]) Linear regression with combined L1 and L2 priors as regularizer.
                - ElasticNetCV(*[, l1_ratio, ...]) Elastic Net model with iterative fitting along a regularization path.
                - Lars(*[, fit_intercept, ...]) Least Angle Regression model a.k.a.
                - LarsCV(*[, fit_intercept, ...]) Cross-validated Least Angle Regression model.
                - Lasso([alpha, fit_intercept, ...]) Linear Model trained with L1 prior as regularizer (aka the Lasso).
                - LassoCV(*[, eps, n_alphas, ...]) Lasso linear model with iterative fitting along a regularization path.
                - LassoLars([alpha, ...]) Lasso model fit with Least Angle Regression a.k.a.
                - LassoLarsCV(*[, fit_intercept, ...]) Cross-validated Lasso, using the LARS algorithm.
                - LassoLarsIC([criterion, ...]) Lasso model fit with Lars using BIC or AIC for model selection.
                - OrthogonalMatchingPursuit(*[, ...]) Orthogonal Matching Pursuit model (OMP).
                - OrthogonalMatchingPursuitCV(*) Cross-validated Orthogonal Matching Pursuit model (OMP).

            - `bayesian`: (Linear) Bayesian regressors
                - ARDRegression(*[, n_iter, tol, ...]) Bayesian ARD regression.
                - BayesianRidge(*[, n_iter, tol, ...]) Bayesian ridge regression.

            - `multitask`: (Linear) Multi-task linear regressors with variable selection. These estimators fit multiple regression problems (or tasks) jointly, while inducing sparse coefficients. While the inferred coefficients may differ between the tasks, they are constrained to agree on the features that are selected (non-zero coefficients).
                - MultiTaskElasticNet([alpha, ...]) Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer.
                - MultiTaskElasticNetCV(*[, ...]) Multi-task L1/L2 ElasticNet with built-in cross-validation.
                - MultiTaskLasso([alpha, ...]) Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
                - MultiTaskLassoCV(*[, eps, ...]) Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.

            - `outlier-robust`: (Linear) Outlier-robust regressors. Any estimator using the Huber loss would also be robust to outliers, e.g. SGDRegressor with loss='huber'.
                - HuberRegressor(*[, epsilon, ...]) Linear regression model that is robust to outliers.
                - QuantileRegressor(*[, ...]) Linear regression model that predicts conditional quantiles.
                - RANSACRegressor([...]) RANSAC (RANdom SAmple Consensus) algorithm.
                - TheilSenRegressor(*[, ...]) Theil-Sen Estimator: robust multivariate regression model.

            - `glm`: (Linear) Generalized linear models (GLM) for regression. These models allow for response variables to have error distributions other than a normal distribution
                - PoissonRegressor(*[, alpha, ...])Generalized Linear Model with a Poisson distribution.
                - TweedieRegressor(*[, power, ...])Generalized Linear Model with a Tweedie distribution.
                - GammaRegressor(*[, alpha, ...])Generalized Linear Model with a Gamma distribution.

            - `kernel_ridge`
                - kernel_ridge.KernelRidge([alpha, kernel, ...])

            - `isotonic`: Isotonic regression. IsotonicRegression produces a series of predictions y^ for the training data which are the closest to the targets y in terms of mean squared error. These predictions are interpolated for predicting to unseen data
                - IsotonicRegression(*[, y_min, ...]) Isotonic regression model.
      
        """
        self.DEFAULT_MODELS = {
            #tree
            'DecisionTreeRegressor': ('tree',DecisionTreeRegressor()),
            'ExtraTreeRegressor': ('tree',ExtraTreeRegressor()),
            # svm
            'LinearSVR': ('svm',LinearSVR()),
            'NuSVR': ('svm',NuSVR()),
            'SVR': ('svm',SVR()),
            # neighbors
            'KNeighborsRegressor': ('neighbors',KNeighborsRegressor()),
            # ensemble
            'AdaBoostRegressor': ('ensemble',AdaBoostRegressor()),
            'ExtraTreesRegressor':('ensemble',ExtraTreesRegressor()),
            'GradientBoostingRegressor':('ensemble',GradientBoostingRegressor()),
            'RandomForestRegressor':('ensemble',RandomForestRegressor()),
            # gaussian_process
            'GaussianProcessRegressor': ('gaussian_process',GaussianProcessRegressor()),
            # neural_network
            'MLPRegressor': ('neural_network',MLPRegressor()),
            # linear
            'LinearRegression': ('linear',LinearRegression()),
            'Ridge': ('linear',Ridge()),
            'RidgeCV': ('linear',RidgeCV()),
            'SGDRegressor': ('linear',SGDRegressor()),
            # linear_variable_selection
            'ElasticNet': ('linear_variable_selection',ElasticNet()),
            'ElasticNetCV': ('linear_variable_selection',ElasticNetCV()),
            'Lars': ('linear_variable_selection',Lars()),
            'LarsCV': ('linear_variable_selection',LarsCV()),
            'Lasso': ('linear_variable_selection',Lasso()),
            'LassoCV': ('linear_variable_selection',LassoCV()),
            'LassoLars': ('linear_variable_selection',LassoLars()),
            'LassoLarsCV': ('linear_variable_selection',LassoLarsCV()),
            'LassoLarsIC': ('linear_variable_selection',LassoLarsIC()),
            'OrthogonalMatchingPursuit': ('linear_variable_selection',OrthogonalMatchingPursuit()),
            'OrthogonalMatchingPursuitCV': ('linear_variable_selection',OrthogonalMatchingPursuitCV()),
            # bayesian
            'ARDRegression': ('bayesian',ARDRegression()),
            'BayesianRidge': ('bayesian',BayesianRidge()),
            # outlier_robust
            'HuberRegressor': ('outlier_robust',HuberRegressor()),
            'RANSACRegressor': ('outlier_robust',RANSACRegressor()),
            'TheilSenRegressor': ('outlier_robust',TheilSenRegressor()),
            # glm
            'PoissonRegressor': ('glm',PoissonRegressor()),
            'TweedieRegressor': ('glm',TweedieRegressor()),
            'GammaRegressor': ('glm',GammaRegressor()),
            # kernel_ridge
            'KernelRidge': ('kernel_ridge',KernelRidge()),

        }
        self.verbose = verbose
        self.__init_models__(models)
        self.__filter_warnings__(ignore_warnings)

    def score(self,X,y):
        y_pred = self.predict(X)
        scores = pd.DataFrame()
        for (n,m),y_i in zip(self.models,y_pred):
            iter_score = regression_scores(y,y_i)
            iter_score["model"] = type_of(m)
            iter_score["params"] = tryf_return(str({}),str(m.get_params))
            iter_score["group"] = n
            scores = pd.concat([scores,pd.DataFrame([iter_score])])
        scores = scores.set_index(["model","params","group"])
        return scores.reset_index(["model","params","group"])







