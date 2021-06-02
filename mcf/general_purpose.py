"""General purpose procedures.

# -*- coding: utf-8 -*-.
Created on Thu Apr  2 17:55:24 2020

@author: MLechner
"""
from concurrent import futures
import copy
import math
import itertools
import pickle
import importlib.util
from datetime import datetime, timedelta
import sys
from itertools import chain
import os.path
import gc
import scipy.stats as sct
from sympy.ntheory import primefactors
import pandas as pd
import numpy as np
import psutil
from numba import njit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def delete_file_if_exists(file_name):
    """Delete existing file."""
    if os.path.exists(file_name):
        os.remove(file_name)


def auto_garbage_collect(pct=80.0):
    """
    Call garbage collector if memory used > pct% of total available memory.

    This is called to deal with an issue in Ray not freeing up used memory.
    pct - Default value of 80%.  Amount of memory in use that triggers
          the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()


# define an Output class for simultaneous console - file output
class OutputTerminalFile():
    """Output class for simultaneous console/file output."""

    def __init__(self, file_name):

        self.terminal = sys.stdout
        self.output = open(file_name, "w")

    def write(self, message):
        """Write both into terminal and file."""
        self.terminal.write(message)
        self.output.write(message)

    def flush(self):
        """Python 3 compatibility."""


def print_dic(dic):
    """Print dictionary in a simple way."""
    for keys, values in dic.items():
        if isinstance(values, list):
            print(keys, ':  ', *values)
        else:
            print(keys, ':  ', values)


def save_load(file_name, object_to_save=None, save=True, output=True):
    """
    Save and load objects via pickle.

    Parameters
    ----------
    file_name : String. File to save to or to load from.
    object_to_save : any python object that can be pickled, optional.
                     The default is None.
    save : Boolean., optional The default is True. False for loading.

    Returns
    -------
    object_to_load : Unpickeled Python object (if save=False).
    """
    if save:
        delete_file_if_exists(file_name)
        with open(file_name, "wb+") as file:
            pickle.dump(object_to_save, file)
        object_to_load = None
        text = '\nObject saved to '
    else:
        with open(file_name, "rb") as file:
            object_to_load = pickle.load(file)
        text = '\nObject loaded from '
    if output:
        print(text + file_name)
    return object_to_load


def csv_var_names_upper(path, csvfile):
    """
    Convert all variable names in csv-file to upper case.

    Parameters
    ----------
    path : Str. Directory of location of csv_file.
    csvfile: Str.

    Returns
    -------
    csvfile_new : Str.

    """
    indatafile = path + '/' + csvfile + '.csv'
    data_df = pd.read_csv(indatafile)
    names = data_df.columns.tolist()
    any_change = False
    rename_dic = {}
    for name in names:
        if name != name.upper():
            any_change = True
            name_new = name.upper()
        else:
            name_new = name
        rename_dic.update({name: name_new})
    if any_change:
        csvfile_new = csvfile + 'UP'
        outdatafile = path + '/' + csvfile_new + '.csv'
        data_df.rename(rename_dic)
        delete_file_if_exists(outdatafile)
        data_df.to_csv(outdatafile, index=False)
    else:
        csvfile_new = csvfile
    return csvfile_new


def adjust_var_name(var_to_check, var_names):
    """
    Check if upper or lower case version of name is in namelist and adjust.

    Parameters
    ----------
    var_to_check : Str.
    var_names : List of Str.

    Returns
    -------
    var_to_check : Str.

    """
    if var_to_check not in var_names:
        for name in var_names:
            if (var_to_check.upper() == name) or (
                    var_to_check.lower() == name):
                var_to_check = name
                break
    return var_to_check


def RandomForest_scikit(
        x_train, y_train, x_pred, x_name=None, y_name='y', boot=1000,
        n_min=2, no_features='sqrt', workers=-1,  max_depth=None, alpha=0,
        max_leaf_nodes=None, pred_p_flag=True, pred_t_flag=False,
        pred_oob_flag=False, with_output=True, variable_importance=False,
        var_im_groups=None, pred_uncertainty=False, pu_ci_level=0.9,
        pu_skew_sym=0.5, var_im_with_output=True,
        variable_importance_oob_flag=False, return_forest_object=False):
    """
    Compute Random Forest predictions with OOB-optimal parameters & var import.

    Based on sklearn, but with many additional features.
    Prediction intervals are based on:
        Zhang, Zimmerman, Nettleton, and Nordman (2020): Random Forest
        Prediction Intervals, THE AMERICAN STATISTICIAN, 2020, VOL. 74, NO. 4,
        392–406: Data Science, https://doi.org/10.1080/00031305.2019.1585288
        (Type III intervals given; symmetric intervals used if skewness of
         distribution is below user specified cut-off-value of skewness)
    Variable importance measures are based on:
        Williamson, Gilbert, Carone, Simon (2020): Nonparametric variable
        importance assessment using machine learning techniques, Biometrics.
        (no crossfitting used, no uncertainty measure given)

    Parameters
    ----------
    x_train : Numy array. Training features.
    y_train : Numpy array. Training outcome.
    x_pred : Numpy array. Features for which to predict outcomes.
    y_name: String. Name of dependende variable. Default is 'y'.
    x_name : List of Str. Name of features used in RF.
    boot : Int, optional. No of trees. The default is 1000.
    n_min : List of Int, optional. Minimum leaf size. The default is [2].
    no_features : List of Int, optional. M_try. The default is ['sqrt'].
    workers : No of parallel processes, optional. The default is -1.
    pred_p_flag : Boolean. Predit with x_pred. Default is True.
    pred_t_flag : Boolean. Predit with x_train. Default is False.
    pred_oob_flag : Boolean. Use OOB for training prediction. Default is False.
    with_output : Boolean. Print output. Default is True.
    max_depth : Int. Depth of tree. Default is None.
    alpha : Float. Minimum share on each side of split. Default is 0.
    max_leaf_nodes : Int. Maximimum number of leafs. Default is None.
    variable_importance : Boolean. OOB based variable importance (mutated x).
                          Default is False.
    var_im_groups : Lists of lists of strings or None.
    pred_uncertainty : Bool. Compute standard error of prediction.
                          Default is False.
    pu_ci_level : Float. Confidence level for prediction intervals.
                         Default = 0.9
    pu_skew_sym : Float. Cut-off of skewness for symetric CIs. Default is 0.5.
    variable_importance_oobx : Bool. Use OOB obs for VI (computational costly).
                                    Default is False.
    return_forest_object: Boolean. Forest object.

    Returns
    -------
    pred_p : 1d Numpy array. Predictions prediction data.
    pred_t: 1d Numpy array. Predictions training data.
    oob_best: Float. OOB based R2
    pu_ci_p: 2d Numpy array. Confidence interval prediction data
    pu_ci_t: 2d Numpy array. Confidence interval training data
    forest: Forest object.

    """
    if with_output:
        var_im_with_output = True
        print('\n')
        print('Computing forests')
    oob_best = -1e15
    pred_p = []
    pred_t = []
    y_1d = y_train.ravel()
    no_of_vars = np.size(x_train, axis=1)
    if no_features is None:
        no_features = no_of_vars
    if isinstance(no_features, str):
        if no_features == 'sqrt':
            no_features = round(math.sqrt(no_of_vars))
        elif no_features == 'auto':
            no_features = no_of_vars
        elif no_features == 'log2':
            no_features = round(math.log2(no_of_vars))
        else:
            no_features = round(no_of_vars/3)
    elif isinstance(no_features, float):
        no_features = round(no_features * no_of_vars)
    if not isinstance(n_min, (list, tuple, np.ndarray)):
        n_min = [n_min]
    if not isinstance(no_features, (list, tuple, np.ndarray)):
        no_features = [no_features]
    if not isinstance(alpha, (list, tuple, np.ndarray)):
        alpha = [alpha]
    for nval in n_min:
        for mval in no_features:
            # add optimisation über alpha --> see honestforests
            for aval in alpha:
                regr = RandomForestRegressor(
                    n_estimators=boot, min_samples_leaf=nval,
                    max_features=mval, bootstrap=True, oob_score=True,
                    random_state=42, n_jobs=workers, max_depth=max_depth,
                    min_weight_fraction_leaf=aval,
                    max_leaf_nodes=max_leaf_nodes)
                regr.fit(x_train, y_1d)
                if with_output and not ((len(no_features) == 1)
                                        and (len(n_min) == 1)
                                        and (len(alpha) == 1)):
                    print(' N_min: {:2}'.format(nval), ' M_Try: {:2}'.format(
                        mval), ' Alpha: {:5.3f}'.format(aval),
                        '|  OOB R2 (in %): {:8.4f}'.format(
                        regr.oob_score_*100))
                if regr.oob_score_ > oob_best:
                    m_opt = copy.copy(mval)
                    n_opt = copy.copy(nval)
                    a_opt = copy.copy(aval)
                    oob_best = copy.copy(regr.oob_score_)
                    regr_best = copy.deepcopy(regr)
    if with_output:
        print('\n')
        print('-' * 80)
        print('Tuning values choosen for forest (if any)')
        print('-' * 80)
        print('Dependent variable: ', y_name[0], 'N_min: {:2}'.format(n_opt),
              'M_Try: {:2}'.format(m_opt), 'Alpha: {:5.3f}'.format(a_opt),
              '|  OOB R2 (in %): {:8.4f}'.format(oob_best * 100))
        print('-' * 80)
    if pred_p_flag:
        pred_p = regr_best.predict(x_pred)
    if variable_importance or pred_uncertainty:
        pred_oob = regr_best.oob_prediction_
    if pred_t_flag:
        if pred_oob_flag:
            if variable_importance or pred_uncertainty:
                pred_t = np.copy(pred_oob)
            else:
                pred_t = regr_best.oob_prediction_
        else:
            pred_t = regr_best.predict(x_train)
    if with_output:
        print(regr_best.get_params())
        print('-' * 80)
        print(y_name, ': ',
              'R2(%) OOB: {:8.4f}'.format(regr_best.oob_score_*100))
        print('-' * 80)
    if variable_importance:
        if variable_importance_oob_flag:
            vi_names_shares = variable_importance_oob(
                x_name, var_im_groups, no_of_vars, y_1d, pred_oob, with_output,
                var_im_with_output, x_train, boot, n_opt, m_opt, workers,
                max_depth, a_opt, max_leaf_nodes)
        else:
            vi_names_shares = variable_importance_testdata(
                x_name, var_im_groups, y_1d, with_output,
                var_im_with_output, x_train, boot, n_opt, m_opt, workers,
                max_depth, a_opt, max_leaf_nodes)
    else:
        vi_names_shares = None
    if pred_uncertainty and (pred_t_flag or pred_p_flag):
        resid_oob = y_1d - pred_oob
        skewness_res_oob = sct.skew(resid_oob)
        if with_output:
            print('Skewness of oob residual:', skewness_res_oob)
        symmetric = np.absolute(skewness_res_oob) < pu_skew_sym
        if symmetric:
            if with_output:
                print('Symmetric intervals used')
            quant = np.quantile(np.absolute(resid_oob), 1-pu_ci_level)
            if pred_t_flag:
                upper_t = pred_t + quant
                lower_t = pred_t - quant
            if pred_p_flag:
                upper_p = pred_p + quant
                lower_p = pred_p - quant
        else:
            if with_output:
                print('Asymmetric intervals used')
            quant = np.quantile(resid_oob, [(1-pu_ci_level)/2,
                                            1-(1-pu_ci_level)/2])
            if pred_t_flag:
                upper_t = pred_t + quant[1]
                lower_t = pred_t + quant[0]
            if pred_p_flag:
                upper_p = pred_p + quant[1]
                lower_p = pred_p + quant[0]
        if pred_t_flag:
            pu_ci_t = (lower_t, upper_t)
        else:
            pu_ci_t = None
        if pred_p_flag:
            pu_ci_p = (lower_p, upper_p)
        else:
            pu_ci_p = None
    else:
        pu_ci_p = None
        pu_ci_t = None
    if return_forest_object:
        forest = regr_best
    else:
        forest = None
    return pred_p, pred_t, oob_best, vi_names_shares, pu_ci_p, pu_ci_t, forest


def variable_importance_testdata(
        x_name, var_im_groups, y_all, with_output, var_im_with_output, x_all,
        boot, n_opt, m_opt, workers, max_depth, a_opt, max_leaf_nodes,
        share_test_data=0.2):
    """Compute variable importance without out-of-bag predictions."""
    if x_name is None:
        raise Exception('Variable importance needs names of features.')
    if var_im_groups is None:
        var_im_groups = []
    if var_im_groups:
        dummy_names = flatten_list(var_im_groups)
        x_to_check = [x for x in x_name if x not in dummy_names]
    else:
        x_to_check = x_name
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=share_test_data, random_state=42)
    regr_vi = RandomForestRegressor(
        n_estimators=boot, min_samples_leaf=n_opt, max_features=m_opt,
        bootstrap=True, oob_score=True, random_state=42, n_jobs=workers,
        max_depth=max_depth, min_weight_fraction_leaf=a_opt,
        max_leaf_nodes=max_leaf_nodes)
    regr_vi.fit(x_train, y_train)
    r2_full = regr_vi.score(x_test, y_test)
    # Start with single variables (without dummies)
    # indices_of_x = range(no_of_vars)
    loss_in_r2_single = np.empty(len(x_to_check))
    if with_output or var_im_with_output:
        print('Variable importance for ', end='')
    for indx, name_to_randomize in enumerate(x_to_check):
        if with_output or var_im_with_output:
            print(name_to_randomize, end=' ')
        r2_vi = get_r2_test(x_test, y_test, name_to_randomize, x_name, regr_vi)
        loss_in_r2_single[indx] = r2_full - r2_vi
    loss_in_r2_dummy = np.empty(len(var_im_groups))
    if var_im_groups:
        for indx, name_to_randomize in enumerate(var_im_groups):
            if with_output or var_im_with_output:
                print(name_to_randomize, end=' ')
            r2_vi = get_r2_test(x_test, y_test, name_to_randomize, x_name,
                                regr_vi)
            loss_in_r2_dummy[indx] = r2_full - r2_vi
    else:
        loss_in_r2_dummy = 0
    vi_names_shares = print_vi(
        x_to_check, var_im_groups, loss_in_r2_single, loss_in_r2_dummy,
        var_im_with_output)
    return vi_names_shares


def get_r2_test(x_test, y_test, name_to_randomize, x_name, regr_vi):
    """Get R2 for variable importance."""
    x_vi = x_test.copy()
    indices_to_r = find_indices_vi(name_to_randomize, x_name)
    x_to_shuffle = x_vi[:, indices_to_r]
    np.random.shuffle(x_to_shuffle)
    x_vi[:, indices_to_r] = x_to_shuffle
    r2_vi = regr_vi.score(x_vi, y_test)
    return r2_vi


def find_indices_vi(names, x_names):
    """Find indices that correspond to names and return list."""
    x_names = list(x_names)
    if isinstance(names, str):
        names = [names]
    indices = []
    for name in names:
        indices_t = x_names.index(name)
        indices.append(indices_t)
    return indices


def variable_importance_oob(x_name, var_im_groups, no_of_vars, y_1d, pred_oob,
                            with_output, var_im_with_output, x_train, boot,
                            n_opt, m_opt, workers, max_depth, a_opt,
                            max_leaf_nodes):
    """Compute variable importance with out-of-bag predictions."""
    if x_name is None:
        raise Exception('Variable importance needs names of features.')
    if var_im_groups is None:
        var_im_groups = []
    if var_im_groups:
        dummy_names = flatten_list(var_im_groups)
        x_to_check = [x for x in x_name if x not in dummy_names]
    else:
        x_to_check = x_name
    # Start with single variables (without dummies)
    indices_of_x = range(no_of_vars)
    var_y = np.var(y_1d)
    mse_ypred = np.mean((y_1d - pred_oob)**2)
    loss_in_r2_single = np.empty(len(x_to_check))
    if with_output or var_im_with_output:
        print('Variable importance for ', end='')
    for indx, name_to_delete in enumerate(x_to_check):
        if with_output or var_im_with_output:
            print(name_to_delete, end=' ')
        mse_ypred_vi = get_r2_for_vi_oob(
            indices_of_x, x_name, name_to_delete, x_train, y_1d, boot,
            n_opt, m_opt, workers, max_depth, a_opt, max_leaf_nodes)
        loss_in_r2_single[indx] = (mse_ypred_vi - mse_ypred) / var_y
    loss_in_r2_dummy = np.empty(len(var_im_groups))
    if var_im_groups:
        for indx, name_to_delete in enumerate(var_im_groups):
            if with_output or var_im_with_output:
                print(name_to_delete, end=' ')
            mse_ypred_vi = get_r2_for_vi_oob(
                indices_of_x, x_name, name_to_delete, x_train, y_1d, boot,
                n_opt, m_opt, workers, max_depth, a_opt, max_leaf_nodes)
            loss_in_r2_dummy[indx] = (mse_ypred_vi - mse_ypred) / var_y
    else:
        loss_in_r2_dummy = 0
    vi_names_shares = print_vi(
        x_to_check, var_im_groups, loss_in_r2_single, loss_in_r2_dummy,
        var_im_with_output)
    return vi_names_shares


def print_vi(x_to_check, var_im_groups, loss_in_r2_single, loss_in_r2_dummy,
             with_output):
    """
    Print variable importance measure.

    Parameters
    ----------
    x_to_check : Non-List of Str. Non-categorial variables.
    var_im_groups : List of lists of Str. Categorical variables as dummies.
    loss_in_r2_single : List. VI measure for non-cat.
    loss_in_r2_dummy : List. VI measure for non-cat.
    with_output : Bool.

    Returns
    -------
    x_names_sorted : List. Sorted Names.
    vim_sorted : List. Sorted VIM.

    """
    if var_im_groups:
        # Get single names and ignore the dummies (last element)
        var_dummies = []
        for name in var_im_groups:
            var_dummies.append(name[-1])
        x_to_check.extend(var_dummies)
        loss_in_r2_single = np.concatenate(
            (loss_in_r2_single, loss_in_r2_dummy))
    var_indices = np.argsort(loss_in_r2_single)
    var_indices = np.flip(var_indices)
    x_names_sorted = np.array(x_to_check, copy=True)
    x_names_sorted = x_names_sorted[var_indices]
    vim_sorted = loss_in_r2_single[var_indices]
    if with_output:
        print('\n')
        print('-' * 80)
        print('Variable importance statistics in %-point of R2 lost',
              'when omitting particular variable')
        print('- ')
        print('Unordered variables are included (in RF and VI) as dummies'
              ' and the variable itself.')
        if var_im_groups:
            print('Here, all elements of these variables are jointly tested ',
                  '(in Table under heading of unordered variable).')
        for idx, vim in enumerate(vim_sorted):
            print('{:<50}: {:>7.2f}'.format(x_names_sorted[idx], vim*100), '%')
        print('-' * 80)
        print('-' * 80, '\n')
    return x_names_sorted, vim_sorted


def get_r2_for_vi_oob(indices_of_x, x_name, name_to_delete, x_train,  y_1d,
                      boot=1000, n_opt=2, m_opt='sqrt', workers=-1,
                      max_depth=None, a_opt=0, max_leaf_nodes=None):
    """
    Estimate r2 for reduced variable set.

    Parameters
    ----------
    indices_of_x : range(no_of_columns_of X).
    x_name: List of Str. All x_names.
    name_to_delete : Str or list of Str. Variables to be deleted.
    x_train : Numpy array. All features.
    boot : Int, optional. No of trees. The default is 1000.
    n_opt : List of Int, optional. Minimum leaf size. The default is 2.
    m_opt : List of Int, optional. M_try. The default is 'sqrt'.
    workers : No of parallel process, optional. The default is -1.
    max_depth : Int. Depth of tree. Default is None.
    a_opt : Float. Minimum share on each side of split. Default is 0..
    max_leaf_nodes : Int. Maximimum number of leafs. Default is None.
    y_1d : 1d-Numpy array.

    Returns
    -------
    mse_ypred_vi : Float. MSE.

    """
    indices_to_delete = find_indices_vi(name_to_delete, x_name)
    indices_to_remain = [i for i in indices_of_x if i not in indices_to_delete]
    regr_vi = RandomForestRegressor(
        n_estimators=boot, min_samples_leaf=n_opt, max_features=m_opt,
        bootstrap=True, oob_score=True, random_state=42, n_jobs=workers,
        max_depth=max_depth, min_weight_fraction_leaf=a_opt,
        max_leaf_nodes=max_leaf_nodes)
    regr_vi.fit(x_train[:, indices_to_remain], y_1d)
    pred_vi = regr_vi.oob_prediction_
    mse_ypred_vi = np.mean((y_1d - pred_vi)**2)
    return mse_ypred_vi


def r_squared(y_dat, y_pred, w_dat=0):
    """
    Compute R2.

    Parameters
    ----------
    y : 1-dim Numpy array. Observed variable.
    y_pred : 1-dim Numpy array. Predicted variable.
    w : 1-dim Numpy array. optional. Weights. The default is 0 (no weighting).

    Returns
    -------
    r2 : Float. R2.

    """
    if np.all(w_dat == 0):
        w_dat = np.ones(len(y_dat))
    else:
        w_dat = w_dat / np.sum(w_dat) * len(w_dat)
        w_dat = w_dat.reshape(-1)
    y_mean = np.average(y_dat, weights=w_dat, axis=0)
    r_2 = 1 - np.dot(w_dat, np.square(y_dat-y_pred)) / np.dot(
        w_dat, np.square(y_dat-y_mean))
    return r_2


def bound_norm_weights(weight, max_weight, renormalize=True):
    """Bound and renormalized weights.

    Parameters
    ----------
    weight : 1d Numpy array. Weights.
    max_weight : Scalar Float. Maximum value of any weight
    renormalize : Boolean, optional. If True renormalize the weights that they
               add to 1. The default is True.

    Returns
    -------
    weight_norm : Numpy array of same size as input. Normalized weights.
    no_censored: NP float. Number of censored observations.
    share_censored: NP float. Share of censored observations (0-1).

    """
    weight_norm = weight.flatten()
    too_large = (weight + 1e-15) > max_weight
    if np.any(too_large):
        no_censored = np.count_nonzero(too_large)
        weight_norm[too_large] = max_weight
    else:
        no_censored = 0
    share_censored = no_censored / len(weight)
    if renormalize:
        sum_w = np.sum(weight_norm)
        if not ((-1e-10 < sum_w < 1e-10) or (1-1e-10 < sum_w < 1+1e-10)):
            weight_norm = weight_norm / sum_w
    return weight_norm, no_censored, share_censored


@njit
def quadratic_form(vec, mat):
    """Quadratic form for Numpy: vec'mat*vec.

    Parameters
    ----------
    vec : 1D Numpy Array.
    mat : 2D quadratic Numpy Array.

    Returns
    -------
    Numpy Float. The Quadratic form.

    """
    return np.dot(vec, mat @ vec)


def waldtest(diff, variance):
    """Compute Wald Chi2 statistics.

    Parameters
    ----------
    diff : 1D Numy array.
    variance : 2D Numpy array.

    Returns
    -------
    stat : Numpy float. Test statistic.
    df : Int. Degrees of freedom.
    p : Numpy float. p-value.

    """
    degfr = int(round(np.linalg.matrix_rank(variance)))
    stat = -1
    pval = -1
    if degfr == len(variance):
        if np.all(np.linalg.eigvals(variance) > 1e-15):
            weight = np.linalg.inv(variance)
            stat = quadratic_form(diff, weight)
            pval = sct.chi2.sf(stat, degfr)
    return stat, degfr, pval


def add_dummies(cat_vars, indatafile):
    """Add dummy variables to csv data file."""
    dat_df = pd.read_csv(indatafile)
    x_dummies = pd.get_dummies(dat_df[cat_vars],
                               columns=cat_vars)
    new_x_name = x_dummies.columns.tolist()
    dat_df = pd.concat([dat_df, x_dummies], axis=1)
    delete_file_if_exists(indatafile)
    dat_df.to_csv(indatafile, index=False)
    return new_x_name


def copy_csv_file_pandas(new_file, old_file):
    """Copy csv file with pandas."""
    dat_df = pd.read_csv(old_file)
    delete_file_if_exists(new_file)
    dat_df.to_csv(new_file, index=False)


def print_effect_z(g_r, gm_r, z_values, gate_str):
    """Print treatment effects.

    Parameters
    ----------
    est : Numpy array. Point estimate.
    se : Numpy array. Standard error.
    t_val : Numpy array. t/z-value.
    p_val : Numpy array. p-value.
    effect_list : List of Int. Treatment values involved in comparison.
    add_title: None or string. Additional title.
    add_info: None or Int. Additional information about parameter.

    Returns
    -------
    None.

    """
    no_of_effect_per_z = np.size(g_r[0][0])
    print('- ' * 40)
    print('                   ', gate_str,
          '                                ', gate_str, '- ATE')
    print('Comparison   Z       Est        SE    t-val   p-val        Est',
          '       SE    t-val  p-val')
    print('- ' * 40)
    for j in range(no_of_effect_per_z):
        for zind, z_val in enumerate(z_values):
            print('{:<3} vs {:>3}'.format(g_r[zind][4][j][0],
                                          g_r[zind][4][j][1]), end=' ')
            print('{:>5.1f}'.format(z_val), end=' ')
            print('{:>8.5f}  {:>8.5f}'.format(g_r[zind][0][j],
                                              g_r[zind][1][j]), end=' ')
            print('{:>5.2f}  {:>5.2f}%'.format(g_r[zind][2][j],
                                               g_r[zind][3][j]*100), end=' ')
            if g_r[zind][3][j] < 0.001:
                print('****', end=' ')
            elif g_r[zind][3][j] < 0.01:
                print(' ***', end=' ')
            elif g_r[zind][3][j] < 0.05:
                print('  **', end=' ')
            elif g_r[zind][3][j] < 0.1:
                print('   *', end=' ')
            else:
                print('    ', end=' ')
            print('{:>8.5f}  {:>8.5f}'.format(gm_r[zind][0][j],
                                              gm_r[zind][1][j]), end=' ')
            print('{:>5.2f}  {:>5.2f}%'.format(gm_r[zind][2][j],
                                               gm_r[zind][3][j]*100), end=' ')
            if gm_r[zind][3][j] < 0.001:
                print('****')
            elif gm_r[zind][3][j] < 0.01:
                print(' ***')
            elif gm_r[zind][3][j] < 0.05:
                print('  **')
            elif gm_r[zind][3][j] < 0.1:
                print('   *')
            else:
                print(' ')
        if j < no_of_effect_per_z-1:
            print('- ' * 40)
    print('-' * 80)
    print('Values of Z may have been recoded into primes.')
    print('-' * 80)


def print_effect(est, stderr, t_val, p_val, effect_list, add_title=None,
                 add_info=None):
    """Print treatment effects.

    Parameters
    ----------
    est : Numpy array. Point estimate.
    stderr : Numpy array. Standard error.
    t_val : Numpy array. t/z-value.
    p_val : Numpy array. p-value.
    effect_list : List of Int. Treatment values involved in comparison.
    add_title: None or string. Additional title.
    add_info: None or Int. Additional information about parameter.

    Returns
    -------
    None.
    """
    if add_title is None:
        print('Comparison    Estimate   Standard error t-value   p-value')
    else:
        print('Comparison ', add_title,
              '   Estimate   Standard error t-value   p-value')
    print('- ' * 40)
    for j in range(np.size(est)):
        print('{:<3} vs {:>3}'.format(effect_list[j][0], effect_list[j][1]),
              end=' ')
        if add_title is not None:
            print('{:6.2f}'.format(add_info), end=' ')
        print('{:12.6f}  {:12.6f}'.format(est[j], stderr[j]), end=' ')
        print('{:8.2f}  {:8.3f}%'.format(t_val[j], p_val[j]*100), end=' ')
        if p_val[j] < 0.001:
            print('****')
        elif p_val[j] < 0.01:
            print(' ***')
        elif p_val[j] < 0.05:
            print('  **')
        elif p_val[j] < 0.1:
            print('   *')
        else:
            print()
    print('- ' * 40)


def effect_from_potential(pot_y, pot_y_var, d_values):
    """Compute effects and stats from potential outcomes.

    Parameters
    ----------
    pot_y_ao : Numpy array. Potential outcomes.
    pot_y_var_ao : Numpy array. Variance of potential outcomes.

    Returns
    -------
    est : Numpy array. Point estimates.
    se : Numpy array. Standard error.
    t_val : Numpy array. t-value.
    p_val : Numpy array.

    """
    no_of_comparisons = round(len(d_values) * (len(d_values) - 1) / 2)
    est = np.empty(no_of_comparisons)
    stderr = np.empty(no_of_comparisons)
    comparison = [None] * no_of_comparisons
    j = 0
    for idx, treat1 in enumerate(d_values):
        for jnd, treat2 in enumerate(d_values):
            if jnd <= idx:
                continue
            est[j] = pot_y[jnd] - pot_y[idx]
            stderr[j] = np.sqrt(pot_y_var[jnd] + pot_y_var[idx])
            comparison[j] = [treat2, treat1]
            j = j + 1
    t_val = np.abs(est / stderr)
    # p_val = sct.t.sf(np.abs(t_val), 1000000) * 2
    p_val = sct.t.sf(t_val, 1000000) * 2
    return est, stderr, t_val, p_val, comparison


def gini_coefficient(x_dat):
    """Compute Gini coefficient of numpy array of values."""
    diffsum = 0
    x_dat = np.sort(x_dat)
    x_mean = np.mean(x_dat)
    if not -1e-15 < x_mean < 1e-15:
        for idx, x_i in enumerate(x_dat[:-1], 1):
            diffsum += np.sum(np.abs(x_i - x_dat[idx:]))
        return diffsum / ((len(x_dat))**2 * x_mean)
    return diffsum


def kernel_density(data, grid, kernel, bandwidth):
    """Compute nonparametric estimate of density of data.

    Parameters
    ----------
    data : Numpy array. Dependent variable.
    grid : grid : Numpy array. Values for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    f_grid : Numpy array. Prediction.

    """
    differ = np.subtract.outer(data, grid)  # This builds a matrix
    y_dach_i = kernel_proc(differ / bandwidth, kernel)
    f_grid = np.mean(y_dach_i, axis=0) / bandwidth
    return f_grid


def kernel_density_y(y_dat, x_dat, grid, kernel, bandwidth):
    """Compute nonparametric estimate of density of data * y.

    Parameters
    ----------
    y_dat : Numpy array. Dependent variable.
    x_dat : Numpy array. Independent variable.
    grid : grid : Numpy array. Values for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    f_grid : Numpy array. Prediction.

    """
    differ = np.subtract.outer(x_dat, grid)  # This builds a matrix
    y_dach_i = kernel_proc(differ / bandwidth, kernel)
    f_grid = np.mean(y_dach_i * np.reshape(y_dat, (len(grid), 1)),
                     axis=0) / bandwidth
    return f_grid


def nadaraya_watson(y_dat, x_dat, grid, kernel, bandwidth):
    """Compute Nadaraya-Watson one dimensional nonparametric regression.

    Parameters
    ----------
    y_dat : Numpy array. Dependent variable.
    x_dat :  Numpy array. Independent variable.
    grid : Numpy array. Values of x for which to create predictions.
    kernel : Int. 1: Epanechikov  2: Normal.
    bandwidth : Float. Bandwidth.

    Returns
    -------
    estimate : Numpy array. Estimated quantity.

    """
    f_yx = kernel_density_y(y_dat, x_dat, grid, kernel, bandwidth)
    f_x = kernel_density(x_dat, grid, kernel, bandwidth)
    estimate = f_yx / f_x
    return estimate


def kernel_proc(data, kernel):
    """Feed data through kernel for nonparametric estimation.

    This function works for matrices and vectors.

    Parameters
    ----------
    data : Numpy array. Data.
    kernel : Int. 1: Epanechikov  2: Normal.

    Returns
    -------
    y : Numpy array. Kernel applied to data.

    """
    if kernel == 1:
        y_dat = np.zeros(np.shape(data))   # this works with matrices
        smaller_than_1 = np.abs(data) < 1  # abs() seems to be efficient
        y_dat[smaller_than_1] = 3/4 * (1 - (data[smaller_than_1] ** 2))
    if kernel == 2:  # This works for matrices
        y_dat = sct.norm.pdf(data)
    return y_dat


def bandwidth_silverman(data, kernel=1):
    """Compute Silvermans rule of thumb for Epanechikov and normal kernels.

    Silvermans rule of thumb, Cameron, Trivedi, p. 304

    Parameters
    ----------
    data : Numpy array 1D. Data.
    kernel : INT. 1: Epanechikov, 2: Normal

    Returns
    -------
    bandwidth : Float. Optimal bandwidth if data is normally distributed

    """
    obs = len(data)
    if obs < 5:
        raise Exception('Only', len(data), ' observations for',
                        'bandwidth selection.')
    iqr = np.quantile(data, (0.25, 0.75))
    iqr = (iqr[1] - iqr[0]) / 1.349
    std = np.std(data)
    if (std < iqr) or (iqr < 1e-15):
        sss = std
    else:
        sss = iqr
    band = 1.3643 * (obs ** (-0.2)) * sss
    if kernel == 1:  # Epanechikov
        bandwidth = band * 1.7188
    elif kernel == 2:  # Normal distribution
        bandwidth = band * 0.7764
    else:
        raise Exception('Wrong type of kernel in Silverman bandwidth')
    if bandwidth < 1e-15:
        bandwidth = 1
    return bandwidth


def bandwidth_nw_rule_of_thumb(data):
    """Compute rule of thumb for Nadaraya Watson Kernel regression.

    Li & Racine, 2004, Nonparametric Econometrics: Theory & Practice,
                       bottom of page 66

    Parameters
    ----------
    data : Numpy array 1D. Data.

    Returns
    -------
    bandwidth : Float. Optimal bandwidth if data is normally distributed

    """
    obs = len(data)
    if obs < 5:
        raise Exception('Only', len(data), ' observations for',
                        'bandwidth selection.')
    iqr = np.quantile(data, (0.25, 0.75))
    iqr = (iqr[1] - iqr[0]) / 1.349
    std = np.std(data)
    if (std < iqr) or (iqr < 1e-15):
        sss = std
    else:
        sss = iqr
    bandwidth = sss * (obs ** (-0.2))
    if bandwidth < 1e-15:
        bandwidth = 1
    return bandwidth


def moving_avg_mean_var(data, k, mean_and_var=True):
    """Compute moving average of mean and std deviation.

    Parameters
    ----------
    y : Numpy array. Dependent variable. Sorted.
    k : Int. Number of neighbours to be considered.

    Returns
    -------
    y_mean_movaverge : numpy array.
    y_var_movaverge : numpy array.
    """
    var = None
    obs = len(data)
    k = int(k)
    if k >= obs:
        mean = np.full(obs, np.mean(data))
        if mean_and_var:
            var = np.full(obs, np.var(data))
    else:
        weights = np.ones(k) / k
        mean = np.convolve(data, weights, mode='valid')
        half_diff_in_len = (obs - len(mean))/2
        add_dim_first = int(np.ceil(half_diff_in_len))
        add_dim_last = int(np.floor(half_diff_in_len))
        firstvalues = np.full(add_dim_first, mean[0])
        lastvalues = np.full(add_dim_last, mean[-1])
        mean = np.concatenate((firstvalues, mean, lastvalues))
        if mean_and_var:
            data_s = data ** 2
            mean_s = np.convolve(data_s, weights, mode='valid')
            firstvalues = np.full(add_dim_first, mean_s[0])
            lastvalues = np.full(add_dim_last, mean_s[-1])
            mean_s = np.concatenate((firstvalues, mean_s, lastvalues))
            var = (mean_s - mean * mean)
    return mean, var


def aggregate_cluster_pos_w(cl_dat, w_dat, y_dat=None, norma=True, w2_dat=None,
                            sweights=None, y2_compute=False):
    """Aggregate weighted cluster means.

    Parameters
    ----------
    cl_dat : Numpy array. Cluster indicator.
    w_dat : Numpy array. Weights.
    y_dat : Numpy array. Outcomes.
    ...

    Returns
    -------
    w_agg : Numpy array. Aggregated weights. Normalised to one.
    y_agg : Numpy array. Aggregated outcomes.
    w2_agg : Numpy array. Aggregated weights. Normalised to one.

    """
    cluster_no = np.unique(cl_dat)
    # no_cluster = np.size(cluster_no)
    no_cluster = len(cluster_no)
    w_pos = np.abs(w_dat) > 1e-15
    if y_dat is not None:
        if y_dat.ndim == 1:
            y_dat = np.reshape(y_dat, (-1, 1))
        q_obs = np.size(y_dat, axis=1)
        y_agg = np.zeros((no_cluster, q_obs))
    else:
        y_agg = None
    if y2_compute:
        y2_agg = np.copy(y_agg)
    else:
        y2_agg = None
    w_agg = np.zeros(no_cluster)
    if w2_dat is not None:
        w2_agg = np.zeros(no_cluster)
        w2_pos = np.abs(w2_dat) > 1e-15
    else:
        w2_agg = None
    for j, cl_ind in enumerate(cluster_no):
        in_cluster = (cl_dat == cl_ind).reshape(-1)
        in_cluster_pos = in_cluster & w_pos
        if y2_compute:
            in_cluster_pos2 = in_cluster & w2_pos
        if np.any(in_cluster_pos):
            w_agg[j] = np.sum(w_dat[in_cluster_pos])
            if w2_dat is not None:
                w2_agg[j] = np.sum(w2_dat[in_cluster])
            if (y_dat is not None) and np.any(in_cluster_pos):
                for odx in range(q_obs):
                    if sweights is None:
                        y_agg[j, odx] = (np.dot(
                            w_dat[in_cluster_pos], y_dat[in_cluster_pos, odx])
                            / w_agg[j])
                        if y2_compute:
                            y2_agg[j, odx] = (np.dot(
                                w2_dat[in_cluster_pos2],
                                y_dat[in_cluster_pos2, odx]) / w2_agg[j])
                    else:
                        y_agg[j, odx] = (np.dot(
                            w_dat[in_cluster_pos] *
                            sweights[in_cluster_pos].reshape(-1),
                            y_dat[in_cluster_pos, odx]) / w_agg[j])
                        if y2_compute:
                            y2_agg[j, odx] = (np.dot(
                                w2_dat[in_cluster_pos2] *
                                sweights[in_cluster_pos2].reshape(-1),
                                y_dat[in_cluster_pos2, odx]) / w2_agg[j])
    if norma:
        sum_w_agg = np.sum(w_agg)
        if not 1-1e-10 < sum_w_agg < 1+1e-10:
            w_agg = w_agg / sum_w_agg
        if w2_dat is not None:
            sum_w2_agg = np.sum(w2_agg)
            if not 1-1e-10 < sum_w2_agg < 1+1e-10:
                w2_agg = w2_agg / sum_w2_agg
    return w_agg, y_agg, w2_agg, y2_agg


def weight_var(w0_dat, y0_dat, cl_dat, c_dict, norm=True, w_for_diff=None,
               weights=None):
    """Generate the weight-based variance.

    Parameters
    ----------
    w_dat : Numpy array. Weights.
    y_dat : Numpy array. Outcomes.
    cl_dat : Numpy array. Cluster indicator.
    c_dict : Dict. Parameters.
    norm : Boolean. Normalisation. Default is True.
    w_for_diff : Numpy array. weights used for difference wheb clustering.
                 Default is None.
    weights : Numpy array. Sampling weights. Clustering only. Default is None.
    no_agg :   Boolean. No aggregation of weights. Default is False.

    Returns
    -------
    variance : Float. Variance.

    """
    w_dat = np.copy(w0_dat)
    y_dat = np.copy(y0_dat)
    if c_dict['cluster_std'] and (cl_dat is not None):
        if not c_dict['w_yes']:
            weights = None
        w_dat, y_dat, _, _ = aggregate_cluster_pos_w(
            cl_dat, w_dat, y_dat, norma=norm, sweights=weights)
    if w_for_diff is not None:
        w_dat = w_dat - w_for_diff
    if norm:
        sum_w_dat = np.sum(w_dat)
        if not ((-1e-15 < sum_w_dat < 1e-15)
                or (1-1e-10 < sum_w_dat < 1+1e-10)):
            w_dat = w_dat / sum_w_dat
    w_ret = np.copy(w_dat)
    w_pos = np.abs(w_dat) > 1e-15  # use non-zero only to speed up
    w_dat2 = w_dat[w_pos]
    y_dat = y_dat[w_pos]
    obs = len(w_dat2)
    if obs < 5:
        return 0, 1, w_ret
    est = np.dot(w_dat2, y_dat)
    if c_dict['cond_var']:
        sort_ind = np.argsort(w_dat2)
        y_s = y_dat[sort_ind]
        w_s = w_dat2[sort_ind]
        if c_dict['knn']:
            k = int(np.round(c_dict['knn_const'] * np.sqrt(obs) * 2))
            if k < c_dict['knn_min_k']:
                k = c_dict['knn_min_k']
            if k > obs/2:
                k = np.floor(obs/2)
            exp_y_cond_w, var_y_cond_w = moving_avg_mean_var(y_s, k)
        else:
            band = bandwidth_nw_rule_of_thumb(w_s) * c_dict['nw_bandw']
            exp_y_cond_w = nadaraya_watson(y_s, w_s, w_s, c_dict['nw_kern'],
                                           band)
            var_y_cond_w = nadaraya_watson((y_s - exp_y_cond_w)**2, w_s, w_s,
                                           c_dict['nw_kern'], band)
        variance = np.dot(w_s**2, var_y_cond_w) + obs * np.var(
            w_s*exp_y_cond_w)
    else:
        variance = len(w_dat2) * np.var(w_dat2 * y_dat)
    return est, variance, w_ret


def flatten_list(list_to_flatten):
    """Flatten a list of lists in one single list.

    Parameters
    ----------
    list_to_flatten : List of list. To flatten.

    Returns
    -------
    flatted_list : List. Single list.

    """
    flatted_list = [item for sublist in list_to_flatten for item in sublist]
    return flatted_list


def grid_log_scale(large, small, number):
    """Define a logarithmic grid.

    Parameters
    ----------
    large : INT or FLOAT: Largest value of grid.
    small : INT or FLOAT: Smallest value of grid.
    number : INT: Number of grid points.

    Returns
    -------
    List with grid.

    """
    if small <= 0.0000001:
        small = 0.00000001
    small = math.log(small)
    large = math.log(large)
    sequence = np.unique(np.round(np.exp(np.linspace(small, large, number))))
    sequence_p = sequence.tolist()
    return sequence_p


def share_completed(current, total):
    """Counter for how much of a task is completed.

    Parameters
    ----------
    current : INT. No of tasks completed.
    total : INT. Total number of tasks.

    Returns
    -------
    None.

    """
    if current == 1:
        print("\nShare completed (%):", end=" ")
    share = current / total * 100

    if total < 20:
        print('{:4.0f}'.format(share), end=" ", flush=True)
    else:
        points_to_print = range(1, total, round(total/20))
        if current in points_to_print:
            print('{:4.0f}'.format(share), end=" ", flush=True)
    if current == total:
        print('Task completed')


def get_key_values_in_list(dic):
    """Create two lists with keys and values of dictionaries.

    Parameters
    ----------
    dic : Dictionary.

    Returns
    -------
    key_list : List of keys.
    value_list : List of values.

    """
    key_list = []
    value_list = []
    for keys in dic.keys():
        key_list += [keys]
        value_list += [dic[keys]]
    return key_list, value_list


def statistics_covariates(indatei, dict_var_type):
    """Analyse covariates wrt to variable type.

    Parameters
    ----------
    indatei : string. File with data.
    dict_var_type : dictionary with covariates and their type.

    Returns
    -------
    None.

    """
    data = pd.read_csv(indatei)
    print('\nCovariates that enter forest estimation', '(categorical',
          'variables are recoded with primes)')
    print('-' * 80)
    print('Name of variable             ', 'Type of variable         ',
          '# unique values', '  Min     ', 'Mean      ', 'Max       ')
    for keys in dict_var_type.keys():
        x_name = keys
        print('{:30}'.format(keys), end='')
        if dict_var_type[keys] == 0:
            print('ordered                     ', end='')
        elif dict_var_type[keys] == 1:
            print('unordered categorical short ', end='')
        elif dict_var_type[keys] == 2:
            print('unordered categorical long  ', end='')
        print('{:<12}'.format(len(data[x_name].unique())), end='')
        print('{:10.5f}'.format(data[x_name].min()), end='')
        print('{:10.5f}'.format(data[x_name].mean()), end='')
        print('{:10.5f}'.format(data[x_name].max()))


def dic_get_list_of_key_by_item(dic, value):
    """Get list of keys by item of a dictionary.

    Parameters
    ----------
    dic : Dictionary.
    value : Particular value of interest.

    Returns
    -------
    key_list : List of keys that have the value VALUE.

    """
    key_list = []
    for keys in dic.keys():
        if dic[keys] in value:
            key_list += [keys]
    if key_list == []:
        print('Retrieving items from list was not succesful')
        sys.exit()
    return key_list


def list_product(factors):
    """Prodcuce a product of a list keeping python data format.

    Parameters
    ----------
    factors : List.

    Returns
    -------
    prod : INT or Float.

    """
    return math.prod(factors)  # should be faster and keep python format


# @njit  Does not help with pure numpy functions
def make_np_product(factors):
    """Compute product of all factors.

    Parameters
    ----------
    factors : INT.
    Returns: INT: product of all factors.

    """
    return np.prod(factors)  # This leads to a numpy format


def primes_reverse(number, int_type=True):
    """Give the prime factors of integers.

    Parameters
    ----------
    number : INT, the variable to split into prime factors
    int_type : Boolean: One of number is of type INT32 or INT64.
                        The default is True.
                        It is easier to use TRUE in other operations, but with
                        False it may be possible to pass (and split) much
                        larger numbers
    Returns
    -------
    list_of_primes : INT (same as input)

    """
    if int_type:
        number = number.tolist()
    list_of_primes = primefactors(number)  # Should be faster
    return list_of_primes


def substitute_variable_name(v_dict, old_name, new_name):
    """Exchanges values in a dictionary.

    Parameters
    ----------
    v_dict : Dictionary
    old_name : string, Value to change
    new_name : string, new value

    Returns
    -------
    v_dict : Dictionary with changed names

    """
    vn_dict = copy.deepcopy(v_dict)
    for i in v_dict:
        list_of_this_dic = v_dict[i]
        if list_of_this_dic is not None:
            for j, _ in enumerate(list_of_this_dic):
                if list_of_this_dic[j] == old_name:
                    list_of_this_dic[j] = new_name
            vn_dict[i] = list_of_this_dic
    return vn_dict


def statistics_by_treatment(indatei, treat_name, var_name):
    """Descriptive statistics by treatment status.

    Parameters
    ----------
    indatei : String. Input data
    treat_name : String. Name of treatment
    var_name : List of strings. Name of variables to describe

    No Returns
    """
    print('\n-------------- Statistics by treatment status ------------------')
    data = pd.read_csv(filepath_or_buffer=indatei, header=0)
    data = data[treat_name+var_name]
    mean = data.groupby(treat_name).mean()
    std = data.groupby(treat_name).std()
    count = data.groupby(treat_name).count()
    count2 = data[treat_name+[var_name[0]]].groupby(treat_name).count()
    pd.set_option('display.max_rows', len(data.columns),
                  'display.max_columns', 10)
    print()
    print('Number of observations:')
    print(count2.transpose())
    print('\nMean')
    print(mean.transpose())
    print('\nMedian')
    print(data.groupby(treat_name).median().transpose())
    print('\nStandard deviation')
    print(std.transpose())
    balancing_tests(mean, std, count)


def balancing_tests(mean, std, count):
    """Compute balancing tests.

    Parameters
    ----------
    mean : Dataframe: Means by treatment groups.
    std : Dataframe: Standard deviation by treatment groups.

    No Returns.

    """
    std = std.replace(to_replace=0, value=-1)
    # no_of_treat = mean.shape[0]
    value_of_treat = list(reversed(mean.index))
    value_of_treat2 = value_of_treat[:]
    for i in value_of_treat:
        if i >= value_of_treat[-1]:
            value_of_treat2.remove(i)
            for j in value_of_treat2:
                mean_diff = mean.loc[i, :] - mean.loc[j, :]
                std_diff = np.sqrt((std.loc[i, :]**2) / count.loc[i]
                                   + (std.loc[j, :]**2) / count.loc[j])
                t_diff = mean_diff.div(std_diff).abs()
                p_diff = 2 * sct.norm.sf(t_diff) * 100
                stand_diff = (mean_diff / np.sqrt((std.loc[i, :]**2 +
                                                   std.loc[j, :]**2) / 2) *
                              100)
                stand_diff.abs()
                print('\nComparing treatments {0:>2}'.format(i),
                      'and {0:>2}'.format(j))
                print('Variable                          Mean       Std',
                      '        t-val   p-val (%) Stand.Difference (%)')
                for jdx, _ in enumerate(mean_diff):
                    print('{:30}'.format(mean_diff.index[jdx]),
                          '{:10.5f}'.format(mean_diff[jdx]),
                          '{:10.5f}'.format(std_diff[jdx]),
                          '{:9.2f}'.format(t_diff[jdx]),
                          '{:9.2f}'.format(p_diff[jdx]),
                          '{:9.2f}'.format(stand_diff[jdx]))


def sample_split_2_3(indatei, outdatei1, share1, outdatei2, share2,
                     outdatei3=None, share3=0, random_seed=None,
                     with_output=True):
    """Split sample in 2 or 3 random subsamples (depending on share3 > 0).

    Parameters
    ----------
    indatei : String. File with data to split
    outdatei*: String. Files with splitted data
    random_seed: Fixes seed of random number generator
    with_output: Some infos about split.

    Returns
    -------
    outdatei1,outdatei2,outdatei3: names of files with splitted data

    """
    if ((share1+share2+share3) > 1.01) or ((share1 + share2 + share3) < 0.99):
        print('Sample splitting: Shares do not add up to 1')
        sys.exit()
    data = pd.read_csv(filepath_or_buffer=indatei, header=0)
    # split into 2 or 3 dataframes
    data1 = data.sample(frac=share1, random_state=random_seed)
    if share3 == 0:
        data2 = data.drop(data1.index)
    else:
        tmp = data.drop(data1.index)
        data2 = tmp.sample(frac=share2/(1-share1), random_state=random_seed)
        data3 = tmp.drop(data2.index)
    delete_file_if_exists(outdatei1)
    delete_file_if_exists(outdatei2)
    data1.to_csv(outdatei1, index=False)
    data2.to_csv(outdatei2, index=False)
    if share3 != 0:
        delete_file_if_exists(outdatei3)
        data3.to_csv(outdatei3, index=False)
    if with_output:
        print('\nRandom sample splitting')
        print('Number of obs. in org. data', indatei,
              '{0:>5}'.format(data.shape[0]))
        print('Number of obs. in', outdatei1,
              '{0:>5}'.format(data1.shape[0]))
        print('Number of obs. in', outdatei2,
              '{0:>5}'.format(data2.shape[0]))
        if share3 != 0:
            print('Number of observations in', outdatei3,
                  '{0:>5}'.format(data3.shape[0]))
    return outdatei1, outdatei2, outdatei3


def adjust_vars_vars(var_in, var_weg):
    """Remove from VAR_IN those strings that are also in VAR_WEG.

    Parameters
    ----------
    var_in : list of strings.
    var_weg : list of strings.

    Returns
    -------
    ohne_var_weg : list of strings

    """
    v_inter = set(var_in).intersection(set(var_weg))
    if not v_inter == set():
        ohne_var_weg = list(set(var_in)-v_inter)
    else:
        ohne_var_weg = copy.deepcopy(var_in)
    return ohne_var_weg


def screen_variables(indatei, var_names, perfectcorr, min_dummy, with_output):
    """Screen variables to decide whether they could be deleted.

    Parameters
    ----------
    indatei : string
    var_names : list of strings
    perfectcorr : Boolean
        Check for too high correlation among variables
    min_dummy : integer
        Check for minimum number of observation on both side of dummies
    with_output : string

    Returns
    -------
    variables_remain : List of strings
    variables_delete : list of strings

    """
    data = pd.read_csv(filepath_or_buffer=indatei, header=0)
    data = data[var_names]
    k = data.shape[1]
    all_variable = set(data.columns)
    # Remove variables without any variation
    to_keep = data.std(axis=0) > 1e-10
    datanew = data.loc[:, to_keep].copy()  # Keeps the respective columns
    if with_output:
        print('\n')
        print('-' * 80)
        print('Control variables checked')
        kk1 = datanew.shape[1]
        all_variable1 = set(datanew.columns)
        if kk1 != k:
            deleted_vars = list(all_variable - all_variable1)
            print('Variables without variation: ', deleted_vars)
    if perfectcorr:
        corr_matrix = datanew.corr().abs()
        # Upper triangle of corr matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                          k=1).astype(np.bool))
        to_delete = [c for c in upper.columns if any(upper[c] > 0.999)]
        if not to_delete == []:
            datanew = datanew.drop(columns=to_delete)
    if with_output:
        kk2 = datanew.shape[1]
        all_variable2 = set(datanew.columns)
        if kk2 != kk1:
            deleted_vars = list(all_variable1-all_variable2)
            print('Correlation with other variable > 99.9%: ', deleted_vars)
    if min_dummy > 1:
        shape = datanew.shape
        to_keep = []
        for cols in range(shape[1]):
            value_count = datanew.iloc[:, cols].value_counts()
            if len(value_count) == 2:  # Dummy variable
                if value_count.min() < min_dummy:
                    to_keep = to_keep + ([False])
                else:
                    to_keep = to_keep + ([True])
            else:
                to_keep = to_keep + ([True])
        datanew = datanew.loc[:, to_keep]
    if with_output:
        kk3 = datanew.shape[1]
        all_variable3 = set(datanew.columns)
        if kk3 != kk2:
            deleted_vars = list(all_variable2-all_variable3)
            print('Dummy variables with too few 0 or 1: ', deleted_vars)
    kk2 = datanew.shape[1]
    variables_remain = datanew.columns
    variables_delete = list(all_variable-set(datanew.columns))
    if with_output:
        if kk2 == k:
            print('All control variables have been retained')
        else:
            print('\nThe following variables have been deleted:',
                  variables_delete)
            desc_stat = data[variables_delete].describe()
            to_print = desc_stat.transpose()
            pd.set_option('display.max_rows', len(desc_stat.columns),
                          'display.max_columns', len(to_print.columns))
            print(to_print)
            pd.reset_option('display.max_rows', 'display.max_columns')
            print('\nThe following variables have been retained:',
                  variables_remain)
    return variables_remain, variables_delete


def clean_reduce_data(infile, outfile, names_to_inc, with_output, desc_stat,
                      print_to_file):
    """Remove obs. with missings and variables not included in 'names_to_inc'.

    Parameters     (does not check whether all variables exist)
    ----------
    infile : STRING.        CSV-file with input data.
    outfile : STRING.        CSV-file with output data.
    names_to_inc : list of strings.        Variables to be kept in data.
    with_output : Boolean.        Give some information
    desc_stat : Boolean.        Show descriptive stats of new file.

    Returns
    -------
    outfile : string
        Output file.
    """
    data = pd.read_csv(filepath_or_buffer=infile)
    shape = data.shape
    data.columns = [s.upper() for s in data.columns]
    datanew = data[names_to_inc]
    datanew.dropna()
    shapenew = datanew.shape
    delete_file_if_exists(outfile)
    datanew.to_csv(outfile, index=False)
    if with_output:
        print("\nCheck for missing and unnecessary variables.")
        if shape == shapenew:
            print('File has not been changed')
        else:
            if shapenew[0] == shape[0]:
                print('  No observations deleted')
            else:
                print('  {0:<5}'.format(shape[0]-shapenew[0]),
                      'observations deleted')
            if shapenew[1] == shape[1]:
                print('No variables deleted')
            else:
                print('  {0:<4}'.format(shape[1]-shapenew[1]),
                      'variables deleted:')
                liste = list(data.columns.difference(datanew.columns))
                print(*liste)
        if desc_stat:
            print_descriptive_stats_file(outfile, 'all', print_to_file)
    return outfile


def cleaned_var_names(var_name):
    """Clean variable names.

    Cleaning variable by removing empty list and zero and None and putting
    everything to upper case & removing duplicates

    Parameters
    ----------
    var_name : List with variable names

    Returns
    -------
    var_name2 : List with variable names

    """
    var_name1 = [s.upper() for s in var_name]
    var_name2 = []
    for var in var_name1:
        if (var not in var_name2) and (var != '0') and (var != 0) and (
                var != []) and (var is not None):
            var_name2.append(var)
    return var_name2


def add_var_names(names1, names2=None, names3=None, names4=None, names5=None,
                  names6=None, names7=None, names8=None, names9=None,
                  names10=None):
    """Return a list of strings with unique entries.

    Parameters
    ----------
    names1 to 10: lists of strings
        Names to merge.

    Returns
    -------
    new_names : list of strings
        All unique names in one list.

    """
    if names2 is None:
        names2 = []
    if names3 is None:
        names3 = []
    if names4 is None:
        names4 = []
    if names5 is None:
        names5 = []
    if names6 is None:
        names6 = []
    if names7 is None:
        names7 = []
    if names8 is None:
        names8 = []
    if names9 is None:
        names9 = []
    if names10 is None:
        names10 = []
    new_names = copy.deepcopy(names1)
    new_names.extend(names2)
    new_names.extend(names3)
    new_names.extend(names4)
    new_names.extend(names5)
    new_names.extend(names6)
    new_names.extend(names7)
    new_names.extend(names8)
    new_names.extend(names9)
    new_names.extend(names10)
    new_names = cleaned_var_names(new_names)
    return new_names


def print_descriptive_stats_file(indata, varnames='all', to_file=False):
    """Print descriptive statistics of a dataset on file.

    Parameters
    ----------
    indata : string, Name of input file
    varnames : List of strings, Variable names of selected variables
              The default is 'all'.

    Returns
    -------
    None.
    """
    data = pd.read_csv(filepath_or_buffer=indata, header=0)
    if varnames != 'all':
        data_sel = data[varnames]
    else:
        data_sel = data
    desc_stat = data_sel.describe()
    if (varnames == 'all') or len(varnames) > 10:
        to_print = desc_stat.transpose()
        rows = len(desc_stat.columns)
        cols = len(desc_stat.index)
    else:
        to_print = desc_stat
        rows = len(desc_stat.index)
        if isinstance(desc_stat, pd.DataFrame):
            cols = len(desc_stat.columns)
        else:
            cols = 1
    print('\nData set:', indata)
    if to_file:
        expand = False
    else:
        expand = True
    with pd.option_context('display.max_rows', rows,
                           'display.max_columns', cols+1,
                           'display.expand_frame_repr', expand,
                           'chop_threshold', 1e-13):
        print(to_print)


def check_all_vars_in_data(indata, variables):
    """Check whether all variables are contained in data set.

    Parameters
    ----------
    indata : string. Input data.
    variables : list of strings. Variables to check indata for.

    Returns
    -------
    None. Programme terminates if some variables are not found

    """
    headers = pd.read_csv(filepath_or_buffer=indata, nrows=0)
    header_list = [s.upper() for s in list(headers.columns.values)]
    all_available = all(i in header_list for i in variables)
    if not all_available:
        missing_variables = [i for i in variables if i not in header_list]
        print('\nVariables not in ', indata, ':', missing_variables)
        sys.exit()
    else:
        print("\nAll variables found in ", indata)


def load_module_path(name_modul, path_modul):
    """Load modules with given path and name.

    Parameters
    ----------
    name_modul : string. Name of module to be used
    path_modul : string. Full name of file that contains modul

    Returns
    -------
    modul : name space of modul

    """
    spec = importlib.util.spec_from_file_location(name_modul, path_modul)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def print_timing(text, time_diff):
    """Show date and duration of a programme and its different parts.

    Parameters
    ----------
    text : list of strings
        Explanation for the particular time difference
    time_diff :time differnce in time2-time1 [time.time() format]
        Time difference

    """
    print('\n\n')
    print('-' * 80)
    print("Programme executed at: ", datetime.now())
    print('-' * 80)
    for i in range(0, len(text), 1):
        print(text[i], timedelta(seconds=time_diff[i]))
    print('-' * 80, '\n')


def randomsample(datapath, indatafile, outdatafile, fraction,
                 replacement=False):
    """Generate a random sample of the data in a file.

    Parameters
    ----------
    datapath : string.        location of files.
    indatafile : string.        input file as csv file.
    outdatafile : string.        output file as csv file.
    fraction : float.  share of sample to be randomly put into output file.
    replacement : boolean, optinal.
        True: Sampling with replacement ; False: without rplm.

    """
    indatafile = datapath + '/' + indatafile
    outdatafile = datapath + '/' + outdatafile
    data = pd.read_csv(filepath_or_buffer=indatafile, header=0)
    data = data.sample(frac=fraction, replace=replacement)
    delete_file_if_exists(outdatafile)
    data.to_csv(outdatafile, index=False)


def primeposition(x_values, start_with_1=False):
    """
    Give position of elements of x_values in list of primes.

    Parameters
    ----------
    x_values : List of int.

    Returns
    -------
    position : List of int.

    """
    if start_with_1:
        add = 1
    else:
        add = 0
    primes = primes_list(1000)
    position = []
    for val in x_values:
        position.append(primes.index(val)+add)
    return position


def primes_list(number=1000):
    """List the first 1000 prime numbers.

    Parameters
    ----------
    number : INT. length of vector with the first primes.

    Returns
    -------
    primes: list of INT, prime numbers

    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
              127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
              193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
              269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347,
              349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
              431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
              503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593,
              599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661,
              673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757,
              761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839,
              853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937,
              941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019,
              1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091,
              1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171,
              1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249,
              1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319,
              1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429,
              1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489,
              1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571,
              1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637,
              1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733,
              1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823,
              1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907,
              1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999,
              2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083,
              2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153,
              2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267,
              2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341,
              2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411,
              2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521,
              2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617,
              2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689,
              2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753,
              2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843,
              2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939,
              2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037,
              3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137,
              3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229,
              3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323,
              3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407,
              3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511,
              3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571, 3581,
              3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659, 3671,
              3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761,
              3767, 3769, 3779, 3793, 3797, 3803, 3821, 3823, 3833, 3847, 3851,
              3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919, 3923, 3929,
              3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021,
              4027, 4049, 4051, 4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127,
              4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201, 4211, 4217, 4219,
              4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289,
              4297, 4327, 4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409,
              4421, 4423, 4441, 4447, 4451, 4457, 4463, 4481, 4483, 4493, 4507,
              4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597,
              4603, 4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679,
              4691, 4703, 4721, 4723, 4729, 4733, 4751, 4759, 4783, 4787, 4789,
              4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903,
              4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973,
              4987, 4993, 4999, 5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059,
              5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153, 5167,
              5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273,
              5279, 5281, 5297, 5303, 5309, 5323, 5333, 5347, 5351, 5381, 5387,
              5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441, 5443, 5449,
              5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531,
              5557, 5563, 5569, 5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651,
              5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701, 5711, 5717, 5737,
              5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827,
              5839, 5843, 5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897,
              5903, 5923, 5927, 5939, 5953, 5981, 5987, 6007, 6011, 6029, 6037,
              6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121,
              6131, 6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217,
              6221, 6229, 6247, 6257, 6263, 6269, 6271, 6277, 6287, 6299, 6301,
              6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373,
              6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491,
              6521, 6529, 6547, 6551, 6553, 6563, 6569, 6571, 6577, 6581, 6599,
              6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691, 6701,
              6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793,
              6803, 6823, 6827, 6829, 6833, 6841, 6857, 6863, 6869, 6871, 6883,
              6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967, 6971, 6977,
              6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069,
              7079, 7103, 7109, 7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193,
              7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247, 7253, 7283, 7297,
              7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417,
              7433, 7451, 7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517,
              7523, 7529, 7537, 7541, 7547, 7549, 7559, 7561, 7573, 7577, 7583,
              7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681,
              7687, 7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759,
              7789, 7793, 7817, 7823, 7829, 7841, 7853, 7867, 7873, 7877, 7879,
              7883, 7901, 7907, 7919]
    return primes[0:number]


def pause():
    """Pause."""
    inp = input("Press the <ENTER> key to continue...")
    return inp


def print_size_weight_matrix(weights, weight_as_sparse, no_of_treat):
    """
    Print size of weight matrix in MB.

    Parameters
    ----------
    weights : Sparse (CSR) or dense 2D Numpy array. Weight matrix.
    weight_as_sparse : Boolean.
    no_of_treat : Int. Number of treatments.

    Returns
    -------
    None.

    """
    total_bytes = total_size(weights)
    if weight_as_sparse:
        for d_idx in range(no_of_treat):
            total_bytes += (weights[d_idx].data.nbytes
                            + weights[d_idx].indices.nbytes
                            + weights[d_idx].indptr.nbytes)
    print('Size of weight matrix: ', round(total_bytes / (1024 * 1024), 2),
          ' MB')


def no_of_boot_splits_fct(size_of_object_mb, workers, with_output=True):
    """
    Compute size of chunks for MP.

    Parameters
    ----------
    size_of_forest_MB : Float. Size of the object in MB.
    workers : Int. Number of workers in MP.
    with_output : Boolean. optional. Print or not. The default is True.

    Returns
    -------
    no_of_splits : Int. Number of splits.

    """
    basic_size_mb = 53
    _, available, _, _ = memory_statistics(with_output=False)
    if size_of_object_mb > basic_size_mb:
        multiplier = 1/8 * (14 / workers)
        chunck_size_mb = basic_size_mb * (1 + (available - 33000) / 33000
                                          * multiplier)
        if chunck_size_mb > 2000:
            chunck_size_mb = 2000
        if chunck_size_mb < 10:
            chunck_size_mb = 10
        no_of_splits = math.ceil(size_of_object_mb / chunck_size_mb)
    else:
        no_of_splits = 1
        chunck_size_mb = size_of_object_mb
    if with_output:
        print()
        print('Automatic determination of tree batches')
        print('Size of object:   {:6} MB '.format(round(size_of_object_mb, 2)),
              'Available RAM: {:6} MB '.format(available),
              'Number of workers {:2} '.format(workers),
              'No of splits: {:2} '.format(no_of_splits),
              'Size of chunk:  {:6} MB '.format(round(chunck_size_mb, 2)))
        memory_statistics()
    return no_of_splits


def find_no_of_workers(maxworkers, sys_share=0):
    """
    Find the optimal number of workers for MP such that system does not crash.

    Parameters
    ----------
    maxworkers : Int. Maximum number of workers allowed.

    Returns
    -------
    workers : Int. Workers used.
    sys_share: Float. System share.
    max_cores: Bool. Limit to number of physical(not logical cores)

    """
    share_used = getattr(psutil.virtual_memory(), 'percent') / 100
    if sys_share >= share_used:
        sys_share = 0.9 * share_used
    sys_share = sys_share / 2
    workers = (1-sys_share) / (share_used-sys_share)
    if workers > maxworkers:
        workers = maxworkers
    elif workers < 1.9:
        workers = 1
    else:
        workers = maxworkers
    workers = math.floor(workers + 1e-15)
    return workers


def clean_futures():
    """Clean up workers in memory."""
    workers = psutil.cpu_count()
    with futures.ProcessPoolExecutor() as fpp:
        ret_fut = {fpp.submit(do_almost_nothing, i): i for i in range(workers)}
    del ret_fut


def do_almost_nothing(value):
    """Do almost nothing."""
    value += 1
    return value


def memory_statistics(with_output=True):
    """
    Give memory statistics.

    Parameters
    ----------
    with_output : Boolean. Print output. The default is True.

    Returns
    -------
    total : Float. Total memory in GB.
    available : Float. Available memory in GB.
    used : Float. Used memory in GB.
    free : Float. Free memory in GB.

    """
    memory = psutil.virtual_memory()
    total = round(memory.total / (1024 * 1024), 2)
    available = round(memory.available / (1024 * 1024), 2)
    used = round(memory.used / (1024 * 1024), 2)
    free = round(memory.free / (1024 * 1024), 2)
    if with_output:
        print(
            'RAM total: {:6} MB, '.format(total),
            'used: {:6} MB, '.format(used),
            'available: {:6} MB, '.format(available),
            'free: {:6} MB'.format(free))
    return total, available, used, free


def total_size(ooo, handlers=None, verbose=False):
    """Return the approximate memory footprint an object & all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, (deque), dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
        https://code.activestate.com/recipes/577504/

    """
    #  dict_handler = lambda d: chain.from_iterable(d.items())
    if handlers is None:
        handlers = {}

    def dict_handler(ddd):
        return chain.from_iterable(ddd.items())

    all_handlers = {tuple: iter,
                    list: iter,
                    # deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter}
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()               # track which object id's have already been seen
    default_size = sys.getsizeof(0)
    # estimate sizeof object without __sizeof__

    def sizeof(ooo):
        if id(ooo) in seen:       # do not double count the same object
            return 0
        seen.add(id(ooo))
        sss = sys.getsizeof(ooo, default_size)

        if verbose:
            print(sss, type(ooo), repr(ooo), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(ooo, typ):
                sss += sum(map(sizeof, handler(ooo)))
                break
        return sss

    return sizeof(ooo)


def total_sample_splits_categorical(no_of_values):
    """
    Compute total number of sample splits that can generated by categoricals.

    Parameters
    ----------
    no_of_values : Int.

    Returns
    -------
    no_of_splits: Int.

    """
    no_of_splits = 0
    for i in range(1, no_of_values):
        no_of_splits += math.factorial(no_of_values) / (
            math.factorial(no_of_values-i) * math.factorial(i))
    return no_of_splits/2  # no complements


def all_combinations_no_complements(values):
    """Create all possible combinations of list elements, removing complements.

    Parameters
    ----------
    values : List. Elements to be combined.

    Returns
    -------
    list_without_complements : List of tuples.

    """
    list_all = []
    # This returns a list with tuples of all possible combinations of tuples
    for length in range(1, len(values)):
        list_all.extend(list(itertools.combinations(values, length)))
    # Next, the complements to each list will be removed
    list_wo_compl, _ = drop_complements(list_all, values)
    return list_wo_compl


def drop_complements(list_all, values):
    """
    Identify and remove complements.

    Parameters
    ----------
    list_all : List of tuples. Tuples with combinations.
    values : List. All relevant values.

    Returns
    -------
    list_wo_compl : List of Tuples. List_all with complements removed.

    """
    list_w_compl = []
    list_wo_compl = []
    for i in list_all:
        if i not in list_w_compl:
            list_wo_compl.append(i)
            compl_of_i = values[:]
            for drop_i in i:
                compl_of_i.remove(drop_i)
            list_w_compl.append(tuple(compl_of_i))
    return list_wo_compl, list_w_compl


def resort_list(liste, idx_list, n_x):
    """
    Make sure list is in the same order as some index_variable.

    Parameters
    ----------
    liste : List of lists.
    idx_list : List with indices as finished by multiprocessing.
    n_x : Int. Number of observations.

    Returns
    -------
    weights : As above, but sorted.

    """
    check_idx = list(range(n_x))
    if check_idx != idx_list:
        liste = [liste[i] for i in idx_list]
    return liste
