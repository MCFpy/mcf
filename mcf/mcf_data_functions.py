"""Created on Fri Apr  3 11:05:15 2020.

Contains the functions needed for data manipulation
@author: MLechner
-*- coding: utf-8 -*-
"""
import copy
import math
from concurrent import futures
from dask.distributed import Client, as_completed

import numpy as np
import pandas as pd
import ray

from mcf import general_purpose as gp
from mcf import general_purpose_estimation as gp_est
from mcf import mcf_general_purpose as mcf_gp


def variable_features(var_x_type, var_x_values):
    """
    Show variables and their key features.

    Parameters
    ----------
    var_x_type : Dict. Name and type of variable.
    var_x_values : Dict. Name and values of variables.

    Returns
    -------
    None.

    """
    print('\n')
    print(80 * '=')
    print('Features used to build causal forests')
    print(80 * '-')
    for name in var_x_type.keys():
        print(f'{name:20} ', end=' ')
        if var_x_type[name] == 0:
            print('Ordered   ', end='')
            if var_x_values[name]:
                if isinstance(var_x_values[name][0], float):
                    for val in var_x_values[name]:
                        print(f'{val:>6.2f}', end=' ')
                    print(' ')
                else:
                    print(var_x_values[name])
            else:
                print('Continuous')
        else:
            print('Unordered ', len(var_x_values[name]), ' different values')
    print(80 * '-')


def prepare_data_for_forest(indatei, v_dict, v_x_type, v_x_values, c_dict,
                            no_y_nn=False, regrf=False):
    """Prepare data for Forest and variable importance estimation.

    Parameters
    ----------
    indatei : String. CSV file.
    v_dict : DICT. Variables.
    v_x_type : List. Type of variables.
    v_x_values : List. Values of variables (if not continuous).
    c_dict : DICT. Parameters.
    no_y_nn : Bool.
    regrf : Bool.

    Returns
    -------
    x_name : Dict.
    x_type :Dict.
    x_values : Dict.
    c : Dict. Parameters (updated)
    pen_mult : INT. Multiplier for penalty.
    data_np : Numpy array. Data for estimation.
    y_i : INT. Index of position of y in data_np.
    y_nn_i : Numpy array of INT.
    x_i : Numpy array of INT.
    x_ind : Numpy array of INT.
    x_ai_ind : Numpy array of INT.
    d_i : INT.
    w_i : INT.
    cl_i : INT.

    """
    x_name, x_type = gp.get_key_values_in_list(v_x_type)
    x_type = np.array(x_type)
    x_name2, x_values = gp.get_key_values_in_list(v_x_values)
    pen_mult = 0
    if x_name != x_name2:
        raise Exception('Wrong order of names', x_name, x_name2)
    p_x = len(x_name)     # Number of variables
    c_dict = m_n_grid(c_dict, p_x)  # Grid for # of var's used for split
    x_ind = np.array(range(p_x))    # Indices instead of names of variable
    if not v_dict['x_name_always_in'] == []:
        x_ai_ind = np.array(
            [i for i in range(p_x) if x_name[i] in v_dict['x_name_always_in']])
    else:
        x_ai_ind = []              # Indices of variables used for all splits
    data = pd.read_csv(indatei)
    y_dat = data[v_dict['y_tree_name']].to_numpy()
    if not regrf:
        if c_dict['mtot'] in (1, 4):
            pen_mult = c_dict['mtot_p_diff_penalty'] * np.var(y_dat)
    y_i = [0]
    if regrf:
        d_dat = d_i = None
    else:
        d_dat, d_i = data[v_dict['d_name']].to_numpy(), [1]
    if regrf:
        y_nn = y_nn_i = None
    else:
        y_nn = (np.zeros((len(d_dat), len(v_dict['y_match_name'])))
                if no_y_nn else data[v_dict['y_match_name']].to_numpy())
        y_nn_i = np.arange(2, 2 + len(v_dict['y_match_name']))
    x_dat = data[x_name].to_numpy()
    for col_indx in range(np.shape(x_dat)[1]):
        if x_type[col_indx] > 0:
            x_dat[:, col_indx] = np.around(x_dat[:, col_indx])
    if regrf:
        x_i = np.arange(1, 1+len(v_dict['x_name']))
        data_np = np.concatenate((y_dat, x_dat), axis=1)  # easier handling
    else:
        x_i = np.arange(
            2 + len(v_dict['y_match_name']),
            2 + len(v_dict['y_match_name']) + len(v_dict['x_name']))
        data_np = np.concatenate((y_dat, d_dat, y_nn, x_dat), axis=1)
    if c_dict['w_yes']:
        w_dat = data[v_dict['w_name']].to_numpy()
        data_np = np.concatenate((data_np, w_dat), axis=1)
        w_i = data_np.shape[1] - 1
    else:
        w_i = None
    if c_dict['panel_in_rf']:
        cl_dat = data[v_dict['cluster_name']].to_numpy()
        data_np = np.concatenate((data_np, cl_dat), axis=1)
        cl_i = data_np.shape[1] - 1
    else:
        cl_i = None
    if c_dict['d_type'] == 'continuous':
        d_grid_dat = data[v_dict['d_grid_nn_name']].to_numpy()
        data_np = np.concatenate((data_np, d_grid_dat), axis=1)
        d_grid_i = data_np.shape[1] - 1
    else:
        d_grid_i = None
    return (x_name, x_type, x_values, c_dict, pen_mult, data_np,
            y_i, y_nn_i, x_i, x_ind, x_ai_ind, d_i, w_i, cl_i, d_grid_i)


def m_n_grid(c_dict, no_vars):
    """Generate the grid for the # of coefficients (log-scale).Sort n_min grid.

    Parameters
    ----------
    c_dict : Dict. Parameters of MCF estimation
    no_vars : INT. Number of x-variables used for tree building

    Returns
    -------
    c : Dict. Updated (globally) dictionary with parameters.

    """
    m_min = round(c_dict['m_min_share'] * no_vars)
    m_min = max(m_min, 1)
    m_max = round(c_dict['m_max_share'] * no_vars)
    if m_min == m_max:
        c_dict['m_grid'] = 1
        grid_m = m_min
    else:
        if c_dict['m_grid'] == 1:
            grid_m = round((m_min + m_max)/2)
        else:
            grid_m = gp.grid_log_scale(m_max, m_min, c_dict['m_grid'])
            grid_m = [int(idx) for idx in grid_m]
    if np.size(c_dict['grid_n_min']) > 1:
        c_dict['grid_n_min'] = sorted(c_dict['grid_n_min'], reverse=True)
    c_dict.update({'grid_m': grid_m})  # changes this dictionary globally
    return c_dict


def nn_neighbour_mcf(y_dat, x_dat, d_dat, obs, cov_x_inv, treat_values,
                     i_treat):
    """Find nearest neighbour-y in subsamples by value of d.

    Parameters
    ----------
    y_dat : Numpy array: Outcome variable
    x_dat : Numpy array: Covariates
    d_dat : Numpy array: Treatment
    obs : INT64: Sample size
    cov_x_inv : Numpy array: inverse of covariance matrix
    treat_values : Numpy array: possible values of D
    i_treat : Position treat_values investigated

    Returns
    -------
    y_all : Numpy series with matched values.
    i_treat: see above (included to ease mulithreading which may confuse
                        positions).
    """
    treat_value = treat_values[i_treat]
    cond = d_dat[:, 0] == treat_value
    x_t = x_dat[cond, :]
    y_t = y_dat[cond, :]
    y_all = np.empty([obs, 1])
    for i in range(obs):
        if treat_value == d_dat[i, 0]:
            y_all[i, 0] = y_dat[i, 0]
        else:
            diff = x_t - x_dat[i, :]
            dist = np.sum(np.dot(diff, cov_x_inv) * diff, axis=1)
            min_ind = np.argmin(dist)
            y_all[i, 0] = np.copy(y_t[min_ind, 0])
    return y_all, i_treat  # i_treat is returned for multithreading


@ray.remote
def ray_nn_neighbour_mcf2(y_dat, x_dat, d_dat, obs, cov_x_inv, treat_value,
                          i_treat=None):
    """Make procedure compatible for Ray."""
    return nn_neighbour_mcf2(y_dat, x_dat, d_dat, obs, cov_x_inv, treat_value,
                             i_treat)


def nn_neighbour_mcf2(y_dat, x_dat, d_dat, obs, cov_x_inv, treat_value,
                      i_treat=None):
    """Find nearest neighbour-y in subsamples by value of d.

    Parameters
    ----------
    y_dat : Numpy array: Outcome variable
    x_dat : Numpy array: Covariates
    d_dat : Numpy array: Treatment
    obs : INT64: Sample size
    cov_x_inv : Numpy array: inverse of covariance matrix
    treat_values : Numpy array: possible values of D
    i_treat : Position treat_values investigated

    Returns
    -------
    y_all : Numpy series with matched values.
    i_treat: see above (included to ease mulithreading which may confuse
                        positions).

    """
    cond = (d_dat == treat_value).reshape(-1)
    x_t = x_dat[cond, :]
    y_t = y_dat[cond]
    y_all = np.empty(obs)
    for i in range(obs):
        if treat_value == d_dat[i]:
            y_all[i] = y_dat[i]
        else:
            diff = x_t - x_dat[i, :]
            dist = np.sum(np.dot(diff, cov_x_inv) * diff, axis=1)
            min_ind = np.nonzero(dist <= (dist.min() + 1e-15))
            y_all[i] = np.mean(y_t[min_ind])
    if i_treat is None:
        return y_all  # i_treat is returned for multithreading
    return y_all, i_treat  # i_treat is returned for multithreading


def nn_matched_outcomes(indatei, v_dict, v_type, c_dict):
    """Nearest neighbor matching for outcome variables.

    Parameters
    ----------
    indatei : string.Data input
    v_dict : Dict with variables
    v_type : Dict with variable types (for dummy creation)
    c_dict : Dict with control parameters

    Returns
    -------
    string with name of new data file
    v_dict : Updated dictionary with names including

    """
    # read as pandas data
    data = pd.read_csv(filepath_or_buffer=indatei)
    if c_dict['x_type_1'] or c_dict['x_type_2']:  # Expand cat var's to dummies
        var_names_unordered = gp.dic_get_list_of_key_by_item(v_type, [1, 2])
        x_dummies = data[var_names_unordered].astype('category')
        x_dummies = pd.get_dummies(x_dummies)
    if c_dict['x_type_0']:
        var_names_ordered = gp.dic_get_list_of_key_by_item(v_type, [0])
        x_ord = data[var_names_ordered]
    if c_dict['x_type_0'] and (c_dict['x_type_1'] or c_dict['x_type_2']):
        x_df = pd.concat([x_dummies, x_ord], axis=1)
    elif c_dict['x_type_0'] and not (c_dict['x_type_1'] or c_dict['x_type_2']):
        x_df = x_ord
    else:
        x_df = x_dummies
    x_dat = x_df.to_numpy()
    y_dat = data[v_dict['y_tree_name']].to_numpy()
    if c_dict['d_type'] == 'continuous':
        d_name = v_dict['d_grid_nn_name']
        d_values = c_dict['ct_grid_nn_val']
    else:
        d_name = v_dict['d_name']
        d_values = c_dict['d_values']
    no_of_treat = len(d_values)
    d_dat = data[d_name].to_numpy()
    obs = len(x_dat)                        # Reduce x_dat to prognostic scores
    if (c_dict['match_nn_prog_score']
        and ((c_dict['mtot'] == 1) or (c_dict['mtot'] == 4))
            and (len(x_df.columns) >= 2 * no_of_treat)):
        if c_dict['with_output'] and c_dict['verbose']:
            print('Computing prognostic score for matching')
        c_dict['nn_main_diag_only'] = False   # Use full Mahalanobis matrix
        x_dat_neu = np.empty((obs, no_of_treat))
        for midx, d_val in enumerate(d_values):
            d_m = d_dat == d_val
            d_m = d_m.reshape(len(d_m))
            x_dat_m = x_dat[d_m, :].copy()
            y_dat_m = y_dat[d_m].copy()
            ret_rf = gp_est.random_forest_scikit(
                x_dat_m, y_dat_m, x_dat, boot=c_dict['boot'],
                n_min=c_dict['grid_n_min']/2,
                pred_p_flag=True, pred_t_flag=False,
                pred_oob_flag=False, with_output=False,
                variable_importance=False, x_name=x_df.columns,
                var_im_with_output=False, return_forest_object=False)
            x_dat_neu[:, midx] = np.copy(ret_rf[0])
        x_dat = x_dat_neu
    if (c_dict['match_nn_prog_score']
        and ((c_dict['mtot'] == 1) or (c_dict['mtot'] == 4))
            and (len(x_df.columns) < 2 * no_of_treat)):
        if c_dict['with_output'] and c_dict['verbose']:
            print('Prognostic scores not used for matching',
                  ' (too few covariates')
    if (c_dict['mtot'] == 1) or (c_dict['mtot'] == 4):
        y_match = np.empty((obs, no_of_treat))
        # determine distance metric of Mahalanobis matching
        k = np.shape(x_dat)
        if k[1] > 1:
            cov_x = np.cov(x_dat, rowvar=False)
            if c_dict['nn_main_diag_only']:
                cov_x = cov_x * np.eye(k[1])   # only main diag
            rank_not_ok = True
            counter = 0
            while rank_not_ok:
                if counter == 20:
                    cov_x *= np.eye(k[1])
                if counter > 20:
                    cov_x_inv = np.eye(k[1])
                    break
                if np.linalg.matrix_rank(cov_x) < k[1]:
                    cov_x += 0.5 * np.diag(cov_x) * np.eye(k[1])
                    counter += 1
                else:
                    cov_x_inv = np.linalg.inv(cov_x)
                    rank_not_ok = False
        else:
            cov_x_inv = 1
        if c_dict['no_parallel'] < 1.5:
            maxworkers = 1
        else:
            if c_dict['mp_automatic']:
                maxworkers = mcf_gp.find_no_of_workers(c_dict['no_parallel'],
                                                       c_dict['sys_share'])
            else:
                maxworkers = c_dict['no_parallel']
        if c_dict['with_output'] and c_dict['verbose']:
            print('Number of parallel processes: ', maxworkers)
        if maxworkers == 1:
            for i_treat, i_value in enumerate(d_values):
                y_match[:, i_treat] = nn_neighbour_mcf2(
                    y_dat, x_dat, d_dat, obs, cov_x_inv, i_value)
        else:  # Later on more workers needed
            if c_dict['_mp_ray_shutdown'] or c_dict['_ray_or_dask'] == 'dask':
                if maxworkers > math.ceil(no_of_treat/2):
                    maxworkers = math.ceil(no_of_treat/2)
            if c_dict['_ray_or_dask'] == 'ray':
                if not ray.is_initialized():
                    ray.init(num_cpus=maxworkers, include_dashboard=False)
                x_dat_ref = ray.put(x_dat)
                still_running = [ray_nn_neighbour_mcf2.remote(
                    y_dat, x_dat_ref, d_dat, obs, cov_x_inv, d_values[idx],
                    idx) for idx in range(no_of_treat)]
                while len(still_running) > 0:
                    finished, still_running = ray.wait(still_running)
                    finished_res = ray.get(finished)
                    for res in finished_res:
                        y_match[:, res[1]] = res[0]
                if 'refs' in c_dict['_mp_ray_del']:
                    del x_dat_ref
                if 'rest' in c_dict['_mp_ray_del']:
                    del finished_res, finished
                if c_dict['_mp_ray_shutdown']:
                    ray.shutdown()
            elif c_dict['_ray_or_dask'] == 'dask':
                with Client(n_workers=maxworkers) as client:
                    x_dat_ref = client.scatter(x_dat)
                    y_dat_ref = client.scatter(y_dat)
                    d_dat_ref = client.scatter(d_dat)
                    ret_fut = [client.submit(
                        nn_neighbour_mcf2, y_dat_ref, x_dat_ref, d_dat_ref,
                        obs, cov_x_inv, d_values[idx], idx)
                        for idx in range(no_of_treat)]
                    for _, res in as_completed(ret_fut, with_results=True):
                        y_match[:, res[1]] = res[0]
            else:
                with futures.ProcessPoolExecutor(max_workers=maxworkers
                                                 ) as fpp:
                    ret_fut = {fpp.submit(
                        nn_neighbour_mcf2, y_dat, x_dat, d_dat, obs, cov_x_inv,
                        d_values[idx], idx):
                        idx for idx in range(no_of_treat)}
                    for fut in futures.as_completed(ret_fut):
                        results_fut = fut.result()
                        y_match[:, results_fut[1]] = results_fut[0]
    else:
        y_match = np.zeros((obs, no_of_treat))
    treat_val_str = [str(i) for i in d_values]
    y_match_name = [v_dict['y_tree_name'][0] + "_NN" + treat_val_str[i]
                    for i in range(no_of_treat)]
    v_dict.update({'y_match_name': y_match_name})
    y_match_df = pd.DataFrame(data=y_match, columns=y_match_name,
                              index=data.index)
    datanew = pd.concat([data, y_match_df], axis=1)
    gp.delete_file_if_exists(c_dict['tree_sample_nn'])
    datanew.to_csv(c_dict['tree_sample_nn'], index=False)
    if c_dict['with_output']:
        gp.print_descriptive_stats_file(
            c_dict['tree_sample_nn'], 'all', c_dict['print_to_file'])
    return c_dict['tree_sample_nn'], v_dict


def adjust_variables(v_dict, v_x_type, v_x_values, c_dict, del_var,
                     regrf=False):
    """Remove DEL_VAR variables from other list of variables in which they app.

    Parameters
    ----------
    v_dict : Dict with variables
    v_x_type: Dict.
    v_x_values: Dict.
    c_dict : Dict. Controls.
    del_var : List of string with variables to remove
    regrf : Boolean. Regression RF. Default is False.

    Returns
    -------
    vn : Updated dictionary

    """
    # Do note remove variables in v['x_name_remain']
    del_var = gp.adjust_vars_vars(del_var, v_dict['x_name_remain'])
    vnew_dict = copy.deepcopy(v_dict)
    vnew_dict['x_name'] = gp.adjust_vars_vars(v_dict['x_name'], del_var)
    vnew_dict['z_name'] = gp.adjust_vars_vars(v_dict['z_name'], del_var)
    vnew_dict['name_ordered'] = gp.adjust_vars_vars(v_dict['name_ordered'],
                                                    del_var)
    vnew_dict['name_unordered'] = gp.adjust_vars_vars(v_dict['name_unordered'],
                                                      del_var)
    if not regrf:
        vnew_dict['x_balance_name'] = gp.adjust_vars_vars(
            v_dict['x_balance_name'], del_var)
    vnew_dict['x_name_always_in'] = gp.adjust_vars_vars(
        v_dict['x_name_always_in'], del_var)
    vn_x_type = {key: v_x_type[key] for key in v_x_type if key not in del_var}
    vn_x_values = {key: v_x_values[key] for key in v_x_values
                   if key not in del_var}
    (c_dict['type_0'], c_dict['type_1'], c_dict['type_2']
     ) = unordered_types_overall(list(vn_x_type.values()))
    return vnew_dict, vn_x_type, vn_x_values, c_dict


def create_xz_variables(
        v_dict, c_dict, regrf=False, d_indat_values=None, no_val_dict=None,
        q_inv_dict=None, q_inv_cr_dict=None, prime_values_dict=None,
        unique_val_dict=None, z_new_name_dict=None, z_new_dic_dict=None):
    """Create the variables for GATE & add to covariates, recode X.

    Parameters
    ----------
    v : Dictionary of variables.
    c : Dictionary of control parameters.
    regrf: Boolean. Regression Forest if True. Default is False.
    d1_indat_values: Numpy vector. Unique values of treatment in training data.
               Default is None.
    no_val_dict: Dict. Number of values for variables in z_list (training).
    q_inv_dict: Dict. Discretisation values for z_list. Default is None.
    q_inv_cr_dict: Dict. Discretisation values of continuous x.
               Default is None.
    prime_values_dict, unique_val_dict: Dict. Correspondence table from
               variable to prime (unordered variable). Default is None.

    Returns
    -------
    vn: Updated dictionary of variables
    cn: Updated dictionary of control parameters
    indata_with_z: File that also contains the newly created variables
    predata_with_z: 2nd File that also contains the newly created variables
    d1_indat_values: Numpy vector. Unique values of treatment in training data.
               Default is None.
    no_val_dict: Dict. Number of values for variables in z_list (training).
    q_inv_dict: Dict. Discretisation values for z_list. Default is None.
    q_inv_cr_dict: Dict. Discretatisation values of continuous x.
               Default is None.
    prime_values_dict, unique_val_dict: Dict. Correspondence table from
               variable to prime (unordered variable). Default is None.
    """
    def check_for_nan(data):
        vars_with_nans = []
        for name in data.columns:
            temp_pd = data[name].squeeze()
            if temp_pd.dtype == 'object':
                vars_with_nans.append(name)
        if vars_with_nans:
            print('-' * 80)
            print('WARNING: The following variables are not numeric:')
            print(vars_with_nans)
            print('WARNING: They have to be recoded as numerical if used in',
                  'estimation.')
            print('-' * 80)

    def get_d_discr(data, d_dat_np, grid_val, d_grid_name):
        d_discr = np.zeros_like(d_dat_np)
        for idx, val in enumerate(d_dat_np):
            jdx = (np.abs(val - grid_val)).argmin()
            d_discr[idx] = grid_val[jdx]
        d_discr = np.where((d_dat_np > 1e-15) & (d_discr == 0), grid_val[1],
                           d_discr)
        d_discr_pd = pd.DataFrame(data=d_discr, columns=d_grid_name,
                                  index=data.index)
        return d_discr_pd

    vn_dict, cn_dict = copy.deepcopy(v_dict), copy.deepcopy(c_dict)
    if c_dict['train_mcf']:
        indata_with_z = c_dict['indat_temp']
        not_same = c_dict['indata'] != c_dict['preddata']
        data1 = pd.read_csv(filepath_or_buffer=c_dict['indata'])
        data1.columns = data1.columns.str.upper()
        data1new = data1.copy()   # pylint: disable=E1101
        data1new.replace({False: 0, True: 1}, inplace=True)
        if c_dict['with_output']:
            check_for_nan(data1new)
        if not regrf:
            d_dat_pd = data1new[v_dict['d_name']].squeeze()
            if d_dat_pd.dtype == 'object':
                d_dat_pd = d_dat_pd.astype('category')
                print(d_dat_pd.cat.categories)
                if c_dict['with_output']:
                    print('Automatic recoding of treatment variable')
                    numerical_codes = pd.unique(d_dat_pd.cat.codes)
                    print(numerical_codes)
                d_dat_new_pd = d_dat_pd.cat.codes
                data1new[v_dict['d_name']] = d_dat_new_pd.to_frame()
            if c_dict['d_type'] == 'continuous':
                d1_unique = c_dict['ct_grid_nn_val']
                d_dat_np = data1new[v_dict['d_name']].to_numpy()
                data1new[v_dict['d_name']] = (data1new[v_dict['d_name']] -
                                              data1new[v_dict['d_name']].min())
                d_dat_np = np.where(d_dat_np < 1e-15, 0, d_dat_np)
                d_discr_nn_pd = get_d_discr(data1new, d_dat_np,
                                            c_dict['ct_grid_nn_val'],
                                            v_dict['d_grid_nn_name'])
                d_discr_w_pd = get_d_discr(data1new, d_dat_np,
                                           c_dict['ct_grid_w_val'],
                                           v_dict['d_grid_w_name'])
                d_discr_dr_pd = get_d_discr(data1new, d_dat_np,
                                            c_dict['ct_grid_dr_val'],
                                            v_dict['d_grid_dr_name'])
                data1new = pd.concat([data1new, d_discr_nn_pd, d_discr_w_pd,
                                      d_discr_dr_pd], axis=1)
            else:
                d1_np = data1new[v_dict['d_name']].to_numpy()
                d1_unique = np.round(np.unique(d1_np))
        else:
            d1_unique = None
    else:
        not_same = True
        d1_unique = d_indat_values if not regrf else None
        indata_with_z = None
    if not_same and c_dict['pred_mcf']:
        predata_with_z = c_dict['pred_eff_temp']
        data2 = pd.read_csv(filepath_or_buffer=c_dict['preddata'])
        data2.columns = data2.columns.str.upper()
        data2new = data2.copy()      # pylint: disable=E1101
        data2new.replace({False: 0, True: 1}, inplace=True)
        if c_dict['with_output'] and c_dict['train_mcf']:
            check_for_nan(data1new)
        if not regrf:
            if c_dict['gatet_flag'] or c_dict['atet_flag']:
                text = 'Treatment variable differently coded in both datasets.'
                text += 'Set ATET_FLAG and GATET_FLAG to 0.'
                d_dat_pd = data2new[v_dict['d_name']].squeeze()
                if d_dat_pd.dtype == 'object':
                    raise Exception('Treatment in predicted file not coded' +
                                    'as integer. Change this and use same' +
                                    'coding as in training data.')
                d2_np = data2new[v_dict['d_name']].to_numpy()
                d2_unique = np.round(np.unique(d2_np))
                if len(d1_unique) == len(d2_unique):
                    assert np.all(d1_unique == d2_unique), text
                else:
                    raise Exception(text)
    else:
        predata_with_z = c_dict['indat_temp']
    # Part 1: Recoding and adding Z variables
    if not regrf:
        if c_dict['agg_yes']:
            if not v_dict['z_name_list'] == []:
                z_name_ord_new = []
                if c_dict['train_mcf']:
                    no_val_dict, q_inv_dict, z_new_name_dict = {}, {}, {}
                    z_new_dic_dict = {}
                for z_name in v_dict['z_name_list']:
                    if c_dict['train_mcf']:
                        no_val = data1[z_name].unique()
                        if c_dict['save_forest']:
                            no_val_dict.update({z_name: no_val})
                    else:
                        no_val = no_val_dict[z_name]
                    if len(no_val) > c_dict['max_cats_z_vars']:
                        groups = c_dict['max_cats_z_vars']
                    else:
                        groups = len(no_val)
                    # Variables are discretized because too many groups
                    if len(no_val) > groups:  # Else, existing categ.s are used
                        if c_dict['train_mcf']:
                            quant = np.linspace(1/groups, 1-1/groups, groups-1)
                            q_t = data1[z_name].quantile(quant)  # Returns DF
                            std = data1[z_name].std()
                            q_inv = ([data1[z_name].min() - 0.001*std])
                            q_inv.extend(q_t)   # This is a list
                            q_inv.extend([data1[z_name].max() + 0.001 * std])
                            q_inv = np.unique(q_inv)
                            data1s = pd.cut(data1[z_name], q_inv, right=True,
                                            labels=False)
                            new_name = data1s.name + "CATV"
                            z_name_ord_new.extend([new_name])
                            data1new[new_name] = data1s.copy()
                            means = data1new.groupby(new_name).mean(
                                numeric_only=True)
                            new_dic = means[data1s.name].to_dict()
                            data1new = data1new.replace({new_name: new_dic})
                            if c_dict['save_forest']:
                                z_new_name_dict.update({z_name: new_name})
                                q_inv_dict.update({z_name: q_inv})
                                z_new_dic_dict.update({z_name: new_dic})
                            if c_dict['with_output'] and c_dict['verbose']:
                                print("Training: Variable recoded: ",
                                      data1s.name, "->", new_name)
                        else:
                            q_inv = q_inv_dict[z_name]
                            new_name = z_new_name_dict[z_name]
                            new_dic = z_new_dic_dict[z_name]
                            z_name_ord_new.extend([new_name])
                        if not_same and c_dict['pred_mcf']:
                            std2 = data2[z_name].std()
                            q_inv[0] = data2[z_name].min() - 0.001 * std2
                            q_inv[-1] = data2[z_name].max() + 0.001 * std2
                            data2s = pd.cut(data2[z_name], q_inv,
                                            right=True, labels=False)
                            new_name = data2s.name + "CATV"
                            data2new[new_name] = data2s.copy()
                            data2new = data2new.replace({new_name: new_dic})
                            if c_dict['with_output'] and c_dict['verbose']:
                                print("Prediction: Variable recoded: ",
                                      data2s.name, "->", new_name)
                    else:
                        z_name_ord_new.extend([z_name])
                vn_dict['z_name_ord'].extend(z_name_ord_new)
                vn_dict['z_name'].extend(z_name_ord_new)
            vn_dict['x_name'].extend(vn_dict['z_name'])
            vn_dict['x_name_remain'].extend(vn_dict['z_name'])
            vn_dict['x_balance_name'].extend(vn_dict['z_name'])
            vn_dict['name_ordered'].extend(vn_dict['z_name_ord'])
            vn_dict['name_ordered'].extend(vn_dict['z_name_list'])
            vn_dict['name_unordered'].extend(vn_dict['z_name_unord'])
            vn_dict['x_name'] = gp.cleaned_var_names(vn_dict['x_name'])
            vn_dict['x_name_remain'] = gp.cleaned_var_names(
                vn_dict['x_name_remain'])
            vn_dict['x_balance_name'] = gp.cleaned_var_names(
                vn_dict['x_balance_name'])
            vn_dict['name_ordered'] = gp.cleaned_var_names(
                vn_dict['name_ordered'])
            vn_dict['name_unordered'] = gp.cleaned_var_names(
                vn_dict['name_unordered'])
    # Part 2: Recoding ordered and unordered variables
    x_name_type_unord = []  # Contains INT
    x_name_values = []      # Contains Lists which may be empty
    if c_dict['train_mcf']:
        q_inv_cr_dict, prime_values_dict, unique_val_dict = {}, {}, {}
    for variable in vn_dict['x_name']:
        if variable in vn_dict['name_ordered']:  # Ordered variable
            if c_dict['train_mcf']:
                unique_val = data1new[variable].unique()  # Sorted small > larg
            else:
                unique_val = data2new[variable].unique()  # Sorted small > larg
            unique_val.sort()  # unique_val: Sorted from smallest to largest
            k = len(unique_val)
            # Recode continuous variables to fewer values to speed up programme
            x_name_type_unord += [0]
            # Determine whether this has to be recoded
            if c_dict['max_cats_cont_vars'] < (k-2):
                groups = c_dict['max_cats_cont_vars']
                quant = np.linspace(1/groups, 1-1/groups, groups-1)
                if c_dict['train_mcf']:
                    q_t = data1new[variable].quantile(quant)  # Returns DF
                    std = data1new[variable].std()
                    q_inv = ([data1new[variable].min() - 0.001*std])
                    q_inv.extend(q_t)   # This is a list
                    q_inv.extend([data1new[variable].max() + 0.001*std])
                    data1s = pd.cut(x=data1new[variable], bins=q_inv,
                                    right=True, labels=False)
                    new_variable = data1s.name + "CR"
                    data1new[new_variable] = data1s
                    means = data1new.groupby(new_variable).mean(
                        numeric_only=True)
                    new_dic = means[data1s.name].to_dict()
                    data1new = data1new.replace({new_variable: new_dic})
                    vn_dict = gp.substitute_variable_name(vn_dict, variable,
                                                          new_variable)
                    if c_dict['save_forest']:
                        q_inv_cr_dict.update({variable: q_inv})
                else:
                    q_inv = q_inv_cr_dict[variable]
                if not_same and c_dict['pred_mcf']:
                    data2s = pd.cut(x=data2new[variable], bins=q_inv,
                                    right=True, labels=False)
                    new_variable = data2s.name + "CR"
                    data2new[new_variable] = data2s
                    means = data2new.groupby(new_variable).mean(
                        numeric_only=True)
                    new_dic = means[data2s.name].to_dict()
                    data2new = data2new.replace({new_variable: new_dic})
                if c_dict['with_output'] and c_dict['verbose']:
                    print("Variable recoded: ", variable, "->", new_variable)
                if c_dict['train_mcf']:
                    values_pd = data1new[new_variable].unique()
                else:
                    values_pd = data2new[new_variable].unique()
            else:
                values_pd = unique_val
            values = values_pd.tolist()
            if len(values) < c_dict['max_save_values']:
                x_name_values.append(sorted(values))
            else:
                x_name_values.append([])  # Add empty list to avoid excess mem
        else:   # Unordered variable
            if c_dict['train_mcf']:
                unique_val = data1new[variable].unique()  # Sorted small > larg
                unique_val.sort()  # unique_val: Sorted from smallest to larg
                k = len(unique_val)
                # Recode categorical variables by running integers such that
                # groups of them can be efficiently translated into primes
                prime_values = gp.primes_list(k)
                if len(prime_values) != len(unique_val):
                    raise Exception(
                        'Not enough prime values available for recoding.' +
                        'Most likely reason: Continuous variables coded as' +
                        ' unordered. Program stopped.')
                prime_variable = data1new[variable].name + "PR"
                data1new[prime_variable] = data1new[variable].replace(
                    unique_val, prime_values)
                if c_dict['save_forest']:
                    prime_values_dict.update({prime_variable: prime_values})
                    unique_val_dict.update({variable: unique_val})
            else:
                prime_values = prime_values_dict[variable + "PR"]  # List
                unique_val = unique_val_dict[variable]
            if not_same and c_dict['pred_mcf']:
                unique_val_pred = data2new[variable].unique()
                bad_vals = list(np.setdiff1d(unique_val_pred, unique_val))
                if bad_vals:    # List is not empty
                    print('Prediction file contains values that were not used',
                          'for training: ', variable, ':', bad_vals)
                    raise Exception('Too many values in unordered variable.')
                prime_variable = data2new[variable].name + "PR"
                data2new[prime_variable] = data2new[variable].replace(
                    unique_val, prime_values)
            if k < 19:
                x_name_type_unord += [1]  # <= 18 categories: dtype=int64
            else:
                x_name_type_unord += [2]  # > 18 categories: dtype=object
#                       Object wird durch Operation evt. zu INT. Evt müssen
#                       diese PrimeProducts getrennt in structured array
#                       gespeichert werden
#                       Auch noch wichtig beim Speichern für Prediction!
            if c_dict['train_mcf']:
                values_pd = data1new[prime_variable].unique()
            else:
                values_pd = data2new[prime_variable].unique()
            x_name_values.append(sorted(values_pd.tolist()))
            vn_dict = gp.substitute_variable_name(vn_dict, variable,
                                                  prime_variable)
            if c_dict['with_output'] and c_dict['verbose']:
                print("Variable recoded: ", variable, "->", prime_variable)
    # Define dummy to see if particular type of UO exists at all in data
    type_0, type_1, type_2 = unordered_types_overall(x_name_type_unord)
    vn_x_type = dict(zip(vn_dict['x_name'], x_name_type_unord))
    vn_x_values = dict(zip(vn_dict['x_name'], x_name_values))
    cn_add = {'x_type_0': type_0, 'x_type_1': type_1,
              'x_type_2': type_2}  # Not needed for prediction
    cn_dict.update(cn_add)
    if c_dict['with_output'] and c_dict['verbose']:
        if type_1:
            print('\nType 1 unordered variable detected')
        if type_2:
            print('\nType 2 unordered variable detected')
    if c_dict['train_mcf']:
        gp.delete_file_if_exists(indata_with_z)
        data1new.to_csv(indata_with_z, index=False)
    if not_same and c_dict['pred_mcf']:
        gp.delete_file_if_exists(predata_with_z)
        data2new.to_csv(predata_with_z, index=False)
    if not regrf:
        if c_dict['with_output'] and c_dict['agg_yes']:
            print('\n')
            print('-' * 80)
            print('Short analysis of policy variables (variable to aggregate',
                  'the effects; effect sample).', 'Each value of a variable',
                  'defines an independent stratum')
            if c_dict['train_mcf']:
                print('Training data')
                print('Name                             # of cat ')
                for i in vn_dict['z_name']:
                    print(f'{i:<32} {len(data1new[i].unique()):>6}')
                print('-' * 80, '\n')
            if c_dict['pred_mcf'] and not_same:
                print('Prediction data')
                print('Name                             # of cat ')
                for i in vn_dict['z_name']:
                    print(f' {i:<32}', f' {len(data2new[i].unique()):>6}')
                print('-' * 80, '\n')
    if c_dict['with_output'] and c_dict['desc_stat'] and c_dict['train_mcf']:
        print("\nAugmented training data set: ", indata_with_z)
        gp.print_descriptive_stats_file(
            indata_with_z, 'all', c_dict['print_to_file'])
        if not_same and c_dict['pred_mcf']:
            print('\nAugmented data set used to predict the effects',
                  predata_with_z)
            gp.print_descriptive_stats_file(
                predata_with_z, 'all', c_dict['print_to_file'])
    return (vn_dict, vn_x_type, vn_x_values, cn_dict, indata_with_z,
            predata_with_z, d1_unique, no_val_dict, q_inv_dict, q_inv_cr_dict,
            prime_values_dict, unique_val_dict, z_new_name_dict,
            z_new_dic_dict)


def unordered_types_overall(x_name_type_unord):
    """Create dummies capturing if particular types of unordered vars exit.

    Parameters
    ----------
    x_name_type_unord : list of 0,1,2

    Returns
    -------
    type_0, type_1, type_2 : Boolean. Type exist

    """
    type_2 = bool(2 in x_name_type_unord)
    type_1 = bool(1 in x_name_type_unord)
    type_0 = bool(0 in x_name_type_unord)
    return type_0, type_1, type_2


def adjust_y_names(var_dict, y_name_old, y_name_new, with_output):
    """
    Switch variables names of y in dictionary.

    Parameters
    ----------
    var_dict : Dictionary. Variables.
    y_name_old : List of strings. Old variable names.
    y_name_new : List of strings. New variable names.
    with_output : Boolean.

    Returns
    -------
    var_dict : Dict. Modified variable names.

    """
    for indx, y_name in enumerate(y_name_old):
        if (var_dict['y_tree_name'] is None
            or var_dict['y_tree_name'] == []
                or y_name == var_dict['y_tree_name'][0]):
            var_dict['y_tree_name'] = [y_name_new[indx]]
            break
    var_dict['y_name'] = y_name_new
    if with_output:
        print('\n')
        print('New variable to build trees in RF: ', var_dict['y_tree_name'])
    return var_dict


def random_obs_reductions(infiles, outfiles=None, fraction=0.5,
                          replacement=False, seed=123445):
    """
    Write random subsamples on file.

    Parameters
    ----------
    indata_list : List-like of file names. Input.
    out_data_list : List-like of file names. Output.
    fraction : Share of observations to include.

    Returns
    -------
    None.

    """
    if outfiles is None:
        outfiles = infiles
    for file_idx, file in enumerate(infiles):
        data = pd.read_csv(filepath_or_buffer=file, header=0)
        data = data.sample(frac=fraction, replace=replacement,
                           random_state=seed)
        gp.delete_file_if_exists(outfiles[file_idx])
        data.to_csv(outfiles[file_idx], index=False)


def random_obs_reductions_treatment(infiles, outfiles=None, fraction=0.5,
                                    d_name=None, seed=123344):
    """
    Write random subsamples on file - treatment specific.

    Parameters
    ----------
    indata_list : List-like of file names. Input.
    out_data_list : List-like of file names. Output.
    fraction : Share of observations to include.
    d_name: Name of treatment variable

    Returns
    -------
    None.

    """
    def find_reduction(d_pd, fraction_to_keep):
        largest_count = d_pd.value_counts()
        group = int(largest_count.index[0][0])
        if largest_count.iloc[0] * fraction_to_keep > largest_count.iloc[1]:
            no_of_obs_to_keep = round(largest_count.iloc[0] * fraction_to_keep)
        else:
            no_of_obs_to_keep = largest_count.iloc[1]
        no_of_obs_to_drop = largest_count.iloc[0] - no_of_obs_to_keep
        return group, no_of_obs_to_drop

    def reduce_group(data, d_name, group, no_of_obs_to_drop, seed):
        treatment_group = data.loc[data[d_name].eq(group).squeeze()]
        drop_index = treatment_group.sample(
            no_of_obs_to_drop, random_state=seed).index
        data = data.drop(drop_index)
        print()
        return data

    def write_to_file(data, file):
        gp.delete_file_if_exists(file)
        data.to_csv(file, index=False)

    if outfiles is None:
        outfiles = infiles
    for file_idx, file in enumerate(infiles):
        data = pd.read_csv(filepath_or_buffer=file, header=0)
        group, no_of_obs_to_drop = find_reduction(data[d_name], fraction)
        data = reduce_group(data, d_name, group, no_of_obs_to_drop, seed)
        write_to_file(data, outfiles[file_idx])
