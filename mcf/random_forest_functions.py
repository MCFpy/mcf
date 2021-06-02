"""
Functions for Random Forests based on scikitlearn.

# -*- coding: utf-8 -*-
Created on Fri Nov  6 16:19:23 2020

@author: MLechner
"""


import copy
import math
import time
import pprint
import sys
import pandas as pd
import numpy as np
import psutil
if ('gp' not in sys.modules) and ('general_purpose' not in sys.modules):
    import general_purpose as gp


def randomforest_sk(
        pfad, outpfad, temppfad, datpfad, indata, preddata, outfiletext,
        id_name, y_name, x_name_ord, x_name_unord,
        predictions_for_predfile=None, predictions_for_trainfile=None,
        boot=None, mp_parallel=None, save_forest=None,
        direct_output_to_file=None, with_output=True, descriptive_stats=None,
        screen_covariates=None, check_perfectcorr=None,
        clean_data_flag=None, min_dummy_obs=None, n_min_grid=None,
        n_min_min=None, n_min_max=None, m_min_share=None, m_max_share=None,
        m_grid=None, alpha_reg_min=None, alpha_reg_max=None,
        alpha_reg_grid=None, pred_oob_flag=None, max_depth=None,
        max_leaf_nodes=None, variable_importance=None,
        prediction_uncertainty=None, fontsize=None, dpi=None, ci_level=None,
        pu_ci_level=None, pu_skew_sym=None):
    """
    Estimate random forests based on sklearn with additional features.

    Parameters
    ----------
    pfad : String. All descriptions are contained in RandomForestSKL.py.
    outpfad : String.
    temppfad : String.
    datpfad : String.
    indata : String.
    preddata : String.
    outfiletext : String.
    id_name : List of one string.
    y_name : List of one string.
    x_name_ord : List of strings.
    x_name_unord : List of strings.
    predictions_for_predfile : BOOL, optional. The default is None.
    predictions_for_trainfile : BOOL, optional. The default is None.
    boot : Int, optinal. The default is None.
    mp_parallel : Int, optional. The default is None.
    save_forest : BOOL, optional. The default is None.
    direct_output_to_file : BOOL, optional. The default is None.
    with_output : BOOL, optional. The default is True.
    descriptive_stats : BOOL, optional. The default is None.
    screen_covariates : BOOL, optional. The default is None.
    check_perfectcorr : BOOL, optional. The default is None.
    clean_data_flag : BOOL, optional. The default is None.
    min_dummy_obs : BOOL, optional. The default is None.
    n_min_grid : Int, optional. The default is None.
    n_min_min : Int, optional. The default is None.
    n_min_max : Int, optional. The default is None.
    m_min_share : Float,. The default is None.
    m_max_share : Float,. The default is None.
    m_grid : Float, optional. The default is None.
    alpha_reg_min : Float, optional. The default is None.
    alpha_reg_max : Float, The default is None.
    alpha_reg_grid : Float, The default is None.
    pred_oob_flag : BOOL, optional. The default is None.
    max_depth : Int, optional. The default is None.
    max_leaf_nodes : Int, optional. The default is None.
    variable_importance : BOOL, optional. The default is None.
    prediction_uncertainty : BOOL, optional. The default is None.
    fontsize : Int, optional. The default is None.
    dpi : Int, optional. The default is None.
    ci_level : Float, optional. The default is None.
    pu_ci_level : Float. Confidence level for prediction intervals.
                         Default = None
    pu_skew_sym : Float. Cut-off of skewness for symetric CIs. Default is None.

    Returns
    -------
    None.

    """
    time1 = time.time()
    v_dict = {'id_name': id_name, 'y_name': y_name, 'x_name_ord': x_name_ord,
              'x_name_unord': x_name_unord}
    c_dict = {
        'print_to_file': direct_output_to_file, 'with_output': with_output,
        'pfad': pfad, 'outpfad': outpfad, 'temppfad': temppfad,
        'datpfad': datpfad, 'indata': indata, 'preddata': preddata,
        'outfiletext': outfiletext, 'desc_stat': descriptive_stats,
        'screen_covariates': screen_covariates, 'save_forest': save_forest,
        'clean_data_flag': clean_data_flag, 'min_dummy_obs': min_dummy_obs,
        'check_perfectcorr': check_perfectcorr, 'boot': boot, 'no_parallel':
        mp_parallel, 'n_min_min': n_min_min, 'n_min_max': n_min_max,
        'n_min_grid': n_min_grid, 'm_min_share': m_min_share, 'm_max_share':
        m_max_share, 'm_grid': m_grid, 'alpha_reg_min': alpha_reg_min,
        'alpha_reg_max': alpha_reg_max, 'alpha_reg_grid': alpha_reg_grid,
        'max_depth': max_depth, 'max_leaf_nodes': max_leaf_nodes,
        'pred_oob_flag': pred_oob_flag, 'pred_for_preddata':
        predictions_for_predfile, 'pred_for_traindata':
        predictions_for_trainfile, 'var_import': variable_importance,
        'pred_uncertainty': prediction_uncertainty, 'pu_ci_level': pu_ci_level,
        'pu_skew_sym': pu_skew_sym, 'fig_fontsize': fontsize, 'fig_dpi': dpi,
        'fig_ci_level': ci_level}
    c_dict, v_dict = get_defaults(c_dict, v_dict)

# Some descriptive stats of input and direction of output file
    if c_dict['with_output']:
        if c_dict['print_to_file']:
            orig_stdout = sys.stdout
            outfiletext = open(c_dict['outfiletext'], 'w')
            sys.stdout = outfiletext
        print('\nParameters for Honest Forest Estimation:')
        pprint.pprint(c_dict)
        pprint.pprint(v_dict)
        if c_dict['desc_stat']:
            gp.print_descriptive_stats_file(c_dict['indata'], 'all',
                                            c_dict['print_to_file'])
            gp.check_all_vars_in_data(c_dict['indata'],
                                      v_dict['names_to_check_train'])
        if c_dict['indata'] != c_dict['preddata']:
            if c_dict['desc_stat']:
                gp.print_descriptive_stats_file(
                    c_dict['preddata'], 'all', c_dict['print_to_file'])
                gp.check_all_vars_in_data(c_dict['preddata'],
                                          v_dict['names_to_check_pred'])
    else:
        c_dict['print_to_file'] = False
    x_train, y_train, x_pred, dummy_group_names = prepare_data_for_rf(
        c_dict, v_dict)
    time2 = time.time()

    returns_from_forest = gp.RandomForest_scikit(
        x_train, y_train, x_pred, v_dict['x_name'], v_dict['y_name'],
        boot=c_dict['boot'], n_min=c_dict['grid_n_min'],
        no_features=c_dict['grid_m'], workers=c_dict['no_parallel'],
        max_depth=c_dict['max_depth'], alpha=c_dict['grid_alpha_reg'],
        max_leaf_nodes=c_dict['max_leaf_nodes'],
        pred_p_flag=c_dict['pred_for_preddata'],
        pred_t_flag=c_dict['pred_for_traindata'],
        pred_oob_flag=c_dict['pred_oob_flag'],
        with_output=c_dict['with_output'],
        variable_importance=c_dict['var_import'],
        var_im_groups=dummy_group_names,
        pred_uncertainty=c_dict['pred_uncertainty'],
        pu_ci_level=c_dict['pu_ci_level'], pu_skew_sym=c_dict['pu_skew_sym'])
    # Saving predictions and descriptive stats
    pu_ci_level = [round((1-c_dict['pu_ci_level'])/2, ndigits=2),
                   round(1-(1-c_dict['pu_ci_level'])/2, ndigits=2)]
    if c_dict['pred_for_preddata']:
        pred_p = returns_from_forest[0]
        name_p = [v_dict['y_name'][0] + 'pred']
        n_obs_p = np.size(pred_p)
        data_to_write = np.empty([n_obs_p, 1])
        data_to_write[:, 0] = pred_p
        if c_dict['pred_uncertainty']:
            pu_ci_p = returns_from_forest[4]
            pu_ci_np = np.empty([n_obs_p, 2])
            pu_ci_np[:, 0] = pu_ci_p[0]
            pu_ci_np[:, 1] = pu_ci_p[1]
            data_to_write = np.concatenate((data_to_write, pu_ci_np), axis=1)
            name_p.append(v_dict['y_name'][0] + 'pred' + str(pu_ci_level[0]))
            name_p.append(v_dict['y_name'][0] + 'pred' + str(pu_ci_level[1]))
        data_to_write_df = pd.DataFrame(data=data_to_write, columns=name_p)
        gp.delete_file_if_exists(c_dict['outfile_pred'])
        data_to_write_df.to_csv(c_dict['outfile_pred'], index=False)
        gp.print_descriptive_stats_file(c_dict['outfile_pred'])
    if c_dict['pred_for_traindata']:
        pred_t = returns_from_forest[1]
        n_obs_t = np.size(pred_t)
        data_to_write = np.empty([n_obs_t, 2])
        data_to_write[:, 0] = y_train[:, 0]
        data_to_write[:, 1] = pred_t
        name_t = [v_dict['y_name'][0], v_dict['y_name'][0] + 'pred']
        if c_dict['pred_uncertainty']:
            pu_ci_t = returns_from_forest[5]
            pu_ci_np = np.empty([np.size(pu_ci_t[0]), 2])
            pu_ci_np[:, 0] = pu_ci_t[0]
            pu_ci_np[:, 1] = pu_ci_t[1]
            data_to_write = np.concatenate((data_to_write, pu_ci_np), axis=1)
            name_t.append(v_dict['y_name'][0] + 'pred' + str(pu_ci_level[0]))
            name_t.append(v_dict['y_name'][0] + 'pred' + str(pu_ci_level[1]))
        data_to_write_df = pd.DataFrame(data=data_to_write, columns=name_t)
        gp.delete_file_if_exists(c_dict['outfile_train'])
        data_to_write_df.to_csv(c_dict['outfile_train'], index=False)
        gp.print_descriptive_stats_file(c_dict['outfile_train'])
    time3 = time.time()
    time_string = ['Data preparation:                ',
                   'Forests estimation:              ',
                   'Total time:                      ']
    time_difference = [time2 - time1, time3 - time2, time3 - time1]
    if c_dict['with_output']:
        print('\n')
        if c_dict['no_parallel'] < 2:
            print('No parallel processing')
        else:
            print('Multiprocessing')
            print('Number of cores used: ', c_dict['no_parallel'])
        gp.print_timing(time_string, time_difference)  # print to file
    if c_dict['print_to_file']:
        sys.stdout = orig_stdout
        outfiletext.close()  # last line of procedure


def prepare_data_for_rf(c_dict, v_dict):
    """
    Prepare data for RF estimation.

    Parameters
    ----------
    c_dict : Dict, Controls.
    v_dict : Dict, Variables.

    Returns
    -------
    x_train : Numpy 2d-array. Training data (features).
    y_train : Numpy 1d-array. Training data (outcomes)
    x_pred : Numpy 2d-array. Prediction/test data (features)
    dummy_lists : List of lists. Names of variables belonging to same
                                 categorical variable.

    """
    if c_dict['clean_data_flag']:
        namen1_to_inc = gp.add_var_names(v_dict['id_name'], v_dict['y_name'],
                                         v_dict['x_name'])
        indata2 = gp.clean_reduce_data(
            c_dict['indata'], c_dict['indata2_temp'], namen1_to_inc,
            c_dict['with_output'], c_dict['desc_stat'],
            c_dict['print_to_file'])
        if c_dict['indata'] != c_dict['preddata']:
            namen2_to_inc = gp.add_var_names(v_dict['id_name'],
                                             v_dict['x_name'])
            preddata2 = gp.clean_reduce_data(
                c_dict['preddata'], c_dict['preddata2_temp'], namen2_to_inc,
                c_dict['with_output'], c_dict['desc_stat'],
                c_dict['print_to_file'])
        else:
            preddata2 = c_dict['preddata']
    else:
        indata2 = c_dict['indata']
        preddata2 = c_dict['preddata']
    train_df = pd.read_csv(indata2)
    y_train_df = train_df[v_dict['y_name']]
    x_train_df = train_df[v_dict['x_name']]
    if c_dict['indata'] != c_dict['preddata']:
        pred_df = pd.read_csv(preddata2)
        x_pred_df = pred_df[v_dict['x_name']]
    dummy_group_names = []
    dummy_names = []
    if v_dict['x_name_unord']:  # List is not empty
        for name in v_dict['x_name_unord']:
            x_t_d = pd.get_dummies(x_train_df[name], prefix=name)
            if c_dict['indata'] != c_dict['preddata']:
                x_p_d = pd.get_dummies(x_pred_df[name], prefix=name)
                if len(x_t_d.columns) != len(x_p_d.columns):
                    raise Exception('Dummies cannot consistently coded',
                                    ' because'' of differences in training',
                                    ' and test data.')
            this_dummy_names = x_t_d.columns.tolist()
            dummy_names.extend(this_dummy_names)
            this_dummy_names.append(name)
            dummy_group_names.append(this_dummy_names)
            x_train_df = pd.concat([x_train_df, x_t_d], axis=1)
            if c_dict['indata'] != c_dict['preddata']:
                x_pred_df = pd.concat([x_pred_df, x_p_d], axis=1)
        v_dict['x_name'].extend(dummy_names)
        x_train = x_train_df.to_numpy(copy=True)
        y_train = y_train_df.to_numpy(copy=True)
        if c_dict['indata'] != c_dict['preddata']:
            x_pred = x_pred_df.to_numpy(copy=True)
        else:
            x_pred = x_train
        if c_dict['with_output']:
            print('The following dummy variables have been created',
                  dummy_names)
            print('\n')
            print('Training data')
            print_df_descr(x_train_df)
            if c_dict['indata'] != c_dict['preddata']:
                print('\n')
                print('Prediction data')
                print_df_descr(x_pred_df)
        p_x = np.size(x_train, axis=1)
        m_min = round(c_dict['m_min_share'] * p_x)
        if m_min < 1:
            m_min = 1
        m_max = round(c_dict['m_max_share'] * p_x)
        if m_min == m_max:
            c_dict['m_grid'] = 1
            grid_m = m_min
        else:
            if c_dict['m_grid'] == 1:
                grid_m = round((m_min + m_max)/2)
            else:
                grid_m = gp.grid_log_scale(m_min, m_max, c_dict['m_grid'])
                grid_m = [int(i) for i in grid_m]
        c_dict.update({'grid_m': grid_m})  # this changes dict. globally
    return x_train, y_train, x_pred, dummy_group_names


def get_defaults(c_dict, v_dict):
    """Large procedure to get all parameters and variables needed for HF.

    Parameters
    ----------
    c_dict : Dictory of parameters specified by user
    v_dict : Dictory of list of variable names specified by user

    Returns
    -------
    cn: Updated dictionary of parameters
    vn: Updated list of variables.

    """
    cnew_dict = copy.deepcopy(c_dict)
    # Define full path for various new files
    pred_sample_with_pred = (c_dict['datpfad'] + '/' + c_dict['indata']
                             + 'Predpred' + '.csv')
    cnew_dict['outfiletext'] = (c_dict['outpfad'] + '/' + c_dict['outfiletext']
                                + '.txt')
    outfile_predict = c_dict['outpfad'] + '/' + c_dict['indata'] + 'pred.csv'
    outfile_train = c_dict['outpfad'] + '/' + c_dict['indata'] + 'train.csv'
    out_allindat = c_dict['outpfad'] + '/' + c_dict['indata'] + 'Predall.csv'
    cnew_dict['indata'] = c_dict['datpfad'] + '/' + c_dict['indata'] + '.csv'
    cnew_dict['preddata'] = (c_dict['datpfad'] + '/' + c_dict['preddata']
                             + '.csv')
    indat_with_pred = c_dict['outpfad'] + '/' + c_dict['indata'] + '_a_p.csv'
    save_forest_file = c_dict['outpfad'] + '/' + c_dict['indata'] + '.pickle'
    indat_temp = c_dict['temppfad'] + '/' + 'indat_temp' + '.csv'
    indata2_temp = c_dict['temppfad'] + '/' + 'indata2_temp' + '.csv'
    preddata2_temp = c_dict['temppfad'] + '/' + 'preddata2_temp' + '.csv'
    preddata3_temp = c_dict['temppfad'] + '/' + 'preddata3_temp' + '.csv'
    pred_eff_temp = c_dict['temppfad'] + '/' + 'pred_eff_temp' + '.csv'
    temporary_file = c_dict['temppfad'] + '/' + 'temporaryfile' + '.csv'
    if c_dict['with_output'] is not False:
        cnew_dict['with_output'] = True
    if c_dict['print_to_file'] is not False:
        cnew_dict['print_to_file'] = True
    mp_automatic = False
    if c_dict['no_parallel'] is None:
        cnew_dict['no_parallel'] = round(psutil.cpu_count() * 0.9)
        mp_automatic = True
    else:
        if c_dict['no_parallel'] <= 1.5:
            cnew_dict['no_parallel'] = 1
        else:
            cnew_dict['no_parallel'] = round(c_dict['no_parallel'])
    sys_share = 0.7 * getattr(psutil.virtual_memory(), 'percent') / 100
    if c_dict['screen_covariates'] is not False:
        cnew_dict['screen_covariates'] = True
    if c_dict['check_perfectcorr'] is not False:
        cnew_dict['check_perfectcorr'] = True
    if c_dict['min_dummy_obs'] is None:
        cnew_dict['min_dummy_obs'] = 10
    else:
        cnew_dict['min_dummy_obs'] = round(c_dict['min_dummy_obs'])
    if cnew_dict['min_dummy_obs'] < 1:
        cnew_dict['min_dummy_obs'] = 10
    if c_dict['clean_data_flag'] is not False:
        cnew_dict['clean_data_flag'] = True

    if c_dict['alpha_reg_min'] is None:
        cnew_dict['alpha_reg_min'] = 0
    if (cnew_dict['alpha_reg_min'] < 0) or (cnew_dict['alpha_reg_min'] >= 0.4):
        cnew_dict['alpha_reg_min'] = 0
        if (cnew_dict['alpha_reg_min'] >= 0.4) and cnew_dict['with_output']:
            print('Values of 0.4 and larger do not make sense for alpha',
                  'regularisation. alpha regularisation parameter set to ',
                  'default value. {:8.3f}'.format(cnew_dict['alpha_reg_min']))
    if c_dict['alpha_reg_max'] is None:
        cnew_dict['alpha_reg_max'] = 0.1
    if (cnew_dict['alpha_reg_max'] < 0) or (cnew_dict['alpha_reg_max'] >= 0.4):
        cnew_dict['alpha_reg_max'] = 0.1
        if (cnew_dict['alpha_reg_max'] >= 0.4) and cnew_dict['with_output']:
            print('Values of 0.4 and larger do not make sense for alpha',
                  'regularisation. alpha regularisation parameter set to ',
                  'default value. {:8.3f}'.format(cnew_dict['alpha_reg_max']))
    if c_dict['alpha_reg_grid'] is None:
        cnew_dict['alpha_reg_grid'] = 3
    else:
        cnew_dict['alpha_reg_grid'] = round(c_dict['alpha_reg_grid'])
    if cnew_dict['alpha_reg_grid'] < 1:
        cnew_dict['alpha_reg_grid'] = 3
    if cnew_dict['alpha_reg_min'] >= cnew_dict['alpha_reg_max']:
        cnew_dict['alpha_reg_grid'] = 1
        cnew_dict['alpha_reg_max'] = cnew_dict['alpha_reg_min']
    if cnew_dict['alpha_reg_grid'] == 1:
        alpha_reg = (cnew_dict['alpha_reg_max']
                     + cnew_dict['alpha_reg_min']) / 2
    elif cnew_dict['alpha_reg_grid'] == 2:
        alpha_reg = np.hstack((cnew_dict['alpha_reg_min'],
                               cnew_dict['alpha_reg_max']))
    else:
        alpha_reg = np.linspace(
            cnew_dict['alpha_reg_min'], cnew_dict['alpha_reg_max'],
            cnew_dict['alpha_reg_grid'])
    alpha_reg = list(np.unique(alpha_reg))

    data = pd.read_csv(cnew_dict['indata'], header=0)
    n_sample = len(data.index)

    if c_dict['n_min_min'] is None:
        cnew_dict['n_min_min'] = round((n_sample**0.4)/10)
    else:
        cnew_dict['n_min_min'] = round(c_dict['n_min_min'])
    if cnew_dict['n_min_min'] < 2:
        cnew_dict['n_min_min'] = 2
    if c_dict['n_min_max'] is None:
        cnew_dict['n_min_max'] = round(math.sqrt(n_sample)/5)
    else:
        cnew_dict['n_min_max'] = round(c_dict['n_min_max'])
    if cnew_dict['n_min_max'] < 2:
        cnew_dict['n_min_max'] = 2
    if c_dict['n_min_grid'] is None:
        cnew_dict['n_min_grid'] = 3
    else:
        cnew_dict['n_min_grid'] = round(c_dict['n_min_grid'])
    if cnew_dict['n_min_grid'] < 1:
        cnew_dict['n_min_grid'] = 1
    if cnew_dict['n_min_min'] == cnew_dict['n_min_max']:
        cnew_dict['n_min_grid'] = 1
    elif cnew_dict['n_min_max'] < cnew_dict['n_min_min']:
        cnew_dict['n_min_grid'] = 1
        cnew_dict['n_min_max'] = cnew_dict['n_min_min']
    if cnew_dict['n_min_grid'] == 1:
        n_min = round((cnew_dict['n_min_max'] + cnew_dict['n_min_min']) / 2)
    elif cnew_dict['n_min_grid'] == 2:
        n_min = np.hstack((cnew_dict['n_min_min'], cnew_dict['n_min_max']))
    else:
        n_min = np.linspace(cnew_dict['n_min_min'], cnew_dict['n_min_max'],
                            cnew_dict['n_min_grid'])
    n_min = np.round(n_min)
    n_min = np.unique(n_min)
    n_min = n_min.astype(int)
    n_min = n_min.tolist()
    # Select grid for number of parameters
    if c_dict['m_min_share'] is None:
        cnew_dict['m_min_share'] = 0.1
    if (cnew_dict['m_min_share'] <= 0) or (cnew_dict['m_min_share'] > 1):
        cnew_dict['m_min_share'] = 0.1
    if c_dict['m_max_share'] is None:
        cnew_dict['m_max_share'] = 0.66
    if (cnew_dict['m_max_share'] <= 0) or (cnew_dict['m_max_share'] > 1):
        cnew_dict['m_max_share'] = 0.66
    if c_dict['m_grid'] is None:
        cnew_dict['m_grid'] = 3
    else:
        cnew_dict['m_grid'] = round(c_dict['m_grid'])
    if cnew_dict['m_grid'] < 1:
        cnew_dict['m_grid'] = 3
    if c_dict['boot'] is None:
        cnew_dict['boot'] = 1000
    else:
        cnew_dict['boot'] = round(c_dict['boot'])

    if c_dict['desc_stat'] is False:            # Decriptive statistics
        cnew_dict['desc_stat'] = False
    else:
        cnew_dict['desc_stat'] = True
    if c_dict['var_import'] is False:
        cnew_dict['var_import'] = False
    else:                    # importance measure
        cnew_dict['var_import'] = True
    if c_dict['save_forest'] is False:
        cnew_dict['save_forest'] = False
    else:
        cnew_dict['save_forest'] = True
    if c_dict['fig_fontsize'] is None:
        cnew_dict['fig_fontsize'] = 2
    else:
        cnew_dict['fig_fontsize'] = round(c_dict['fig_fontsize'])
    if (cnew_dict['fig_fontsize'] < 0.5) or (cnew_dict['fig_fontsize'] > 7.5):
        cnew_dict['fig_fontsize'] = 2
    all_fonts = ('xx-small', 'x-small', 'small', 'medium', 'large', 'x-large',
                 'xx-large')
    for idx, i_lab in enumerate(all_fonts):
        if cnew_dict['fig_fontsize'] == idx + 1:
            cnew_dict['fig_fontsize'] = i_lab
    if c_dict['fig_dpi'] is None:
        cnew_dict['fig_dpi'] = 500
    else:
        cnew_dict['fig_dpi'] = round(c_dict['fig_dpi'])
    if cnew_dict['fig_dpi'] < 10:
        cnew_dict['fig_dpi'] = 500
    if c_dict['fig_ci_level'] is None:
        cnew_dict['fig_ci_level'] = 0.90
    if (cnew_dict['fig_ci_level'] < 0.5) or (
            cnew_dict['fig_ci_level'] > 0.999999):
        cnew_dict['fig_ci_level'] = 0.90

    if c_dict['max_depth'] is not None:
        cnew_dict['max_depth'] = round(c_dict['max_depth'])
    if c_dict['max_leaf_nodes'] is not None:
        cnew_dict['max_leaf_nodes'] = round(c_dict['max_leaf_nodes'])
    if c_dict['pred_oob_flag'] is not False:
        cnew_dict['pred_oob_flag'] = True
    if c_dict['pred_for_preddata'] is not False:
        cnew_dict['pred_for_preddata'] = True
    if c_dict['pred_for_traindata'] is not False:
        cnew_dict['pred_for_traindata'] = True
    if c_dict['var_import'] is not False:
        cnew_dict['var_import'] = True
    if c_dict['pred_uncertainty'] is not False:
        cnew_dict['pred_uncertainty'] = True
    if c_dict['pu_skew_sym'] is None:
        cnew_dict['pu_skew_sym'] = 0.5
    if cnew_dict['pu_skew_sym'] < 0:
        cnew_dict['pu_skew_sym'] = 0.5
    if c_dict['pu_ci_level'] is None:
        cnew_dict['pu_ci_level'] = 0.9
    if not 0.5 < cnew_dict['pu_ci_level'] < 1:
        cnew_dict['pu_ci_level'] = 0.9

    cn_add = {'pred_sample_with_pred': pred_sample_with_pred,
              'outfile_pred':       outfile_predict,
              'outfile_train':      outfile_train,
              'out_allindat':       out_allindat,
              'indat_with_pred':    indat_with_pred,
              'indat_temp':         indat_temp,
              'indata2_temp':       indata2_temp,
              'preddata2_temp':     preddata2_temp,
              'preddata3_temp':     preddata3_temp,
              'pred_eff_temp':      pred_eff_temp,
              'temporary_file':     temporary_file,
              'grid_n_min':         n_min,
              'save_forest_file':   save_forest_file,
              'add_pred_to_data_file': True,
              'sys_share':          sys_share,
              'mp_automatic':       mp_automatic,
              'grid_alpha_reg':     alpha_reg}
    cnew_dict.update(cn_add)
    vnew_dict = copy.deepcopy(v_dict)
    vnew_dict['y_name'] = gp.cleaned_var_names(vnew_dict['y_name'])
    vnew_dict['x_name_ord'] = gp.cleaned_var_names(vnew_dict['x_name_ord'])
    vnew_dict['x_name_unord'] = gp.cleaned_var_names(vnew_dict['x_name_unord'])
    vnew_dict['id_name'] = gp.cleaned_var_names(vnew_dict['id_name'])

    x_name = copy.deepcopy(vnew_dict['x_name_ord'] + vnew_dict['x_name_unord'])
    x_name = gp.cleaned_var_names(x_name)
    if not ((vnew_dict['x_name_ord'] == [])
            or (vnew_dict['x_name_unord'] == [])):
        if any([value for value in vnew_dict['x_name_ord']
                if value in vnew_dict['x_name_unord']]):
            print('Remove overlap in ordered and unordered variables')
            sys.exit()
    # Define variables for consistency check in data sets
    names_to_check_train = vnew_dict['y_name'] + x_name
    names_to_check_pred = x_name
    vn_add = {
        'x_name':               x_name,
        'names_to_check_train': names_to_check_train,
        'names_to_check_pred':  names_to_check_pred}
    vnew_dict.update(vn_add)
    return cnew_dict, vnew_dict


def print_df_descr(dataf, to_file=True):
    """
    Print nice outout of descriptive stats of dataframe.

    Parameters
    ----------
    dataf : Dataframe.
    to_file : Boolean. Print output to output file. Default is True.

    Returns
    -------
    None.

    """
    desc_stat = dataf.describe()
    varnames = dataf.columns.to_list()
    if (varnames == 'all') or len(varnames) > 10:
        to_print = desc_stat.transpose()
        rows = len(desc_stat.columns)
        cols = len(desc_stat.index)
    else:
        to_print = desc_stat
        rows = len(desc_stat.index)
        cols = len(desc_stat.columns)
    if to_file:
        expand = False
    else:
        expand = True
    with pd.option_context('display.max_rows', rows,
                           'display.max_columns', cols+1,
                           'display.expand_frame_repr', expand,
                           'chop_threshold', 1e-13):
        print(to_print)
