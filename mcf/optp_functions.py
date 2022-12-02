"""Created on Fri Jun 26 12:42:02 2020.

Optimal Policy Trees: Functions - Python implementation

Can be used under Creative Commons Licence CC BY-SA
Michael Lechner, SEW, University of St. Gallen, Switzerland

# -*- coding: utf-8 -*-
"""
import copy
import time
import sys
import random
from multiprocessing import freeze_support
from pathlib import Path
import os
import ray

import numpy as np
from pandas import read_csv
from psutil import cpu_count

from mcf import optp_tree_functions as optp_t
from mcf import optp_tree_add_functions as optp_ta
from mcf import optp_blackbox_functions as optp_bb
from mcf import optp_print as optp_p
from mcf import general_purpose as gp


def optpoltree(
    indata=None, preddata=None, datpath=None, outpath=None,
    save_pred_to_file=True, id_name=None, polscore_name=None,
    polscore_desc_name=None, d_name=None, x_ord_name=None,
    x_unord_name=None, effect_vs_0=None, effect_vs_0_se=None, output_type=2,
    outfiletext=None, _ray_or_dask='ray', parallel_processing=True,
    how_many_parallel=None, with_numba=True, screen_covariates=True,
    check_perfectcorr=True, min_dummy_obs=10, clean_data_flag=True,
    ft_yes=True, ft_no_of_evalupoints=100, ft_depth=3, ft_min_leaf_size=None,
    st_yes=True, st_depth=5, st_min_leaf_size=None, max_shares=None,
    costs_of_treat=0, costs_of_treat_mult=1, only_if_sig_better_vs_0=False,
    sig_level_vs_0=0.05, bb_yes=True, bb_rest_variable=None,
    ft_bootstraps=0, st_bootstraps=0, bb_bootstraps=499, bb_stochastic=False,
        _smaller_sample=None, _with_output=False):
    """Compute the optimal policy tree."""
    freeze_support()
    time1 = time.time()
    _seed = 1232434
    random.seed(_seed)
    # -------------------------------------------------------------------------
    # use smaller random sample (usually for testing purposes)
    if _smaller_sample is not None:
        if 0 < _smaller_sample < 1:
            gp.randomsample(datpath, indata + '.csv', 'smaller_indata.csv',
                            _smaller_sample, True, seed=_seed)
            indata = 'smaller_indata'

    # set values for control variables
    controls = controls_into_dic(
        how_many_parallel, parallel_processing, output_type, outpath, datpath,
        indata, preddata, outfiletext, _with_output, screen_covariates,
        check_perfectcorr, clean_data_flag, min_dummy_obs,
        ft_no_of_evalupoints, max_shares, ft_depth, costs_of_treat,
        costs_of_treat_mult, with_numba, ft_min_leaf_size,
        only_if_sig_better_vs_0, sig_level_vs_0, save_pred_to_file,
        _ray_or_dask, st_yes, st_depth, st_min_leaf_size, ft_yes, bb_yes,
        ft_bootstraps, st_bootstraps, bb_bootstraps, bb_stochastic)
    variables = variable_dict(
        id_name, polscore_name, polscore_desc_name, d_name, x_ord_name,
        x_unord_name, effect_vs_0, effect_vs_0_se, bb_rest_variable)
    # Set defaults for many control variables of the MCF & define variables
    c_dict, v_dict = get_controls(controls, variables)

    # Some descriptive stats of input and redirection of output file
    if c_dict['with_output']:
        if c_dict['print_to_file']:
            orig_stdout = sys.stdout
            # gp.delete_file_if_exists(c_dict['outfiletext'])
            if c_dict['print_to_terminal']:
                sys.stdout = gp.OutputTerminalFile(c_dict['outfiletext'])
            else:
                outfiletext = open(c_dict['outfiletext'], 'a')
                sys.stdout = outfiletext
    if c_dict['with_output'] and (c_dict['ft_yes'] or dict['st_yes']):
        print('\nParameter for Optimal Policy Tree:')
        gp.print_dic(c_dict)
        print('\nVariables used:')
        gp.print_dic(v_dict)
        gp.print_descriptive_stats_file(
            c_dict['indata'], to_file=c_dict['print_to_file'])
        if c_dict['indata'] != c_dict['preddata']:
            gp.print_descriptive_stats_file(
                c_dict['preddata'], to_file=c_dict['print_to_file'])
    else:
        c_dict['print_to_file'] = False
    names_to_check_train = (v_dict['id_name'] + v_dict['polscore_name']
                            + v_dict['x_ord_name'] + v_dict['x_unord_name'])
    if v_dict['polscore_desc_name'] is not None:
        names_to_check_train += v_dict['polscore_desc_name']
    names_to_check_pred = (v_dict['id_name']
                           + v_dict['x_ord_name'] + v_dict['x_unord_name'])
    if c_dict['bb_yes']:
        names_to_check_train = names_to_check_train + v_dict[
            'bb_rest_variable']
        names_to_check_pred = names_to_check_pred + v_dict[
            'bb_rest_variable']
    if c_dict['only_if_sig_better_vs_0'] or c_dict['bb_stochastic']:
        names_to_check_train = names_to_check_train + v_dict[
            'effect_vs_0'] + v_dict['effect_vs_0_se']
    if v_dict['d_name'] is not None:
        d_in = optp_bb.vars_in_data_file(c_dict['indata'], v_dict['d_name'])
        txt = ('Treatment is not available in training data. Either add it '
               + 'or set D_NAME to None.')
        assert d_in, txt
        names_to_check_train += v_dict['d_name']
    if c_dict['indata'] != c_dict['preddata'] and c_dict['with_output']:
        gp.check_all_vars_in_data(
            c_dict['preddata'], v_dict['x_ord_name'] + v_dict['x_unord_name'])
        gp.print_descriptive_stats_file(
            c_dict['preddata'], varnames=v_dict['x_ord_name']
            + v_dict['x_unord_name'], to_file=c_dict['print_to_file'])
    # Prepare data
    # Remove missings and keep only variables needed for further analysis
    if c_dict['clean_data_flag']:
        indata2 = gp.clean_reduce_data(
            c_dict['indata'], c_dict['indata_temp'], names_to_check_train,
            c_dict['with_output'], c_dict['with_output'],
            c_dict['print_to_file'])
    if c_dict['indata'] != c_dict['preddata']:
        preddata2 = gp.clean_reduce_data(
            c_dict['preddata'], c_dict['preddata_temp'], names_to_check_pred,
            c_dict['with_output'], c_dict['with_output'],
            c_dict['print_to_file'])
    else:
        preddata2 = indata2
    # Remove variables that do not have enough independent variation
    if c_dict['screen_covariates']:
        x_variables_in, _ = gp.screen_variables(
            indata2, v_dict['x_ord_name'] + v_dict['x_unord_name'],
            c_dict['check_perfectcorr'], c_dict['min_dummy_obs'],
            c_dict['with_output'])
    time2 = time.time()

    if c_dict['bb_yes']:
        black_box_alloc_pred = optp_bb.black_box_allocation(
            indata2, preddata2, c_dict, v_dict, _seed)
    else:
        black_box_alloc_pred = None
    time3 = time.time()

    if c_dict['ft_yes'] or dict['st_yes']:
        x_type, x_value, c_dict = optp_ta.classify_var_for_pol_tree(
            indata2, v_dict, c_dict, list(x_variables_in))
        if np.any(np.array(c_dict['max_shares']) < 1):
            c_dict['costs_of_treat'] = optp_ta.automatic_cost(indata2, v_dict,
                                                              c_dict)
        if c_dict['ft_yes']:
            optimal_tree, _, _ = optp_t.optimal_tree_proc(
                indata2, x_type, x_value, v_dict, c_dict)
        else:
            optimal_tree = None
        if c_dict['st_yes']:
            sequential_tree, _, _ = optp_t.sequential_tree_proc(
                indata2, x_type, x_value, v_dict, c_dict)
        else:
            sequential_tree = None
        # Prozedur um den Output darzustellen
        if c_dict['with_output'] and c_dict['ft_yes']:
            print('\n' + '=' * 80)
            print('OPTIMAL Policy Tree in training sample')
            print('=' * 80)
            optp_p.descr_policy_tree(indata2, optimal_tree, x_type.keys(),
                                     x_value, v_dict, c_dict)
            if v_dict['d_name'] is not None:
                no_of_treat = len(v_dict['polscore_name'])
                treat_pred, treat_act = optp_ta.pred_policy_allocation(
                    optimal_tree, x_value, v_dict, c_dict, no_of_treat,
                    indata2)
                data = read_csv(indata2)
                po_np = data[v_dict['polscore_name']].to_numpy()
                changers = np.int64(treat_pred) != np.int64(treat_act)
                optp_p.describe_alloc_other_outcomes(
                    v_dict['polscore_name'], po_np, no_of_treat,
                    treat_pred, changers=changers, changers_only=True,
                    alloc_act=treat_act)
                if v_dict['polscore_desc_name'] is not None:
                    po_descr_np = data[v_dict['polscore_desc_name']].to_numpy()
                    optp_p.describe_alloc_other_outcomes(
                        v_dict['polscore_desc_name'], po_descr_np, no_of_treat,
                        treat_pred, changers=changers, changers_only=True,
                        alloc_act=treat_act)
        if c_dict['with_output'] and c_dict['st_yes']:
            print('\n' + '=' * 80)
            print('SEQUENTIAL Policy Tree in training sample')
            print('=' * 80)
            optp_p.descr_policy_tree(indata2, sequential_tree, x_type.keys(),
                                     x_value, v_dict, c_dict)
            if v_dict['d_name'] is not None:
                no_of_treat = len(v_dict['polscore_name'])
                treat_pred, treat_act = optp_ta.pred_policy_allocation(
                    sequential_tree, x_value, v_dict, c_dict, no_of_treat,
                    indata2)
                data = read_csv(indata2)
                po_np = data[v_dict['polscore_name']].to_numpy()
                changers = np.int64(treat_pred) != np.int64(treat_act)
                optp_p.describe_alloc_other_outcomes(
                    v_dict['polscore_name'], po_np, no_of_treat,
                    treat_pred, changers=changers, changers_only=True,
                    alloc_act=treat_act)
                if v_dict['polscore_desc_name'] is not None:
                    po_descr_np = data[v_dict['polscore_desc_name']].to_numpy()
                    optp_p.describe_alloc_other_outcomes(
                        v_dict['polscore_desc_name'], po_descr_np, no_of_treat,
                        treat_pred, changers=changers, changers_only=True,
                        alloc_act=treat_act)
        if c_dict['ft_yes']:
            print('\n' + '=' * 80)
            print('OPTIMAL Policy Tree in prediction sample')
            print('=' * 80)
            opt_alloc_pred = optp_ta.pred_policy_allocation(
                optimal_tree, x_value, v_dict, c_dict,
                len(v_dict['polscore_name']))
        else:
            opt_alloc_pred = None
        if c_dict['st_yes']:
            print('\n' + '=' * 80)
            print('SEQUENTIAL Policy Tree in prediction sample')
            print('=' * 80)
            seq_alloc_pred = optp_ta.pred_policy_allocation(
                sequential_tree, x_value, v_dict, c_dict,
                len(v_dict['polscore_name']))
        else:
            sequential_tree = seq_alloc_pred = None
        time4 = time.time()
        # Analysing observed allocation in terms of potential outcomes

        if v_dict['d_name'] is not None and c_dict['with_output']:
            print('\nObserved allocation (training) based on potential',
                  'outcomes')
            print('- ' * 40)
            no_of_treat = len(v_dict['polscore_name'])
            data = read_csv(indata2)
            treat_act = np.int64(data[v_dict['d_name']].to_numpy().flatten())
            po_np = data[v_dict['polscore_name']].to_numpy()
            optp_p.describe_alloc_other_outcomes(
                v_dict['polscore_name'], po_np, no_of_treat, treat_act)
            if v_dict['polscore_desc_name'] is not None:
                po_descr_np = data[v_dict['polscore_desc_name']].to_numpy()
                optp_p.describe_alloc_other_outcomes(
                    v_dict['polscore_desc_name'], po_descr_np, no_of_treat,
                    treat_act)
    time5 = time.time()
    # Print timing information
    time_string = ['Data preparation:  ', 'Black-Box rules    ',
                   'Tree building      ', 'Observed allocation',
                   'Total time:        ']
    time_difference = [time2 - time1, time3 - time2, time4 - time3,
                       time5 - time4, time5 - time1]
    if ray.is_initialized():
        ray.shutdown()
    if c_dict['with_output']:
        if c_dict['parallel']:
            print('\nMultiprocessing')
        else:
            print('\nNo Multiprocessing')
        gp.print_timing(time_string, time_difference)  # print to file
        if c_dict['print_to_file']:
            if c_dict['print_to_terminal']:
                sys.stdout.output.close()
            else:
                outfiletext.close()
            sys.stdout = orig_stdout
    return opt_alloc_pred, seq_alloc_pred, black_box_alloc_pred


def get_controls(c_dict, v_dict):
    """Update defaults for controls and variables.

    Parameters
    ----------
    c : Dict. Parameters.
    v : Dict. Variables.

    Returns
    -------
    con : Dict. Parameters.
    var : Dict. Variables.

    """
    def check_adjust_varnames(variable_name, datafile_name):
        """Adjust variable name to its spelling in file. Check if in file.

        Parameters
        ----------
        variable_name : List of strings.
        file_name : String. CSV file.

        Returns
        -------
        new_variable_name : List of strings.

        """
        data = read_csv(datafile_name)
        variable_in_file = list(data)   # List of variable names
        new_variable_name = variable_name[:]
        variable_in_file_upper = [s.upper() for s in variable_in_file]
        missing_variables = []
        for name_i, name in enumerate(variable_name):
            name_upper = name.upper()
            if name_upper in variable_in_file_upper:
                posi = variable_in_file_upper.index(name_upper)
                new_variable_name[name_i] = variable_in_file[posi]
            else:
                missing_variables.append(name)
        if not missing_variables:       # List is empty
            return new_variable_name
        print('The following variables are not in the data: ',
              missing_variables)
        print('The data consists only of these variables:', variable_in_file)
        raise Exception('Programme stopped because of missing variables.')

    def recode_bootstrap(bootstraps, default):
        if isinstance(bootstraps, int):
            if bootstraps < 1:
                bootstraps = default
        else:
            bootstraps = default
        return bootstraps

    path_programme_run = str(Path(__file__).parent.absolute())
    if c_dict['datpfad'] is None:
        c_dict['datpfad'] = path_programme_run
    if c_dict['outpfad'] is None:
        c_dict['outpfad'] = path_programme_run + '/out'
    if os.path.isdir(c_dict['outpfad']):
        if not c_dict['with_output'] is False:
            print(f'Directory for output {c_dict["outpfad"]:s} already exists')
    else:
        try:
            os.mkdir(c_dict['outpfad'])
        except OSError as error:
            raise Exception(f'Creation of the directory {c_dict["outpfad"]:s}',
                            ' failed') from error
        else:
            if not c_dict['with_output'] is False:
                print('Successfully created the directory',
                      f' {c_dict["outpfad"]:s}')
    if c_dict['indata'] is None:
        raise Exception('Filename of indata must be specified')
    if c_dict['outfiletext'] is None:
        c_dict['outfiletext'] = c_dict['indata']
    temppfad = c_dict['outpfad'] + '/_tempoptp_'
    if os.path.isdir(temppfad):
        file_list = os.listdir(temppfad)
        if file_list:
            for temp_file in file_list:
                os.remove(os.path.join(temppfad, temp_file))
        if not c_dict['with_output'] is False:
            print(f'Temporary directory  {temppfad:s} already exists')
            if file_list:
                print('All files deleted.')
    else:
        try:
            os.mkdir(temppfad)
        except OSError as error:
            raise Exception(f'Creation of the directory {temppfad:s} failed'
                            ) from error
        else:
            if not c_dict['with_output'] is False:
                print(f'Successfully created the directory {temppfad:s}')
    c_dict['outfiletext'] = (c_dict['outpfad'] + '/' + c_dict['outfiletext']
                             + '.txt')
    c_dict['indata'] = c_dict['datpfad'] + '/' + c_dict['indata'] + '.csv'
    if c_dict['preddata'] is None:
        c_dict['preddata'] = c_dict['indata']
    else:
        c_dict['preddata'] = c_dict['datpfad'] + '/' + c_dict['preddata'
                                                              ] + '.csv'
    indata_temp = temppfad + '/' + 'indat_temp' + '.csv'
    preddata_temp = temppfad + '/' + 'preddat_temp' + '.csv'
    c_dict['save_pred_to_file'] = c_dict['save_pred_to_file'] is not False
    c_dict['pred_save_file'] = c_dict['outfiletext'][:-4] + 'OptTreat.csv'

    if c_dict['output_type'] is None:
        c_dict['output_type'] = 2
    if c_dict['output_type'] == 0:
        print_to_file = False
        print_to_terminal = True
    elif c_dict['output_type'] == 1:
        print_to_file = True
        print_to_terminal = False
    else:
        print_to_file = print_to_terminal = True

    if c_dict['parallel'] is not False:
        c_dict['parallel'] = True
    c_dict['parallel'] = not c_dict['parallel'] == 0
    if c_dict['no_parallel'] is None:
        c_dict['no_parallel'] = 0
    if c_dict['no_parallel'] < 0.5:
        c_dict['no_parallel'] = round(cpu_count(logical=True) * 0.8)
    else:
        c_dict['no_parallel'] = round(c_dict['no_parallel'])

    if c_dict['_ray_or_dask'] not in ('dask', 'ray'):
        c_dict['_ray_or_dask'] = 'ray'

    if c_dict['screen_covariates'] is not False:
        c_dict['screen_covariates'] = True
    if c_dict['check_perfectcorr'] is not True:
        c_dict['check_perfectcorr'] = False
    if c_dict['min_dummy_obs'] is None:
        c_dict['min_dummy_obs'] = 0
    c_dict['min_dummy_obs'] = 10 if c_dict['min_dummy_obs'] < 1 else round(
        c_dict['min_dummy_obs'])
    if c_dict['clean_data_flag'] is not False:
        c_dict['clean_data_flag'] = True

    c_dict['ft_yes'] = c_dict['ft_yes'] is not False
    c_dict['st_yes'] = c_dict['st_yes'] is not False
    c_dict['bb_yes'] = c_dict['bb_yes'] is not False
    if c_dict['ft_no_of_evalupoints'] is None:
        c_dict['ft_no_of_evalupoints'] = 0
    c_dict['ft_no_of_evalupoints'] = 100 if c_dict[
        'ft_no_of_evalupoints'] < 5 else round(c_dict['ft_no_of_evalupoints'])
    if c_dict['max_shares'] is None:
        c_dict['max_shares'] = [1 for i in range(len(v_dict['polscore_name']))]
    if len(c_dict['max_shares']) != len(v_dict['polscore_name']):
        raise Exception('# of policy scores different from # of restrictions.')
    if c_dict['ft_depth'] is None:
        c_dict['ft_depth'] = 0
    c_dict['ft_depth'] = 4 if c_dict['ft_depth'] < 1 else int(
        round(c_dict['ft_depth']) + 1)
    if c_dict['st_depth'] is None:
        c_dict['st_depth'] = 0
    c_dict['st_depth'] = 3 if c_dict['st_depth'] < 1 else int(
        round(c_dict['st_depth']))
    zeros = 0
    for i in c_dict['max_shares']:
        assert 0 <= i <= 1, 'Restrictions not between 0 and 1.'
        if i == 0:
            zeros += 1
    if zeros == len(c_dict['max_shares']):
        raise Exception('All restrictions are zero. No allocation possible.')
    if sum(c_dict['max_shares']) < 1:
        raise Exception('Sum of restrictions < 1. No allocation possible.')
    restricted = bool(np.any(np.array(c_dict['max_shares']) < 1))
    if c_dict['costs_of_treat'] is None:
        c_dict['costs_of_treat'] = 0
    if isinstance(c_dict['costs_of_treat'], (int, float)):
        if ((c_dict['costs_of_treat'] == 0) or
                np.all(np.array(c_dict['max_shares']) >= 1)):
            c_dict['costs_of_treat'] = np.zeros(len(c_dict['max_shares']))
    else:
        if len(c_dict['costs_of_treat']) != len(c_dict['max_shares']):
            c_dict['costs_of_treat'] = np.zeros(len(c_dict['max_shares']))
        else:
            c_dict['costs_of_treat'] = np.array(c_dict['costs_of_treat'])
    if c_dict['costs_mult'] is None:
        c_dict['costs_mult'] = 1
    if isinstance(c_dict['costs_mult'], (int, float)):
        if c_dict['costs_mult'] < 0:
            c_dict['costs_mult'] = 1
    else:
        if len(c_dict['costs_mult']) != len(c_dict['max_shares']):
            c_dict['costs_mult'] = 1
        else:
            c_dict['costs_mult'] = np.array(c_dict['costs_mult'])
            c_dict['costs_mult'][c_dict['costs_mult'] < 0] = 0
    if c_dict['with_numba'] is not False:
        c_dict['with_numba'] = True
    v_dict_new = copy.deepcopy(v_dict)
    if not c_dict['clean_data_flag']:  # otherwise all vars are capital
        for key in v_dict.keys():   # Check if variables are ok
            v_dict_new[key] = check_adjust_varnames(v_dict[key],
                                                    c_dict['indata'])
    with open(c_dict['indata']) as file:
        n_of_obs = sum(1 for line in file)-1
    max_by_treat = np.ceil(n_of_obs * np.array(c_dict['max_shares']))
    for i, val in enumerate(max_by_treat):
        if val > n_of_obs:
            max_by_treat = n_of_obs
    no_of_treatments = len(v_dict_new['polscore_name'])

    if c_dict['ft_min_leaf_size'] is None:
        c_dict['ft_min_leaf_size'] = 0
    if c_dict['ft_min_leaf_size'] < 1:
        c_dict['ft_min_leaf_size'] = int(
            0.1 * n_of_obs / (2 ** c_dict['ft_depth']))
    else:
        c_dict['ft_min_leaf_size'] = int(c_dict['ft_min_leaf_size'])
    if c_dict['st_min_leaf_size'] is None:
        c_dict['st_min_leaf_size'] = 0
    if c_dict['st_min_leaf_size'] < 1:
        c_dict['st_min_leaf_size'] = int(
            0.1 * n_of_obs / (2 ** c_dict['st_depth']))
    else:
        c_dict['st_min_leaf_size'] = int(c_dict['st_min_leaf_size'])
    if c_dict['only_if_sig_better_vs_0'] is not True:
        c_dict['only_if_sig_better_vs_0'] = False
    if c_dict['only_if_sig_better_vs_0']:
        if c_dict['sig_level_vs_0'] is None:
            c_dict['sig_level_vs_0'] = 0.05
        if not 0 < c_dict['sig_level_vs_0'] < 1:
            c_dict['sig_level_vs_0'] = 0.05
            if len(v_dict['effect_vs_0']) != (no_of_treatments-1):
                raise Exception('Wrong dimension of variables effect_vs_0')
            if len(v_dict['effect_vs_0_se']) != (no_of_treatments-1):
                raise Exception('Wrong dimension of variables effect_vs_0_se')
    if c_dict['bb_stochastic'] is not True:
        c_dict['bb_stochastic'] = False
    if c_dict['bb_stochastic']:
        if v_dict['effect_vs_0_se'] in (None, [], ()):
            raise Exception('effect_vs_0_se must be available for stochastic' +
                            ' simulations.')
        if len(v_dict['effect_vs_0_se']) != (no_of_treatments-1):
            raise Exception('Wrong dimension of variables effect_vs_0')
    c_dict['bb_bootstraps'] = recode_bootstrap(c_dict['bb_bootstraps'], 499)
    c_dict['ft_bootstraps'] = recode_bootstrap(c_dict['ft_bootstraps'], 0)
    c_dict['st_bootstraps'] = recode_bootstrap(c_dict['st_bootstraps'], 0)

    add_c = {
        'temppfad': temppfad, 'indata_temp': indata_temp,
        'preddata_temp': preddata_temp,
        'print_to_file': print_to_file, 'print_to_terminal': print_to_terminal,
        'max_by_treat': max_by_treat, 'no_of_treatments': no_of_treatments,
        'restricted': restricted}
    c_dict.update(add_c)
    return c_dict, v_dict_new


def variable_dict(id_name, polscore_name, polscore_desc_name, d_name,
                  x_ord_name, x_unord_name, effect_vs_0, effect_vs_0_se,
                  bb_rest_variable):
    """Pack variable names into a dictionary.

    Parameters
    ----------
    id_name : List or Tuple of string or string.
    polscore : List or Tuple of strings.
    d_name : List or Tuple of string or string.
    x_ord_name : List or Tuple of strings.
    x_unord_name : List or Tuple of strings.
    effect_vs_0 : None or list.
    effect_vs_0_se :  None or list.
    bb_rest_variabl : List or Tuple of strings.

    Returns
    -------
    var : Dictionary. Variable names

    """
    def capital_letter_and_list(string_list):
        if not isinstance(string_list, list):
            string_list = [string_list]
        string_list = [s.upper() for s in string_list]
        return string_list

    id_name = [] if id_name is None else capital_letter_and_list(id_name)
    if d_name is not None:
        d_name = capital_letter_and_list(d_name)
    if (polscore_name is None) or (polscore_name == []):
        raise Exception('Policy Score must be specified.')
    polscore_name = capital_letter_and_list(polscore_name)
    if polscore_desc_name is not None:
        polscore_desc_name = capital_letter_and_list(polscore_desc_name)
    x_ord_name = [] if x_ord_name is None else capital_letter_and_list(
        x_ord_name)
    x_unord_name = [] if x_unord_name is None else capital_letter_and_list(
        x_unord_name)
    if (x_ord_name == []) and x_unord_name:
        raise Exception('x_ord_name or x_unord_name must contain names.')
    effect_vs_0 = [] if effect_vs_0 is None else capital_letter_and_list(
        effect_vs_0)
    effect_vs_0_se = [] if effect_vs_0_se is None else capital_letter_and_list(
        effect_vs_0_se)
    bb_rest_variable = [] if bb_rest_variable is None else (
        capital_letter_and_list(bb_rest_variable))
    var = {
        'id_name': id_name, 'd_name': d_name, 'polscore_name': polscore_name,
        'polscore_desc_name': polscore_desc_name,
        'x_ord_name': x_ord_name, 'x_unord_name': x_unord_name,
        'effect_vs_0': effect_vs_0, 'effect_vs_0_se': effect_vs_0_se,
        'bb_rest_variable': bb_rest_variable}
    return var


def controls_into_dic(how_many_parallel, parallel_processing,
                      output_type, outpfad, datpfad, indata, preddata,
                      outfiletext, with_output, screen_covariates,
                      check_perfectcorr, clean_data_flag, min_dummy_obs,
                      ft_no_of_evalupoints, max_shares, ft_depth,
                      costs_of_treat, costs_of_treat_mult, with_numba,
                      ft_min_leaf_size, only_if_sig_better_vs_0,
                      sig_level_vs_0, save_pred_to_file, _ray_or_dask,
                      st_yes, st_depth, st_min_leaf_size, ft_yes, bb_yes,
                      ft_bootstraps, st_bootstraps, bb_bootstraps,
                      bb_stochastic):
    """Build dictionary containing control parameters for later easier use."""
    control_dic = {
        'no_parallel': how_many_parallel, 'parallel': parallel_processing,
        'output_type': output_type, 'outpfad': outpfad, 'datpfad': datpfad,
        'save_pred_to_file': save_pred_to_file, 'indata': indata,
        'preddata': preddata, 'with_output': with_output,
        'outfiletext': outfiletext, 'screen_covariates': screen_covariates,
        'check_perfectcorr': check_perfectcorr, 'min_dummy_obs': min_dummy_obs,
        'clean_data_flag': clean_data_flag, '_ray_or_dask': _ray_or_dask,
        'with_numba': with_numba, 'max_shares': max_shares,
        'costs_of_treat': costs_of_treat, 'costs_mult': costs_of_treat_mult,
        'only_if_sig_better_vs_0': only_if_sig_better_vs_0,
        'sig_level_vs_0': sig_level_vs_0, 'ft_yes': ft_yes,
        'ft_depth': ft_depth, 'ft_no_of_evalupoints': ft_no_of_evalupoints,
        'ft_min_leaf_size': ft_min_leaf_size,  'st_yes': st_yes,
        'st_min_leaf_size': st_min_leaf_size, 'st_depth': st_depth,
        'bb_yes': bb_yes, 'ft_bootstraps': ft_bootstraps,
        'st_bootstraps': st_bootstraps, 'bb_bootstraps': bb_bootstraps,
        'bb_stochastic': bb_stochastic
        }
    return control_dic
