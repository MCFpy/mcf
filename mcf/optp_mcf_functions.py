"""
Combining mcf estimation of IATEs with optimal policy implementation.

This file contains the necessary functions.

# -*- coding: utf-8 -*-

Created on Thu Sep 22 08:31:20 2022

@author: MLechner

"""
import sys
import time
from pathlib import Path

import pandas as pd

from mcf import general_purpose as gp
from mcf import mcf_functions as mcf
from mcf import optp_functions as optp
from mcf import mcf_iate_cv_functions as mcf_iate_cv

def optpolicy_with_mcf(
        bb_bootstraps=499,  bb_rest_variable=None, bb_stochastic=False,
        boot=1000,
        costs_of_treat=0, costs_of_treat_mult=1, cv_iate=5,
        datapath=None, d_name=None,
        evaluate_what='blackb',
        ft_depth=3, ft_no_of_evalupoints=100, ft_min_leaf_size=None,
        ft_max_train=1e100,
        id_name=None, indata=None,
        l_centering=True, l_centering_undo_iate=False,
        max_shares=None,
        only_if_sig_better_vs_0=False,
        opt_x_ord_name=None, opt_x_unord_name=None,
        outpath=None, outfiletext='OptPol_with_mcf.txt',
        print_to_file=True, print_to_terminal=True,
        polscore_desc_name=None,
        sig_level_vs_0=0.05,
        support_check=1, support_quantil=1,
        train_share=0.8,
        with_output=True,
        x_name_always_in_ord=None, x_name_always_in_unord=None,
        x_name_ord=None, x_name_unord=None,  x_name_remain_ord=None,
        y_name=None,
        _seed_sample_split=122435467,
        ):
    """
    Compute optimal policy allocation based on mcf estimates of IATEs.

    Procedures used:
    A.	Evaluation of Black-Box procedures ('blackb'):
        There is no E-T sample split needed. We compute the IATEs via
        cross-validation with a standard mcf including standard errors.

    B.	Evaluation of policy tree ('poltree')
    a.	E-T split (80-20)
    b.	Estimate IATEs-8020 on E and predict for T. This will also
        automatically
        take account of common support. The programme will output a
        common-support adjusted T that will be used further.
    c.	Using E only: For all observations in E (inside common support),
        estimate IATEs by CV. No standard errors, cross-fitting in side each
        mcf estimation.
    d.	Using E only: For all observations in E (inside common support), build
        policy tree based on IATEs estimated in step c).
    e.	Using T only: Predict allocations for all observations in T
        (inside common support).
    f.	Using T only: Compare predicted outcome to observed and random
        allocations (as we did in the paper) for the policy tree. For this
        evaluation use the IATEs computed in step b).

    Returns
    -------
    None.

    """
    # Preparations
    time_start_all = time.time()
    om_dict = om_dictionary(indata, train_share, with_output, print_to_file,
                            print_to_terminal, outfiletext, datapath, outpath,
                            cv_iate, evaluate_what)
    if om_dict['with_output']:
        orig_stdout, print_f = start_printing(om_dict['print_to_file'],
                                              om_dict['print_to_terminal'],
                                              om_dict['outfiletext'])
    iate_user_c_dict, iate_user_v_dict = iate_user_params(
        d_name=d_name, y_name=y_name, id_name=id_name, x_name_ord=x_name_ord,
        x_name_unord=x_name_unord, x_name_always_in_ord=x_name_always_in_ord,
        x_name_always_in_unord=x_name_always_in_unord,
        x_name_remain_ord=x_name_remain_ord, boot=boot,
        l_centering=l_centering, l_centering_undo_iate=l_centering_undo_iate,
        support_check=support_check, support_quantil=support_quantil)

    for policy_tool in om_dict['evaluate_what']:
        time_start = time.time()
        # 1) Training-test sample split
        if policy_tool == 'poltree':
            gp.sample_split_2_3(om_dict['indata'], om_dict['train_sample'],
                                om_dict['train_share'], om_dict['test_sample'],
                                1-om_dict['train_share'],
                                random_seed=_seed_sample_split,
                                with_output=om_dict['with_output'])
            indata_file = om_dict['train_sample']
        else:
            indata_file = om_dict['indata']
        # 2) Cross-validated estimates of IATEs
        time_1 = time.time()
        iate_c_dict, iate_v_dict = params_for_iate_estimation(
            iate_user_c_dict, iate_user_v_dict, policy_tool)
        if policy_tool == 'poltree':
            iate_c_dict, iate_dict, om_dict = mcf_on_train_test(
                iate_c_dict, iate_v_dict, om_dict)
        cv_dict = params_for_cv(om_dict, indata_file)
        if om_dict['with_output']:
            print(f'Computing X-validated mcf predictions {policy_tool}')
            if policy_tool == 'poltree':
                print('No inference, but more efficient IATE predictions.')
            else:
                print('With inference.')
        max_train = ft_max_train if policy_tool == 'poltree' else 1e100
        names_pot_iate = mcf_iate_cv.estim_iate_cv(
            cv_dict, iate_c_dict, iate_v_dict, _seed_sample_split,
            om_dict['with_output'], max_train, called_by_mcf=False)
        if om_dict['with_output']:
            data_iate_save = pd.read_csv(om_dict['iate_sample'])
            data_iate_save.to_csv(om_dict['iate_sample_out'])
        time_2 = time.time()
        optp_dict = params_for_optp(om_dict, names_pot_iate, iate_dict,
                                    policy_tool, id_name, d_name,
                                    opt_x_ord_name, opt_x_unord_name,
                                    bb_rest_variable, bb_bootstraps,
                                    bb_stochastic, ft_no_of_evalupoints,
                                    ft_depth, ft_min_leaf_size, max_shares,
                                    costs_of_treat, costs_of_treat_mult,
                                    only_if_sig_better_vs_0, sig_level_vs_0,
                                    polscore_desc_name)
        if isinstance(optp_dict['polscore_desc_name'], (list, tuple)):
            vars_to_add = (optp_dict['d_name'],
                           *optp_dict['polscore_desc_name'],)
        else:
            vars_to_add = (optp_dict['d_name'],
                           optp_dict['polscore_desc_name'],)
        add_vars_to_input_file(om_dict['iate_sample'], om_dict['indata'],
                               optp_dict['id_name'], vars_to_add)
        gp.print_descriptive_stats_file(om_dict['iate_sample'],
                                        to_file=om_dict['print_to_file'])
        if policy_tool == 'poltree':
            add_vars_to_input_file(om_dict['test_sample'], om_dict['indata'],
                                   optp_dict['id_name'], vars_to_add)
            gp.print_descriptive_stats_file(om_dict['test_sample'],
                                            to_file=om_dict['print_to_file'])
        end_printing(om_dict['print_to_file'], om_dict['print_to_terminal'],
                     print_f, orig_stdout)
        optp.optpoltree(**optp_dict)
        orig_stdout, print_f = start_printing(om_dict['print_to_file'],
                                              om_dict['print_to_terminal'],
                                              om_dict['outfiletext'])
        time_end = time.time()
        time_string = ['Data preparation:        ',
                       'IATE estimation:         ',
                       'Test sample performance: ',
                       'Total time:              ']
        time_difference = [time_1-time_start, time_2-time_1, time_end-time_2,
                           time_end-time_start]
        if om_dict['with_output']:
            gp.print_timing(time_string, time_difference)
            print(' ', flush=True)   # clear print buffer
    time_end_all = time.time()
    gp.print_timing(['Total time all:       '], [time_end_all-time_start_all])
    mcf_iate_cv.remove_dir(om_dict['temppath'], om_dict['with_output'])
    if om_dict['with_output']:
        end_printing(om_dict['print_to_file'], om_dict['print_to_terminal'],
                     print_f, orig_stdout)


def mcf_on_train_test(iate_c_dict, iate_v_dict, om_dict):
    """Estimate mcf on test and training sample."""
    if om_dict['with_output']:
        print('Policy Tree: Computing mcf on training sample')
    iate_c_dict['datpfad'] = str(
        Path(om_dict['train_sample']).parent.resolve())
    iate_c_dict['indata'] = str(Path(om_dict['train_sample']).stem)
    iate_c_dict['preddata'] = str(Path(om_dict['test_sample']).stem)
    iate_c_dict['outpfad'] = om_dict['temppath']
    iate_dict = {**iate_c_dict, **iate_v_dict}
    (ate, _, _, _, _, _, _, _, pred_outfile, _, _, _, _, _, _
     ) = mcf.modified_causal_forest(**iate_dict)
    om_dict['test_sample'] = pred_outfile   # CS adjusted
    if om_dict['with_output']:
        print(f'Estimated ATE(s) from this step: {ate}')
    return iate_c_dict, iate_dict, om_dict


def add_vars_to_input_file(add_file, source_file, id_name, add_name):
    """Add variables from another file using an identifier."""
    data_add = pd.read_csv(add_file)
    data_source = pd.read_csv(source_file)
    data_add.columns = data_add.columns.str.upper()
    data_source.columns = data_source.columns.str.upper()
    if isinstance(add_name, (list, tuple)):
        add_name = [name.upper() for name in add_name
                    if name not in data_add.columns]
        if add_name:
            add_variables = True
    else:
        add_name = add_name.upper()
        if add_name in data_add.columns:
            add_variables = True
    if add_variables:
        if isinstance(id_name, (list, tuple)):
            id_name = id_name[0]
        id_name = id_name.upper()
        if isinstance(add_name, (list, tuple)):
            names_source = [id_name, *add_name]
        else:
            names_source = [id_name, add_name]
        data_source = data_source[names_source]
        data_new = data_add.merge(data_source, left_on=id_name, right_on=id_name)
        gp.delete_file_if_exists(add_file)
        data_new.to_csv(add_file, index=False)


def start_printing(print_to_file, print_to_terminal, outfiletext):
    """Open printing file and terminal."""
    orig_stdout = print_f = None
    if print_to_file:
        orig_stdout = sys.stdout
        if print_to_terminal:
            sys.stdout = gp.OutputTerminalFile(outfiletext)
        else:
            print_f = open(outfiletext, 'a', encoding="utf-8")
    return orig_stdout, print_f


def end_printing(print_to_file, print_to_terminal, print_f, orig_stdout):
    """Reset printing options and close file."""
    if print_to_file:
        if print_to_terminal:
            sys.stdout.output.close()
        else:
            print_f.close()
        sys.stdout = orig_stdout


def params_for_optp(om_dict, names_pot_iate, iate_dict, policy_tool, id_name,
                    d_name, x_ord_name, x_unord_name, bb_rest_variable,
                    bb_bootstraps, bb_stochastic, ft_no_of_evalupoints,
                    ft_depth, ft_min_leaf_size, max_shares, costs_of_treat,
                    costs_of_treat_mult, only_if_sig_better_vs_0,
                    sig_level_vs_0, polscore_desc_name):
    """Get parameters of optimal policy analysis"""
    names_pol_dict = names_of_policy_vars(names_pot_iate, iate_dict)
    optp_sys_dict = optp_system_params(om_dict, policy_tool)
    optp_user_dict = optp_user_params(
        id_name, d_name, x_ord_name, x_unord_name,
        bb_rest_variable, bb_bootstraps, bb_stochastic, ft_no_of_evalupoints,
        ft_depth, ft_min_leaf_size, max_shares, costs_of_treat,
        costs_of_treat_mult, only_if_sig_better_vs_0, sig_level_vs_0,
        polscore_desc_name)
    optp_dict = {**names_pol_dict, **optp_user_dict, **optp_sys_dict}
    return optp_dict


def optp_system_params(om_dict, policy_tool):
    """User defined parameters of Optimal Policy Modul."""
    optp_sys_dict = {}
    optp_sys_dict['save_pred_to_file'] = False
    optp_sys_dict['output_type'] = 2
    optp_sys_dict['_with_output'] = True
    optp_sys_dict['screen_covariates'] = True
    optp_sys_dict['check_perfectcorr'] = True
    optp_sys_dict['min_dummy_obs'] = None
    optp_sys_dict['clean_data_flag'] = True
    optp_sys_dict['bb_yes'] = True
    optp_sys_dict['st_yes'] = False
    optp_sys_dict['st_depth'] = 4
    optp_sys_dict['st_min_leaf_size'] = 4
    optp_sys_dict['outpath'] = om_dict['outpath']
    optp_sys_dict['datpath'] = str(
        Path(om_dict['iate_sample']).parent.resolve())
    optp_sys_dict['indata'] = str(Path(om_dict['iate_sample']).stem)
    optp_sys_dict['ft_yes'] = policy_tool == 'poltree'
    if policy_tool == 'blackb':
        optp_sys_dict['preddata'] = optp_sys_dict['indata']
    else:
        optp_sys_dict['preddata'] = str(Path(om_dict['test_sample']).stem)
    optp_sys_dict['outfiletext'] = str(Path(om_dict['outfiletext']).stem)
    return optp_sys_dict


def optp_user_params(id_name=None, d_name=None, x_ord_name=None,
                     x_unord_name=None, bb_rest_variable=None,
                     bb_bootstraps=499, bb_stochastic=False,
                     ft_no_of_evalupoints=100, ft_depth=3,
                     ft_min_leaf_size=None, max_shares=None,
                     costs_of_treat=0, costs_of_treat_mult=1,
                     only_if_sig_better_vs_0=False, sig_level_vs_0=0.05,
                     polscore_desc_name=None):
    """User defined parameters of Optimal Policy Modul."""
    optp_user_dict = {}
    optp_user_dict['id_name'] = id_name
    optp_user_dict['d_name'] = d_name
    optp_user_dict['x_ord_name'] = x_ord_name
    optp_user_dict['x_unord_name'] = x_unord_name
    optp_user_dict['bb_rest_variable'] = bb_rest_variable
    optp_user_dict['bb_bootstraps'] = bb_bootstraps
    optp_user_dict['bb_stochastic'] = bb_stochastic
    optp_user_dict['ft_no_of_evalupoints'] = ft_no_of_evalupoints
    optp_user_dict['ft_depth'] = ft_depth
    optp_user_dict['ft_min_leaf_size'] = ft_min_leaf_size
    optp_user_dict['max_shares'] = max_shares
    optp_user_dict['costs_of_treat'] = costs_of_treat
    optp_user_dict['costs_of_treat_mult'] = costs_of_treat_mult
    optp_user_dict['sig_level_vs_0'] = sig_level_vs_0
    optp_user_dict['polscore_desc_name'] = polscore_desc_name
    return optp_user_dict


def names_of_policy_vars(names_pot_iate, iate_dict):
    """Get the names of the variable for policy analysis."""
    pol_vars = {}
    if iate_dict['l_centering'] and iate_dict['l_centering_undo_iate']:
        pol_vars['polscore_name'] = names_pot_iate[0]['names_pot_y_uncenter']
    else:
        pol_vars['polscore_name'] = names_pot_iate[0]['names_pot_y']
    pol_vars['effect_vs_0'] = names_pot_iate[1]['names_iate']
    pol_vars['effect_vs_0_se'] = names_pot_iate[1]['names_iate_se']
    return pol_vars


def params_for_cv(om_dict, indata_file):
    """Get parameters for cv estimation from om_dict."""
    cv_dict = {}
    cv_dict['indata'] = indata_file
    cv_dict['folds'] = om_dict['cv_iate']
    cv_dict['est_temp_file'] = om_dict['est_temp_file']
    cv_dict['pred_temp_file'] = om_dict['pred_temp_file']
    cv_dict['temppath'] = om_dict['temppath']
    cv_dict['with_output'] = om_dict['with_output']
    cv_dict['iate_sample'] = om_dict['iate_sample']
    return cv_dict


def params_for_iate_estimation(iate_user_c_dict, iate_user_v_dict,
                               policy_tool):
    """Get dictionary for parameters of iate estimation."""
    iate_c_dict = iate_user_c_dict.copy()
    iate_c_dict['d_type'] = 'discrete'
    iate_c_dict['atet_flag']=False
    iate_c_dict['gatet_flag'] = False
    iate_c_dict['balancing_test'] = False
    iate_c_dict['_descriptive_stats'] = False
    iate_c_dict['iate_flag'] = True
    iate_c_dict['post_est_stats'] = False
    iate_c_dict['relative_to_first_group_only'] = True
    iate_c_dict['se_boot_iate'] = False
    iate_c_dict['support_check'] = 1
    iate_c_dict['variable_importance_oob'] = False
    iate_c_dict['_verbose'] = False
    iate_c_dict['_with_output'] = False
    iate_c_dict['_return_iate_sp'] = True
    iate_c_dict['iate_se_flag'] = policy_tool == 'blackb'
    iate_c_dict['iate_eff_flag'] = policy_tool == 'poltree'
    iate_c_dict['iate_cv_flag'] = False
    iate_v_dict = iate_user_v_dict.copy()
    return iate_c_dict, iate_v_dict


def iate_user_params(d_name=None, y_name=None, id_name=None, x_name_ord=None,
                     x_name_unord=None, x_name_always_in_ord=None,
                     x_name_always_in_unord=None, x_name_remain_ord=None,
                     boot=1000, l_centering=True, l_centering_undo_iate=False,
                     support_check=None, support_quantil=None):
    """Pack user parameters for IATE estimation into dictionary."""
    iate_user_v_dict = {}
    iate_user_v_dict['id_name'] = id_name
    iate_user_v_dict['d_name'] = d_name
    if isinstance(y_name, (list, tuple)):
        y_name = [y_name[0]]
    iate_user_v_dict['y_name'] = y_name
    iate_user_v_dict['x_name_ord'] = x_name_ord
    iate_user_v_dict['x_name_unord'] = x_name_unord
    iate_user_v_dict['x_name_always_in_ord'] = x_name_always_in_ord
    iate_user_v_dict['x_name_always_in_unord'] = x_name_always_in_unord
    iate_user_v_dict['x_name_remain_ord'] = x_name_remain_ord
    iate_user_c_dict = {}
    iate_user_c_dict['boot'] = boot
    iate_user_c_dict['l_centering'] = l_centering
    iate_user_c_dict['l_centering_undo_iate'] = l_centering_undo_iate
    iate_user_c_dict['support_check'] = support_check
    iate_user_c_dict['support_quantil'] = support_quantil
    return iate_user_c_dict, iate_user_v_dict


def om_dictionary(indata, train_share, with_output, print_to_file,
                  print_to_terminal, outfiletext, datapath, outpath, cv_iate,
                  evaluate_what):
    """Put variables and controls into om_dict. and prepare."""
    om_dict = {}
    om_dict['indata'] = indata
    om_dict['train_share'] = train_share
    om_dict['with_output'] = with_output
    om_dict['print_to_file'] = print_to_file
    om_dict['print_to_terminal'] = print_to_terminal
    om_dict['outfiletext'] = outfiletext
    om_dict['train_sample'] = 'TRAIN' + om_dict['indata']
    om_dict['test_sample'] = 'TEST' + om_dict['indata']
    if outpath is None:
        outpath = str(Path(__file__).parent.absolute())
    om_dict['outpath'] = outpath
    om_dict['outpath'] = mcf_iate_cv.create_dir(
        om_dict['outpath'], om_dict['with_output'], create_new_if_exists=True)
    if datapath is None:
        datapath = str(Path(__file__).parent.absolute())
    om_dict['datapath'] = datapath
    # Add path and extensions to files
    om_dict['outfiletext'] = (om_dict['outpath'] + '/' + om_dict['outfiletext']
                              + '.txt')
    om_dict['temppath'] = om_dict['outpath'] + '/_tempoptmcf_'
    om_dict['indata'] = om_dict['datapath'] + '/' + om_dict['indata'] + '.csv'
    om_dict['iate_sample'] = (om_dict['temppath'] + '/'
                              + om_dict['train_sample'] + '.csv')
    om_dict['iate_sample_out'] = (om_dict['outpath'] + '/'
                              + om_dict['train_sample'] + '.csv')
    om_dict['pred_temp_file'] = (om_dict['temppath'] + '/'
                                 + om_dict['train_sample'] + 'PRED.csv')
    om_dict['est_temp_file'] = (om_dict['temppath'] + '/'
                                + om_dict['train_sample'] + 'EST.csv')
    om_dict['train_sample'] = (om_dict['temppath'] + '/'
                               + om_dict['train_sample'] + '.csv')
    om_dict['test_sample'] = (om_dict['temppath'] + '/'
                               + om_dict['test_sample'] + '.csv')
    mcf_iate_cv.create_dir(om_dict['temppath'], om_dict['with_output'],
                           create_new_if_exists=False)

    om_dict['cv_iate'] = 5 if cv_iate is None else cv_iate
    if isinstance(evaluate_what, str):
        evaluate_what = tuple(evaluate_what)
    if not isinstance(evaluate_what, (list, tuple)):
        raise TypeError(f'{evaluate_what} must be a List or Tuple.')
    if not ('poltree' in evaluate_what or 'blackb' in evaluate_what):
        raise ValueError(f'{evaluate_what} is not a valid OptP method.')
    om_dict['evaluate_what'] = evaluate_what
    return om_dict
