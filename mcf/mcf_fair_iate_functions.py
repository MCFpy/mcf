"""
Created on Tue Sep 12 13:45:30 2023.

Contains the class and the functions needed for running the mcf.
@author: MLechner
-*- coding: utf-8 -*-
"""
import numpy as np
import pandas as pd

import ray

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_general_sys as mcf_sys


def make_fair_iates(mcf_, data_df, with_output=None):
    """Compute fair iates by rerunning mcf with reference samples."""
    blind_dic = mcf_.blind_dict
    potout_dic = {}
    mcf_.gen_dict['with_output'] = mcf_.int_dict['with_output'] = False
    mcf_.p_dict['atet'] = mcf_.p_dict['gatet'] = False
    mcf_.p_dict['cbgate'] = False
    mcf_.p_dict['bt_yes'] = False
    mcf_.p_dict['bgate'] = mcf_.p_dict['bt_yes'] = False
    mcf_.p_dict['cluster_std'] = mcf_.p_dict['gates_smooth'] = False
    mcf_.p_dict['iate_se'] = mcf_.p_dict['iate_m_ate'] = False
    mcf_.p_dict['se_boot_ate'] = mcf_.p_dict['se_boot_gate'] = False
    mcf_.p_dict['se_boot_iate'] = False
    mcf_.int_dict['return_iate_sp'] = True
    if len(mcf_.var_dict['y_name']) > 1:
        raise ValueError('Blinded method runs only with a single outcome.')
    # Labels for the dictionary with the results
    polscore_labels_dic = ['pol_score_weight' + str(weight)
                           for weight in blind_dic['weights_of_blind']]
    # Labels for the (un-) adjusted potential outcomes in DataFrame
    # Step 1: Check and define variables
    (var_not_blind_ord, var_not_blind_unord, var_blind_ord, var_blind_unord,
     var_policy_ord, var_policy_unord) = check_fair_vars(mcf_, with_output)
    var_not_blind = var_not_blind_ord + var_not_blind_unord
    var_blind = var_blind_ord + var_blind_unord
    var_all = var_blind + var_not_blind

    # Step 2: Compute (unadjusted, unblinded) IATEs
    mcf_.p_dict['iate'] = True
    if with_output:
        print('\n' + 'Computing unblinded (standard) potential outcomes')
    results_dic = mcf_.predict(data_df)
    data_all_df = results_dic['iate_data_df']
    data_df = data_all_df[var_all]
    potout_dic[polscore_labels_dic[0]], polscore_names = pols_names_from_res(
        mcf_, results_dic)

    # Step 3. Compute Blind IATEs
    compute_blind = ((len(blind_dic['weights_of_blind']) > 1)
                     or (len(blind_dic['weights_of_blind']) == 1
                         and blind_dic['weights_of_blind'][0] > 0))
    if compute_blind:
        mcf_.p_dict['ate_no_se_only'] = True
        mcf_.p_dict['iate'] = mcf_.cs_dict['yes'] = False
        mp_parallel = mcf_.gen_dict['mp_parallel']
        mcf_.gen_dict['mp_parallel'] = 1
        data_reference_df = compute_reference_data(mcf_, data_df,
                                                   with_output=with_output)
        data_not_blind_df = data_df[var_not_blind]
        # Define empty Dataframe to collect the blinded policy scores
        blind_pol_score_df = pd.DataFrame(0, columns=polscore_names,
                                          index=range(len(data_df)),
                                          dtype=float)
        n_x = len(data_df)
        no_of_treat = mcf_.gen_dict['no_of_treat']
        # ATEs will returned relative to treatment 0 (which is normalized to 0)
        y_pot_0 = potout_dic[polscore_labels_dic[0]].iloc[:, 0].to_numpy()
        blind_ate_np = np.ones((len(data_df), no_of_treat)) * y_pot_0.reshape(
            -1, 1)
        if with_output:
            print('\n' + 'Computing blinded potential outcomes')
        if mp_parallel < 1.5:
            for row_no in range(n_x):
                blind_pol_score_df.iloc[row_no] = ate_for_blinded(
                    mcf_, row_no, data_reference_df, data_not_blind_df,
                    no_of_treat, blind_ate_np)
                if with_output:
                    mcf_gp.share_completed(row_no, n_x)
        else:
            if not ray.is_initialized():
                mcf_sys.init_ray_with_fallback(
                    mp_parallel, mcf_.int_dict, mcf_.gen_dict,
                    ray_err_txt='Ray initialisation error in fairness '
                    'adjustment of IATEs.'
                    )
            data_reference_df_ref = ray.put(data_reference_df)
            data_not_blind_df_ref = ray.put(data_not_blind_df)
            blind_ate_np_ref = ray.put(blind_ate_np)
            mcf_ref = ray.put(mcf_)
            still_running = [ray_ate_for_blinded.remote(
                mcf_ref, row_no, data_reference_df_ref, data_not_blind_df_ref,
                no_of_treat, blind_ate_np_ref) for row_no in range(n_x)]
            jdx = 0
            while len(still_running) > 0:
                finished, still_running = ray.wait(still_running)
                finished_res = ray.get(finished)
                for res in finished_res:
                    iix = res[1]
                    blind_pol_score_df.iloc[iix] = res[0]
                    if with_output:
                        mcf_gp.share_completed(jdx, n_x)
                        jdx += 1
        potout_dic[polscore_labels_dic[-1]] = blind_pol_score_df

    # 4. Linear combinations (falls weights nicht nur 0,1)
    for idx, weight in enumerate(blind_dic['weights_of_blind']):
        if 0.00001 < weight < 0.99999:
            pol_score_df = (weight * blind_pol_score_df + (1 - weight)
                            * potout_dic[polscore_labels_dic[0]]
                            )
            potout_dic[polscore_labels_dic[idx]] = pol_score_df.copy()

    # 5. Descriptive stats of potential outcomes
    if with_output:
        descriptives_of_allocation(mcf_, potout_dic, polscore_labels_dic)
    return (potout_dic, data_df, var_policy_ord,  var_policy_unord,
            var_blind_ord, var_blind_unord)


def descriptives_of_allocation(mcf_, potout_dic, polscore_labels_dic):
    """Create descriptive stats of policy scores."""
    txt = ('\n' * 2 + '-' * 100 + 'Descriptive statistics of policy scores\n'
           + '- ' * 50)
    mcf_ps.print_mcf(mcf_.gen_dict, txt, summary=True)
    for label in polscore_labels_dic:
        mcf_ps.print_mcf(mcf_.gen_dict, '\n' + str(label) + '\n', summary=True)
        with pd.option_context(
                'display.max_rows', 500, 'display.max_columns', 500,
                'display.expand_frame_repr', True, 'display.width', 150,
                'chop_threshold', 1e-13):
            data = potout_dic[label].copy()
            mcf_ps.print_mcf(mcf_.gen_dict, data.describe().transpose(),
                             summary=True)


@ray.remote
def ray_ate_for_blinded(mcf_, row_no, data_reference_df, data_not_blind_df,
                        no_of_treat, ate_np):
    """Make ate_for_blinded ready for ray."""
    ate2_np = ate_np.copy()
    return (ate_for_blinded(mcf_, row_no, data_reference_df, data_not_blind_df,
                            no_of_treat, ate2_np), row_no)


def ate_for_blinded(mcf_, row_no, data_reference_df, data_not_blind_df,
                    no_of_treat, ate_np):
    """Compute ate for blinding adjustment for single observation."""
    data_index_df = data_index_df_fct(row_no, data_reference_df,
                                      data_not_blind_df)
    results_dic = mcf_.predict(data_index_df)
    ate = np.zeros(no_of_treat)
    ate[1:] = results_dic['ate'][0, 0, :no_of_treat-1].squeeze()
    ate_ret = ate_np[row_no, :] + ate
    return ate_ret


def data_index_df_fct(row_no, data_reference_df, data_not_blind_df):
    """Modify reference data."""
    data_row_no_np = data_not_blind_df.iloc[row_no]
    no_blind_vars = data_not_blind_df.columns
    # data_np = np.ones_like(data_reference_df[no_blind_vars]) * data_row_no_np
    data_np = np.zeros_like(data_reference_df[no_blind_vars])
    data_np[:] = data_row_no_np.values
    data_index_df = data_reference_df.copy()
    data_index_df[no_blind_vars] = data_np
    return data_index_df


def compute_reference_data(mcf_, data_df, with_output=True):
    """Compute reference data set."""
    data_reference_df = data_df.sample(
        n=mcf_.blind_dict['obs_ref_data'], replace=False,
        random_state=mcf_.blind_dict['seed'])
    data_reference_df.reset_index(drop=True, inplace=True)
    if with_output:
        mcf_ps.print_mcf(mcf_.gen_dict, '\n' * 2
                         + 'Reference data set (randomly drawn)\n' + '- ' * 50,
                         summary=False)
        with pd.option_context(
                'display.max_rows', 500, 'display.max_columns', 500,
                'display.expand_frame_repr', True, 'display.width', 150,
                'chop_threshold', 1e-13):
            mcf_ps.print_mcf(mcf_.gen_dict, data_df.describe().transpose(),
                             summary=False)
    return data_reference_df


def check_fair_vars(mcf_, with_output):
    """Check and sort variables."""
    # Put variables in list, upper case, empty lists if None
    x_type = mcf_.var_x_type

    # Clean variable names
    mcf_.blind_dict['var_x_protected_name'] = mcf_gp.cleaned_var_names(
        mcf_.blind_dict['var_x_protected_name'])
    mcf_.blind_dict['var_x_policy_name'] = mcf_gp.cleaned_var_names(
        mcf_.blind_dict['var_x_policy_name'])
    mcf_.blind_dict['var_x_unrestricted_name'] = mcf_gp.cleaned_var_names(
        mcf_.blind_dict['var_x_unrestricted_name'])

    # Classify variables
    var_not_blind = (mcf_.blind_dict['var_x_policy_name']
                     + mcf_.blind_dict['var_x_unrestricted_name'])
    var_blind = mcf_.blind_dict['var_x_protected_name'].copy()
    var_policy = mcf_.blind_dict['var_x_policy_name']

    # Add uncategorized variables to list of protected variables
    all_names = remove_end_str(list(x_type.keys()), 'catv')
    all_names = remove_end_str(all_names, '_prime')
    add_vars_to_blind = [var for var in all_names
                         if var not in var_not_blind and var not in var_blind
                         and var not in var_policy]

    if add_vars_to_blind:
        var_blind.extend(add_vars_to_blind)

    # Split into ordered and unordered variables
    var_blind_ord = [var for var in var_blind
                     if ((var in x_type and x_type[var] == 0)
                         or (var + 'catv') in x_type)]
    var_blind_unord = [var for var in var_blind
                       if ((var in x_type and x_type[var] > 0)
                           or (var + '_prime') in x_type)]
    var_not_blind_ord = [var for var in var_not_blind
                         if ((var in x_type and x_type[var] == 0)
                             or (var + 'catv') in x_type)]
    var_not_blind_unord = [var for var in var_not_blind
                           if ((var in x_type and x_type[var] > 0)
                               or (var + '_prime') in x_type)]
    var_policy_ord = [var for var in var_policy
                      if ((var in x_type and x_type[var] == 0)
                          or (var + 'catv') in x_type)]
    var_policy_unord = [var for var in var_policy
                        if ((var in x_type and x_type[var] > 0)
                            or (var + '_prime') in x_type)]
    var_blind_ord = remove_end_str(var_blind_ord, 'catv')
    var_not_blind_ord = remove_end_str(var_not_blind_ord, 'catv')
    var_policy_ord = remove_end_str(var_policy_ord, 'catv')
    var_blind_unord = remove_end_str(var_blind_unord, '_prime')
    var_not_blind_unord = remove_end_str(var_not_blind_unord, '_prime')
    var_policy_unord = remove_end_str(var_policy_unord, '_prime')

    # Print
    if with_output:
        print_variable_output(mcf_, var_blind=var_blind, var_policy=var_policy)
    return (var_not_blind_ord, var_not_blind_unord, var_blind_ord,
            var_blind_unord, var_policy_ord, var_policy_unord)


def remove_end_str(var_list, str_to_remove='_prime'):
    """Remove ending that have been added by the mcf estimation programme."""
    var_list_red = []
    for var in var_list:
        if var.endswith('catv'):
            var_list_red.append(var[:-4])
            continue
        if var.endswith('_prime'):
            var_list_red.append(var[:-6])
            continue
        if var.endswith(str_to_remove):
            var_list_red.append(var[:-len(str_to_remove)])
            continue
        var_list_red.append(var)
    return var_list_red


def print_variable_output(mcf_, var_blind=None, var_policy=None):
    """Print a summary of the variables and their roles."""
    x_type = mcf_.var_x_type
    txt = ('\n' + '=' * 100
           + '\nBlinding protected variables for optimal policy analysis'
           + '\n' + '-' * 100)

    txt += '\nPolicy_variables:                        ' + ' '.join(
        mcf_.blind_dict['var_x_policy_name']) + '\n' + '- ' * 50
    txt += '\nProtected variables (specified by user): ' + ' '.join(
        mcf_.blind_dict['var_x_protected_name']) + '\n' + '- ' * 50
    txt += '\nUnrestricted variables:                  ' + ' '.join(
        mcf_.blind_dict['var_x_unrestricted_name']) + '\n' + '- ' * 50
    mcf_ps.print_mcf(mcf_.gen_dict, txt, summary=True)

    txt = ('\n' + '-' * 100
           + '\nClassification of all features used for IATE estimation'
           + '\n' + '- ' * 50)

    for var in x_type:
        ordered = 'ordered' if x_type[var] == 0 else 'unordered'
        if ordered == 'ordered':
            if var.endswith('catv'):
                var = var[:-4]
        else:
            if var.endswith('_prime'):
                var = var[:-6]
        if var in var_blind:
            if var in mcf_.blind_dict['var_x_protected_name']:
                blind = 'Protected - blinded (as decided by user)'
            else:
                blind = 'Protected - blinded (automatically decided)'
        elif var in var_policy:
            blind = 'Policy (decision)'
        else:
            blind = 'Other unblinded'
        txt += f'\n{var:30s} {ordered:15s} {blind}'
    txt += ('\n' + '- ' * 50 + '\nVariables not explicity designated by user'
            ' as "Other unblinded" will be treated as "Protected."\n'
            + '-' * 100)
    mcf_ps.print_mcf(mcf_.gen_dict, txt, summary=False)


def pols_names_from_res(mcf_, results_dic):
    """Get the right names for the data contained in the results dic."""
    if mcf_.gen_dict['iate_eff']:
        name_pot = ('names_y_pot_uncenter_eff'
                    if mcf_.lc_dict['yes'] else 'names_y_pot_eff')
    else:
        name_pot = ('names_y_pot_uncenter'
                    if mcf_.lc_dict['yes'] else 'names_y_pot')
    polscore_names = results_dic['iate_names_dic'][0][name_pot]
    polscore_df = results_dic['iate_data_df'][polscore_names]
    return polscore_df, polscore_names
