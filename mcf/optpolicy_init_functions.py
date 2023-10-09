"""Created on Sat July 15 10:03:15 2023.

Contains the functions needed for initialising the parameters.
@author: MLechner
-*- coding: utf-8 -*-
"""
import psutil

import numpy as np

from mcf import mcf_general as gp
from mcf import mcf_general_sys as mcf_sys


def init_int(how_many_parallel=None, parallel_processing=None,
             output_no_new_dir=None, with_numba=None, with_output=None):
    """Initialise basic technical pamameters."""
    dic = {}
    dic['parallel_processing'] = parallel_processing is not False
    if dic['parallel_processing']:
        if how_many_parallel is None or how_many_parallel < 0.5:
            dic['mp_parallel'] = round(psutil.cpu_count(logical=True)*0.8)
        else:
            dic['mp_parallel'] = round(how_many_parallel)
    else:
        dic['mp_parallel'] = 1
    dic['output_no_new_dir'] = output_no_new_dir is True
    dic['with_numba'] = with_numba is not False
    dic['with_output'] = with_output is not False
    return dic


def init_gen(method=None, outfiletext=None, outpath=None, output_type=None,
             variable_importance=None, with_output=None, new_outpath=None):
    """Initialise general parameters."""
    dic = {}
    dic['method'] = 'best_policy_score' if method is None else method
    if dic['method'] not in ('best_policy_score', 'policy tree',
                             'policy tree eff',):
        raise ValueError(f'{dic["method"]} is not a valid method.')
    if method == 'best_policy_score':
        dir_nam = 'BPS'
    elif method == 'policy tree':
        dir_nam = 'PT'
    elif method == 'policy tree eff':
        dir_nam = 'PT_EFF'
    dic['variable_importance'] = variable_importance is True

    dic['output_type'] = 2 if output_type is None else output_type
    if dic['output_type'] == 0:
        dic['print_to_file'], dic['print_to_terminal'] = False, True
    elif dic['output_type'] == 1:
        dic['print_to_file'], dic['print_to_terminal'] = True, False
    else:
        dic['print_to_file'] = dic['print_to_terminal'] = True
    if not with_output:
        dic['print_to_file'] = dic['print_to_terminal'] = False
    dic['with_output'] = with_output
    if outpath is None:
        dic['outpath'] = mcf_sys.define_outpath(None, new_outpath
                                                ) if with_output else None
    else:
        dic['outpath'] = mcf_sys.define_outpath(outpath + dir_nam, new_outpath)
    dic['outfiletext'] = ('txtFileWithOutput'
                          if outfiletext is None else outfiletext)
    dic['outfiletext'] = dic['outpath'] + '/' + dic['outfiletext'] + '.txt'
    dic['outfilesummary'] = dic['outfiletext'][:-4] + '_Summary.txt'
    if with_output:
        mcf_sys.delete_file_if_exists(dic['outfiletext'])
        mcf_sys.delete_file_if_exists(dic['outfilesummary'])
    return dic


def init_gen_solve(optp_, data_df):
    """Add and update some dictionary entry based on data."""
    var_dic, gen_dic = optp_.var_dict, optp_.gen_dict
    if ('d_name' in var_dic and (var_dic['d_name'] is not None)
            and var_dic['d_name'][0] in data_df.columns):
        d_dat = data_df[var_dic['d_name']].to_numpy()
        gen_dic['d_values'] = np.int16(np.round(np.unique(d_dat))).tolist()
    else:
        gen_dic['d_values'] = list(np.int16(
            range(len(var_dic['polscore_name']))))
    gen_dic['no_of_treat'] = len(gen_dic['d_values'])
    optp_.gen_dict = gen_dic


def init_dc(check_perfectcorr=None, clean_data=None, min_dummy_obs=None,
            screen_covariates=None):
    """Initialise data cleaning related parameters."""
    dic = {}
    dic['check_perfectcorr'] = check_perfectcorr is not False
    dic['clean_data'] = clean_data is not False
    dic['screen_covariates'] = screen_covariates is not False
    dic['min_dummy_obs'] = (10 if min_dummy_obs is None or min_dummy_obs < 1
                            else round(min_dummy_obs))
    return dic


def init_pt(depth=None, enforce_restriction=None, eva_cat_mult=None,
            no_of_evalupoints=None, min_leaf_size=None,
            select_values_cat=None):
    """Initialise parameters related to policy tree."""
    dic = {}
    if no_of_evalupoints is None or no_of_evalupoints < 5:
        dic['no_of_evalupoints'] = 100
    else:
        dic['no_of_evalupoints'] = round(no_of_evalupoints)
    if depth is None or depth < 1:
        dic['depth'] = 4
    else:
        dic['depth'] = round(depth + 1)
    dic['min_leaf_size'] = min_leaf_size   # To be initialized later
    dic['select_values_cat'] = select_values_cat is True
    dic['enforce_restriction'] = enforce_restriction is True
    if (eva_cat_mult is None or not isinstance(eva_cat_mult, (float, int))
            or eva_cat_mult < 0.1):
        dic['eva_cat_mult'] = 1
    else:
        dic['eva_cat_mult'] = eva_cat_mult
    return dic


def init_pt_solve(optp_, no_of_obs):
    """Initialise parameters related to policy tree."""
    optp_.pt_dict['min_leaf_size'] = 0.1 * no_of_obs / (
        (optp_.pt_dict['depth'] - 1) * 2)


def init_other_solve(optp_):
    """Initialise treatment costs (needs info on number of treatments."""
    no_of_treat = optp_.gen_dict['no_of_treat']
    ot_dic = optp_.other_dict
    if ot_dic['max_shares'] is None or len(ot_dic['max_shares']) < no_of_treat:
        ot_dic['max_shares'] = [1] * no_of_treat
    no_zeros = sum(1 for share in ot_dic['max_shares'] if share == 0)
    if no_zeros == len(ot_dic['max_shares']):
        raise ValueError('All restrictions are zero. No allocation possible.')
    if sum(ot_dic['max_shares']) < 1:
        raise ValueError('Sum of restrictions < 1. No allocation possible.')
    ot_dic['restricted'] = any(share < 1 for share in ot_dic['max_shares'])
    if ot_dic['costs_of_treat'] is None or len(ot_dic['costs_of_treat']
                                               ) < no_of_treat:
        ot_dic['costs_of_treat'] = [0] * no_of_treat

    if (ot_dic['costs_of_treat_mult'] is None
            or len(ot_dic['costs_of_treat_mult']) < no_of_treat):
        ot_dic['costs_of_treat_mult'] = [1] * no_of_treat
    if any(cost <= 0 for cost in ot_dic['costs_of_treat_mult']):
        raise ValueError('Cost multiplier must be positive.')
    optp_.other_dict = ot_dic


def init_rnd_shares(optp_, data_df, d_in_data):
    """Reinitialise the shares if they are not consistent with use."""
    no_of_treat = optp_.gen_dict['no_of_treat']
    rnd_dic, var_dic = optp_.rnd_dict, optp_.var_dict
    if rnd_dic['shares'] is None or len(rnd_dic['shares']) < no_of_treat:
        if d_in_data:
            obs_shares = data_df[var_dic['d_name']].value_counts(normalize=True
                                                                 ).sort_index()
            rnd_dic['shares'] = obs_shares.tolist()
        else:
            rnd_dic['shares'] = [1/no_of_treat] * no_of_treat
    if sum(rnd_dic['shares']) < 0.999999 or sum(rnd_dic['shares']) > 1.0000001:
        raise ValueError('"random shares" do not add to 1.')
    optp_.rnd_dict = rnd_dic


def init_var(bb_restrict_name=None, d_name=None, effect_vs_0=None,
             effect_vs_0_se=None, id_name=None, polscore_desc_name=None,
             polscore_name=None, vi_x_name=None, vi_to_dummy_name=None,
             x_ord_name=None, x_unord_name=None):
    """Initialise variables."""
    var_dic = {}
    var_dic['bb_restrict_name'] = check_var(bb_restrict_name)
    var_dic['d_name'] = check_var(d_name)
    var_dic['effect_vs_0'] = check_var(effect_vs_0)
    var_dic['effect_vs_0_se'] = check_var(effect_vs_0_se)
    var_dic['id_name'] = check_var(id_name)
    var_dic['polscore_desc_name'] = check_var(polscore_desc_name)
    var_dic['polscore_name'] = check_var(polscore_name)
    var_dic['x_ord_name'] = check_var(x_ord_name)
    var_dic['x_unord_name'] = check_var(x_unord_name)
    var_dic['vi_x_name'] = check_var(vi_x_name)
    var_dic['vi_to_dummy_name'] = check_var(vi_to_dummy_name)
    var_dic['name_ordered'] = var_dic['z_name'] = var_dic['x_name_remain'] = []
    var_dic['name_unordered'] = var_dic['x_balance_name'] = []
    var_dic['x_name_always_in'] = []
    return var_dic


def check_var(variable):
    """Capitalise and clean variable names."""
    if variable is None or variable == []:
        return variable
    variable = gp.to_list_if_needed(variable)
    variable = gp.cleaned_var_names(variable)
    return variable
