"""
Provides the data related functions.

Created on Sun Jul 16 13:10:45 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
import numpy as np

from mcf import mcf_data_functions as mcf_data
from mcf import mcf_general as mcf_gp


def var_available(variable_all, var_names, needed='nice_to_have'):
    """Check if variable is available and unique in list of variable names."""
    if not isinstance(variable_all, (list, tuple)):
        variable = [variable_all]
    count = [var_names.count(variable) for variable in variable_all]
    for idx, variable in enumerate(variable_all):
        if count[idx] == 0:
            if needed == 'must_have':
                raise ValueError(f'Rquired variable/s {variable} is/are not'
                                 ' available. Available variables are'
                                 f'{var_names}')
            return False   # returns False if at least one variable is missing
        if count[idx] > 1:
            raise ValueError(f'{variable} appears twice in data. All '
                             'variables are transformed to upper case. Maybe '
                             'variable names are case sensitive. In this case,'
                             ' change names.')
    return True


def prepare_data_bb_pt(optp_, data_df, black_box=True):
    """Prepare and check data for Black-Box allocations."""
    var_dic, gen_dic = optp_.var_dict, optp_.gen_dict
    data_df, var_names = mcf_data.data_frame_vars_upper(data_df)
    # Check status of variables, available and in good shape
    var_available(var_dic['polscore_name'], var_names, needed='must_have')
    names_to_inc = var_dic['polscore_name'].copy()
    if black_box:
        bb_rest_variable = var_available(var_dic['bb_restrict_name'],
                                         var_names, needed='nice_to_have')
        if bb_rest_variable:
            names_to_inc.extend(var_dic['bb_restrict_name'])
    else:
        x_ordered = var_available(var_dic['x_ord_name'], var_names)
        x_unordered = var_available(var_dic['x_unord_name'], var_names)
        if not (x_ordered or x_unordered):
            raise ValueError('No features specified for tree building')
        optp_.var_dict['x_name'] = []
        if x_ordered:
            names_to_inc.extend(var_dic['x_ord_name'])
            optp_.var_dict['x_name'].extend(var_dic['x_ord_name'])
        if x_unordered:
            names_to_inc.extend(var_dic['x_unord_name'])
            optp_.var_dict['x_name'].extend(var_dic['x_unord_name'])
    data_new_df, optp_.var_dict['id_name'] = mcf_data.clean_reduce_data(
        data_df, names_to_inc, gen_dic, var_dic['id_name'],
        descriptive_stats=gen_dic['with_output'])
    if black_box:
        return data_new_df, bb_rest_variable
    (optp_.var_x_type, optp_.var_x_values, optp_.gen_dic
     ) = classify_var_for_pol_tree(optp_, data_new_df,
                                   optp_.var_dict['x_name'])
    (optp_.gen_dict, optp_.var_dict, optp_.var_x_type, optp_.var_x_values
     ) = mcf_data.screen_adjust_variables(optp_, data_new_df)
    return data_new_df, None


def classify_var_for_pol_tree(optp_, data_df, all_var_names):
    """Classify variables as most convenient for policy trees building."""
    var_dic, pt_dic, gen_dic = optp_.var_dict, optp_.pt_dict, optp_.gen_dict
    x_continuous = x_ordered = x_unordered = False
    x_type_dic, x_value_dic = {}, {}
    for var in all_var_names:
        values = np.unique(data_df[var].to_numpy())  # Sorted values
        if var in var_dic['x_ord_name']:
            if len(values) > pt_dic['no_of_evalupoints']:
                x_type_dic.update({var: 'cont'})
                x_value_dic.update({var: None})
                x_continuous = True
            else:
                x_type_dic.update({var: 'disc'})
                x_value_dic.update({var: values.tolist()})
                x_ordered = True
        elif var in var_dic['x_unord_name']:
            values_round = np.round(values)
            a_str = 'Categorical variables must be coded as integers.'
            assert np.sum(np.abs(values-values_round)) <= 1e-10, a_str
            x_type_dic.update({var: 'unord'})
            x_value_dic.update({var: values.tolist()})
            x_unordered = True
        else:
            raise ValueError(var + 'is neither ordered nor unordered.')
    gen_dic.update({'x_cont_flag': x_continuous, 'x_ord_flag': x_ordered,
                   'x_unord_flag': x_unordered})
    return x_type_dic, x_value_dic, gen_dic


def prepare_data_eval(optp_, data_df):
    """Prepare and check data for evaluation."""
    var_dic = optp_.var_dict
    data_df, var_names = mcf_data.data_frame_vars_upper(data_df)
    var_available(var_dic['polscore_name'], var_names, needed='must_have')
    d_ok = var_available(var_dic['d_name'], var_names, needed='nice_to_have')
    polscore_desc_ok = var_available(var_dic['polscore_desc_name'], var_names,
                                     needed='nice_to_have')
    desc_var_list = []
    if var_available(var_dic['bb_restrict_name'], var_names,
                     needed='nice_to_have'):
        desc_var_list.extend(var_dic['bb_restrict_name'])
    if var_available(var_dic['x_ord_name'], var_names, needed='nice_to_have'):
        desc_var_list.extend(var_dic['x_ord_name'])
    if var_available(var_dic['x_unord_name'], var_names,
                     needed='nice_to_have'):
        desc_var_list.extend(var_dic['x_unord_name'])
    return data_df, d_ok, polscore_desc_ok, mcf_gp.remove_dupl_keep_order(
        desc_var_list)
