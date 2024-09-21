"""
Provides the data related functions.

Created on Sun Jul 16 13:10:45 2023
# -*- coding: utf-8 -*-
@author: MLechner
"""
from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mcf import mcf_data_functions as mcf_data
from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as mcf_ps


def prepare_data_fair(optp_, data_df):
    """Prepare data for fairness correction of policy scores."""
    var_dic = optp_.var_dict

    # Recode all variables to lower case
    var_dic['protected_ord_name'] = case_insensitve(
        var_dic['protected_ord_name'].copy())
    var_dic['protected_unord_name'] = case_insensitve(
        var_dic['protected_unord_name'].copy())
    var_dic['polscore_name'] = case_insensitve(
        var_dic['polscore_name'].copy())
    data_df.columns = case_insensitve(data_df.columns.tolist())

    # Check if variables are available in data_df
    protected_ord = var_available(
        var_dic['protected_ord_name'], list(data_df.columns),
        needed='must_have')
    protected_unord = var_available(
        var_dic['protected_unord_name'], list(data_df.columns),
        needed='must_have')
    material_ord = var_available(
        var_dic['material_ord_name'], list(data_df.columns),
        needed='must_have')
    material_unord = var_available(
        var_dic['material_unord_name'], list(data_df.columns),
        needed='must_have')

    if not (protected_ord or protected_unord):
        raise ValueError('Neither ordered nor unordered protected features '
                         'specified. Fairness adjustment is impossible '
                         'without specifying at least one protected feature.')

    prot_list = [*var_dic['protected_ord_name'],
                 *var_dic['protected_unord_name']]
    mat_list = [*var_dic['material_ord_name'], *var_dic['material_unord_name']]
    common_elements = [elem for elem in prot_list if elem in mat_list]
    if common_elements:
        raise ValueError(f'Fairness adjustment: {" ".join(common_elements)} '
                         'are included among protected and '
                         'materially relevant features. This is logically '
                         'inconsistent.')

    var_available(var_dic['polscore_name'], list(data_df.columns),
                  needed='must_have')

    if optp_.gen_dict['with_output']:
        txt_print = ('\n' + '-' * 100
                     + '\nFairness adjusted score '
                     f'(method: {optp_.fair_dict["adj_type"]})'
                     + '\n' + '- ' * 50
                     + f'\nProtected features: {" ".join(prot_list)}'
                     )
        if mat_list:
            txt_print += f'\nMaterially relevant features: {" ".join(mat_list)}'
        txt_print += '\n' + '- ' * 50
        mcf_ps.print_mcf(optp_.gen_dict, txt_print, summary=True)

    # Delete protected variables from x_ord and x_unord and create dummies
    del_x_var_list = []
    if protected_ord:
        del_x_var_list = [var for var in var_dic['x_ord_name']
                          if var in var_dic['protected_ord_name']
                          ]
        optp_.var_dict['x_ord_name'] = [
            var for var in var_dic['x_ord_name']
            if var not in var_dic['protected_ord_name']]
        optp_.var_dict['protected_name'] = var_dic['protected_ord_name'].copy()
    else:
        optp_.var_dict['protected_name'] = []
    if protected_unord:
        del_x_var_list.extend([var for var in var_dic['x_unord_name']
                               if var in var_dic['protected_unord_name']
                               ])
        optp_.var_dict['x_unord_name'] = [
            var for var in var_dic['x_unord_name']
            if var not in var_dic['protected_unord_name']]

        dummies_df = pd.get_dummies(data_df[var_dic['protected_unord_name']],
                                    columns=var_dic['protected_unord_name'],
                                    dtype=int)
        optp_.var_dict['protected_name'].extend(dummies_df.columns)
        # Add dummies to data_df
        data_df = pd.concat((data_df, dummies_df), axis=1)

    if not (protected_ord or protected_unord):
        raise ValueError('No features available for fairness corrections.')

    if material_ord:
        optp_.var_dict['material_name'] = var_dic['material_ord_name'].copy()
    else:
        optp_.var_dict['material_name'] = []
    if material_unord:
        dummies_df = pd.get_dummies(data_df[var_dic['material_unord_name']],
                                    columns=var_dic['material_unord_name'],
                                    dtype=int)
        optp_.var_dict['material_name'].extend(dummies_df.columns)
        # Add dummies to data_df
        data_df = pd.concat((data_df, dummies_df), axis=1)

    if del_x_var_list and optp_.gen_dict['with_output']:
        optp_.report['fairscores_delete_x_vars_txt'] = (
            'The following variables will not be used as decision variables '
            'because they are specified as protected (fairness) by user: '
            f'{", ".join(del_x_var_list)}.')
        mcf_ps.print_mcf(optp_.gen_dict,
                         optp_.report['fairscores_delete_x_vars_txt'],
                         summary=True)
    else:
        optp_.report['fairscores_delete_x_vars_txt'] = None
    return data_df


def prepare_data_for_classifiers(data_df, var_dic, scaler=None,
                                 x_name_train=None):
    """Prepare the data to be used in the classifier."""
    # ensure case insensitivity of variable names
    var_dic['x_ord_name'] = case_insensitve(var_dic['x_ord_name'].copy())
    var_dic['x_unord_name'] = case_insensitve(var_dic['x_unord_name'].copy())
    data_df.columns = case_insensitve(data_df.columns.tolist())
    x_name = []
    x_ordered = var_available(var_dic['x_ord_name'], list(data_df.columns),
                              needed='must_have')
    x_unordered = var_available(var_dic['x_unord_name'], list(data_df.columns),
                                needed='must_have')
    if x_ordered:
        x_name.extend(var_dic['x_ord_name'])
        x_ord_np = data_df[var_dic['x_ord_name']].to_numpy()

    if x_unordered:
        x_dummies_df = pd.get_dummies(data_df[var_dic['x_unord_name']],
                                      columns=var_dic['x_unord_name'],
                                      dtype=int)
        x_name.extend(x_dummies_df.columns)
        x_dummies_np = x_dummies_df.to_numpy()

    if x_name_train is not None:
        if x_name != x_name_train:
            x_name_not = [var for var in x_name if var not in x_name_train]
            x_name_pred_not = [var for var in x_name_train if var not in x_name]
            raise ValueError(
                'Names (order) of features in transformed data does not fit to '
                'training names.'
                f'\nNames used in training: {" ".join(x_name)}'
                f'\nNames used in prediction: {" ".join(x_name_train)}'
                '\nVariables in training data that are not in prediction data: '
                f'{" ".join(x_name_not)}'
                '\nVariables in prediction data that are not in training data: '
                f'{" ".join(x_name_pred_not)}'
                '\nWarning: Note that a potential problem could be that the '
                'the categorical values of the prediction data do not have all '
                'values which are observed for the training data. In this case '
                ', (as a hack) artificial observations with this observations '
                'can be added for the allocation method (and subsequently '
                'removed from the resulting datafrome containing the '
                'allocation (before using the evaluate method).')

    if x_ordered and x_unordered:
        x_dat_np = np.concatenate((x_ord_np, x_dummies_np), axis=1)
    elif x_ordered:
        x_dat_np = x_ord_np
    elif x_unordered:
        x_dat_np = x_dummies_np
    else:
        raise ValueError('No features available for bps_classifier.')

    # Rescaling features by subtracting mean and dividing by std
    # (save in scaler object for later use in allocation method)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(x_dat_np)
    x_dat_trans_np = scaler.transform(x_dat_np)
    return x_dat_trans_np, x_name, scaler


def prepare_data_bb_pt(optp_, data_df):
    """Prepare and check data for Black-Box allocations."""
    var_dic, gen_dic = optp_.var_dict, optp_.gen_dict
    data_df, var_names = mcf_data.data_frame_vars_lower(data_df)
    # Check status of variables, available and in good shape
    if var_available(var_dic['polscore_name'], var_names, needed='must_have'):
        names_to_inc = var_dic['polscore_name'].copy()
    else:
        raise ValueError('Policy scores not in data. Cannot train model.')
    if var_dic['x_ord_name'] is None:
        var_dic['x_ord_name'] = []
    if var_dic['x_unord_name'] is None:
        var_dic['x_unord_name'] = []

    if gen_dic['method'] != 'best_policy_score':
        if var_dic['x_ord_name']:
            x_ordered = var_available(var_dic['x_ord_name'], var_names,
                                      needed='must_have')
        else:
            x_ordered = False

        if var_dic['x_unord_name']:
            x_unordered = var_available(var_dic['x_unord_name'], var_names,
                                        needed='must_have')
        else:
            x_unordered = False

        if not (x_ordered or x_unordered):
            raise ValueError('No features specified for tree building')

        optp_.var_dict['x_name'] = []
        if x_ordered:
            names_to_inc.extend(var_dic['x_ord_name'])
            optp_.var_dict['x_name'].extend(var_dic['x_ord_name'])
        if x_unordered:
            names_to_inc.extend(var_dic['x_unord_name'])
            optp_.var_dict['x_name'].extend(var_dic['x_unord_name'])

    if gen_dic['method'] in ('best_policy_score', 'bps_classifier'):
        bb_rest_variable = var_available(var_dic['bb_restrict_name'],
                                         var_names, needed='nice_to_have')
        if (bb_rest_variable
                and var_dic['bb_restrict_name'][0] not in names_to_inc):
            names_to_inc.extend(var_dic['bb_restrict_name'])

    data_new_df, optp_.var_dict['id_name'] = mcf_data.clean_reduce_data(
        data_df, names_to_inc, gen_dic, var_dic['id_name'],
        descriptive_stats=gen_dic['with_output'])
    if gen_dic['method'] == 'best_policy_score':
        return data_new_df, bb_rest_variable
    (optp_.var_x_type, optp_.var_x_values, optp_.gen_dict
     ) = classify_var_for_pol_tree(optp_, data_new_df,
                                   optp_.var_dict['x_name'],
                                   eff=gen_dic['method'] == 'policy tree')
    (optp_.gen_dict, optp_.var_dict, optp_.var_x_type, optp_.var_x_values,
     optp_.report['removed_vars']
     ) = mcf_data.screen_adjust_variables(optp_, data_new_df)
    return data_new_df, None


def classify_var_for_pol_tree(optp_, data_df, all_var_names, eff=False):
    """Classify variables as most convenient for policy trees building."""
    var_dic, pt_dic, gen_dic = optp_.var_dict, optp_.pt_dict, optp_.gen_dict
    x_continuous = x_ordered = x_unordered = False
    x_type_dic, x_value_dic = {}, {}
    for var in all_var_names:
        values = np.unique(data_df[var].to_numpy())  # Sorted values
        if var in var_dic['x_ord_name']:
            if len(values) > pt_dic['no_of_evalupoints']:
                x_type_dic.update({var: 'cont'})
                if eff:
                    x_value_dic.update({var: values.tolist()})
                else:
                    x_value_dic.update({var: None})
                x_continuous = True
            else:
                x_type_dic.update({var: 'disc'})
                x_value_dic.update({var: values.tolist()})
                x_ordered = True
        elif var in var_dic['x_unord_name']:
            if len(values) < 3:
                raise ValueError(f'{var} has only {len(values)}'
                                 ' different values. Remove it from the '
                                 'list of unorderd variables and add it '
                                 'to the list of ordered variables.')
            values_round = np.round(values)
            if np.sum(np.abs(values-values_round)) > 1e-10:
                raise ValueError('Categorical variables must be coded as'
                                 ' integers.')
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
    data_df, var_names = mcf_data.data_frame_vars_lower(data_df)
    var_available(var_dic['polscore_name'], var_names, needed='nice_to_have')
    d_ok = var_available(var_dic['d_name'], var_names, needed='nice_to_have')
    polscore_desc_ok = var_available(var_dic['polscore_desc_name'],
                                     var_names, needed='nice_to_have')
    polscore_ok = var_available(var_dic['polscore_name'], var_names,
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
    if var_available(var_dic['protected_ord_name'], var_names,
                     needed='nice_to_have'):
        desc_var_list.extend(var_dic['protected_ord_name'])
    if var_available(var_dic['protected_unord_name'], var_names,
                     needed='nice_to_have'):
        desc_var_list.extend(var_dic['protected_unord_name'])
    if optp_.gen_dict['variable_importance']:
        x_in = var_available(optp_.var_dict['vi_x_name'], var_names,
                             needed='nice_to_have')
        dum_in = var_available(optp_.var_dict['vi_to_dummy_name'], var_names,
                               needed='nice_to_have')
        if not (x_in or dum_in):
            print('WARNING: Variable importance requires the specification '
                  'of at least "var_vi_x_name" or "vi_to_dummy_name"'
                  'Since they are not specified, variable_importance'
                  'is not conducted.')
            optp_.gen_dict['variable_importance'] = False
    return (data_df, d_ok, polscore_ok, polscore_desc_ok,
            mcf_gp.remove_dupl_keep_order(desc_var_list))


def var_available(variable_all, var_names, needed='nice_to_have',
                  error_message=None):
    """Check if variable is available and unique in list of variable names."""
    if variable_all is None or variable_all == []:
        return False
    if not isinstance(variable_all, (list, tuple)):
        variable_all = [variable_all]
    # ensure case insensitive comparisons
    variable_all_ci = [variable.casefold() for variable in variable_all]
    var_names_ci = [variable.casefold() for variable in var_names]

    count = [var_names_ci.count(variable) for variable in variable_all_ci]
    for idx, variable in enumerate(variable_all_ci):
        if count[idx] == 0:
            if needed == 'must_have':
                if error_message is None:
                    raise ValueError(f'Required variable/s {variable} is/are '
                                     'not available. Available variables are'
                                     f'{var_names}')
                else:
                    raise ValueError(error_message + f'{var_names}')
            return False   # returns False if at least one variable is missing
        if count[idx] > 1:
            raise ValueError(f'{variable} appears more than once in data '
                             f'{(" ".join(var_names_ci))}. All '
                             'variables are transformed to lower case. Maybe '
                             'variable names are case sensitive. In this case,'
                             ' change names.')
    return True


def case_insensitve(variables):
    """Return list or string of lowercase."""
    if variables is not None and variables != [] and variables != ():
        if isinstance(variables, (list, tuple)):
            return [var.casefold() for var in variables]
        return variables.casefold()
    return variables


def dataframe_checksum(data_df):
    """Get a checksum for dataframe."""
    # Convert the DataFrame to a string representation
    df_string = data_df.to_string()

    # Use hashlib to create a hash of the string
    hash_object = sha256(df_string.encode())
    return hash_object.hexdigest()
