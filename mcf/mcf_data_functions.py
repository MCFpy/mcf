"""Created on Mon May 15 08:05:30 2023.

Contains the functions needed for cleaning the data.

@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mcf import mcf_general as mcf_gp
from mcf import mcf_print_stats_functions as mcf_ps


def split_sample_for_mcf(self, data_df):
    """Split sample for mcf forest estimation."""
    tree_df, fill_y_df = train_test_split(
        data_df, test_size=self.int_dict['share_forest_sample'],
        random_state=42)
    return tree_df, fill_y_df


def screen_adjust_variables(mcf_, data_df):
    """Screen and adjust variables."""
    # Used by mcf and optpolicy. Therefore, instance is not modified directly
    _, variables_deleted = screen_variables(mcf_, data_df)

    if variables_deleted:
        gen_dic, var_dic, var_x_type, var_x_values = adjust_variables(
            mcf_, variables_deleted)
    else:
        gen_dic, var_dic = mcf_.gen_dict, mcf_.var_dict
        var_x_type, var_x_values = mcf_.var_x_type, mcf_.var_x_values

    return gen_dic, var_dic, var_x_type, var_x_values, variables_deleted


def screen_variables(mcf_, data_df):
    """Screen variables to decide whether they could be deleted."""
    gen_dic, dc_dic = mcf_.gen_dict, mcf_.dc_dict
    var_names = mcf_.var_dict['x_name']
    data = data_df[var_names].copy()
    mcf_gp.check_if_not_number(data, var_names)
    k = data.shape[1]
    #    all_variable = set(data.columns)
    all_variable = mcf_gp.remove_dupl_keep_order(data.columns)
    # Remove variables without any variation
    to_keep = data.std(axis=0) > 1e-10
    datanew = data.loc[:, to_keep].copy()  # Keeps the respective columns
    txt = ''
    if gen_dic['with_output']:
        txt += '\n' + '-' * 100 + '\nControl variables checked'
        kk1 = datanew.shape[1]
        all_variable1 = mcf_gp.remove_dupl_keep_order(datanew.columns)
        if kk1 != k:
            deleted_vars = [var for var in all_variable
                            if var not in all_variable1]
            txt += '\nVariables without variation: ' + ' '.join(deleted_vars)
    if dc_dic['check_perfectcorr']:
        corr_matrix = datanew.corr().abs()
        # Upper triangle of corr matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1
                                          ).astype(bool))
        to_delete = [c for c in upper.columns if any(upper[c] > 0.999)]
        if not to_delete == []:
            datanew = datanew.drop(columns=to_delete)
    if gen_dic['with_output']:
        kk2 = datanew.shape[1]
        all_variable2 = mcf_gp.remove_dupl_keep_order(datanew.columns)
        if kk2 != kk1:
            deleted_vars = [var for var in all_variable1
                            if var not in all_variable2]
            txt += '\nCorrelation with other variable > 99.9%: ' + ' '.join(
                deleted_vars)
    if dc_dic['min_dummy_obs'] > 1:
        shape = datanew.shape
        to_keep = []
        for cols in range(shape[1]):
            value_count = datanew.iloc[:, cols].value_counts()
            if len(value_count) == 2:  # Dummy variable
                if value_count.min() < dc_dic['min_dummy_obs']:
                    to_keep = to_keep + ([False])
                else:
                    to_keep = to_keep + ([True])
            else:
                to_keep = to_keep + ([True])
        datanew = datanew.loc[:, to_keep]
    if gen_dic['with_output']:
        kk3 = datanew.shape[1]
        all_variable3 = mcf_gp.remove_dupl_keep_order(datanew.columns)
        if kk3 != kk2:
            deleted_vars = [var for var in all_variable2
                            if var not in all_variable3]
            txt += '\nDummy variables with too few 0 or 1: ' + ' '.join(
                deleted_vars)
    kk2 = datanew.shape[1]
    variables_remain = datanew.columns
    variables_delete = [var for var in all_variable
                        if var not in variables_remain]
    if gen_dic['with_output']:
        mcf_ps.print_mcf(gen_dic, txt, summary=False)
        if kk2 == k:
            txt = '\nAll control variables have been retained'
            mcf_ps.print_mcf(gen_dic, txt, summary=False)
        else:
            txt = '\nThe following variables have been deleted:' + ' '.join(
                variables_delete) + '\n'
            mcf_ps.print_mcf(gen_dic, txt, summary=True)
            desc_stat = data[variables_delete].describe()
            with pd.option_context('display.max_rows', None,
                                   'display.max_columns', None,
                                   'display.width', 120,
                                   'display.expand_frame_repr', True,
                                   'chop_threshold', 1e-13):
                mcf_ps.print_mcf(gen_dic, desc_stat.transpose(), summary=False)
            txt = '\nThe following variables have been retained:' + ' '.join(
                variables_remain)
            mcf_ps.print_mcf(gen_dic, txt, summary=False)
    return variables_remain, variables_delete


def adjust_variables(mcf_, del_var):
    """Remove DEL_VAR variable from other list of vars in which they appear."""
    var_dic, gen_dic = mcf_.var_dict, mcf_.gen_dict
    var_x_type, var_x_values = mcf_.var_x_type, mcf_.var_x_values
    # Do note remove variables in v['x_name_remain']
    del_var = mcf_gp.adjust_vars_vars(del_var, var_dic['x_name_remain'])
    vnew_dic = deepcopy(var_dic)
    vnew_dic['x_name'] = mcf_gp.adjust_vars_vars(var_dic['x_name'], del_var)
    vnew_dic['z_name'] = mcf_gp.adjust_vars_vars(var_dic['z_name'], del_var)
    vnew_dic['name_ordered'] = mcf_gp.adjust_vars_vars(var_dic['name_ordered'],
                                                       del_var)
    vnew_dic['name_unordered'] = mcf_gp.adjust_vars_vars(
        var_dic['name_unordered'], del_var)
    vnew_dic['x_name_balance_test'] = mcf_gp.adjust_vars_vars(
            var_dic['x_name_balance_test'], del_var)
    vnew_dic['x_name_always_in'] = mcf_gp.adjust_vars_vars(
        var_dic['x_name_always_in'], del_var)
    vn_x_type = {key: var_x_type[key] for key in var_x_type
                 if key not in del_var}
    vn_x_values = {key: var_x_values[key] for key in var_x_values
                   if key not in del_var}
    (gen_dic['x_type_0'], gen_dic['x_type_1'], gen_dic['x_type_2']
     ) = unordered_types_overall(list(vn_x_type.values()))
    return gen_dic, vnew_dic, vn_x_type, vn_x_values


def clean_data(mcf_, data_df, train=True, d_alloc_names=None):
    """Clean the data."""
    int_dic, var_dic, gen_dic = mcf_.int_dict, mcf_.var_dict, mcf_.gen_dict
    obs = len(data_df)
    if train:
        namen_to_inc = mcf_gp.add_var_names(
            var_dic['id_name'], var_dic['y_name'], var_dic['cluster_name'],
            var_dic['w_name'], var_dic['d_name'], var_dic['y_tree_name'],
            var_dic['x_name_balance_test'], var_dic['x_name'])
        if gen_dic['d_type'] == 'continuous':
            namen_to_inc.append(*var_dic['grid_nn_name'])
        if int_dic['iv']:
            namen_to_inc = mcf_gp.add_var_names(namen_to_inc,
                                                var_dic['iv_name'])
    else:
        namen_to_inc = mcf_gp.add_var_names(
            var_dic['id_name'], var_dic['w_name'],
            var_dic['x_name_balance_test'], var_dic['x_name'])
        if mcf_.p_dict['d_in_pred']:
            namen_to_inc = mcf_gp.add_var_names(namen_to_inc, var_dic['d_name'])
            if gen_dic['d_type'] == 'continuous':
                namen_to_inc.append(*var_dic['grid_nn_name'])
        if mcf_.p_dict['cluster_std']:
            namen_to_inc = mcf_gp.add_var_names(namen_to_inc,
                                                var_dic['cluster_name'])
        if d_alloc_names is not None:
            namen_to_inc = mcf_gp.add_var_names(namen_to_inc, d_alloc_names)

    namen_to_inc = mcf_gp.include_org_variables(namen_to_inc[:],
                                                data_df.columns)
    data_df, var_dic['id_name'] = clean_reduce_data(
        data_df, namen_to_inc, gen_dic, var_dic['id_name'],
        int_dic['descriptive_stats'], obs_bigdata=int_dic['obs_bigdata'])
    obs_del = obs - len(data_df)
    mcf_.var_dict = var_dic
    return data_df, (obs, obs_del,)


def clean_reduce_data(data_df, names_to_inc_, gen_dic, id_name_,
                      descriptive_stats=None, add_missing_vars=False,
                      obs_bigdata=10000000):
    """Remove obs. with missings and variables not needed."""
    shape = data_df.shape
    data_df.columns = [var.casefold() for var in data_df.columns]
    names_to_inc = [var.casefold() for var in names_to_inc_]
    if id_name_ is None or id_name_ == []:
        id_name_ = 'id_mcf'
        # Do not use names that already exist
        while True:
            if id_name_ in list(data_df.columns):
                id_name_ += 'x'
            else:
                break
    if not isinstance(id_name_, (list, tuple)):
        id_name_ = [id_name_]
    id_name = [var.casefold() for var in id_name_]
    if id_name[0] not in data_df.columns:
        data_df[id_name[0]] = np.arange(len(data_df))
    if add_missing_vars:
        add_df = add_missing_vars_fct(data_df.columns, names_to_inc,
                                      len(data_df), gen_dic['with_output'])
        if add_df is not None:
            data_df = pd.concat([data_df, add_df], axis=1)
    missing_vars = [x for x in names_to_inc if x not in data_df.columns]
    if missing_vars:
        raise NameError('The following variables are needed for prediction '
                        'but are not contained in the provided data: '
                        f'{" ".join(missing_vars)}')
    data_new_df = data_df[names_to_inc].copy()
    data_new_df.dropna(inplace=True, axis=0, how='any')
    nan_mask = (data_new_df.astype(str) == 'nan').any(axis=1)
    data_new_df = data_new_df.loc[~nan_mask]
    stop_if_df_contains_nan(data_new_df, obs_bigdata)
    shapenew = data_new_df.shape
    if gen_dic['with_output']:
        txt = '\nCheck for missing and unnecessary variables.'
        if shape == shapenew:
            txt += '\nFile has not been changed'
        else:
            if shapenew[0] == shape[0]:
                txt += '\n  No observations deleted'
            else:
                txt += f'\n{shape[0]-shapenew[0]:<5} observations deleted'
            if shapenew[1] == shape[1]:
                txt += '\nNo variables deleted'
            else:
                txt += f'\n{shape[1]-shapenew[1]:<4} variables deleted:'
                all_var = mcf_gp.remove_dupl_keep_order(data_df.columns)
                keep_var = mcf_gp.remove_dupl_keep_order(data_new_df.columns)
                del_var = [var for var in all_var if var not in keep_var]
                vars_str = ' '.join(del_var)
                txt += vars_str
        mcf_ps.print_mcf(gen_dic, txt + '\n', summary=False)
        if descriptive_stats:
            mcf_ps.print_descriptive_df(gen_dic, data_new_df, varnames='all',
                                        summary=False)
    return data_new_df, id_name


def stop_if_df_contains_nan(data_df, obs_bigdata=10000000):
    """Raise informative exception if array contains NaN."""
    data_np = mcf_gp.to_numpy_big_data(data_df, obs_bigdata)
    is_nan_np = np.isnan(data_np)
    if is_nan_np.any():   # False means there are NaN somewhere
        loc_nan = np.nonzero(is_nan_np)[1]
        vars_with_nan = [data_df.columns[i] for i in loc_nan]
        txt = ('The following variables have NaN after removing all '
               'missing values: ' ' '.join(vars_with_nan))
        raise ValueError(txt)


def add_missing_vars_fct(names_in_data, names_to_inc, obs, gen_dic):
    """Include missing variables as zeros."""
    missing_vars = [var for var in names_to_inc if var not in names_in_data]
    # list(set(names_to_inc).difference(names_in_data))
    if missing_vars:
        if gen_dic['with_output']:
            txt = '-' * 100 + 'The following variables will be addded with'
            txt += ' zero values: ' ' '.join(missing_vars)
            mcf_ps.print_mcf(gen_dic, txt, summary=False)
        add_df = pd.DataFrame(0, index=np.arange(obs), columns=missing_vars)
        return add_df
    return None


def print_prime_value_corr(data_train_dic, gen_dic, summary=False):
    """Print how primes correspond to underlying values."""
    if gen_dic['with_output'] and gen_dic['verbose']:
        txt = '\nDictionary for recoding of categorical variables into primes'
        key_list_unique_val_dict = list(data_train_dic['unique_values_dict'])
        key_list_prime_values_dict = list(data_train_dic['prime_values_dict'])
        for key_nr, key in enumerate(key_list_unique_val_dict):
            txt += '\n' + key
            txt += f'\n{data_train_dic["unique_values_dict"][key]}'
            txt += '\n' + key_list_prime_values_dict[key_nr]
            val = data_train_dic['prime_values_dict'][
                key_list_prime_values_dict[key_nr]]
            txt += f'\n{val}'
        if key_list_unique_val_dict:
            mcf_ps.print_mcf(gen_dic, txt, summary=summary)


def create_xz_variables(mcf_, data_df, train=True):
    """Create the variables for GATE & add to covariates, recode X.

    Parameters
    ----------
    mcf_c : Instance of the Modified Causal Forest class.
    data_df : DataFrame.
           Data for prediction or training.
    train : Bool. train is True if function is used in training part. Default
               is True

    Returns
    -------
    data_df : DataFrame.
        Updated data.

    """

    def check_for_nan(data_df, gen_dic, summary=False):
        vars_with_nans_mask = data_df.isna().any()
        vars_with_nans = vars_with_nans_mask[vars_with_nans_mask
                                             ].index.tolist()

        if vars_with_nans:
            txt = ('-' * 100 + '\nWARNING: The following variables are not '
                   'numeric or may contain missing values:\n'
                   f' {vars_with_nans} '
                   '\nWARNING: Rows with missing values will be deleted.\n'
                   ' Other non-numeric values have to be recoded as numerical'
                   ' if variable is used in the mcf.\n' + '-' * 100)
            mcf_ps.print_mcf(gen_dic, txt, summary=summary)

    var_dic, gen_dic, ct_dic = mcf_.var_dict, mcf_.gen_dict, mcf_.ct_dict
    p_dic, int_dic = mcf_.p_dict, mcf_.int_dict
    data_train_dic = mcf_.data_train_dict
    if train:
        no_val_dict, q_inv_dict, z_new_name_dict = {}, {}, {}
        z_new_dic_dict = {}
    else:
        d_indat_values = data_train_dic['d_indat_values']
        no_val_dict = data_train_dic['no_val_dict']
        q_inv_dict = data_train_dic['q_inv_dict']
        q_inv_cr_dict = data_train_dic['q_inv_cr_dict']
        prime_values_dict = data_train_dic['prime_values_dict']
        unique_val_dict = data_train_dic['unique_values_dict']
        z_new_name_dict = data_train_dic['z_new_name_dict']
        z_new_dic_dict = data_train_dic['z_new_dic_dict']
        prime_new_name_dict = data_train_dic['prime_new_name_dict']
        prime_old_name_dict = data_train_dic['prime_old_name_dict']

    data_df.columns = data_df.columns.str.casefold()
    temp_df = data_df.replace({False: 0, True: 1})
    data_df = temp_df
    if gen_dic['with_output']:
        check_for_nan(data_df, gen_dic, summary=False)
    if len(data_df.columns) > len(set(data_df.columns)):
        raise RuntimeError(
            'Duplicate variable names. Names are NOT case sensitive')
    # Get and check treatment variable
    if train:
        if gen_dic['d_type'] == 'continuous':
            d1_unique = ct_dic['grid_nn_val']
            data_df = recode_treat_cont(data_df, var_dic, ct_dic)
        else:
            d1_unique = np.unique(np.int32(np.round(
                data_df[var_dic['d_name']].to_numpy())))
    else:
        # This loads information from training iteration.
        d1_unique = d_indat_values
        if p_dic['gatet'] or p_dic['atet']:  # Treatment variable required
            text = 'Treatment variable differently coded in both datasets.'
            text += 'Set ATET and GATET to False.'
            d2_unique = np.unique(np.int32(np.round(
                data_df[var_dic['d_name']].to_numpy())))
            # Treatment identical in training and prediction?
            if len(d1_unique) == len(d2_unique):
                if not np.all(d1_unique == d2_unique):
                    raise ValueError(text)
            else:
                raise RuntimeError(text)
    # Part 1: Recoding and adding Z variables
    if gen_dic['agg_yes']:
        if not var_dic['z_name_cont'] == []:
            z_name_ord_new = []
            z_name_cont_remove = []  # changed 25.8.23
            if train:
                no_val_dict, q_inv_dict, z_new_name_dict = {}, {}, {}
                z_new_dic_dict = {}
            for z_name in var_dic['z_name_cont']:
                if train:
                    no_val = data_df[z_name].unique()
                    no_val_dict.update({z_name: no_val})
                else:
                    no_val = no_val_dict[z_name]
                if len(no_val) > p_dic['max_cats_z_vars']:
                    groups = p_dic['max_cats_z_vars']
                else:
                    groups = len(no_val)
                # Variables are discretized because too many groups
                if len(no_val) > groups:  # Else, existing categ.s are used
                    if train:
                        quant = np.linspace(1/groups, 1-1/groups, groups-1)
                        q_t = data_df[z_name].quantile(quant)  # Returns DF
                        std = data_df[z_name].std()
                        q_inv = [data_df[z_name].min() - 0.001 * std]
                        q_inv.extend(q_t)   # This is a list
                        q_inv.extend([data_df[z_name].max() + 0.001 * std])
                        q_inv = np.unique(q_inv)
                    else:
                        q_inv = q_inv_dict[z_name]
                        std = data_df[z_name].std()
                        q_inv[0] = min(q_inv[0],
                                       data_df[z_name].min() - 0.001 * std)
                        q_inv[-1] = max(q_inv[-1],
                                        data_df[z_name].max() + 0.001 * std)
                    data_s = pd.cut(data_df[z_name], q_inv, right=True,
                                    labels=False)
                    if train:
                        cont_name = data_s.name
                        new_name = cont_name + 'catv'
                        # z_name_ord_new.extend([new_name])
                    else:
                        new_name = z_new_name_dict[z_name]
                        new_dic = z_new_dic_dict[z_name]
                        cont_name = new_name[:-4]
                    z_name_ord_new.append(new_name)
                    z_name_cont_remove.append(cont_name)
                    data_df[new_name] = data_s.copy()
                    if train:
                        means = data_df.groupby(new_name).mean(
                            numeric_only=True)
                        new_dic = means[data_s.name].to_dict()
                        z_new_name_dict.update({z_name: new_name})
                        q_inv_dict.update({z_name: q_inv})
                        z_new_dic_dict.update({z_name: new_dic})
                    data_df = data_df.replace({new_name: new_dic})
                    if gen_dic['with_output'] and gen_dic['verbose']:
                        txt = (f'Training: Variable recoded: {data_s.name}'
                               f'-> {new_name}')
                        mcf_ps.print_mcf(gen_dic, txt, summary=False)
                else:
                    z_name_ord_new.extend([z_name])
            var_dic['z_name_ord'].extend(z_name_ord_new)
            var_dic['z_name'].extend(z_name_ord_new)
            if z_name_cont_remove:  # changed 25.8.23
                z_name_ord = [name for name in var_dic['z_name_ord']
                              if name not in z_name_cont_remove]
                z_name = [name for name in var_dic['z_name']
                          if name not in z_name_cont_remove]
                var_dic['z_name_ord'] = z_name_ord
                var_dic['z_name'] = z_name
        var_dic['z_name_ord'] = mcf_gp.cleaned_var_names(var_dic['z_name_ord'])
        var_dic['z_name'] = mcf_gp.cleaned_var_names(var_dic['z_name'])
        var_dic['x_name'].extend(var_dic['z_name'])
        if var_dic['x_name_balance_bgate']:
            bad_vars = [var for var in var_dic['x_name_balance_bgate']
                        if var not in var_dic['x_name']]
            if bad_vars:
                raise ValueError(f'{bad_vars} not included among features. Add'
                                 ' them.')
            var_dic['x_name_remain'].extend(var_dic['x_name_balance_bgate'])
        var_dic['x_name_remain'].extend(var_dic['z_name'])
        var_dic['x_name_balance_test'].extend(var_dic['z_name'])
        var_dic['name_ordered'].extend(var_dic['z_name_ord'])
        var_dic['name_ordered'].extend(var_dic['z_name_cont'])
        var_dic['name_unordered'].extend(var_dic['z_name_unord'])
        var_dic['x_name'] = mcf_gp.cleaned_var_names(var_dic['x_name'])
        var_dic['x_name_remain'] = mcf_gp.cleaned_var_names(
            var_dic['x_name_remain'])
        var_dic['x_name_balance_test'] = mcf_gp.cleaned_var_names(
            var_dic['x_name_balance_test'])
        var_dic['name_ordered'] = mcf_gp.cleaned_var_names(
            var_dic['name_ordered'])
        var_dic['name_unordered'] = mcf_gp.cleaned_var_names(
            var_dic['name_unordered'])
    # Part 2: Recoding ordered and unordered variables
    x_name_type_unord,  x_name_values = [], []
    if train:
        q_inv_cr_dict, prime_values_dict, unique_val_dict = {}, {}, {}
        prime_new_name_dict, prime_old_name_dict = {}, {}
    list_of_missing_vars = []
    for variable in var_dic['x_name']:
        if variable in var_dic['name_ordered']:  # Ordered variable
            unique_val = data_df[variable].unique()
            unique_val.sort()  # unique_val: Sorted from smallest to largest
            k = len(unique_val)
            # Recode continuous variables to fewer values to speed up programme
            x_name_type_unord += [0]
            # Determine whether this has to be recoded
            if int_dic['max_cats_cont_vars'] < (k-2):
                groups = int_dic['max_cats_cont_vars']
                quant = np.linspace(1/groups, 1-1/groups, groups-1)
                if train:
                    q_t = data_df[variable].quantile(quant)  # Returns DF
                    std = data_df[variable].std()
                    q_inv = [data_df[variable].min() - 0.001*std]
                    q_inv.extend(q_t)   # This is a list
                    q_inv.extend([data_df[variable].max() + 0.001*std])
                    q_inv_cr_dict.update({variable: q_inv})
                else:
                    q_inv = q_inv_cr_dict[variable]
                data_s = pd.cut(x=data_df[variable], bins=q_inv, right=True,
                                labels=False)
                new_variable = data_s.name + "CR"
                data_df[new_variable] = data_s.copy()
                means = data_df.groupby(new_variable).mean(numeric_only=True)
                new_dic = means[data_s.name].to_dict()
                data_df = data_df.replace({new_variable: new_dic})
                var_dic = mcf_gp.substitute_variable_name(
                    var_dic, variable, new_variable
                    )
                if gen_dic['with_output'] and gen_dic['verbose']:
                    txt = f'Variable recoded: {variable} -> {new_variable}'
                    mcf_ps.print_mcf(gen_dic, txt, summary=False)
                values_pd = data_df[new_variable].unique()
            else:
                values_pd = unique_val
            values = values_pd.tolist()
            if len(values) < int_dic['max_save_values']:
                x_name_values.append(sorted(values))
            else:
                x_name_values.append([])  # Add empty list to avoid excess mem
        elif variable in var_dic['name_unordered']:   # Unordered variable
            if train:
                unique_val = data_df[variable].unique()
                unique_val.sort()  # unique_val: Sorted from smallest to large
                k = len(unique_val)
                if k < 3:
                    raise ValueError(f'{variable} has only {k}'
                                     ' different values. Remove it from the '
                                     'list of unorderd variables and add it '
                                     'to the list of ordered variables.')

                # Recode categorical variables by running integers such that
                # groups of them can be efficiently translated into primes
                prime_values = mcf_gp.primes_list(k)
                if len(prime_values) != len(unique_val):
                    raise RuntimeError(
                        'Not enough prime values available for recoding. Most '
                        'likely reason: Continuous variables coded as'
                        ' unordered. Program stopped.')
            else:
                variable = variable[:-6]   # Remove _prime from variable name
                prime_values = prime_values_dict[variable + '_prime']  # List
                unique_val = unique_val_dict[variable]
                unique_val_pred = data_df[variable].unique()
                bad_vals = list(np.setdiff1d(unique_val_pred, unique_val))
                if bad_vals:    # List is not empty
                    data_df.drop(
                        data_df[data_df[variable].isin(bad_vals)].index,
                        inplace=True)
                    if gen_dic['with_output'] and gen_dic['verbose']:
                        # raise RuntimeError
                        txt = ('Too many values in unordered variable.'
                               ' Prediction file contains values that were not'
                               f' used for training: {variable} : {bad_vals}. '
                               'These obervations will be removed from '
                               'prediction data.')
                        mcf_ps.print_mcf(gen_dic, txt, summary=True)
            prime_variable = data_df[variable].name + '_prime'
            data_df[prime_variable] = data_df[variable].replace(
                    unique_val, prime_values)
            if train:
                prime_values_dict.update({prime_variable: prime_values})
                prime_new_name_dict.update({variable: prime_variable})
                prime_old_name_dict.update({prime_variable: variable})
                unique_val_dict.update({variable: tuple(unique_val)})
            if k < 19:
                x_name_type_unord += [1]  # <= 18 categories: dtype=int64
            else:
                x_name_type_unord += [2]  # > 18 categories: dtype=object
            values_pd = data_df[prime_variable].unique()
            x_name_values.append(sorted(values_pd.tolist()))
            var_dic = mcf_gp.substitute_variable_name(
                var_dic, variable, prime_variable
                )
            if gen_dic['with_output'] and gen_dic['verbose']:
                txt = f'Variable recoded: {variable} -> {prime_variable}'
                mcf_ps.print_mcf(gen_dic, txt, summary=False)
        else:
            list_of_missing_vars.extend(variable)
    if list_of_missing_vars:
        raise ValueError(f'{" ".join(list_of_missing_vars)} is neither '
                         'contained in list of ordered nor list of unordered '
                         'variables.')
    # Define dummy to see if particular type of UO exists at all in data
    type_0, type_1, type_2 = unordered_types_overall(x_name_type_unord)
    var_x_type = dict(zip(var_dic['x_name'], x_name_type_unord))
    var_x_values = dict(zip(var_dic['x_name'], x_name_values))
    gen_dic['x_type_0'], gen_dic['x_type_1'] = type_0, type_1
    gen_dic['x_type_2'] = type_2
    if gen_dic['with_output'] and gen_dic['verbose'] and (type_1 or type_2):
        txt = ''
        if type_1:
            txt = '\nType 1 unordered variable detected'
        if type_2:
            txt = '\nType 2 unordered variable detected'
        mcf_ps.print_mcf(gen_dic, txt, summary=False)
    if gen_dic['with_output'] and gen_dic['agg_yes']:
        txt = ('\n' + '-' * 100 + '\nShort analysis of policy variables'
               ' (variable to aggregate the effects; effect sample). \nEach'
               ' value of a variable defines an independent stratum.')
        if train:
            txt += '\nTraining data\n'
        else:
            txt += '\nPrediction data\n'
        txt += 'Name                             # of cat \n'
        for i in var_dic['z_name']:
            txt += f'{i:<32} {len(data_df[i].unique()):>6}\n'
        txt += '-' * 100 + '\n'
        mcf_ps.print_mcf(gen_dic, txt, summary=False)
    if not isinstance(var_dic['id_name'], (list, tuple)):
        var_dic['id_name'] = [var_dic['id_name']]
    if not var_dic['id_name'] or isinstance(var_dic['id_name'][0], str):
        # Add identifier to data
        var_dic['id_name'] = ['id_mcf']
        # data_df[var_dic['id_name'][0]] = np.arange(len(data_df))
        data_df.loc[:, var_dic['id_name'][0]] = np.arange(len(data_df))
    if gen_dic['with_output'] and int_dic['descriptive_stats'] and train:
        mcf_ps.print_descriptive_df(gen_dic, data_df, varnames='all',
                                    summary=False)
    mcf_.data_train_dict = {
        'd_indat_values': d1_unique, 'no_val_dict': no_val_dict,
        'q_inv_dict': q_inv_dict, 'q_inv_cr_dict': q_inv_cr_dict,
        'prime_values_dict': prime_values_dict,
        'unique_values_dict': unique_val_dict,
        'prime_new_name_dict': prime_new_name_dict,
        'prime_old_name_dict': prime_old_name_dict,
        'z_new_name_dict': z_new_name_dict, 'z_new_dic_dict': z_new_dic_dict
        }

    mcf_.var_dict = var_dic
    mcf_.var_x_type,  mcf_.var_x_values = var_x_type, var_x_values
    mcf_.gen_dict = gen_dic
    return data_df


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


def check_recode_treat_variable(mcf_, data_df, treat_name=None):
    """Recode treatment variable if categorical or does not start with 0."""
    gen_dic = mcf_.gen_dict
    d_name = mcf_.var_dict['d_name'] if treat_name is None else treat_name
    n_old_df = len(data_df)
    if isinstance(d_name, (list, tuple)):
        d_name = d_name[0]
    temp_df = data_df.dropna(subset=d_name)
    data_df = temp_df
    n_new_df = len(data_df)
    if n_new_df < n_old_df:
        if gen_dic['with_output']:
            mcf_ps.print_mcf(gen_dic, f'{n_old_df - n_new_df} observations',
                             f' dropped due to missing values in {d_name}',
                             summary=True)
    d_dat_pd = data_df[d_name].squeeze()  # pylint: disable=E1136
    if d_dat_pd.dtype == 'object':
        d_dat_pd = d_dat_pd.astype('category')
        mcf_ps.print_mcf(gen_dic, d_dat_pd.cat.categories, summary=False)
        if gen_dic['with_output']:
            mcf_ps.print_mcf(gen_dic,
                             'Automatic recoding of treatment variable',
                             summary=True)
            numerical_codes = pd.unique(d_dat_pd.cat.codes)
            mcf_ps.print_mcf(gen_dic, numerical_codes, summary=True)
        data_df[d_name] = d_dat_pd.cat.codes
    if data_df[d_name].min() != 0:
        data_df[d_name] = data_df[d_name] - data_df[d_name].min()
    return data_df


def recode_treat_cont(data_df, var_dic, ct_dic):
    """Recode treatment variable for continuous treatments."""
    d_dat_np = data_df[var_dic['d_name']].to_numpy()
    data_df[var_dic['d_name']] = (data_df[var_dic['d_name']] -
                                  data_df[var_dic['d_name']].min())
    d_dat_np = np.where(d_dat_np < 1e-15, 0, d_dat_np)
    d_discr_w_pd = get_d_discr(data_df, d_dat_np, ct_dic['grid_w_val'],
                               var_dic['grid_w_name'])
    d_discr_dr_pd = get_d_discr(data_df, d_dat_np, ct_dic['grid_dr_val'],
                                var_dic['grid_dr_name'])
    if np.all(ct_dic['grid_nn_val'] == ct_dic['grid_w_val']):
        data_df = pd.concat([data_df, d_discr_w_pd, d_discr_dr_pd], axis=1)
    else:
        d_discr_nn_pd = get_d_discr(data_df, d_dat_np, ct_dic['grid_nn_val'],
                                    var_dic['grid_nn_name'])
        data_df = pd.concat([data_df, d_discr_nn_pd, d_discr_w_pd,
                             d_discr_dr_pd], axis=1)
    return data_df


def get_d_discr(data, d_dat_np, grid_val, d_grid_name):
    """Recode continuous II."""
    d_discr = np.zeros_like(d_dat_np)
    for idx, val in enumerate(d_dat_np):
        jdx = (np.abs(val - grid_val)).argmin()
        d_discr[idx] = grid_val[jdx]
    d_discr = np.where((d_dat_np > 1e-15) & (d_discr == 0), grid_val[1],
                       d_discr)
    return pd.DataFrame(data=d_discr, columns=d_grid_name, index=data.index)


def dummies_for_unord(x_df, names_unordered, primes=True,
                      data_train_dict=None):
    """Add dummies and remove unordered variables."""
    unordered_dummy_names = {}
    for idx, name in enumerate(names_unordered):
        x_dummies = pd.get_dummies(x_df[name], columns=[name],  dtype=int)
        x_dummies_names = [name + str(ddd) for ddd in x_dummies.columns]
        unordered_dummy_names[name] = x_dummies.columns = x_dummies_names
        if data_train_dict is not None:  # Check prediction data if complete
            vals_in_x = x_df[name].unique()
            if primes:
                vals_all = data_train_dict['prime_values_dict'][name]
            else:
                vals_all = data_train_dict['unique_values_dict'][name]
            vals_not_in = [val for val in vals_all if val not in vals_in_x]
            bad_vals = [val for val in vals_in_x if val not in vals_all]
            if bad_vals:
                raise ValueError(f'{name}: {bad_vals} are in the prediction'
                                 'data but not in the training data.')
            if vals_not_in:
                x_not_in_df = pd.DataFrame(data=vals_not_in, columns=[name])
                x_help = pd.get_dummies(x_not_in_df[name], columns=[name],
                                        dtype=int)
                add_name = [name + str(ddd) for ddd in x_help.columns]
                x_not_in_np = np.zeros((len(x_dummies), len(add_name)))
                x_not_in_df = pd.DataFrame(data=x_not_in_np, columns=add_name)
                x_not_in_df.index = x_dummies.index
                x_dummies = pd.concat((x_dummies, x_not_in_df), axis=1,
                                      copy=True)
        x_dummies_s = x_dummies.reindex(sorted(x_dummies.columns), axis=1)
        if idx == 0:
            x_all_dummies = x_dummies_s
        else:
            x_all_dummies = pd.concat([x_all_dummies, x_dummies_s], axis=1,
                                      copy=True)
    # Remove names of unordered variables
    names_wo = [name for name in x_df.columns if name not in names_unordered]
    x_new_df = pd.concat((x_df[names_wo], x_all_dummies), axis=1, copy=True)
    return x_new_df, unordered_dummy_names


def get_x_data(data_df, x_name):
    """Get features."""
    x_all = data_df[x_name]    # deep copies
    obs = len(x_all.index)
    return x_all, obs


def get_treat_info(mcf_):
    """Get some basic treatment info frequently needed."""
    if mcf_.gen_dict['d_type'] == 'continuous':
        d_name = mcf_.var_dict['grid_nn_name']
        d_values = mcf_.ct_dict['grid_nn_val']
        no_of_treat = len(mcf_.ct_dict['grid_nn_val'])
    else:
        d_name = mcf_.var_dict['d_name']
        d_values = mcf_.gen_dict['d_values']
        no_of_treat = mcf_.gen_dict['no_of_treat']
    return d_name, d_values, no_of_treat


def data_frame_vars_lower(data_df):
    """Make sure all variables in dataframe are capitalized."""
    var_names = [var.casefold() for var in data_df.columns]
    data_df.columns = var_names
    return data_df, var_names
