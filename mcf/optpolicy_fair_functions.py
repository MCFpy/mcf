"""Created on Wed May  1 16:35:19 2024.

Functions for correcting scores w.r.t. protected variables.

@author: MLechner

# -*- coding: utf-8 -*-
"""
import numpy as np
import pandas as pd

from mcf import mcf_estimation_generic_functions as mcf_gf
from mcf import mcf_print_stats_functions as mcf_ps


def adjust_scores(optp_, data_df, seed=1246546):
    """Remove effect of protected variables from policy score."""
    gen_dic, fair_dic = optp_.gen_dict, optp_.fair_dict
    # Change data to numpy arrays
    scores_np = data_df[optp_.var_dict['polscore_name']].to_numpy()
    protect_np = data_df[optp_.var_dict['protected_name']].to_numpy()
    y_cond_mean_protect_np = np.zeros_like(scores_np)
    if fair_dic['adj_type'] == 'MeanVar':
        y_cond_var_protect_np = np.zeros_like(scores_np)
    obs, no_of_scores = scores_np.shape
    cross_validation_k = 5
    boot = 1000
    if gen_dic['with_output']:
        txt_report = ('\nMethod selected for Fairness adjustment: '
                      f'{fair_dic["adj_type"]}')
        if gen_dic['with_output']:
            mcf_ps.print_mcf(gen_dic, txt_report, summary=True)
    else:
        txt_report = ''

    if fair_dic['adj_type'] in ('Mean', 'MeanVar',):
        if fair_dic['adj_type'] == 'Mean':
            adjust_ment_set = ('mean', )
        else:
            adjust_ment_set = ('mean', 'variance')

        # Loop over scores to obtain prediction of conditonal expectation of y
        for idx in range(no_of_scores):
            for mean_var in adjust_ment_set:
                if gen_dic['with_output']:
                    print(f'\nCurrently adjusting {mean_var}')
                if mean_var == 'mean':
                    # Adjust conditional mean by residualisation
                    y_np = scores_np[:, idx]
                    if np.std(y_np) < 1-8:
                        y_cond_mean_protect_np[:, idx] = y_np.copy()
                        continue
                else:
                    # Adjust conditional variance by rescaling
                    y_np = (scores_np[:, idx]
                            - y_cond_mean_protect_np[:, idx])**2
                    if np.std(y_np) < 1-8:
                        y_cond_var_protect_np[:, idx] = y_np.copy()**2
                        continue

                # Find best estimator for outcome
                (estimator, params, best_lable, _, transform_x, txt_mse
                 ) = mcf_gf.best_regression(
                    protect_np,  y_np.ravel(),
                    estimator=fair_dic['regression_method'],
                    boot=boot, seed=seed+12435,
                    max_workers=optp_.int_dict['mp_parallel'],
                    cross_validation_k=cross_validation_k,
                    absolute_values_pred=mean_var == 'variance')
                if gen_dic['with_output']:
                    text = (f'\nAdjustment for {mean_var} of '
                            + optp_.var_dict['polscore_name'][idx])
                    txt_mse = text + txt_mse
                    mcf_ps.print_mcf(gen_dic, txt_mse, summary=False)
                    if mean_var == 'mean':
                        txt_report += '\n'
                    txt_report += text + ' by ' + best_lable

                # Obtain out-of-sample prediction by k-fold cross-validation
                index = np.arange(obs)      # indices
                rng = np.random.default_rng(seed=seed)
                rng.shuffle(index)
                index_folds = np.array_split(index, cross_validation_k)
                for fold_pred in range(cross_validation_k):
                    fold_train = [x for indx, x in enumerate(index_folds)
                                  if indx != fold_pred]
                    index_train = np.hstack(fold_train)
                    index_pred = index_folds[fold_pred]
                    if transform_x:
                        _, x_train, x_pred = mcf_gf.scale(
                            protect_np[index_train], protect_np[index_pred])
                    else:
                        x_train = protect_np[index_train]
                        x_pred = protect_np[index_pred]
                    y_train = y_np[index_train].ravel()
                    y_obj = mcf_gf.regress_instance(estimator, params)
                    if y_obj is None:
                        if mean_var == 'mean':
                            y_cond_mean_protect_np[index_pred, idx
                                                   ] = np.average(y_train)
                        else:
                            y_cond_var_protect_np[index_pred, idx
                                                  ] = np.average(y_train)
                    else:
                        y_obj.fit(x_train, y_train)
                        if mean_var == 'mean':
                            y_cond_mean_protect_np[index_pred, idx
                                                   ] = y_obj.predict(x_pred)
                        else:
                            y_cond_var_protect_np[index_pred, idx
                                                  ] = y_obj.predict(x_pred)
    fair_score_np = scores_np - y_cond_mean_protect_np
    if fair_dic['adj_type'] == 'MeanVar':
        constant = 0.000001
        mask_var = y_cond_var_protect_np < constant
        if mask_var.any():
            y_cond_var_protect_np = np.where(mask_var, constant,
                                             y_cond_var_protect_np.copy())
        # Remove variability due to heteroscedasticity
        y_cond_std_protect_np = np.sqrt(y_cond_var_protect_np)
        fair_score_np /= y_cond_std_protect_np
        # Rescale to retain the variability of the original scores
        std_fair_score_np = np.mean(1 / y_cond_std_protect_np, axis=0
                                    ).reshape(-1, 1).T
        fair_score_np /= std_fair_score_np
    # Correct scores, but keep score specific mean
    fair_score_np += np.mean(y_cond_mean_protect_np, axis=0).reshape(-1, 1).T
    fairscore_name = [name + '_fair'
                      for name in optp_.var_dict['polscore_name']]
    fair_score_df = pd.DataFrame(fair_score_np, columns=fairscore_name)
    data_df = data_df.reset_index(drop=True)
    data_fair_df = pd.concat((data_df, fair_score_df), axis=1)

    # Descriptive statistics
    if gen_dic['with_output']:
        txt_report += fair_stats(optp_, data_fair_df, fairscore_name)
    # Change variable names
    if optp_.var_dict['polscore_desc_name'] is not None:
        optp_.var_dict['polscore_desc_name'].extend(
            optp_.var_dict['polscore_name'].copy())
    else:
        optp_.var_dict['polscore_desc_name'] = optp_.var_dict[
            'polscore_name'].copy()
    if (optp_.var_dict['vi_x_name'] is not None
            and optp_.var_dict['protected_ord_name'] is not None):
        optp_.var_dict['vi_x_name'].extend(
            optp_.var_dict['protected_ord_name'])
    elif (optp_.var_dict['vi_x_name'] is None
          and optp_.var_dict['protected_ord_name'] is not None):
        optp_.var_dict['vi_x_name'] = optp_.var_dict['protected_ord_name']

    if (optp_.var_dict['vi_to_dummy_name'] is not None
            and optp_.var_dict['protected_unord_name'] is not None):
        optp_.var_dict['vi_to_dummy_name'].extend(
            optp_.var_dict['protected_unord_name'])
    elif (optp_.var_dict['vi_to_dummy_name'] is None
          and optp_.var_dict['protected_ord_name'] is not None):
        optp_.var_dict['vi_to_dummy_name'] = optp_.var_dict[
            'protected_unord_name']

    if optp_.var_dict['vi_x_name'] is not None:
        optp_.var_dict['vi_x_name'] = unique_list(optp_.var_dict['vi_x_name'])
    if optp_.var_dict['vi_to_dummy_name'] is not None:
        optp_.var_dict['vi_to_dummy_name'] = unique_list(
            optp_.var_dict['vi_to_dummy_name'])
    optp_.var_dict['polscore_name'] = fairscore_name
    optp_.report['fairscores_build_stats'] = txt_report
    return data_fair_df, fairscore_name


def unique_list(list_tuple):
    """Remove duplicate elements from list without changing order."""
    unique = []
    _ = [unique.append(item)
         for item in list_tuple if item not in unique]
    return unique


def fair_stats(optp_, data_df, fairscore_names):
    """Compute descriptive statistics of fairness adjustments."""
    prot_names = optp_.var_dict['protected_name']
    score_names = optp_.var_dict['polscore_name']
    txt = '\n\nDescriptive statistics of adjusted and unadjusted scores'
    all_scores = [*score_names, *fairscore_names]
    stats = np.round(data_df[all_scores].describe().T, 3)
    corr = np.round(data_df[all_scores].corr() * 100)
    txt += '\n' + str(stats)
    txt += '\n\nCorrelation of adjusted and unadjusted scores in %'
    txt += '\n' + str(corr)
    correlations_np = np.zeros((len(all_scores), len(prot_names)))
    correlations_df = pd.DataFrame(correlations_np,
                                   index=all_scores, columns=prot_names)
    for score in all_scores:
        for prot in prot_names:
            corr = data_df[[score, prot]].corr()
            correlations_df.loc[score, prot] = round(corr.iloc[1, 0] * 100, 2)
    txt += ('\n' * 2 + 'Bivariate Correlations in %'
            + '\n' + str(correlations_df))

    mcf_ps.print_mcf(optp_.gen_dict, txt, summary=True)
    return txt
