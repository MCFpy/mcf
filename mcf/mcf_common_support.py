"""
Contains functions for common support adjustments.

Created on Thu May 11 16:30:11 2023

@author: MLechner
# -*- coding: utf-8 -*-
"""
from copy import deepcopy
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mcf import mcf_data
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats as mcf_ps
from mcf.mcf_estimation import predict_proba_aligned
from mcf.mcf_variable_importance import print_variable_importance

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.mcf_init import GenCfg
    from mcf.mcf_init_train import CsCfg


def common_support_p_score(mcf_: 'ModifiedCausalForest',
                           tree_df: pd.DataFrame,
                           fill_y_df: pd.DataFrame,
                           train: bool = True,
                           p_score_only: bool = False,
                           ) -> tuple[pd.DataFrame, pd.DataFrame,
                                      float, int,
                                      list[str],
                                      pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None,
                                      ]:
    """Remove observations from data files that are off-support."""
    gen_cfg, int_cfg = mcf_.gen_cfg, mcf_.int_cfg
    lc_cfg, var_x_type, cs_cfg = mcf_.lc_cfg, mcf_.var_x_type, mcf_.cs_cfg
    data_train_dic = mcf_.data_train_dict
    cs_tree_prob_df = cs_fill_y_prob_df = predicted_probs_df = x_cs_np = None
    probs_df = None
    d_name, d_values, no_of_treat = mcf_data.get_treat_info(mcf_)  # Only main treatments
    x_name, x_type = mcf_gp.get_key_values_in_list(var_x_type)
    len1 = 0 if tree_df is None else len(tree_df)
    len2 = 0 if fill_y_df is None else len(fill_y_df)
    obs = len1 + len2
    names_unordered = [x_name[j] for j, val in enumerate(x_type) if val > 0]
    txt_cs = 'Propensity score estimation' if p_score_only else 'Common support analysis'
    if gen_cfg.with_output and gen_cfg.verbose:
        mcf_ps.print_mcf(gen_cfg, '\n' + '=' * 100 + '\n' + txt_cs, summary=True)
    if train:
        if cs_cfg.detect_const_vars_stop and not p_score_only:
            check_const_single_variable(list(var_x_type.keys()),
                                        d_name,
                                        (tree_df, fill_y_df,)
                                        )
        if lc_cfg.cs_cv:   # Crossvalidate ... only tree data is used
            tree_mcf_df, fill_y_mcf_df = tree_df.copy(), fill_y_df.copy()
            d_tree_mcf_np = mcf_gp.to_numpy_big_data(tree_mcf_df[d_name], int_cfg.obs_bigdata
                                                     ).ravel()
        else:  # Use lc_cfg.cs_share of data for common support estim. only
            # Take the same share of obs. from both input samples
            tree_mcf_df, tree_cs_df = train_test_split(tree_df,
                                                       test_size=lc_cfg.cs_share, random_state=42
                                                       )
            fill_y_mcf_df, fill_y_cs_df = train_test_split(fill_y_df,
                                                           test_size=lc_cfg.cs_share,
                                                           random_state=42
                                                           )
            data_cs_df = pd.concat([tree_cs_df, fill_y_cs_df], axis=0)
            x_cs_df, _ = mcf_data.get_x_data(data_cs_df, x_name)
            d_cs_np = mcf_gp.to_numpy_big_data(data_cs_df[d_name], int_cfg.obs_bigdata).ravel()
        x_fy_df, _ = mcf_data.get_x_data(fill_y_mcf_df, x_name)
    else:
        tree_mcf_df, fill_y_mcf_df = tree_df, None
    x_mcf_df, obs_mcf = mcf_data.get_x_data(tree_mcf_df, x_name)  # train,adj.
    txt = ''
    file_list_jpeg = file_list_d_jpeg = None
    if train:
        if names_unordered:  # List is not empty
            x_fy_df, _ = mcf_data.dummies_for_unord(
                x_fy_df, names_unordered, data_train_dict=data_train_dic)
            x_mcf_df, dummy_names = mcf_data.dummies_for_unord(
                x_mcf_df, names_unordered, data_train_dict=data_train_dic)
            if not lc_cfg.cs_cv:
                x_cs_df, _ = mcf_data.dummies_for_unord(
                    x_cs_df, names_unordered, data_train_dict=data_train_dic)
        else:
            dummy_names = None
        if not lc_cfg.cs_cv:
            x_cs_np = mcf_gp.to_numpy_big_data(x_cs_df, int_cfg.obs_bigdata)
        x_fy_np = mcf_gp.to_numpy_big_data(x_fy_df, int_cfg.obs_bigdata)
        x_mcf_np = mcf_gp.to_numpy_big_data(x_mcf_df, int_cfg.obs_bigdata)
        if gen_cfg.with_output and gen_cfg.verbose:
            txt += '\n' + '-' * 100 + '\n'
            if p_score_only:
                txt += 'Computing random forest based propensity score\n'
            else:
                txt += 'Computing random forest based common support\n'
            mcf_ps.print_mcf(gen_cfg, txt, summary=False)
        max_workers = 1 if int_cfg.replication else gen_cfg.mp_parallel
        classif_parameters = {'n_estimators': mcf_.cf_cfg.boot,
                              'criterion': 'gini',
                              'max_features': 'sqrt',
                              'bootstrap': True,
                              'oob_score': False,
                              'n_jobs': max_workers,
                              'min_samples_split': 5,
                              'random_state': 42,
                              'verbose': False,
                              'min_samples_leaf': 5,
                              }
        classif = RandomForestClassifier(**classif_parameters)
        if gen_cfg.with_output and gen_cfg.verbose and int_cfg.memory_print:
            mcf_sys.print_mememory_statistics(gen_cfg, 'Training: ' + txt_cs + ' before CV')
        if lc_cfg.cs_cv:   # Crossvalidate
            index = np.arange(obs_mcf)       # indices
            rng = np.random.default_rng(seed=9324561)
            rng.shuffle(index)
            index_folds = np.array_split(index, lc_cfg.cs_cv_k)
            pred_mcf_np = np.empty((len(index), no_of_treat))
            pred_fy_np_fold = np.zeros((len(x_fy_np), no_of_treat))
            forests = []
            for fold_pred in range(lc_cfg.cs_cv_k):
                fold_train = [x for idx, x in enumerate(index_folds) if idx != fold_pred]
                index_train = np.hstack(fold_train)
                index_pred = index_folds[fold_pred]
                x_pred, x_train = x_mcf_np[index_pred], x_mcf_np[index_train]
                d_train = d_tree_mcf_np[index_train]
                classif.fit(x_train, d_train)
                forests.append(deepcopy(classif))
                pred_mcf_np[index_pred, :] = predict_proba_aligned(classif, x_pred, d_values,
                                                                   dtype=pred_mcf_np.dtype,
                                                                   zero_tol=int_cfg.zero_tol,
                                                                   )
                pred_fy_np_fold += predict_proba_aligned(classif, x_fy_np, d_values,
                                                         dtype=pred_fy_np_fold.dtype,
                                                         zero_tol=int_cfg.zero_tol,
                                                         )
                if gen_cfg.with_output and gen_cfg.verbose and int_cfg.memory_print:
                    mcf_sys.print_mememory_statistics(gen_cfg, 'Training: ' + txt_cs + ' during CV')
            pred_cs_np, d_cs_np = pred_mcf_np, d_tree_mcf_np  # To get cut-offs
            pred_fy_np = pred_fy_np_fold / lc_cfg.cs_cv_k
        else:
            x_train, x_test, d_train, d_test = train_test_split(x_cs_np, d_cs_np, test_size=0.25,
                                                                random_state=42
                                                                )
            classif.fit(x_train, d_train)
            d_cs_np = d_test
            proba_dtype = np.float32 if len(x_mcf_np) >= int_cfg.obs_bigdata else np.float64
            pred_cs_np = predict_proba_aligned(classif, x_test, d_values,
                                               dtype=proba_dtype, zero_tol=int_cfg.zero_tol,
                                               )
            pred_mcf_np = predict_proba_aligned(classif, x_mcf_np, d_values,
                                                dtype=proba_dtype, zero_tol=int_cfg.zero_tol,
                                                )
            pred_fy_np = predict_proba_aligned(classif, x_fy_np, d_values,
                                               dtype=proba_dtype, zero_tol=int_cfg.zero_tol,
                                               )
            forests = [classif]
            if gen_cfg.with_output and gen_cfg.verbose and int_cfg.memory_print:
                mcf_sys.print_mememory_statistics(gen_cfg, 'Prediction: ' + txt_cs)
        cs_cfg.forests = forests
        if gen_cfg.with_output:
            if names_unordered:
                names_unordered_print = data_train_dic['prime_old_name_dict']
            else:
                names_unordered_print = None

            print_variable_importance(deepcopy(classif), x_mcf_df, tree_mcf_df[d_name],
                                      x_name=x_name, names_uo= names_unordered,
                                      unordered_dummy_names=dummy_names, gen_cfg=gen_cfg,
                                      summary=True,
                                      name_label_dict=names_unordered_print,
                                      obs_bigdata=int_cfg.obs_bigdata,
                                      classification=True, scaler=None,
                                      )
        # Normalize estimated probabilities to add up to 1
        if not p_score_only:
            pred_cs_np_sum = pred_cs_np.sum(axis=1, keepdims=True)
            pred_mcf_np_sum = pred_mcf_np.sum(axis=1, keepdims=True)
            pred_fy_np_sum = pred_fy_np.sum(axis=1, keepdims=True)
            pred_cs_np /= pred_cs_np_sum
            pred_mcf_np /= pred_mcf_np_sum
            pred_fy_np /= pred_fy_np_sum

            # Determine cut-offs nased on pred_cs_np
            cs_cfg.cut_offs = get_cut_off_probs(mcf_, pred_cs_np, d_cs_np)
            mcf_.cs_cfg = cs_cfg   # Update instance with cut-off prob's
            # Descriptive stats
            if gen_cfg.with_output:
                titel_ = 'Training - fill mcf with y data'
                file_list_jpeg, file_list_d_jpeg = plot_support(mcf_, pred_cs_np, d_cs_np, titel_)
                descriptive_stats_on_off_support(mcf_, pred_fy_np, fill_y_mcf_df, titel_)
            # Reduce samples
            fill_y_mcf_df, _, cs_fill_y_prob_df = on_off_support_df(
                mcf_.cs_cfg, mcf_.var_cfg.id_name,
                pred_fy_np, fill_y_mcf_df,
                return_predicted=gen_cfg.with_output,
                )
    else:  # Reduce prediction sample
        # Predict treatment probabilities
        if names_unordered:  # List is not empty
            x_mcf_df, _ = mcf_data.dummies_for_unord(x_mcf_df, names_unordered,
                                                     data_train_dict=data_train_dic
                                                     )
        if len(x_mcf_df) < int_cfg.obs_bigdata:
            pred_mcf_np = np.zeros((len(x_mcf_df), no_of_treat))
        else:
            pred_mcf_np = np.zeros((len(x_mcf_df), no_of_treat), dtype=np.float32)
        # If cross-validation, take average of forests in folds
        for forest in cs_cfg.forests:
            # pred_mcf_np += forest.predict_proba(mcf_gp.to_numpy_big_data(x_mcf_df,
            #                                                              int_cfg.obs_bigdata)
            #                                     )
            x_pred_np = mcf_gp.to_numpy_big_data(x_mcf_df, int_cfg.obs_bigdata)
            pred_mcf_np += predict_proba_aligned(forest, x_pred_np, d_values,
                                                 dtype=pred_mcf_np.dtype, zero_tol=int_cfg.zero_tol,
                                                 )
        pred_mcf_np /= len(cs_cfg.forests)
        # Normalize estimated probabilities to add up to 1
        pred_mcf_np_sum = pred_mcf_np.sum(axis=1, keepdims=True)
        pred_mcf_np /= pred_mcf_np_sum

    # Delete observation off support
    if not p_score_only:
        if gen_cfg.with_output:
            titel = 'Training - build mcf data' if train else 'Prediction data'
            descriptive_stats_on_off_support(mcf_, pred_mcf_np, tree_mcf_df, titel)
        tree_mcf_df, _, probs_df = on_off_support_df(mcf_.cs_cfg, mcf_.var_cfg.id_name,
                                                     pred_mcf_np, tree_mcf_df,
                                                     return_predicted=gen_cfg.with_output,
                                                     )
    if train and not p_score_only:
        cs_tree_prob_df = probs_df
    else:
        predicted_probs_df = probs_df

    if p_score_only:
        predicted_probs_df = predicted_probs_frame(tree_mcf_df, mcf_.var_cfg.id_name, pred_mcf_np)

    len1 = 0 if tree_mcf_df is None else len(tree_mcf_df)
    len2 = 0 if fill_y_mcf_df is None else len(fill_y_mcf_df)
    if train and not p_score_only:
        check_if_too_many_deleted(mcf_, len1, len(pred_mcf_np) - len1)
        check_if_too_many_deleted(mcf_, len2, len(pred_fy_np) - len2)
    obs_remain = len1 + len2
    share_deleted = (obs - obs_remain) / obs

    return (tree_mcf_df, fill_y_mcf_df, share_deleted, obs_remain,
            (file_list_jpeg, file_list_d_jpeg,),
            cs_tree_prob_df, cs_fill_y_prob_df, predicted_probs_df,
            )


def check_if_too_many_deleted(mcf_: 'ModifiedCausalForest', obs_keep: int, obs_del: int) -> None:
    """Check if too many obs are deleted and raise Exception if so."""
    max_del_train = mcf_.cs_cfg.max_del_train
    share_del = obs_del / (obs_keep + obs_del)
    if share_del > max_del_train:
        err_str = (f'{share_del:3.1%} observation deleted in common support, but only'
                   f' {max_del_train:3.1%} observations of training data are allowed'
                   ' to be deleted in support check. Programme is terminated. Improve'
                   ' balance of input data or change share allowed to be deleted.'
                   )
        raise ValueError(err_str)


def descriptive_stats_on_off_support(mcf_: 'ModifiedCausalForest',
                                     probs_np: NDArray[Any],
                                     data_df: pd.DataFrame,
                                     titel: str = ''
                                     ) -> None:
    """Compute descriptive stats for deleted and retained observations."""
    keep_df, delete_df, _ = on_off_support_df(mcf_.cs_cfg,
                                              mcf_.var_cfg.id_name,
                                              probs_np,
                                              data_df,
                                              return_predicted=False
                                              )
    gen_cfg, var_cfg = mcf_.gen_cfg, mcf_.var_cfg
    titel = '\n' + '-' * 100 + f'\nData investigated for common support: {titel}\n' + '-' * 100
    if delete_df.empty:
        mcf_ps.print_mcf(gen_cfg,
                         titel + '\nNo observations deleted in common support check', summary=True
                         )
    else:
        d_name, _, _ = mcf_data.get_treat_info(mcf_)
        data_train_dic = mcf_.data_train_dict
        keep_df = mcf_ps.change_name_value_df(keep_df,
                                              data_train_dic['prime_old_name_dict'],
                                              data_train_dic['prime_values_dict'],
                                              data_train_dic['unique_values_dict']
                                              )
        delete_df = mcf_ps.change_name_value_df(delete_df, data_train_dic['prime_old_name_dict'],
                                                data_train_dic['prime_values_dict'],
                                                data_train_dic['unique_values_dict']
                                                )
        x_name = [data_train_dic['prime_old_name_dict'].get(item, item)
                  for item in var_cfg.x_name.copy()
                  ]
        obs_del, obs_keep = len(delete_df), len(keep_df)
        obs = obs_del + obs_keep
        txt = f'\nObservations deleted: {obs_del:4} ({obs_del/obs:.2%})'
        txt += '\n' + '-' * 100
        mcf_ps.print_mcf(gen_cfg, titel + txt, summary=True)
        txt = ''
        with pd.option_context('display.max_rows', 500, 'display.max_columns', 500,
                               'display.expand_frame_repr', True, 'display.width', 150,
                               'chop_threshold', 1e-13
                               ):
            all_var_names = [name.casefold() for name in data_df.columns]
            if d_name[0].casefold() in all_var_names:
                d_keep = keep_df[d_name]
                d_delete = delete_df[d_name]
                d_keep_count = d_keep.value_counts(sort=False)
                d_delete_count = d_delete.value_counts(sort=False)
                d_keep_count = pd.concat([d_keep_count, np.round(d_keep_count / obs_keep * 100, 2)],
                                         axis=1
                                         )
                d_delete_count = pd.concat([d_delete_count,
                                            np.round(d_delete_count / obs_del * 100, 2)],
                                           axis=1
                                           )
                d_keep_count.columns = ['Obs.', 'Share in %']
                d_delete_count.columns = ['Obs.', 'Share in %']
                if gen_cfg.panel_data:
                    cluster_id = data_df[var_cfg.cluster_name].squeeze()
                    cluster_keep = keep_df[var_cfg.cluster_name].squeeze()
                    cluster_delete = delete_df[var_cfg.cluster_name].squeeze()
                k_str = '\nObservations kept, by treatment\n    '
                d_str = '\nObservations deleted, by treatment\n '
                k_str += d_keep_count.to_string()
                d_str += d_delete_count.to_string()
                txt += k_str + '\n' + '-   ' * 20 + d_str
                if gen_cfg.panel_data:
                    txt += '-   ' * 20
                    txt += '\nTotal number of panel units:'
                    txt += f'{len(cluster_id.unique())}'
                    txt += '\nObservations belonging to '
                    txt += f'{len(cluster_keep.unique())} panel units that are'
                    txt += '  ON support\nObservations belonging to '
                    txt += f'{len(cluster_delete.unique())} panel units are'
                    txt += ' OFF support'
                mcf_ps.print_mcf(gen_cfg, txt, summary=True)
            else:
                txt = f'\nData investigated for common support: {titel}\n'
                txt += '-' * 100 + '\nTreatment not in prediction data.\n'
                txt += '-' * 100
                mcf_ps.print_mcf(gen_cfg, txt, summary=False)
            txt = '\n' + '-' * 100
            txt += '\nFull sample (Data ON and OFF support)' + '\n' + '-' * 100
            mcf_ps.print_mcf(gen_cfg, txt, summary=False)
            mcf_ps.print_mcf(gen_cfg, data_df[x_name].describe().transpose(),  summary=False)
            if d_name[0].casefold() in all_var_names:
                mean_by_treatment(data_df[d_name], data_df[x_name], gen_cfg, summary=False)
            txt = '\n' + '-' * 100 + '\nData ON support' + '\n' + '-' * 100
            mcf_ps.print_mcf(gen_cfg, txt, summary=False)
            mcf_ps.print_mcf(gen_cfg, keep_df[x_name].describe().transpose(), summary=False)
            if d_name[0].casefold() in all_var_names and len(keep_df) > 5:
                mean_by_treatment(keep_df[d_name], keep_df[x_name], gen_cfg, summary=False)
            txt = '\n' + '-' * 100 + '\nData OFF support' + '\n' + '-' * 100
            mcf_ps.print_mcf(gen_cfg, txt, summary=False)
            mcf_ps.print_mcf(gen_cfg, delete_df[x_name].describe().transpose(), summary=False)
            if d_name[0].casefold() in all_var_names and len(delete_df) > 5:
                mean_by_treatment(delete_df[d_name], delete_df[x_name], gen_cfg, summary=False)


def mean_by_treatment(treat_df: pd.DataFrame,
                      data_df: pd.DataFrame,
                      gen_cfg: 'GenCfg',
                      summary: str = False
                      ) -> None:
    """Compute mean by treatment status."""
    treat_df = treat_df.squeeze()
    treat_vals = pd.unique(treat_df)
    txt = '\n------------------ Mean by treatment status ---------------------'
    mcf_ps.print_mcf(gen_cfg, txt, summary=summary)
    if len(treat_vals) > 0:
        mean = data_df.groupby(treat_df).mean(numeric_only=True)
        mcf_ps.print_mcf(gen_cfg, mean.transpose(), summary=summary)
    else:
        txt = f'\nAll obs have same treatment: {treat_vals}'
        mcf_ps.print_mcf(gen_cfg, txt, summary=summary)


def on_off_support_df(cs_cfg: 'CsCfg',
                      id_name: str,
                      probs_np: NDArray[Any],
                      data_df: pd.DataFrame,
                      return_predicted: bool = False,
                      ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Split DataFrame into retained and deleted part."""
    # _, _, no_of_treat = mcf_data.get_treat_info(mcf_)
    lower, upper = cs_cfg.cut_offs['lower'], cs_cfg.cut_offs['upper']
    obs = len(probs_np)
    off_support = np.empty(obs, dtype=bool)
    for i in range(obs):
        off_upper = np.any(probs_np[i, :] > upper)
        off_lower = np.any(probs_np[i, :] < lower)
        off_support[i] = off_upper or off_lower
    data_on_df = data_df[~off_support].copy()
    data_off_df = data_df[off_support].copy()

    if return_predicted:
        id_df = data_df[id_name]
        # on_support_df = 1 - pd.DataFrame({'on_support': off_support.astype(int)})
        on_support_df = pd.DataFrame({'on_support': (~off_support).astype(int)},
                                     index=data_df.index
                                     )
        col_names = [f"P{i}" for i in range(probs_np.shape[1])]
        # probs_df = pd.DataFrame(probs_np, columns=col_names)
        probs_df = pd.DataFrame(probs_np, columns=col_names, index=data_df.index)
        predicted_df = pd.concat((id_df.reset_index(drop=True),
                                  on_support_df.reset_index(drop=True),
                                  probs_df.reset_index(drop=True)
                                  ),
                                 axis=1
                                 )
    else:
        predicted_df = None

    return data_on_df, data_off_df, predicted_df


def smoothed_propensity_density(data: NDArray[Any], *,
                                zero_tol: float = 1e-10,
                                ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate a binned Gaussian KDE with reflection at probability boundaries."""
    values = np.asarray(data, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError('Propensity scores must be finite.')
    if np.any((values < 0) | (values > 1)):
        raise ValueError('Propensity scores must lie between 0 and 1.')

    n_bins = 1024
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    grid = np.concatenate(([0.0], centers, [1.0]))
    missing = np.full(grid.shape, np.nan, dtype=np.float64)
    if values.size < 2:
        return grid, missing

    sigma = float(np.std(values, ddof=1))
    if sigma <= zero_tol:
        return grid, missing

    q25, q75 = np.percentile(values, [25, 75])
    iqr_scale = float(q75 - q25) / 1.349
    scale = min(sigma, iqr_scale) if iqr_scale > zero_tol else sigma
    bandwidth = max(0.9 * scale * values.size ** (-0.2), 1.0 / n_bins)

    counts, _ = np.histogram(values, bins=edges)
    density = gaussian_filter1d(counts.astype(np.float64), bandwidth * n_bins,
                                mode='reflect')
    density *= n_bins / values.size
    density = np.concatenate((density[:1], density, density[-1:]))
    density /= np.trapezoid(density, grid)

    return grid, density


def plot_support(mcf_: 'ModifiedCausalForest',
                 probs_np: NDArray[Any],
                 d_np: NDArray[Any],
                 titel_data: str | None = None
                 ) -> tuple[list[Path], list[Path]]:
    """Plot propensity-score distributions and export the underlying data."""
    cs_cfg, int_cfg = mcf_.cs_cfg, mcf_.int_cfg
    lower, upper = cs_cfg.cut_offs['lower'], cs_cfg.cut_offs['upper']
    zero_tol = int_cfg.zero_tol
    _, d_values, _ = mcf_data.get_treat_info(mcf_)
    labels = ['Treat ' + str(d) for d in d_values]
    file_list_jpeg: list[Path] = []
    file_list_d_jpeg: list[Path] = []

    for idx_p, ival_p in enumerate(d_values):
        treat_prob = probs_np[:, idx_p]
        titel = f'Probability of treatment {ival_p} in different subsamples'
        f_titel = f'common_support_pr_treat{ival_p}'
        data_hist = [treat_prob[np.isclose(d_np, val, rtol=zero_tol, atol=zero_tol)]
                     for val in d_values]
        hist_bins = np.histogram_bin_edges(treat_prob, bins='auto')
        fit_line_all, bins_all = [], []
        plot_notes: list[str] = []

        for idx, dat in enumerate(data_hist):
            if dat.size == 0:
                bins_all.append(np.empty(0, dtype=np.float64))
                fit_line_all.append(np.empty(0, dtype=np.float64))
                plot_notes.append(f'{labels[idx]}: no observations.')
                continue

            grid, fit_line = smoothed_propensity_density(dat, zero_tol=zero_tol)
            if not np.all(np.isfinite(fit_line)):
                plot_notes.append(
                    f'{labels[idx]}: density smoothing omitted '
                    '(insufficient variation or sample).')
            bins_all.append(grid.copy())
            fit_line_all.append(fit_line.copy())

        if plot_notes:
            txt = f'\nCommon-support plot: probability of treatment {ival_p}\n'
            txt += '\n'.join(plot_notes)
            mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=False)

        file_name_csv = cs_cfg.paths['common_support_fig_pfad_csv'] / f'{f_titel}.csv'
        padded_arrays = padded_array_from_list(data_hist)
        save_df = pd.DataFrame(padded_arrays, columns=labels).fillna(value='NaN')
        mcf_sys.delete_file_if_exists(file_name_csv)
        save_df.to_csv(file_name_csv, index=False)

        file_name_csv_d = cs_cfg.paths['common_support_fig_pfad_csv'] / f'{f_titel}_d.csv'
        np_list = [*fit_line_all, *bins_all]
        label_list = [*['line_' + lab for lab in labels],
                      *['bin_' + lab for lab in labels]]
        padded_arrays = padded_array_from_list(np_list)
        save_df = pd.DataFrame(padded_arrays, columns=label_list).fillna(value='NaN')
        mcf_sys.delete_file_if_exists(file_name_csv_d)
        save_df.to_csv(file_name_csv_d, index=False)

        for density, file_list in ((False, file_list_jpeg), (True, file_list_d_jpeg)):
            files = plot_support_pages(mcf_, data_hist, hist_bins, bins_all, fit_line_all,
                                       labels, titel, f_titel, float(lower[idx_p]),
                                       float(upper[idx_p]), density=density,
                                       titel_data=titel_data)
            file_list.extend(files)

    return file_list_jpeg, file_list_d_jpeg


def plot_support_pages(mcf_: 'ModifiedCausalForest',
                       data_hist: list[NDArray[Any]],
                       hist_bins: NDArray[Any],
                       grids: list[NDArray[Any]],
                       densities: list[NDArray[Any]],
                       labels: list[str],
                       title: str,
                       file_stem: str,
                       lower: float,
                       upper: float,
                       *,
                       density: bool,
                       titel_data: str | None = None
                       ) -> list[Path]:
    """Use an overlay for up to four groups, otherwise at most six panels per page."""
    cs_cfg, int_cfg = mcf_.cs_cfg, mcf_.int_cfg
    panel_mode = len(data_hist) > 4
    groups_per_page = 6 if panel_mode else len(data_hist)
    no_of_pages = (len(data_hist) + groups_per_page - 1) // groups_per_page
    colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:purple')
    hist_values = []
    smooth_ok = []

    for dat, fit_line in zip(data_hist, densities, strict=True):
        heights = np.histogram(dat, bins=hist_bins)[0].astype(np.float64)
        if density and dat.size > 0:
            heights /= dat.size * np.diff(hist_bins)
        hist_values.append(heights)
        smooth_ok.append(fit_line.size > 0 and np.all(np.isfinite(fit_line)))

    y_max = max(float(heights.max()) for heights in hist_values)
    if density:
        for fit_line, valid in zip(densities, smooth_ok, strict=True):
            if valid:
                y_max = max(y_max, float(fit_line.max()))
    y_upper = 1.05 * y_max if y_max > 0 else 1.0

    stem = file_stem + ('_d' if density else '')
    file_list: list[Path] = []
    for page in range(no_of_pages):
        start = page * groups_per_page
        stop = min(start + groups_per_page, len(data_hist))
        n_panels = stop - start if panel_mode else 1
        n_cols = 2 if n_panels > 1 else 1
        n_rows = (n_panels + n_cols - 1) // n_cols
        figsize = (10, 3 * n_rows + 2) if panel_mode else (8, 6)
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, sharex=True, sharey=True,
                                 figsize=figsize
                                 )
        axes_flat = axes.ravel()
        legend_handles = []

        for local_idx, group_idx in enumerate(range(start, stop)):
            axe = axes_flat[local_idx] if panel_mode else axes_flat[0]
            dat = data_hist[group_idx]
            color = 'tab:blue' if panel_mode else colors[group_idx]
            if dat.size > 0:
                artist = axe.stairs(hist_values[group_idx], hist_bins, fill=True,
                                    color=color, alpha=0.5, label=labels[group_idx],
                                    )
                if not panel_mode:
                    legend_handles.append(artist)
                if density and smooth_ok[group_idx]:
                    axe.plot(grids[group_idx], densities[group_idx], '--', color=color)

            if panel_mode:
                axe.set_title(f'{labels[group_idx]} (n={dat.size:,})',
                              fontsize=int_cfg.fontsize,
                              )
                note = ''
                if dat.size == 0:
                    note = 'No observations'
                elif density and not smooth_ok[group_idx]:
                    note = 'Smoothing omitted'
                if note:
                    axe.text(0.5, 0.9, note, transform=axe.transAxes, ha='center',
                             va='top', fontsize=int_cfg.fontsize,
                             )

        for axe in axes_flat[:n_panels]:
            axe.set_xlim(0, 1)
            axe.set_ylim(0, y_upper)
            axe.set_xlabel('Treatment probability')
            axe.set_ylabel('Density' if density else 'Observations')
            axe.tick_params(labelbottom=True)
            axe.axvline(lower, color='black', linewidth=0.9, linestyle='--')
            axe.axvline(upper, color='black', linewidth=0.9, linestyle=':')

        for axe in axes_flat[n_panels:]:
            axe.set_visible(False)

        if panel_mode:
            legend_handles.append(Patch(facecolor='tab:blue', alpha=0.5, label='Histogram'))
        if density and any(smooth_ok[start:stop]):
            legend_handles.append(Line2D([0], [0], color='gray', linestyle='--',
                                         label='Smoothed density'))
        legend_handles.extend([
            Line2D([0], [0], color='black', linestyle='--', label='Lower cutoff'),
            Line2D([0], [0], color='black', linestyle=':', label='Upper cutoff')])
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.04),
                   ncol=min(3, len(legend_handles)), fontsize=int_cfg.fontsize,
                   )
        page_title = title
        if panel_mode:
            page_title += f'\nPage {page + 1} of {no_of_pages}'
        fig.suptitle(page_title, fontsize=int_cfg.fontsize)
        if titel_data is not None:
            fig.text(0.02, 0.01, 'Data: ' + titel_data, ha='left',
                     fontsize=int_cfg.fontsize)
        fig.tight_layout(rect=(0, 0.16, 1, 0.9))

        page_suffix = f'_page{page + 1:02d}' if panel_mode else ''
        image_stem = stem + page_suffix
        file_jpeg = cs_cfg.paths['common_support_fig_pfad_jpeg'] / f'{image_stem}.jpeg'
        file_pdf = cs_cfg.paths['common_support_fig_pfad_pdf'] / f'{image_stem}.pdf'
        mcf_sys.delete_file_if_exists(file_jpeg)
        mcf_sys.delete_file_if_exists(file_pdf)
        fig.savefig(file_jpeg, dpi=int_cfg.dpi, bbox_inches='tight')
        fig.savefig(file_pdf, dpi=int_cfg.dpi, bbox_inches='tight')
        file_list.append(file_jpeg)
        if int_cfg.show_plots:
            plt.show()
        plt.close(fig)

    current_stems = {path.stem for path in file_list}
    for path_key, extension in (('common_support_fig_pfad_jpeg', '.jpeg'),
                                ('common_support_fig_pfad_pdf', '.pdf')):
        directory = cs_cfg.paths[path_key]
        old_overlay = directory / f'{stem}{extension}'
        if old_overlay.stem not in current_stems:
            mcf_sys.delete_file_if_exists(old_overlay)
        for old_page in directory.glob(f'{stem}_page*{extension}'):
            page_number = old_page.stem[len(stem) + len('_page'):]
            if page_number.isdigit() and old_page.stem not in current_stems:
                mcf_sys.delete_file_if_exists(old_page)

    return file_list


def padded_array_from_list(list_of_arrays: list[NDArray[Any]]) -> NDArray[Any]:
    """Create arrays of equal length."""
    max_length = max(len(arr) for arr in list_of_arrays)
    padded_arrays = np.concatenate([np.pad(arr, (0, max_length - len(arr)), mode='constant',
                                           constant_values=np.nan).reshape(-1, 1)
                                    for arr in list_of_arrays
                                    ],
                                   axis=1
                                   )
    return padded_arrays


def get_cut_off_probs(mcf_: 'ModifiedCausalForest',
                      probs_np: NDArray[Any],
                      d_np: NDArray[Any]
                      ) -> dict:
    """Compute the cut-offs for common support for training only."""
    cs_cfg, gen_cfg = mcf_.cs_cfg, mcf_.gen_cfg
    zero_tol = mcf_.int_cfg.zero_tol
    _, d_values, no_of_treat = mcf_data.get_treat_info(mcf_)
    if cs_cfg.type_ == 1:
        q_s = cs_cfg.quantil
        upper_limit = np.empty((no_of_treat, no_of_treat))
        lower_limit = np.empty_like(upper_limit)
        for idx, ival in enumerate(d_values):
            probs = probs_np[np.isclose(d_np, ival, atol=zero_tol, rtol=zero_tol)]
            if q_s == 1:
                upper_limit[idx, :] = np.max(probs, axis=0)
                lower_limit[idx, :] = np.min(probs, axis=0)
            else:
                upper_limit[idx, :] = np.quantile(probs, q_s, axis=0)
                lower_limit[idx, :] = np.quantile(probs, 1 - q_s, axis=0)
            upper_limit[idx, 0] = 1
            lower_limit[idx, 0] = 0
        txt = ''
        if cs_cfg.adjust_limits != 0:
            upper_limit *= 1 + cs_cfg.adjust_limits
            lower_limit *= 1 - cs_cfg.adjust_limits
            lower_limit = np.clip(lower_limit, a_min=0, a_max=1)
            upper_limit = np.clip(upper_limit, a_min=0, a_max=1)
            if gen_cfg.with_output:
                txt += '\n' + '-' * 100 + '\nCommon support bounds adjusted by'
                txt += f' {cs_cfg.adjust_limits:5.2%}-points\n' + '-' * 100
        if gen_cfg.with_output:
            txt += '\nTreatment sample     Treatment probabilities in %'
            txt += '\n--------------------- Upper limits ----------------'
            for idx, ival in enumerate(d_values):
                txt += f'\nD = {ival:9}          '
                for jdx in range(no_of_treat):
                    txt += f'{upper_limit[idx, jdx]:>7.2%}  '
            txt += '\n--------------------- Lower limits ----------------'
            for idx, ival in enumerate(d_values):
                txt += f'\nD = {ival:9}          '
                for jdx in range(no_of_treat):
                    txt += f'{lower_limit[idx, jdx]:>7.2%}  '
            txt += '\n' + 100 * '-'
            txt += '\nFirst treatment is set to 1 and 0 (ignored) due to'
            txt += ' additivity.' + '\n' + 100 * '-'
        upper, lower = np.min(upper_limit, axis=0), np.max(lower_limit, axis=0)
        if gen_cfg.with_output:
            upper_str = [f'{x:>7.2%}' for x in upper]
            lower_str = [f'{x:>7.2%}' for x in lower]
            txt += '\nUpper limits used: ' + ' '.join(upper_str)
            txt += '\nLower limits used: ' + ' '.join(lower_str)
            txt += '\n' + 100 * '-'
            mcf_ps.print_mcf(gen_cfg, txt, summary=True)
    else:
        # Normalize such that probabilities add up to 1
        upper = np.ones(no_of_treat) * (1 - cs_cfg.min_p)
        lower = np.ones(no_of_treat) * cs_cfg.min_p
    cut_offs = {'upper': upper, 'lower': lower}

    return cut_offs


def check_const_single_variable(x_name: list[str],
                                d_name: str | list[str] | tuple[str, ...],
                                data_in_df: pd.DataFrame | list[pd.DataFrame, ...]
                                            | tuple[pd.DataFrame, ...],
                                ) -> None:
    """Check if variables have no variation for any treatment state."""
    if isinstance(data_in_df, pd.DataFrame):
        data_in_df = [data_in_df,]

    threshold = 1e-8
    var_with_problems = []
    if isinstance(d_name, str):
        group_cols = [d_name]
    else:
        group_cols = list(d_name)

    for data_df in data_in_df:
        std_by_group = data_df.groupby(group_cols)[x_name].std()
        almost_zero_std_cols = std_by_group.columns[(std_by_group < threshold).any()].tolist()

        if almost_zero_std_cols:
            var_with_problems.extend(almost_zero_std_cols)

    if var_with_problems:
        names_no_variation = list(set(var_with_problems))
        raise NoVariationInTreatmentState(
            'The following variables have no variation in some of the '
            f'treatment states: {" ".join(names_no_variation)}.'
            '\nPossible solutions: '
            '\n(1) Remove these variables from the specification (not '
            'recommended if this variable is an important confounder).'
            '\n(2) Identify and remove observations that cause the problem.'
            '\nIf you need to prevent raising this exception, set the mcf '
            "keyword 'cs_detect_const_vars_stop' to False."
            )


class NoVariationInTreatmentState(Exception):
    """Raised when there is no variation of any x in any treatment state."""


def predicted_probs_frame(data_df: pd.DataFrame,
                          id_name: str | list[str] | tuple[str, ...],
                          probs_np: NDArray[Any],
                          ) -> pd.DataFrame:
    """Create DataFrame with IDs and predicted treatment probabilities."""
    id_names = [id_name] if isinstance(id_name, str) else list(id_name)
    col_names = [f'P{i}' for i in range(probs_np.shape[1])]
    probs_df = pd.DataFrame(probs_np, columns=col_names, index=data_df.index)

    return pd.concat((data_df[id_names], probs_df), axis=1)
