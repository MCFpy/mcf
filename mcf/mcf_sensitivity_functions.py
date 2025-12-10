"""
Created on Thu Oct 12 16:34:32 2023.

Contains the functions needed for the sensitivity analysis.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score

from mcf import mcf_ate_functions as mcf_ate
from mcf import mcf_data_functions as mcf_data
from mcf.mcf_estimation_generic_functions import moving_avg_mean_var
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf.mcf_init_predict_sens_functions import SensCfg
from mcf import mcf_init_update_helper_functions as mcf_init_update
from mcf import mcf_print_stats_functions as mcf_ps

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def sensitivity_main(mcf_: 'ModifiedCausalForest',
                     train_df: pd.DataFrame | None,
                     predict_df: pd.DataFrame | None = None,
                     results: dict | None = None,
                     sens_cbgate: bool | None = None,
                     sens_bgate: bool | None = None,
                     sens_gate: bool | None = None,
                     sens_iate: bool | None = None,
                     sens_iate_se: bool | None = None,
                     sens_scenarios: list | tuple | None = None,
                     sens_cv_k: int | None = None,
                     sens_replications: int = 2,
                     sens_reference_population: int | float | None = None
                     ) -> dict:
    """
    Compute simulation based sensitivity indicators.

    Parameters
    ----------
    train_df : DataFrame.
        Data with real outcomes, treatments, and covariates. Data will be
        transformed to compute sensitivity indicators.

    predict_df : DataFrame (or None), optinal.
        Prediction data to compute all effects for. This data will not be
        changed in the computation process. Only covariate information is
        used from this dataset. If predict_df is not a DataFrame,
        train_df will be used instead.

    results : dictionary, optional.
        The standard output dictionary from the
        :meth:`~ModifiedCausalForest.predict` method is expected.
        If this dictionary contains estimated IATEs, the same data as in
        the :meth:`~ModifiedCausalForest.predict` method will be used,
        IATEs are computed under the no effect (basic) scenario and these
        IATEs are compared to the IATEs contained in the results dictionary.
        If the dictionary does not contain estimated IATEs, passing it has
        no consequence.

    sens_cbgate : Boolean (or None), optional
        Compute CBGATEs for sensitivity analysis. Default is False.

    sens_bgate : Boolean (or None), optional
        Compute BGATEs for sensitivity analysis. Default is False.

    sens_gate : Boolean (or None), optional
        Compute GATEs for sensitivity analysis. Default is False.

    sens_iate : Boolean (or None), optional
        Compute IATEs for sensitivity analysis. If the results dictionary
        is passed, and it contains IATEs, then the default value is True,
        and False otherwise.

    sens_iate_se : Boolean (or None), optional
        Compute Standard errors of IATEs for sensitivity analysis. Default
        is False.

    sens_scenarios : List or tuple of strings, optional.
        Different scenarios considered. Default is ('basic',).
        'basic' : Use estimated treatment probabilities for simulations.
        No confounding.

    sens_cv_k : Integer (or None), optional
        Data to be used for any cross-validation: Number of folds in
        cross-validation. Default (or None) is 5.

    sens_replications : Integer (or None), optional.
        Number of replications for simulating placebo treatments. Default
        is 2.

    sens_reference_population: integer or float (or None)
        Defines the treatment status of the reference population used by
        the sensitivity analysis. Default is to use the treatment with most
        observed observations.


    Returns
    -------
    results_avg : Dictionary
        Same content as for the
        :meth:`~ModifiedCausalForest.predict` method but (if applicable)
        averaged over replications.

    """
    if (isinstance(results, dict)
        and 'iate_data_df' in results and 'iate_names_dic' in results
            and isinstance(results['iate_data_df'], pd.DataFrame)):
        predict_df = results['iate_data_df']
        iate_df = predict_df[results['iate_names_dic'][0]['names_iate']]
    else:
        iate_df = None
    if not isinstance(predict_df, pd.DataFrame):
        predict_df = train_df.copy()
    mcf_.sens_cfg = SensCfg.from_args(
        mcf_.p_cfg,
        cbgate=sens_cbgate, bgate=sens_bgate, gate=sens_gate,
        iate=sens_iate, iate_se=sens_iate_se,
        scenarios=sens_scenarios,
        cv_k=sens_cv_k,
        replications=sens_replications,
        reference_population=sens_reference_population,
        iate_df=iate_df
        )
    if mcf_.gen_cfg.with_output:
        time_start = time()
    results_avg, plots_iate, txt_ate = sensitivity_analysis(
        mcf_, train_df, predict_df, mcf_.gen_cfg.with_output, iate_df,
        seed=9345467)
    mcf_.report['sens_plots_iate'] = plots_iate
    mcf_.report['sens_txt_ate'] = txt_ate
    if mcf_.gen_cfg.with_output:
        time_difference = [time() - time_start]
        time_string = ['Total time for sensitivity analysis:            ']
        mcf_ps.print_timing(mcf_.gen_cfg, 'Sensitiviy analysis', time_string,
                            time_difference, summary=True)

    return results_avg


def sensitivity_analysis(mcf_: 'ModifiedCausalForest',
                         train_df: pd.DataFrame,
                         predict_df: pd.DataFrame,
                         with_output: bool,
                         iate_df: pd.DataFrame | None = None,
                         seed: int = 12345,
                         ) -> tuple[dict, list[Path], str]:
    """Check sensitivity with respect to possible violations."""
    mcf_.p_cfg.bt_yes = False
    if with_output:
        mcf_ps.print_mcf(mcf_.gen_cfg,
                         '\n' + '=' * 100 + '\nSensitivity analysis',
                         summary=True)
    if mcf_.gen_cfg.d_type == 'continuous':
        raise NotImplementedError('Sensitivity tests not (yet) implemented'
                                  'for continuous treatments.')

    # 1) Estimate treatment propabilities with cross-fitting and delete treated
    (treat_probs_np, train_reduced_df, treat_name, treat_values
     ) = get_treat_probs_del_treat(mcf_, train_df.copy(), with_output)

    rng = np.random.default_rng(seed=seed)
    results_all_dict = {}
    placebo_iate_plot = txt_ate = None
    for scenario in mcf_.sens_cfg.scenarios:
        if with_output:
            if with_output:
                mcf_ps.print_mcf(mcf_.gen_cfg, '\n' + 'x' * 100
                                 + '\nSensitivity: Scenario: 'f'{scenario}',
                                 summary=True)
        results_repl = []
        for repl in range(mcf_.sens_cfg.replications):
            # simulate pseudo.treated
            train_placebo_df, d_probs = simulate_placebo(
                scenario, treat_probs_np, train_reduced_df, treat_name,
                treat_values, rng)
            if with_output:
                d_probs_str = [str(round(p * 100, 2)) + '%' for p in d_probs]
                mcf_ps.print_mcf(mcf_.gen_cfg,
                                 f'\nSensitivity - Scenario: {scenario}, '
                                 f'Replication: {repl+1}, '
                                 'Simulated treatment shares: '
                                 f'{", ".join(d_probs_str)}',
                                 summary=True)
            mcf_repl = deepcopy(mcf_)
            if not mcf_.sens_cfg.gate:
                mcf_repl.var_cfg.z_name_cont = []
                mcf_repl.var_cfg.z_name_ord = []
                mcf_repl.var_cfg.z_name_unord = []
            mcf_repl.p_cfg.bgate = mcf_.sens_cfg.bgate
            mcf_repl.p_cfg.cbgate = mcf_.sens_cfg.cbgate
            mcf_repl.p_cfg.iate = mcf_.sens_cfg.iate
            mcf_repl.p_cfg.iate_se = mcf_.sens_cfg.iate_se
            if iate_df is not None:
                mcf_repl.cs_cfg.type_ = 0
                # No common support check if iate from main estimation available
            mcf_repl.gen_cfg.with_output = False
            mcf_repl.gen_cfg.verbose = False

            # Estimate and predict
            mcf_repl.train(train_placebo_df)
            results_tmp = mcf_repl.predict(predict_df)
            results_repl.append(results_tmp)
        results_all_dict[scenario] = deepcopy(results_repl)
    results_avg_dict = average_sens_results(mcf_repl, results_all_dict)
    if with_output:
        for scenario in mcf_.sens_cfg.scenarios:
            mcf_ps.print_mcf(mcf_.gen_cfg, '\n' + '-' * 100 +
                             f'\nSensitivity - Result scenario: {scenario}',
                             summary=True)
            txt_ate = print_output_ate(mcf_repl, results_avg_dict, scenario)
            if (mcf_.sens_cfg.gate or mcf_.sens_cfg.bgate
                    or mcf_.sens_cfg.cbgate):
                print_output_gate(mcf_repl, results_avg_dict, scenario)
            if mcf_.sens_cfg.iate:
                placebo_iate_plot = print_iate(
                    mcf_repl, results_avg_dict, iate_df, scenario,
                    )
    return results_avg_dict, placebo_iate_plot, txt_ate


def print_iate(mcf_: 'ModifiedCausalForest',
               results_avg_dict: dict,
               iate_df: pd.DataFrame,
               scenario: str,
               ) -> list[Path]:
    """Print output of sensitivity analysis for IATE."""
    post_cfg = mcf_.post_cfg
    results_scen_dict = results_avg_dict[scenario]
    data_df = results_scen_dict['iate_data_df']
    names_iate = results_scen_dict['iate_names_dic'][0]['names_iate']
    txt = f'\nScenario: {scenario}   Descriptive statistics of IATE'
    txt += '\n' + '- ' * 50
    do_plots = scenario == 'basic' and iate_df is not None
    files_jpeg = [] if do_plots else None

    for name_iate in names_iate:
        txt += f'\nVariable: {name_iate:20} '
        txt += iate_stats_str(data_df[name_iate].to_numpy())
        if do_plots:
            file_name_jpeg = plot_iate(mcf_, data_df, iate_df, name_iate)
            files_jpeg.append(file_name_jpeg)
    mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True)
    names_iate_se = results_scen_dict['iate_names_dic'][0]['names_iate_se']
    cluster_lab_np = None
    if names_iate_se is not None:
        t_values_np, t_values_df = cluster_var_t_value(
            data_df, names_iate, names_iate_se,
            mcf_.int_cfg.obs_bigdata
            )
        silhouette_avg_prev = -1
        txt = '\n' + '- ' * 50 + '\nScenario: {scenario} K-Means++ clustering'
        txt += ' of t-values of IATE \n' + '- ' * 50
        for cluster_no in post_cfg.kmeans_no_of_groups:
            cluster_lab_tmp = KMeans(
                n_clusters=cluster_no,
                n_init=post_cfg.kmeans_replications, init='k-means++',
                max_iter=post_cfg.kmeans_max_tries, algorithm='lloyd',
                random_state=42, tol=1e-5, verbose=0, copy_x=True
                ).fit_predict(t_values_np)
            silhouette_avg = silhouette_score(t_values_np, cluster_lab_tmp)
            txt += (f'\nNumber of clusters: {cluster_no}   '
                    f'Average silhouette score: {silhouette_avg: 8.3f}')
            if silhouette_avg > silhouette_avg_prev:
                cluster_lab_np = np.copy(cluster_lab_tmp)
                silhouette_avg_prev = np.copy(silhouette_avg)

        txt += ('\n\nBest value of average silhouette score:'
                f' {silhouette_avg_prev: 8.3f}')
        # Reorder labels for better visible inspection of results
        cl_means = t_values_df.groupby(by=cluster_lab_np).mean(
            numeric_only=True)
        cl_means_np = np.mean(cl_means.to_numpy(), axis=1)
        sort_ind = np.argsort(cl_means_np)
        cl_group = cluster_lab_np.copy()
        for cl_j, cl_old in enumerate(sort_ind):
            cl_group[cluster_lab_np == cl_old] = cl_j
        txt += ('\n' + '- ' * 50 +
                '\nt-values are ordered w.r.t. their mean size')
        cl_values, cl_obs = np.unique(cl_group, return_counts=True)
        txt += '\n' + '- ' * 50 + '\nNumber of observations in the clusters'
        txt += '\n' + '- ' * 50
        for idx, val in enumerate(cl_values):
            txt += f'\nCluster {val:2}: {cl_obs[idx]:6} '
        txt += '\n' + '-' * 100 + '\nt-values\n' + '- ' * 50
        cl_means = t_values_df.groupby(by=cl_group).mean(numeric_only=True)
        txt += '\n' + cl_means.transpose().to_string()
        mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True)
        txt += '\n' + '-' * 100 + '\nAll other variables\n' + '- ' * 50
        cl_means = data_df.groupby(by=cl_group).mean(numeric_only=True)
        txt = cl_means.transpose().to_string() + '\n' + '-' * 100
        pd.set_option('display.max_rows', 1000, 'display.max_columns', 100)
        mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True)
        pd.set_option('display.max_rows', None, 'display.max_columns', None)

    return files_jpeg


def plot_iate(mcf_: 'ModifiedCausalForest',
              data_df: pd.DataFrame,
              iate_df: pd.DataFrame,
              name_iate: list[str] | tuple[str, ...],
              ) -> Path:
    """Plot IATE under no effects against IATE from main estimation."""
    iate_sim = mcf_gp.to_numpy_big_data(data_df[name_iate],
                                        mcf_.int_cfg.obs_bigdata
                                        )
    iate_est = mcf_gp.to_numpy_big_data(iate_df[name_iate],
                                        mcf_.int_cfg.obs_bigdata
                                        )
    if len(iate_sim) != len(iate_est):
        raise RuntimeError(
            f'Number of observations in simulated IATEs ({len(iate_sim)}) are '
            'different from number of observations in estimated IATEs '
            f'({len(iate_est)}). This may happen when the number of '
            'observations used for predictions is too small for the '
            'sensitivity analysis.')
    # Sort values according to estimated values
    sorted_ind = np.argsort(iate_est)
    iate_sim = iate_sim[sorted_ind]
    iate_est = iate_est[sorted_ind]
    k = np.round(mcf_.p_cfg.knn_const * np.sqrt(len(iate_sim)) * 2)
    iate_sim = moving_avg_mean_var(iate_sim, k, False)[0]

    file_name_jpeg = mcf_.gen_cfg.outpath / f'{name_iate}plac_est.jpeg'
    file_name_pdf = mcf_.gen_cfg.outpath / f'{name_iate}plac_est.pdf'
    file_name_csv = mcf_.gen_cfg.outpath / f'{name_iate}plac_est.csv'

    fig, axe = plt.subplots()
    axe.plot(iate_est, iate_est, 'blue', label='Estimated IATE')
    axe.plot(iate_est, iate_sim, 'red', label='Placebo IATE')
    axe.plot(iate_est, np.zeros_like(iate_est), "black", label='_nolegend_',
             linestyle='--')
    axe.set_ylabel('Value of IATEs')
    axe.set_xlabel('Estimated IATEs (sorted)')
    axe.set_title(f'Placebo vs estimated IATEs: {name_iate}')
    axe.legend(loc=mcf_.int_cfg.legend_loc, shadow=True,
               fontsize=mcf_.int_cfg.fontsize)
    mcf_sys.delete_file_if_exists(file_name_jpeg)
    mcf_sys.delete_file_if_exists(file_name_pdf)
    fig.savefig(file_name_jpeg, dpi=mcf_.int_cfg.dpi)
    fig.savefig(file_name_pdf, dpi=mcf_.int_cfg.dpi)
    plt.show()
    plt.close()
    data_to_save = np.concatenate((iate_sim.reshape(-1, 1),
                                   iate_est.reshape(-1, 1)), axis=1)
    datasave = pd.DataFrame(data=data_to_save,
                            columns=(name_iate+'_sim', name_iate+'_est'))
    mcf_sys.delete_file_if_exists(file_name_csv)
    datasave.to_csv(file_name_csv, index=False)

    return file_name_jpeg


def cluster_var_t_value(data_df: pd.DataFrame,
                        names_iate: list[str] | tuple[str, ...],
                        names_iate_se: list[str] | tuple[str, ...],
                        obs_bigdata: bool,
                        ) -> tuple[NDArray, pd.DataFrame]:
    """Compute t-values to cluster on."""
    est_np = mcf_gp.to_numpy_big_data(data_df[names_iate], obs_bigdata)
    se_np = mcf_gp.to_numpy_big_data(data_df[names_iate_se], obs_bigdata)
    t_val_np = est_np / se_np
    t_val_names = [name + '_t_val' for name in names_iate]
    t_val_df = pd.DataFrame(data=t_val_np, columns=t_val_names)

    return t_val_np, t_val_df


def iate_stats_str(iate_np: NDArray) -> str:
    """Compute mean, median, std., and return string ready to print."""
    mean = np.mean(iate_np)
    median = np.median(iate_np)
    std = np.std(iate_np)
    txt = f' Mean: {mean:10.4f}   Median: {median:10.4f}   Std.: {std:10.4f}'

    return txt


def print_output_gate(mcf_: 'ModifiedCausalForest',
                      results_avg_dict: dict,
                      scenario: str,
                      ) -> None:
    """Print output of sensitivity analysis for gate and bgate."""
    results_scen_dict = results_avg_dict[scenario]
    continuous = mcf_.gen_cfg.d_type == 'continuous'
    d_values = (mcf_.ct_cfg.d_values_dr_np
                if continuous else mcf_.gen_cfg.d_values)
    effect_list = results_scen_dict['ate_effect_list']
    gates = ('gate', 'bgate', 'cbgate')
    gates_se = ('gate_se', 'bgate_se', 'cbgate_se')
    txt = ''
    z_values_dic = results_scen_dict['gate_names_values']
    z_names_order_list = results_scen_dict['gate_names_values']['z_names_list']
    for gidx, gate in enumerate(gates):
        if results_scen_dict[gate] is not None:
            txt = '\n' + '- ' * 50 + '\nGATEs (only if p-value < 10%)'
            gate_se = gates_se[gidx]
            # Iterate over different gate variables (list)
            for zdx, gate_np in enumerate(results_scen_dict[gate]):
                gate_se_np = results_scen_dict[gate_se][zdx]
                z_name = z_names_order_list[zdx]
                # Iterate over different values of Z
                for idx in range(gate_np.shape[0]):
                    z_value = z_values_dic[z_name][idx]
                    gate_z = gate_np[idx, :, :, :]
                    gate_se_z = gate_se_np[idx, :, :, :]
                    for o_idx, out_name in enumerate(mcf_.var_cfg.y_name):
                        if idx == 0:
                            txt += '\n' + '- ' * 50
                        if len(mcf_.var_cfg.y_name) > 1 or idx == 0:
                            txt += f'\nScenario: {scenario} {gate.casefold()} '
                            txt += '(sensitivity) Outcome variable: '
                            txt += f'{out_name}'
                        else:
                            txt += '\n'
                        for a_idx in range(gate_z.shape[1]):
                            if gate_z.shape[1] > 1 or idx == 0:
                                if a_idx == 0:
                                    txt += '   Reference population: All  '
                                else:
                                    txt += '   Reference population: Treatment'
                                    txt += f' group {d_values[a_idx-1]}  '
                            txt += f'Variable: {z_name}'
                            txt += f'  Value: {z_value}'
                            est = gate_z[o_idx, a_idx]
                            stderr = gate_se_z[o_idx, a_idx]
                            t_val, p_val = get_t_val_pal(est, stderr)
                            txt += mcf_ps.print_effect(
                                est, stderr, t_val, p_val, effect_list,
                                continuous=continuous,
                                print_first_line=idx == 0,
                                print_last_line=False,
                                small_p_val_only=True)
                            # if idx == 0:
                            #     txt += '\n'
                            if continuous and a_idx == 0:
                                mcf_ate.dose_response_figure(
                                    out_name, mcf_.var_cfg.d_name[0], est,
                                    stderr, d_values[1:], mcf_.int_cfg,
                                    mcf_.p_cfg,
                                    with_output=mcf_.gen_cfg.with_output,
                                    )
            mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True)


def print_output_ate(mcf_: 'ModifiedCausalForest',
                     results_avg_dict: dict,
                     scenario: str,
                     ) -> str:
    """Print output of sensitivity analysis for ate."""
    results_scen_dict = results_avg_dict[scenario]
    txt = '\n' + '- ' * 50 + '\nATEs (only if p-value < 10%)'
    continuous = mcf_.gen_cfg.d_type == 'continuous'
    ate, ate_se = results_scen_dict['ate'], results_scen_dict['ate_se']
    effect_list = results_scen_dict['ate_effect_list']
    d_values = (mcf_.ct_cfg.d_values_dr_np
                if continuous else mcf_.gen_cfg.d_values)
    for o_idx, out_name in enumerate(mcf_.var_cfg.y_name):
        txt += '\n' + '- ' * 50 + f'\nScenario: {scenario} ATE (sensitivity) '
        txt += f'Outcome variable: {out_name}'
        for a_idx in range(ate.shape[1]):
            if a_idx == 0:
                txt += '   Reference population: All'
            else:
                txt += ('   Reference population: Treatment group '
                        f'{d_values[a_idx-1]}')
            txt += '\n' + '- ' * 50
            est, stderr = ate[o_idx, a_idx], ate_se[o_idx, a_idx]
            t_val, p_val = get_t_val_pal(est, stderr)
            txt += mcf_ps.print_effect(est, stderr, t_val, p_val, effect_list,
                                       continuous=continuous,
                                       small_p_val_only=True)
            if continuous and a_idx == 0:
                mcf_ate.dose_response_figure(
                    out_name, mcf_.var_cfg.d_name[0], est, stderr,
                    d_values[1:], mcf_.int_cfg, mcf_.p_cfg,
                    with_output=mcf_.gen_cfg.with_output
                    )
    mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True)

    return txt


def get_t_val_pal(est: NDArray[Any],
                  stderr: NDArray[Any],
                  ) -> tuple[NDArray[Any], NDArray[Any]]:
    """Get t-value and p-value from estimate and standard error."""
    t_val = np.abs(est / stderr)
    p_val = t.sf(t_val, 1000000) * 2

    return t_val, p_val


def average_sens_results(mcf_: 'ModifiedCausalForest',
                         results_all_dict: dict,
                         ) -> dict:
    """Average results over replications."""
    id_name = mcf_.var_cfg.id_name[0]
    iate = mcf_.p_cfg.iate
    results_dict_avg = {}
    avg_inside_list = ('cbgate', 'cbgate_diff', 'cbgate_se', 'cbgate_diff_se',
                       'bgate', 'bgate_diff', 'bgate_se', 'bgate_diff_se',
                       'gate', 'gate_diff', 'gate_se', 'gate_diff_se',)
    iate_df = 'iate_data_df'
    avg_numpy = ('ate', 'ate_se',)
    # Loops over scenarios
    for scenario in results_all_dict:
        # Loops over list with each replication for scenario to average results
        no_of_replications = len(results_all_dict[scenario])
        iate_df_list = []
        for idx, result_repl_org in enumerate(results_all_dict[scenario]):
            final = idx == len(results_all_dict[scenario])-1
            result_repl = deepcopy(result_repl_org)
            if iate:
                iate_df_list.append(result_repl[iate_df])
            if idx == 0:
                results_dict_avg[scenario] = result_repl
            else:
                # Loops over statistics (in dict) for each replication
                for statistic in result_repl:
                    if statistic in avg_inside_list and (
                            result_repl[statistic] is not None):
                        for list_idx, elem in enumerate(
                                result_repl[statistic]):
                            results_dict_avg[scenario][
                                statistic][list_idx] += elem
                            if final:
                                results_dict_avg[scenario][
                                    statistic][list_idx] /= no_of_replications
                    elif statistic in avg_numpy:   # Average numpy array
                        results_dict_avg[scenario][statistic
                                                   ] += result_repl[statistic]
                        if final:
                            results_dict_avg[scenario][
                                statistic] /= no_of_replications
        if iate:
            results_dict_avg[scenario][iate_df] = average_iate(iate_df_list,
                                                               id_name
                                                               )
    return results_dict_avg


def average_iate(iate_df_list: list[pd.DataFrame],
                 id_name: str,
                 ) -> pd.DataFrame:
    """Average IATE in the subset of those in the same common support."""
    id_list = []
    # Extract iates & reduce to id_values that are contained in id_list
    # find identifiers that always available
    for iate_dat_df in iate_df_list:
        id_list.append(iate_dat_df[id_name])
    id_df = id_list[0]  # Initialize result_df with the first DataFrame
    # Iterate over the remaining DataFrames in df_list
    for df_id_list in id_list[1:]:
        # Perform an inner merge with the previous result_df & the current DF
        id_df = pd.merge(id_df, df_id_list, on=id_name, how='inner')
        # The result_df will contain the intersection of values across all DF
    all_iates = pd.merge(id_df, iate_df_list[0], on=id_name, how='inner')
    # Than average these iates
    for iate in iate_df_list[1:]:
        all_iates = all_iates + pd.merge(id_df, iate, on=id_name, how='inner')
    all_iates = all_iates / len(iate_df_list)

    return all_iates


def simulate_placebo(
        scenario: str,
        treatment_probs_np: NDArray[Any],
        train_df: pd.DataFrame,
        treat_name: str,
        treat_values: list[int | float] | NDArray[np.floating | np.integer],
        rng: np.random.Generator
        ) -> tuple[pd.DataFrame, NDArray[Any]]:
    """Simulate placebo outcomes and return data."""
    d_placebo = np.empty((len(train_df), 1))
    if scenario == 'basic':
        # No unobserved confounding
        for idx, prob in enumerate(treatment_probs_np):
            d_placebo[idx, :] = rng.choice(treat_values, p=prob)
        _, d_count = np.unique(d_placebo, return_counts=True)
        d_prob = d_count / len(d_placebo)
    else:
        # Future:Put estimated prob &confounder in exp-function for confounding
        raise ValueError('Sensitivity: Attempted scenario does not exist.')
    train_placebo_df = train_df.copy()
    train_placebo_df[treat_name] = d_placebo

    return train_placebo_df, d_prob


def get_treat_probs_del_treat(
        mcf_: 'ModifiedCausalForest',
        data_df: pd.DataFrame,
        with_output: bool
        ) -> tuple[NDArray[Any], NDArray[Any], str, NDArray[Any]]:
    """Estimate treatment probabilities in training data."""
    mcf_copy = deepcopy(mcf_)
    mcf_copy.gen_cfg.with_output = False
    data_df, _ = mcf_data.data_frame_vars_lower(data_df)
    data_df = mcf_data.check_recode_treat_variable(mcf_copy, data_df)

    # Initialise again with data information (only relevant for simulation,
    # not the other steps outside this function.)
    mcf_init_update.var_update_train(mcf_copy, data_df)
    mcf_init_update.gen_update_train(mcf_copy, data_df)
    mcf_init_update.ct_update_train(mcf_copy, data_df)
    mcf_init_update.cs_update_train(mcf_copy)
    mcf_init_update.cf_update_train(mcf_copy, data_df)
    mcf_init_update.int_update_train(mcf_copy)
    mcf_init_update.p_update_train(mcf_copy)

    data_df = mcf_data.create_xz_variables(mcf_copy, data_df, train=True)

    # Clean data and remove missings and unncessary variables
    if mcf_copy.dc_cfg.clean_data:
        data_df, _ = mcf_data.clean_data(mcf_copy, data_df, train=True)
    if mcf_copy.dc_cfg.screen_covariates:   # Only training
        (mcf_copy.gen_cfg, mcf_copy.var_cfg, mcf_copy.var_x_type,
         mcf_copy.var_x_values, _
         ) = mcf_data.screen_adjust_variables(mcf_copy, data_df)
    gen_cfg, var_x_type = mcf_copy.gen_cfg, mcf_copy.var_x_type
    int_cfg = mcf_copy.int_cfg
    data_train_dic, sens_cfg = mcf_copy.data_train_dict, mcf_copy.sens_cfg

    # Estimate out-of-sample probabilities by k-fold cross-validation
    if with_output:
        mcf_ps.print_mcf(
            gen_cfg, '\n' + '-' * 100 + '\nSensitivity: Computing '
                     f'treatment probabilities (by {sens_cfg.cv_k}'
                     '-fold cross-validation)', summary=True)
    d_name, _, no_of_treat = mcf_data.get_treat_info(mcf_copy)
    x_name, x_type = mcf_gp.get_key_values_in_list(var_x_type)
    names_unordered = [x_name[j] for j, val in enumerate(x_type) if val > 0]
    x_df, obs = mcf_data.get_x_data(data_df, x_name)
    x_np = mcf_gp.to_numpy_big_data(x_df, int_cfg.obs_bigdata)
    d_np = mcf_gp.to_numpy_big_data(data_df[d_name],
                                    int_cfg.obs_bigdata
                                    ).ravel()
    if names_unordered:  # List is not empty
        x_df, dummy_names = mcf_data.dummies_for_unord(
            x_df, names_unordered, data_train_dict=data_train_dic)
    max_workers = 1 if int_cfg.replication else gen_cfg.mp_parallel
    classif = RandomForestClassifier(
        n_estimators=mcf_copy.cf_cfg.boot, max_features='sqrt',
        bootstrap=True, oob_score=False, n_jobs=max_workers,
        random_state=42, verbose=False, min_samples_split=5)
    index = np.arange(obs)       # indices
    rng = np.random.default_rng(seed=9324561)
    rng.shuffle(index)
    index_folds = np.array_split(index, sens_cfg.cv_k)
    pred_prob_np = np.empty((len(index), no_of_treat))
    forests = []
    for fold_pred in range(sens_cfg.cv_k):
        fold_train = [x for idx, x in enumerate(index_folds)
                      if idx != fold_pred]
        index_train = np.hstack(fold_train)
        index_pred = index_folds[fold_pred]
        x_pred, x_train = x_np[index_pred], x_np[index_train]
        d_train = d_np[index_train]
        classif.fit(x_train, d_train)
        forests.append(deepcopy(classif))
        pred_prob_np[index_pred, :] = classif.predict_proba(x_pred)

    # Keep only observations from largest treatment group
    if mcf_copy.sens_cfg.reference_population is None:
        value_counts = data_df[d_name].value_counts()
        group_value = value_counts.idxmax()[0]
    else:
        group_value = mcf_copy.sens_cfg.reference_population
    keep = (d_np == group_value).ravel()
    data_reduced_df = data_df[keep]
    treatment_prob_np = pred_prob_np[keep, :]

    # Print basic stats
    if with_output:
        obs_keep = np.sum(keep)
        txt = '\n' + '-' * 100
        if mcf_copy.sens_cfg.reference_population is None:
            txt += '\nLargest treatment group (training data): '
        else:
            txt += '\nSelected treatment group (training data): '
        txt += (f'Value {group_value}, # of observations: {obs_keep} '
                '\n' + f'Share of observations kept: {obs_keep/obs:4.2%}'
                '\n' + 'Sensivity analysis continues with observations kept'
                ' only' + '\n' + '-' * 100)
        mcf_ps.print_mcf(gen_cfg, txt, summary=True)

    return treatment_prob_np, data_reduced_df, d_name, np.unique(d_np)
