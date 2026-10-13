"""
Contains the functions for the post estimation analysis.

Created on Thu Jun 29 17:28:31 2023.


# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import deepcopy
from pathlib import Path
from textwrap import fill
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mcf import mcf_estimation_generic as mcf_est_g
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats as mcf_ps

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest


def post_estimation_iate(mcf_: 'ModifiedCausalForest',
                         results: dict[str, Any],
                         iv: bool = False
                         ) -> list[Path]:
    """Analyse IATEs and export correlations, sorted effects, and densities."""
    ct_cfg, int_cfg = mcf_.ct_cfg, mcf_.int_cfg
    gen_cfg, p_cfg, post_cfg = mcf_.gen_cfg, mcf_.p_cfg, mcf_.post_cfg
    var_cfg, var_x_type = mcf_.var_cfg, mcf_.var_x_type
    zero_tol = int_cfg.zero_tol
    figure_list: list[Path] = []

    txt = '\n' + '=' * 100 + '\nPost estimation analysis'
    txt_summary = txt
    if gen_cfg.d_type == 'continuous':
        d_values = ct_cfg.d_values_dr_np
    elif mcf_.gen_tv_cfg.yes:
        d_values = mcf_.gen_tv_cfg.d_all_values_one_label
    else:
        d_values = gen_cfg.d_values

    number_of_runs = 3 if iv else 1
    for iv_ind in range(number_of_runs):
        match iv_ind:
            case 0:  # Always run this
                iate_pot_all_name = results['iate_names_dic']
                ate_all, ate_all_se = results['ate'], results['ate_se']
                effect_list = results['ate_effect_list']
                data_df = results['iate_data_df']
            case 1:  # 1st stage of IV estimation
                iate_pot_all_name = results['iate_1st_names_dic']
                ate_all, ate_all_se = results['ate_1st'], results['ate_1st_se']
                effect_list = results['ate 1st_effect_list']
                data_df = results['iate_1st_pred_df']
            case 2:  # Reduced form of IV estimation
                iate_pot_all_name = results['iate_redf_names_dic']
                ate_all = results['ate_redf']
                ate_all_se = results['ate_redf_se']
                effect_list = results['ate redf_effect_list']
                data_df = results['iate_redf_pred_df']
            case _:
                raise ValueError(f'Invalid IV stage: {iv_ind}.')

        if post_cfg.relative_to_first_group_only or gen_cfg.d_type == 'continuous':
            txt += f'\n Only effects relative to treatment {d_values[0]} are investigated.'
            iate_pot_name = iate_pot_all_name[1]
            selected = [idx for idx, pair in enumerate(effect_list)
                        if np.isclose(pair[1], d_values[0], atol=zero_tol, rtol=zero_tol)]
            ate = ate_all[:, 0, selected]
            ate_se = ate_all_se[:, 0, selected]
        else:
            iate_pot_name = iate_pot_all_name[0]
            ate, ate_se = ate_all[:, 0, :], ate_all_se[:, 0, :]
        ate, ate_se = ate.reshape(-1), ate_se.reshape(-1)
        txt += '\n'
        if mcf_.lc_cfg.uncenter_po:
            y_pot = data_df[iate_pot_name['names_y_pot_uncenter']]
        else:
            y_pot = data_df[iate_pot_name['names_y_pot']]
        iate = data_df[iate_pot_name['names_iate']]
        x_name = delete_x_with_catv(var_x_type.keys())
        x_dat_df = data_df[x_name]

        # Change _primes back to original values
        if mcf_.data_train_dict is not None and len(mcf_.data_train_dict) > 0:
            x_dat_df = mcf_ps.change_name_value_df(x_dat_df,
                                                   mcf_.data_train_dict['prime_old_name_dict'],
                                                   mcf_.data_train_dict['prime_values_dict'],
                                                   mcf_.data_train_dict['unique_values_dict']
                                                   )
        cint = norm.ppf(p_cfg.ci_level + 0.5 * (1 - p_cfg.ci_level))
        if post_cfg.bin_corr_yes:
            txt += '\n' + '=' * 100 + '\nCorrelations of effects with ... in %'
            txt += '\n' + '-' * 100
        ci_level = f'{p_cfg.ci_level:.0%}'
        iterator = range(2) if p_cfg.iate_m_ate else range(1)
        no_of_names = len(iate_pot_name['names_iate'])
        outcome_names = var_cfg.d_name if iv_ind == 1 else var_cfg.y_name
        if post_cfg.relative_to_first_group_only or gen_cfg.d_type == 'continuous':
            effect_pairs = [pair for pair in effect_list
                            if np.isclose(pair[1], d_values[0], atol=zero_tol, rtol=zero_tol)]
        else:
            effect_pairs = list(effect_list)

        treatment_labels = (mcf_ps.treatment_label_map(mcf_)
                            if gen_cfg.d_type == 'discrete' and iv_ind == 0 else {})
        comparison_labels: list[str] = []
        for left, right in effect_pairs:
            if iv_ind > 0:
                comparison = f'Instrument {left} vs {right}'
            elif gen_cfg.d_type == 'continuous':
                comparison = f'Dose {left} vs {right}'
            else:
                comparison = f'{treatment_labels[str(left)]} vs {treatment_labels[str(right)]}'
            comparison_labels.append(comparison)

        plot_labels = [(outcome, comparison) for outcome in outcome_names
                       for comparison in comparison_labels]
        if len(plot_labels) != no_of_names:
            raise ValueError('IATE names do not match the outcome and comparison metadata.')
        eva_points = eva_points_fct(no_of_names, len(outcome_names))
        if iv:
            stage_label = ('LIATE', 'IATE (1st stage)', 'IATE (reduced form)')[iv_ind]
        else:
            stage_label = 'IATE'
        no_of_analysed = sum(gen_cfg.d_type == 'discrete' or idx in eva_points
                            for idx in range(no_of_names))
        txt_summary += (f'\n\n{stage_label}'
                        f'\nPrediction observations: {len(data_df):,}'
                        f'\nOutcomes: {len(outcome_names):,}'
                        f'\nSelected outcome-comparison combinations: {no_of_names:,}'
                        f'\nCombinations in sorted-effect/density output: {no_of_analysed:,}'
                        )
        txt += f'\n{stage_label}\n' + '-' * 100

        for idx in range(no_of_names):
            outcome_label, comparison_label = plot_labels[idx]
            for imate in iterator:
                if imate == 0:
                    name_eff = 'names_iate'
                    ate_t, ate_se_t = ate[idx].copy(), ate_se[idx].copy()
                else:
                    name_eff, ate_t = 'names_iate_mate', 0
                name_iate_t = iate_pot_name[name_eff][idx]
                if p_cfg.iate_se:
                    name_se = name_eff + '_se'
                    name_iate_se_t = iate_pot_name[name_se][idx]
                else:
                    name_iate_se_t = None
                titel = 'Sorted ' + name_iate_t

                # Add heatmaps or similar w.r.t. to single variables
                if post_cfg.plots and imate == 0:
                    relational_graphs(mcf_, data_df, name_iate_t, x_dat_df)

                # Add correlation analysis of IATEs
                if gen_cfg.d_type == 'discrete' or idx in eva_points:
                    if post_cfg.bin_corr_yes and imate == 0:
                        txt += f'\nEffect: {name_iate_t}' + '\n' + '- ' * 50
                        if gen_cfg.d_type == 'discrete':
                            corr = iate.corrwith(data_df[name_iate_t])
                            for jdx in corr.keys():
                                txt += f'\n{jdx:<40} {corr[jdx]:>8.2%}'
                            txt += '\n' + '- ' * 50
                            corr = y_pot.corrwith(data_df[name_iate_t])
                            for jdx in corr.keys():
                                txt += f'\n{jdx:<40} {corr[jdx]:>8.2%}'
                            txt += '\n' + '- ' * 50

                        corr = x_dat_df.corrwith(data_df[name_iate_t])
                        corr = corr.sort_values()
                        for jdx in corr.keys():
                            if np.abs(corr[jdx].item()) > post_cfg.bin_corr_threshold:
                                txt += f'\n{jdx:<40} {corr[jdx]:>8.2%}'
                        txt += '\n' + '- ' * 50
                    iate_temp = mcf_gp.to_numpy_big_data(data_df[name_iate_t],
                                                            int_cfg.obs_bigdata
                                                            )
                    if p_cfg.iate_se:
                        iate_se_temp = mcf_gp.to_numpy_big_data(data_df[name_iate_se_t],
                                                                int_cfg.obs_bigdata
                                                                )
                    else:
                        iate_se_temp = None
                    sorted_ind = np.argsort(iate_temp)
                    iate_temp = iate_temp[sorted_ind]
                    if p_cfg.iate_se:
                        iate_se_temp = iate_se_temp[sorted_ind]
                    x_values = np.arange(len(iate_temp)) + 1
                    x_values = np.around(x_values / x_values[-1] * 100, decimals=1)
                    k = max(1, round(p_cfg.knn_const * np.sqrt(len(iate_temp)) * 2))
                    iate_temp = mcf_est_g.moving_avg_mean_var(iate_temp, k, False)[0]
                    if p_cfg.iate_se:
                        iate_se_temp = mcf_est_g.moving_avg_mean_var(iate_se_temp, k, False)[0]
                    titel_f = titel.replace(' ', '')
                    (file_name_jpeg, file_name_pdf, file_name_csv
                     ) = file_names_helper(iv_ind,
                                           p_cfg.paths['ate_iate_fig_pfad_jpeg'],
                                           p_cfg.paths['ate_iate_fig_pfad_pdf'],
                                           p_cfg.paths['ate_iate_fig_pfad_csv'],
                                           titel_f,
                                           )
                    if p_cfg.iate_se:
                        upper = iate_temp + iate_se_temp * cint
                        lower = iate_temp - iate_se_temp * cint
                    ate_t = ate_t * np.ones(len(iate_temp))
                    if imate == 0:
                        ate_upper = ate_t + ate_se_t * cint * np.ones(len(iate_temp))
                        ate_lower = ate_t - ate_se_t * cint * np.ones(len(iate_temp))
                    line_ate, line_iate = '_-r', '-b'
                    fig, axe = plt.subplots()

                    ivt = bool(iv)
                    match imate:
                        case 0:
                            label_y = 'Effect'
                            match (ivt, iv_ind):
                                case (True, 0):
                                    label_t, label_r = 'LIATE', 'LATE'
                                case (True, 1):
                                    label_t = 'IATE (1st stage)'
                                    label_r = 'ATE (1st stage)'
                                case (True, 2):
                                    label_t = 'IATE (reduced form)'
                                    label_r = 'ATE (reduced form)'
                                case _:
                                    label_t, label_r = 'IATE', 'ATE'
                        case _:
                            label_y = 'Effect - average'
                            label_r = '_nolegend_'
                            match (ivt, iv_ind):
                                case (True, 0):
                                    label_t = 'LIATE-LATE'
                                case (True, 1):
                                    label_t = 'IATE-ATE (1st)'
                                case (True, 2):
                                    label_t = 'IATE-ATE (red.f.)'
                                case _:
                                    label_t = 'IATE-ATE'

                    label_ci_effect = f'{label_t}: smoothed {ci_level} CI limits'
                    label_ci_reference = f'{label_r}: {ci_level} CI'
                    axe.plot(x_values, iate_temp, line_iate, label=label_t)
                    axe.set_ylabel(label_y)
                    axe.plot(x_values, ate_t, line_ate, label=label_r)
                    if imate == 0:
                        axe.fill_between(x_values, ate_upper, ate_lower,
                                         alpha=0.3, color='r', label=label_ci_reference
                                         )
                    title_parts = (f'Sorted {label_t}: {comparison_label}',
                                   f'Outcome: {outcome_label}')
                    display_title = '\n'.join(fill(part, width=70) for part in title_parts)
                    axe.set_title(display_title, fontsize=int_cfg.fontsize)
                    axe.set_xlabel(f'Percentile of sorted {label_t} estimates')
                    if p_cfg.iate_se:
                        axe.fill_between(x_values, upper, lower, alpha=0.3, color='b',
                                         label=label_ci_effect
                                         )
                    axe.legend(loc=int_cfg.legend_loc, shadow=True, fontsize=int_cfg.fontsize)
                    fig.tight_layout()
                    if post_cfg.plots:
                        mcf_sys.delete_file_if_exists(file_name_jpeg)
                        mcf_sys.delete_file_if_exists(file_name_pdf)
                        fig.savefig(file_name_jpeg, dpi=int_cfg.dpi, bbox_inches='tight')
                        fig.savefig(file_name_pdf, dpi=int_cfg.dpi, bbox_inches='tight')
                    if post_cfg.plots and int_cfg.show_plots:
                        plt.show()
                    plt.close(fig)
                    plot_values = {}
                    if p_cfg.iate_se:
                        plot_values['upper'] = upper
                    plot_values['effects'] = iate_temp
                    if p_cfg.iate_se:
                        plot_values['lower'] = lower
                    reference_name = 'ate' if imate == 0 else 'zero_reference'
                    plot_values[reference_name] = ate_t
                    if imate == 0:
                        plot_values['ate_l'] = ate_lower
                        plot_values['ate_u'] = ate_upper
                    plot_values['percentile'] = x_values

                    datasave = pd.DataFrame({name: np.asarray(vals).reshape(-1)
                                            for name, vals in plot_values.items()})
                    mcf_sys.delete_file_if_exists(file_name_csv)
                    datasave.to_csv(file_name_csv, index=False)
                    if post_cfg.plots and imate == iterator[-1] and p_cfg.iate_se:
                        figure_list.append(file_name_jpeg)

                    # Density plots
                    if imate == 0:
                        titel = 'Density ' + iate_pot_name['names_iate'][idx]
                        titel_f = titel.replace(' ', '')
                        match iv_ind:
                            case 0:
                                suf_img, suf_pdf, suf_csv = '', '', 'plotdat'
                            case 1:
                                suf_img, suf_pdf = '1st', '1st'
                                suf_csv = 'plotdat1st'
                            case 2:
                                suf_img, suf_pdf = 'redf', 'redf'
                                suf_csv = 'plotdatredf'
                            case _:
                                raise ValueError(f'invalid iv_ind={iv_ind}; expected 0, 1, or 2')
                        base_jpeg = p_cfg.paths['ate_iate_fig_pfad_jpeg']
                        base_pdf = p_cfg.paths['ate_iate_fig_pfad_pdf']
                        base_csv = p_cfg.paths['ate_iate_fig_pfad_csv']
                        file_name_jpeg = base_jpeg / f'{titel_f}{suf_img}.jpeg'
                        file_name_pdf = base_pdf / f'{titel_f}{suf_pdf}.pdf'
                        file_name_csv = base_csv / f'{titel_f}{suf_csv}.csv'

                        iate_temp = mcf_gp.to_numpy_big_data(data_df[name_iate_t],
                                                             int_cfg.obs_bigdata
                                                             )
                        bandwidth = mcf_est_g.bandwidth_silverman(iate_temp, 1,
                                                                  zero_tol=int_cfg.zero_tol
                                                                  )
                        dist = np.abs(iate_temp.max() - iate_temp.min())
                        low_b = iate_temp.min() - 0.1 * dist
                        up_b = iate_temp.max() + 0.1 * dist
                        grid = np.linspace(low_b, up_b, 1000)
                        density = mcf_est_g.kernel_density(iate_temp, grid, 1, bandwidth)
                        fig, axe = plt.subplots()
                        title_parts = (f'Density of {label_t}: {comparison_label}',
                                       f'Outcome: {outcome_label}')
                        display_title = '\n'.join(fill(part, width=70) for part in title_parts)
                        axe.set_title(display_title, fontsize=int_cfg.fontsize)
                        axe.set_ylabel('Estimated density')
                        axe.set_xlabel(label_t)
                        axe.plot(grid, density, '-b')
                        axe.fill_between(grid, density, alpha=0.3, color='b')
                        fig.tight_layout()
                        if post_cfg.plots:
                            mcf_sys.delete_file_if_exists(file_name_jpeg)
                            mcf_sys.delete_file_if_exists(file_name_pdf)
                            fig.savefig(file_name_jpeg, dpi=int_cfg.dpi, bbox_inches='tight')
                            fig.savefig(file_name_pdf, dpi=int_cfg.dpi, bbox_inches='tight')
                            if int_cfg.show_plots:
                                plt.show()
                        plt.close(fig)
                        density = density.reshape(-1, 1)
                        cols = ['grid', 'density']
                        grid = grid.reshape(-1, 1)
                        density = density.reshape(-1, 1)
                        effects_et_al = np.concatenate((grid, density), axis=1)
                        datasave = pd.DataFrame(data=effects_et_al,
                                                columns=cols)
                        mcf_sys.delete_file_if_exists(file_name_csv)
                        datasave.to_csv(file_name_csv, index=False)
                        if post_cfg.plots and not p_cfg.iate_se:
                            figure_list.append(file_name_jpeg)

        if gen_cfg.d_type == 'continuous' and not iv:
            no_of_y = len(var_cfg.y_name)
            surface_names = iate_pot_all_name[0]
            no_of_iate_y = len(effect_list)
            if len(surface_names['names_iate']) != no_of_y * no_of_iate_y:
                raise ValueError('IATE names do not match the continuous-treatment comparisons.')
            surface_doses = np.asarray([pair[0] for pair in effect_list])
            index_0 = range(no_of_iate_y)
            for idx_y in range(no_of_y):
                for imate in iterator:
                    if imate == 0:
                        name_eff, iate_label = 'names_iate', 'IATE'
                    else:
                        name_eff, iate_label = 'names_iate_mate', 'IATE-ATE'

                    # Preserve the existing filenames.
                    titel = ('Dose response relative to 0 ' + iate_label + ' '
                             + var_cfg.y_name[idx_y])
                    index_t = [i + no_of_iate_y * idx_y for i in index_0]
                    name_iate_t = [surface_names[name_eff][idx] for idx in index_t]
                    iate_temp = mcf_gp.to_numpy_big_data(data_df[name_iate_t], int_cfg.obs_bigdata)
                    indices_sort = np.argsort(np.mean(iate_temp, axis=1))
                    iate_temp = iate_temp[indices_sort]
                    z_plt = np.transpose(iate_temp)
                    fig, axe = plt.subplots(subplot_kw={'projection': '3d'})
                    x_plt, y_plt = np.meshgrid(np.arange(z_plt.shape[1]) + 1, surface_doses)
                    surf = axe.plot_surface(x_plt, y_plt, z_plt, cmap='coolwarm_r', linewidth=0,
                                            antialiased=False
                                            )
                    title_parts = (f'{iate_label}: adjacent-dose comparisons',
                                   f'Outcome: {var_cfg.y_name[idx_y]}')
                    display_title = '\n'.join(fill(part, width=70) for part in title_parts)
                    axe.set_title(display_title, fontsize=int_cfg.fontsize)
                    axe.set_ylabel('Upper dose')
                    axe.set_zlabel(iate_label)
                    axe.set_xlabel('Index of sorted IATEs')
                    fig.colorbar(surf, shrink=0.5, aspect=5)
                    ttt = titel.replace(' ', '')
                    file_name_jpeg = p_cfg.paths['ate_iate_fig_pfad_jpeg'] / f'{ttt}.jpeg'
                    file_name_pdf = p_cfg.paths['ate_iate_fig_pfad_pdf'] / f'{ttt}.pdf'
                    if post_cfg.plots:
                        mcf_sys.delete_file_if_exists(file_name_jpeg)
                        mcf_sys.delete_file_if_exists(file_name_pdf)
                        fig.savefig(file_name_jpeg, dpi=int_cfg.dpi)
                        fig.savefig(file_name_pdf, dpi=int_cfg.dpi)
                        if int_cfg.show_plots:
                            plt.show()
                    plt.close(fig)
                    if post_cfg.plots and imate == iterator[-1]:
                        figure_list.append(file_name_jpeg)

    if post_cfg.bin_corr_yes:
        txt_summary += ('\n\nFull correlation tables are in the detailed TXT output.'
                        '\nCovariate correlations are included only when their absolute value'
                        f' exceeds {post_cfg.bin_corr_threshold:.1%}.'
                        )
    else:
        txt_summary += '\n\nCorrelation analysis was not requested.'

    mcf_ps.print_mcf(gen_cfg, txt, summary=False, non_summary=True)
    mcf_ps.print_mcf(gen_cfg, txt_summary, summary=True, non_summary=False)

    return figure_list


def cluster_mean_table(means: pd.DataFrame) -> str:
    """Format variable-by-cluster means without truncating names or wide tables."""
    if means.empty:
        return 'No numeric variables to report.'

    lines: list[str] = []
    for name, row in means.iterrows():
        lines.append(fill(str(name), width=100))
        for start in range(0, len(row), 6):
            block = row.iloc[start:start + 6]
            labels = ''.join(f'{str(cluster):>14}' for cluster in block.index)
            values = ''.join(f'{value:>14.6g}' for value in block)
            lines.append('Cluster ' + labels)
            lines.append('Mean    ' + values)
        lines.append('')
    return '\n'.join(lines).rstrip()


def k_means_of_x_iate(mcf_: 'ModifiedCausalForest',
                      results_prev: dict[str, Any]
                      ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cluster IATEs and separate concise results from detailed diagnostics."""
    results = deepcopy(results_prev)
    gen_cfg, post_cfg, int_cfg = mcf_.gen_cfg, mcf_.post_cfg, mcf_.int_cfg
    var_cfg, var_x_type = mcf_.var_cfg, mcf_.var_x_type
    if post_cfg.relative_to_first_group_only or gen_cfg.d_type == 'continuous':
        iate_pot_name = results['iate_names_dic'][1]
    else:
        iate_pot_name = results['iate_names_dic'][0]
    data_df = results['iate_data_df']

    data_df, rnd_reduce, txt = mcf_gp.check_reduce_dataframe(data_df, title='k-means clustering',
                                                             max_obs=int_cfg.max_obs_post_kmeans,
                                                             seed=124535, ignore_index=True
                                                             )
    if rnd_reduce:
        mcf_ps.print_mcf(gen_cfg, txt, summary=True)

    if mcf_.lc_cfg.uncenter_po:
        pot_y = data_df[iate_pot_name['names_y_pot_uncenter']]
    else:
        pot_y = data_df[iate_pot_name['names_y_pot']]
    x_name = delete_x_with_catv(var_x_type.keys())
    x_dat = data_df[x_name]
    if mcf_.data_train_dict is not None and len(mcf_.data_train_dict) > 0:
        x_dat = mcf_ps.change_name_value_df(x_dat,
                                            mcf_.data_train_dict['prime_old_name_dict'],
                                            mcf_.data_train_dict['prime_values_dict'],
                                            mcf_.data_train_dict['unique_values_dict']
                                            )
    iate_name_all = iate_pot_name['names_iate']
    reference_names = set(results['iate_names_dic'][1]['names_iate'])
    no_of_kmeans = len(iate_name_all) + 1 if post_cfg.kmeans_single else 1
    report_dic: dict[str, Any] = {}
    txt_summary = '\n' + '=' * 100 + '\nK-Means++ clustering summary'
    single_merges = 0

    for val_idx in range(no_of_kmeans):
        joint = val_idx == 0
        if joint:
            iate_names_cluster = iate_name_all.copy()
            title = 'K-Means++ clustering (all selected effects jointly)'
        else:
            iate_names_cluster = [iate_name_all[val_idx - 1]]
            title = f'K-Means++ clustering ({iate_names_cluster[0]} only)'
        iate = data_df[iate_names_cluster]
        iate_np = mcf_gp.to_numpy_big_data(iate, int_cfg.obs_bigdata)
        silhouette_avg_prev, cluster_lab_np = -1, None
        best_requested: int | None = None
        best_merged = False
        txt = '\n' + '=' * 100 + '\n' + fill(title, width=100)
        txt += '\n' + '-' * 100

        for cluster_no in post_cfg.kmeans_no_of_groups:
            cluster_lab_tmp, silhouette_avg, merge = kmeans_labels(iate_np,
                no_of_clusters=cluster_no,
                kmeans_max_tries=post_cfg.kmeans_max_tries,
                kmeans_replications=post_cfg.kmeans_replications,
                kmeans_min_size_share=post_cfg.kmeans_min_size_share,
                )
            actual_no = len(np.unique(cluster_lab_tmp))
            txt += (f'\nRequested clusters: {cluster_no}; obtained: {actual_no}; '
                    f'average silhouette score: {silhouette_avg:8.3f}'
                    )
            if merge:
                txt += '\nSmall clusters were merged with clusters having the closest centroids.'
            if silhouette_avg > silhouette_avg_prev:
                cluster_lab_np = np.copy(cluster_lab_tmp)
                silhouette_avg_prev = np.copy(silhouette_avg)
                best_requested = int(cluster_no)
                best_merged = bool(merge)
        if cluster_lab_np is None:
            raise ValueError('No candidate clustering has a valid silhouette score.')
        del iate_np

        # Order joint results by the first outcome, and single runs by their selected effect.
        if joint:
            no_of_first = round(len(iate_names_cluster) / len(var_cfg.y_name))
            names_first = iate_names_cluster[:no_of_first]
            order_text = ('Clusters are ordered by the average of the selected effects '
                          'for the first outcome.')
        else:
            names_first = iate_names_cluster
            order_text = 'Clusters are ordered by the selected effect.'
        cl_means = iate[names_first].groupby(by=cluster_lab_np).mean(numeric_only=True)
        cl_means_np = np.mean(cl_means.to_numpy(copy=True), axis=1)
        sort_ind = np.argsort(cl_means_np)
        cl_group = cluster_lab_np.copy()
        for cl_j, cl_old in enumerate(sort_ind):
            cl_group[cluster_lab_np == cl_old] = cl_j

        cl_values, cl_obs = np.unique(cl_group, return_counts=True)
        selected_text = (f'\nSelected clusters: {len(cl_values)} '
                         f'(requested before merging: {best_requested})'
                         f'\nAverage silhouette score: {silhouette_avg_prev:8.3f}'
                         )
        if best_merged:
            selected_text += '\nThe selected solution includes merging of small clusters.'
        selected_text += '\n' + fill(order_text, width=100)
        txt += '\n' + selected_text

        counts_text = '\nNumber of observations in the clusters\n' + '- ' * 50
        txt_report = '\n\nNumber of observations in the clusters\n'
        for val, obs in zip(cl_values, cl_obs, strict=True):
            counts_text += f'\nCluster {val:2}: {obs:6} ({obs / len(data_df):6.1%})'
            txt_report += f'\nCluster {val:2}: {obs:6} '
        txt += counts_text
        if joint:
            report_dic['obs_cluster_str'] = txt_report
            txt_summary += (f'\nObservations used: {len(data_df):,}'
                            f'\nEffect columns used jointly: {len(iate_names_cluster):,}'
                            + selected_text + counts_text
                            )
        else:
            single_merges += int(best_merged)

        iate_cluster_id_name = ('IATE_Cluster' if joint
                                else 'IATE_Cluster_' + iate_names_cluster[0])
        data_df[iate_cluster_id_name] = cl_group
        effects_by_cluster = iate.groupby(by=cl_group).mean(numeric_only=True).transpose()
        if joint:
            report_dic['IATE_df'] = np.round(effects_by_cluster.copy(), 2)
            summary_names = [name for name in iate_names_cluster if name in reference_names]
            if summary_names:
                effect_title = 'Average effects by cluster: comparisons with the first group'
            else:
                summary_names = iate_names_cluster
                effect_title = 'Average effects by cluster: all selected comparisons'
            txt_summary += '\n\n' + effect_title + '\n'
            txt_summary += cluster_mean_table(effects_by_cluster.loc[summary_names])
        txt += '\n' + '-' * 100 + '\nEffects\n' + '- ' * 50 + '\n'
        txt += cluster_mean_table(effects_by_cluster)

        pot_by_cluster = pot_y.groupby(by=cl_group).mean(numeric_only=True).transpose()
        if joint:
            report_dic['PotOutcomes_df'] = np.round(pot_by_cluster, 2)
        txt += '\n' + '-' * 100 + '\nPotential outcomes\n' + '- ' * 50 + '\n'
        txt += cluster_mean_table(pot_by_cluster)

        names_unordered = [name for name in var_x_type if var_x_type[name] > 0]
        if names_unordered:
            names_unordered_org = [mcf_.data_train_dict['prime_old_name_dict'].get(name, name)
                                   for name in names_unordered]
            x_dummies = pd.get_dummies(x_dat, columns=names_unordered_org, dtype=int)
            x_km = pd.concat([x_dat[names_unordered_org], x_dummies], axis=1)
        else:
            x_km = x_dat
        cl_means = x_km.groupby(by=cl_group).mean(numeric_only=True)
        cl_means.columns = [name.replace('_prime', '') for name in cl_means.columns]
        cl_means.columns = [name.replace('_prime_', '_D') for name in cl_means.columns]
        cl_means.columns = [name.replace('.0', '') for name in cl_means.columns]
        features_by_cluster = cl_means.transpose()
        if joint:
            report_dic['Features_df'] = np.round(features_by_cluster, 2)
        txt += '\n' + '-' * 100 + '\nCovariates\n' + '- ' * 50 + '\n'
        txt += cluster_mean_table(features_by_cluster)
        txt += '\n' + '-' * 100 + '\nCluster indicator stored in the returned data:\n'
        txt += fill(iate_cluster_id_name, width=100)
        mcf_ps.print_mcf(gen_cfg, txt, summary=False, non_summary=True)

    if post_cfg.kmeans_single:
        txt_summary += (f'\n\nSeparate single-effect clustering analyses: {no_of_kmeans - 1:,}.'
                        f'\nSelected single-effect solutions with merging: {single_merges:,}.'
                        )
    txt_summary += ('\n\nFull effect, potential-outcome and covariate tables, candidate scores,'
                    '\nand results for any separate single-effect runs are in the detailed TXT.'
                    )
    mcf_ps.print_mcf(gen_cfg, txt_summary, summary=True, non_summary=False)
    results['iate_data_df'] = data_df

    return results, report_dic

def kmeans_labels(iate_np: NDArray[Any],
                  no_of_clusters: int = 5,
                  kmeans_max_tries: int = 1000,
                  kmeans_replications: int = 10,
                  kmeans_min_size_share: int | float = 1,
                  ) -> tuple[NDArray[Any], float, bool]:
    """Compute k-means labels and merge clusters below the minimum size."""
    max_clusters = min(len(iate_np) - 1, len(np.unique(iate_np, axis=0)))
    if max_clusters < 2:
        return np.zeros(len(iate_np), dtype=int), -np.inf, False
    no_of_clusters = min(no_of_clusters, max_clusters)

    kmeans = KMeans(n_clusters=no_of_clusters,
                    n_init=kmeans_replications, init='k-means++',
                    max_iter=kmeans_max_tries, algorithm='lloyd',
                    random_state=42, tol=1e-5, verbose=0, copy_x=True,
                    )
    cluster_labels = kmeans.fit_predict(iate_np)
    labels, counts = np.unique(cluster_labels, return_counts=True)
    min_size = kmeans_min_size_share / 100 * len(cluster_labels)
    merged = False

    if np.any(counts < min_size):
        centroids = np.stack([np.mean(iate_np[cluster_labels == label], axis=0, dtype=np.float64)
                              for label in labels]
                             )
        while len(labels) > 1 and np.any(counts < min_size):
            small_idx = int(np.argmin(counts))
            distances = cdist(centroids[[small_idx], :], centroids).ravel()
            distances[small_idx] = np.inf
            closest_idx = int(np.argmin(distances))

            total = counts[closest_idx] + counts[small_idx]
            small_weight = counts[small_idx] / total
            centroids[closest_idx] += small_weight * (centroids[small_idx] - centroids[closest_idx])
            cluster_labels[cluster_labels == labels[small_idx]] = labels[closest_idx]
            counts[closest_idx] = total
            labels = np.delete(labels, small_idx)
            counts = np.delete(counts, small_idx)
            centroids = np.delete(centroids, small_idx, axis=0)
            merged = True

        # Relabel the remaining clusters consecutively, preserving the label dtype.
        _, relabelled = np.unique(cluster_labels, return_inverse=True)
        cluster_labels = relabelled.astype(cluster_labels.dtype, copy=False)

    if len(labels) < 2:
        return cluster_labels, -np.inf, merged

    silhouette = float(silhouette_score(iate_np, cluster_labels))

    return cluster_labels, silhouette, merged


def eva_points_fct(no_of_names: int | float,
                   no_of_y_name:  int | float,
                   ) -> list[list[int]]:
    """Get evaluation points."""
    no_of_names_y = round(no_of_names / no_of_y_name)
    eva_points_y = [round(no_of_names_y / 3), round(2 * no_of_names_y / 3)]
    eva_points = []
    for idx in range(no_of_y_name):
        eva_points_t = [i + no_of_names_y * idx for i in eva_points_y]
        eva_points.extend(eva_points_t)

    return eva_points


def delete_x_with_catv(names_with_catv: list[str]) -> list[str]:
    """Delete variables which end with catv from list."""
    return [x_name for x_name in names_with_catv if not x_name.endswith('catv')]


def random_forest_tree_of_iate(mcf_: 'ModifiedCausalForest',
                               results: dict[str, Any]
                               ) -> None:
    """Describe estimated IATEs with concise forest and tree summaries."""
    gen_cfg, var_x_type = mcf_.gen_cfg, mcf_.var_x_type
    random_forest = mcf_.post_cfg.random_forest_vi
    tree = mcf_.post_cfg.tree

    if mcf_.post_cfg.relative_to_first_group_only or gen_cfg.d_type == 'continuous':
        iate_pot_name = results['iate_names_dic'][1]
    else:
        iate_pot_name = results['iate_names_dic'][0]
    names_unordered = [name for name in var_x_type if var_x_type[name] > 0]
    x_name = delete_x_with_catv(var_x_type.keys())
    no_of_names = len(iate_pot_name['names_iate'])
    eva_points = eva_points_fct(no_of_names, len(mcf_.var_cfg.y_name))
    data_df = results['iate_data_df']
    x_dat = data_df[x_name]
    iate = data_df[iate_pot_name['names_iate']]
    x_name = x_dat.columns.tolist()
    dummy_group_names: list[list[str]] = []
    header = '\n' + '=' * 100 + '\nRandom-forest and regression-tree analysis of IATEs'
    txt_summary = header
    mcf_ps.print_mcf(gen_cfg, header, summary=False, non_summary=True)
    txt = ''

    if names_unordered:
        dummy_names: list[str] = []
        if mcf_.data_train_dict is not None and len(mcf_.data_train_dict) > 0:
            x_dat = mcf_ps.change_name_value_df(x_dat,
                                              mcf_.data_train_dict['prime_old_name_dict'],
                                              mcf_.data_train_dict['prime_values_dict'],
                                              mcf_.data_train_dict['unique_values_dict']
                                              )
            names_unordered_org = [mcf_.data_train_dict['prime_old_name_dict'].get(name, name)
                                   for name in names_unordered]
            x_name = [mcf_.data_train_dict['prime_old_name_dict'].get(name, name)
                      for name in x_name]
        else:
            names_unordered_org = names_unordered.copy()
        for name in names_unordered_org:
            x_t_d = pd.get_dummies(x_dat[name], prefix=name, dtype=int)
            this_dummy_names = x_t_d.columns.tolist()
            dummy_names.extend(this_dummy_names)
            dummy_group_names.append(this_dummy_names + [name])
            x_dat = pd.concat([x_dat, x_t_d], axis=1)
        x_name.extend(dummy_names)
        txt += '\nCreated dummy variables:\n'
        txt += '\n'.join(fill(name, width=100) for name in dummy_names)

    x_train = x_dat.to_numpy(copy=True)
    txt += '\n\nDescriptive statistics of model inputs\n' + '-' * 100
    descriptions = x_dat.describe()
    for name in descriptions:
        txt += '\n' + fill(str(name), width=100) + '\n'
        txt += descriptions[name].to_string(max_rows=None)
    mcf_ps.print_mcf(gen_cfg, txt, summary=False, non_summary=True)

    selected_indices = [idx for idx in range(no_of_names)
                        if gen_cfg.d_type != 'continuous' or idx in eva_points]
    txt_summary += (f'\nObservations: {len(data_df):,}'
                    f'\nInput columns, including dummies: {x_train.shape[1]:,}'
                    f'\nOutcome-comparison combinations analysed: {len(selected_indices):,}'
                    '\nThese models describe variation in estimated individual effects.'
                    )
    for idx in selected_indices:
        y_name = iate_pot_name['names_iate'][idx]
        y_train = iate[y_name].to_numpy(copy=True)
        effect_heading = '\n\n' + fill(f'Effect: {y_name}', width=100)
        txt = effect_heading
        txt_summary += effect_heading
        if random_forest:
            (_, _, oob_r2, vi_info, _, _, _, txt_rf
             ) = mcf_est_g.random_forest_scikit(x_train, y_train, None,
                                               x_name=x_name, y_name=y_name,
                                               boot=mcf_.cf_cfg.boot, n_min=2,
                                               no_features='sqrt', max_depth=None,
                                               workers=gen_cfg.mp_parallel, alpha=0,
                                               var_im_groups=dummy_group_names,
                                               max_leaf_nodes=None, pred_p_flag=False,
                                               pred_t_flag=True, pred_oob_flag=True,
                                               with_output=True, variable_importance=True,
                                               pred_uncertainty=False,
                                               pu_ci_level=mcf_.p_cfg.ci_level,
                                               pu_skew_sym=0.5, var_im_with_output=True,
                                               )
            # Correct two legacy labels in this helper's detailed text.
            txt_rf = txt_rf.replace(f'Dependent variable: {y_name[0]} N_min:',
                                    f'Dependent variable: {y_name} N_min:', 1)
            old_vi_title = ('Variable importance statistics in %-point of R2 lost '
                            'when omitting particular variable'
                            )
            txt_rf = txt_rf.replace(old_vi_title,
                                    'Permutation importance: test-set R2 loss in percentage points'
                                    )
            txt += '\nRandom forest\n'
            txt += '\n'.join(fill(line, width=100, subsequent_indent='  ')
                             for line in txt_rf.splitlines())
            txt_summary += f'\nRandom forest: out-of-bag R2 = {oob_r2:.2%}'
            if vi_info is not None:
                vi_names, vi_values = vi_info[:2]
                no_to_show = min(5, len(vi_names))
                txt_summary += (f'\nLargest {no_to_show} permutation-importance values'
                                ' (test-set R2 loss, percentage points):'
                                )
                for name, value in zip(vi_names[:no_to_show], vi_values[:no_to_show], strict=True):
                    line = f'{name}: {100 * float(value):.2f}'
                    txt_summary += '\n' + fill(line, width=100, subsequent_indent='  ')
        if tree:
            txt_tree = analyse_iate_tree(mcf_, x_train, y_train, x_name=x_name, y_name=y_name)
            txt += txt_tree
            txt_summary += txt_tree
        mcf_ps.print_mcf(gen_cfg, txt, summary=False, non_summary=True)

    txt_summary += ('\n\nModel settings, input statistics and full variable-importance tables'
                    '\nfor any requested forests are in the detailed TXT output.'
                    )
    mcf_ps.print_mcf(gen_cfg, txt_summary, summary=True, non_summary=False)


def analyse_iate_tree(mcf_: 'ModifiedCausalForest',
                      x_dat: NDArray[Any],
                      y_dat: NDArray[Any],
                      x_name: list[str] | None = None,
                      y_name: str | None = None, *,
                      seed: int = 1223
                      ) -> str:
    """Report tree fit and save, show and close the figures according to the settings."""
    results_dict, _ = mcf_est_g.honest_tree_explainable(x_dat, y_dat,
                                                      tree_type='regression',
                                                      depth_grid=mcf_.post_cfg.tree_depths,
                                                      feature_names=x_name, title=y_name,
                                                      seed=seed
                                                      )
    figures = list(results_dict['plots']) + list(results_dict['plots_h'])
    try:
        txt = '\n\n' + '-' * 100
        txt += '\n' + fill(f'Tree-based evaluation of {y_name}', width=100)
        txt += '\nOut-of-sample R2 for each requested depth\n'
        txt += '\nDepth     Standard tree      Honest tree'
        for depth, fit, fit_h in zip(mcf_.post_cfg.tree_depths,
                                     results_dict['fit'], results_dict['fit_h'], strict=True):
            txt += f'\n{depth:5}     {fit:13.2%}         {fit_h:8.2%}'

        if mcf_.post_cfg.plots:
            for depth, plot, plot_h in zip(mcf_.post_cfg.tree_depths,
                                           results_dict['plots'], results_dict['plots_h'],
                                           strict=True):
                titel_f = 'Tree_D' + str(depth) + y_name
                for prefix, figure in (('', plot), ('Hon_', plot_h)):
                    file_name_jpeg = (mcf_.p_cfg.paths['ate_iate_fig_pfad_jpeg']
                                      / f'{prefix}{titel_f}.jpeg'
                                      )
                    file_name_pdf = (mcf_.p_cfg.paths['ate_iate_fig_pfad_pdf']
                                     / f'{prefix}{titel_f}.pdf'
                                     )
                    for axis in figure.axes:
                        axis.set_title(fill(axis.get_title(), width=100))
                    mcf_sys.delete_file_if_exists(file_name_jpeg)
                    mcf_sys.delete_file_if_exists(file_name_pdf)
                    figure.savefig(file_name_jpeg, dpi=mcf_.int_cfg.dpi, bbox_inches='tight')
                    figure.savefig(file_name_pdf, dpi=mcf_.int_cfg.dpi, bbox_inches='tight')
            if mcf_.int_cfg.show_plots:
                plt.show()
    finally:
        for figure in figures:
            plt.close(figure)
    return txt


def relational_graphs(mcf_: 'ModifiedCausalForest',
                      data_df: pd.DataFrame,
                      iate_name: list[str],
                      x_dat_df: pd.DataFrame
                      ) -> None:
    """Show graphical relationships between iates and single x-variable."""
    first_run = True
    for x_name in x_dat_df.columns:
        iate_x_dat_df = pd.concat((data_df[iate_name].reset_index(drop=True),
                                   x_dat_df[x_name].reset_index(drop=True)),
                                  axis=1,
                                  )
        iate_x_dat_df, rnd_reduce, txt = mcf_gp.check_reduce_dataframe(
            iate_x_dat_df,
            title='Post analysis using relational graphs',
            max_obs=mcf_.int_cfg.max_obs_post_rel_graphs,
            seed=124535,
            ignore_index=True
            )
        if rnd_reduce and first_run:
            mcf_ps.print_mcf(mcf_.gen_cfg, txt, summary=True)
            first_run = False

        if x_dat_df[x_name].nunique() > 20:
            g = sns.relplot(data=iate_x_dat_df, x=x_name, y=iate_name, kind='scatter',)
            ax = g.axes[0, 0]
            # Overlay a regression line (no additional scatter)
            sns.regplot(data=iate_x_dat_df,
                        x=x_name,
                        y=iate_name,
                        lowess=True,
                        scatter=False,   # do not re-draw scatter
                        ci=None,         # disable confidence interval if desired
                        color='red',
                        ax=ax
                        )
        else:
            g = sns.catplot(data=iate_x_dat_df, x=x_name, y=iate_name, kind='box')
        title = f'Relation of {iate_name} and {x_name}'
        plt.title(title)
        plt.subplots_adjust(top=0.90)

        # Save to file
        title_f = f'Univariate_{iate_name}_{x_name}'
        file_name_jpeg = mcf_.p_cfg.paths['ate_iate_fig_pfad_jpeg'] / f'{title_f}.jpeg'
        file_name_pdf = mcf_.p_cfg.paths['ate_iate_fig_pfad_pdf'] / f'{title_f}.pdf'
        file_name_csv = mcf_.p_cfg.paths['ate_iate_fig_pfad_csv'] / f'{title_f}plotdat.csv'
        # Delete files with they already exist (overwrite may be impossible)
        mcf_sys.delete_file_if_exists(file_name_csv)
        mcf_sys.delete_file_if_exists(file_name_jpeg)
        mcf_sys.delete_file_if_exists(file_name_pdf)

        # Save figures and data
        plt.savefig(file_name_jpeg, dpi=mcf_.int_cfg.dpi)
        plt.savefig(file_name_pdf, dpi=mcf_.int_cfg.dpi)
        iate_x_dat_df.to_csv(file_name_csv, index=False)
        if mcf_.int_cfg.show_plots:
            plt.show()
        plt.close(g.fig)


def file_names_helper(iv_ind: int,
                      file_jpeg: Path,
                      file_pdf: Path,
                      file_csv: Path,
                      titel: str,
                      ) -> tuple[Path, Path, Path]:
    """Produce file names for plots."""
    match iv_ind:
        case 0:
            file_name_jpeg = file_jpeg / f'{titel}.jpeg'
            file_name_pdf = file_pdf / f'{titel}.pdf'
            file_name_csv = file_csv / f'{titel}plotdat.csv'
        case 1:
            file_name_jpeg = file_jpeg / f'{titel}1st.jpeg'
            file_name_pdf = file_pdf / f'{titel}1st.pdf'
            file_name_csv = file_csv / f'{titel}plotdat1st.csv'
        case 2:
            file_name_jpeg = file_jpeg / f'{titel}redft.jpeg'
            file_name_pdf = file_pdf / f'{titel}redf.pdf'
            file_name_csv = file_csv / f'{titel}plotdatredf.csv'
        case _:
            file_name_jpeg = file_name_pdf = file_name_csv = None

    return file_name_jpeg, file_name_pdf, file_name_csv
