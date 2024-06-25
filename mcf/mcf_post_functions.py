"""
Contains the functions for the post estimation analysis.

Created on Thu Jun 29 17:28:31 2023.


# -*- coding: utf-8 -*-
@author: MLechner
"""
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mcf import mcf_estimation_generic_functions as mcf_est_g
from mcf import mcf_general as mcf_gp
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as ps


def post_estimation_iate(mcf_, results):
    """Do post-estimation analysis: correlations, k-means, sorted effects."""
    ct_dic, int_dic = mcf_.ct_dict, mcf_.int_dict
    gen_dic, p_dic, post_dic = mcf_.gen_dict, mcf_.p_dict, mcf_.post_dict
    var_dic, var_x_type = mcf_.var_dict, mcf_.var_x_type
    figure_list = []

    txt = '\n' + '=' * 100 + '\nPost estimation analysis'
    if gen_dic['d_type'] == 'continuous':
        d_values = ct_dic['d_values_dr_np']
        no_of_treat = len(d_values)
    else:
        no_of_treat, d_values = gen_dic['no_of_treat'], gen_dic['d_values']
    iate_pot_all_name = results['iate_names_dic']
    ate_all, ate_all_se = results['ate'], results['ate_se']
    effect_list = results['ate effect_list']
    data_df = results['iate_data_df']

    if (post_dic['relative_to_first_group_only']
            or gen_dic['d_type'] == 'continuous'):
        txt += ': Effects only relative to treatment 0 are considered'
        iate_pot_name = iate_pot_all_name[1]
        dim_all = (len(ate_all), no_of_treat-1)
        ate, ate_se = np.empty(dim_all), np.empty(dim_all)
        jdx = 0
        for idx, i_lab in enumerate(effect_list):
            if i_lab[1] == d_values[0]:  # compare to 1st treat only
                ate[:, jdx] = ate_all[:, 0, idx]
                ate_se[:, jdx] = ate_all_se[:, 0, idx]
                jdx += 1
    else:
        iate_pot_name = iate_pot_all_name[0]
        dim_all = (np.size(ate_all, axis=0), np.size(ate_all, axis=2))
        ate, ate_se = ate_all[:, 0, :], ate_all_se[:, 0, :]
    ate, ate_se = ate.reshape(-1), ate_se.reshape(-1)
    txt += '\n'
    if mcf_.lc_dict['uncenter_po']:
        y_pot = data_df[iate_pot_name['names_y_pot_uncenter']]
    else:
        y_pot = data_df[iate_pot_name['names_y_pot']]
    iate = data_df[iate_pot_name['names_iate']]
    x_name = delete_x_with_catv(var_x_type.keys())
    x_dat = data_df[x_name]
    cint = norm.ppf(p_dic['ci_level'] + 0.5 * (1 - p_dic['ci_level']))
    if post_dic['bin_corr_yes']:
        txt += '\n' + '=' * 100 + '\nCorrelations of effects with ... in %'
        txt += '\n' + '-' * 100
    label_ci = f'{p_dic["ci_level"]:2.0%}-CI'
    iterator = range(2) if p_dic['iate_m_ate'] else range(1)
    no_of_names = len(iate_pot_name['names_iate'])
    eva_points = eva_points_fct(no_of_names, len(var_dic['y_name']))
    for idx in range(no_of_names):
        for imate in iterator:
            if imate == 0:
                name_eff = 'names_iate'
                ate_t, ate_se_t = ate[idx].copy(), ate_se[idx].copy()
            else:
                name_eff, ate_t = 'names_iate_mate', 0
            name_iate_t = iate_pot_name[name_eff][idx]
            if p_dic['iate_se']:
                name_se = name_eff + '_se'
                name_iate_se_t = iate_pot_name[name_se][idx]
            else:
                name_se = name_iate_se_t = None
            titel = 'Sorted ' + name_iate_t
            # Add correlation analyis of IATEs
            if gen_dic['d_type'] == 'discrete' or idx in eva_points:
                if post_dic['bin_corr_yes'] and imate == 0:
                    txt += f'\nEffect: {name_iate_t}' + '\n' + '- ' * 50
                    if gen_dic['d_type'] == 'discrete':
                        corr = iate.corrwith(data_df[name_iate_t])
                        for jdx in corr.keys():
                            txt += f'\n{jdx:<20} {corr[jdx]:>8.2%}'
                        txt += '\n' + '- ' * 50
                        corr = y_pot.corrwith(data_df[name_iate_t])
                        for jdx in corr.keys():
                            txt += f'\n{jdx:<20} {corr[jdx]:>8.2%}'
                        txt += '\n' + '- ' * 50
                    corr = x_dat.corrwith(data_df[name_iate_t])
                    corr = corr.sort_values()
                    for jdx in corr.keys():
                        if np.abs(corr[jdx].item()
                                  ) > post_dic['bin_corr_threshold']:
                            txt += f'\n{jdx:<20} {corr[jdx]:>8.2%}'
                    txt += '\n' + '- ' * 50
                iate_temp = data_df[name_iate_t].to_numpy()
                iate_se_temp = (data_df[name_iate_se_t].to_numpy()
                                if p_dic['iate_se'] else None)
                sorted_ind = np.argsort(iate_temp)
                iate_temp = iate_temp[sorted_ind]
                if p_dic['iate_se']:
                    iate_se_temp = iate_se_temp[sorted_ind]
                x_values = np.arange(len(iate_temp)) + 1
                x_values = np.around(x_values / x_values[-1] * 100, decimals=1)
                k = np.round(p_dic['knn_const'] * np.sqrt(len(iate_temp)) * 2)
                iate_temp = mcf_est_g.moving_avg_mean_var(
                    iate_temp, k, False)[0]
                if p_dic['iate_se']:
                    iate_se_temp = mcf_est_g.moving_avg_mean_var(
                        iate_se_temp, k, False)[0]
                titel_f = titel.replace(' ', '')
                file_name_jpeg = (p_dic['ate_iate_fig_pfad_jpeg']
                                  + '/' + titel_f + '.jpeg')
                file_name_pdf = (p_dic['ate_iate_fig_pfad_pdf']
                                 + '/' + titel_f + '.pdf')
                file_name_csv = (p_dic['ate_iate_fig_pfad_csv']
                                 + '/' + titel_f + 'plotdat.csv')
                if p_dic['iate_se']:
                    upper = iate_temp + iate_se_temp * cint
                    lower = iate_temp - iate_se_temp * cint
                ate_t = ate_t * np.ones(len(iate_temp))
                if imate == 0:
                    ate_upper = ate_t + (
                        ate_se_t * cint * np.ones(len(iate_temp)))
                    ate_lower = ate_t - (
                        ate_se_t * cint * np.ones(len(iate_temp)))
                line_ate, line_iate = '_-r', '-b'
                fig, axe = plt.subplots()
                if imate == 0:
                    label_t, label_r, label_y = 'IATE', 'ATE', 'Effect'
                else:
                    label_t, label_r = 'IATE-ATE', '_nolegend_'
                    label_y = 'Effect - average'
                axe.plot(x_values, iate_temp, line_iate, label=label_t)
                axe.set_ylabel(label_y)
                axe.plot(x_values, ate_t, line_ate, label=label_r)
                if imate == 0:
                    axe.fill_between(x_values, ate_upper, ate_lower,
                                     alpha=0.3, color='r', label=label_ci)
                titel_tmp = titel.replace('_iate', '')
                if 'mate' in titel:
                    titel_tmp = titel_tmp.replace('mate', '')
                titel_tmp = titel_tmp[:-4] + ' ' + titel_tmp[-4:]
                titel_tmp = titel_tmp.replace('vs', ' vs ')
                if 'mate' in titel:
                    titel_tmp += ' (- ATE)'
                axe.set_title(titel_tmp)
                axe.set_xlabel('Quantile of sorted IATEs')
                if p_dic['iate_se']:
                    axe.fill_between(x_values, upper, lower, alpha=0.3,
                                     color='b', label=label_ci)
                axe.legend(loc=int_dic['legend_loc'], shadow=True,
                           fontsize=int_dic['fontsize'])
                if post_dic['plots']:
                    mcf_sys.delete_file_if_exists(file_name_jpeg)
                    mcf_sys.delete_file_if_exists(file_name_pdf)
                    fig.savefig(file_name_jpeg, dpi=int_dic['dpi'])
                    fig.savefig(file_name_pdf, dpi=int_dic['dpi'])
                if post_dic['plots']:
                    plt.show()
                else:
                    plt.close()
                iate_temp = iate_temp.reshape(-1, 1)
                if p_dic['iate_se']:
                    upper, lower = upper.reshape(-1, 1), lower.reshape(-1, 1)
                ate_t = ate_t.reshape(-1, 1)
                iate_temp = iate_temp.reshape(-1, 1)
                if imate == 0:
                    ate_upper = ate_upper.reshape(-1, 1)
                    ate_lower = ate_lower.reshape(-1, 1)
                    if p_dic['iate_se']:
                        effects_et_al = np.concatenate(
                            (upper, iate_temp, lower, ate_t, ate_upper,
                             ate_lower), axis=1)
                        cols = ['upper', 'effects', 'lower', 'ate', 'ate_l',
                                'ate_u']
                    else:
                        effects_et_al = np.concatenate(
                            (iate_temp, ate_t, ate_upper, ate_lower), axis=1)
                        cols = ['effects', 'ate', 'ate_l', 'ate_u']
                else:
                    effects_et_al = np.concatenate(
                        (upper, iate_temp, lower, ate_t), axis=1)
                    cols = ['upper', 'effects', 'lower', 'ate']
                datasave = pd.DataFrame(data=effects_et_al, columns=cols)
                mcf_sys.delete_file_if_exists(file_name_csv)
                datasave.to_csv(file_name_csv, index=False)
                if imate == iterator[-1] and p_dic['iate_se']:
                    figure_list.append(file_name_jpeg)
                # density plots
                if imate == 0:
                    titel = 'Density ' + iate_pot_name['names_iate'][idx]
                    titel_f = titel.replace(' ', '')
                    file_name_jpeg = (p_dic['ate_iate_fig_pfad_jpeg']
                                      + '/' + titel_f + '.jpeg')
                    file_name_pdf = (p_dic['ate_iate_fig_pfad_pdf']
                                     + '/' + titel_f + '.pdf')
                    file_name_csv = (p_dic['ate_iate_fig_pfad_csv']
                                     + '/' + titel_f + 'plotdat.csv')
                    iate_temp = data_df[name_iate_t].to_numpy()
                    bandwidth = mcf_est_g.bandwidth_silverman(iate_temp, 1)
                    dist = np.abs(iate_temp.max() - iate_temp.min())
                    low_b = iate_temp.min() - 0.1 * dist
                    up_b = iate_temp.max() + 0.1 * dist
                    grid = np.linspace(low_b, up_b, 1000)
                    density = mcf_est_g.kernel_density(iate_temp, grid, 1,
                                                       bandwidth)
                    fig, axe = plt.subplots()
                    titel_tmp = titel.replace('_iate', '')
                    titel_tmp = titel_tmp[:-4] + ' ' + titel_tmp[-4:]
                    titel_tmp = titel_tmp.replace('vs', ' vs ')
                    axe.set_title(titel_tmp)
                    axe.set_ylabel('Estimated density')
                    axe.set_xlabel('IATE')
                    axe.plot(grid, density, '-b')
                    axe.fill_between(grid, density, alpha=0.3, color='b')
                    if post_dic['plots']:
                        mcf_sys.delete_file_if_exists(file_name_jpeg)
                        mcf_sys.delete_file_if_exists(file_name_pdf)
                        fig.savefig(file_name_jpeg, dpi=int_dic['dpi'])
                        fig.savefig(file_name_pdf, dpi=int_dic['dpi'])
                    if post_dic['plots']:
                        plt.show()
                    else:
                        plt.close()
                    density = density.reshape(-1, 1)
                    cols = ['grid', 'density']
                    grid = grid.reshape(-1, 1)
                    density = density.reshape(-1, 1)
                    effects_et_al = np.concatenate((grid, density), axis=1)
                    datasave = pd.DataFrame(data=effects_et_al, columns=cols)
                    mcf_sys.delete_file_if_exists(file_name_csv)
                    datasave.to_csv(file_name_csv, index=False)
                    if not p_dic['iate_se']:
                        figure_list.append(file_name_jpeg)
    if gen_dic['d_type'] == 'continuous':
        no_of_y = len(var_dic['y_name'])
        no_of_iate_y = round(len(iate_pot_name['names_iate']) / no_of_y)
        index_0 = range(no_of_iate_y)
        for idx_y in range(no_of_y):  # In case there are several outcomes
            for imate in iterator:
                if imate == 0:
                    name_eff, iate_label = 'names_iate', 'IATE'
                else:
                    name_eff, iate_label = 'names_iate_mate', 'IATE-ATE'
                titel = ('Dose response relative to 0 ' + iate_label + ' '
                         + var_dic['y_name'][idx_y])
                index_t = [i + no_of_iate_y * idx_y for i in index_0]
                name_iate_t = [iate_pot_name[name_eff][idx] for idx in index_t]
                iate_temp = data_df[name_iate_t].to_numpy()
                indices_sort = np.argsort(np.mean(iate_temp, axis=1))
                iate_temp = iate_temp[indices_sort]
                z_plt = np.transpose(iate_temp)
                fig, axe = plt.subplots(subplot_kw={"projection": "3d"})
                x_plt, y_plt = np.meshgrid(np.arange(z_plt.shape[1]) + 1,
                                           d_values[1:])
                surf = axe.plot_surface(x_plt, y_plt, z_plt, cmap='coolwarm_r',
                                        linewidth=0, antialiased=False)
                plt.title(titel)
                axe.set_ylabel('Treatment levels')
                axe.set_zlabel(iate_label)
                axe.set_xlabel('Index of sorted IATEs')
                fig.colorbar(surf, shrink=0.5, aspect=5)
                ttt = titel.replace(' ', '')
                file_name_jpeg = (p_dic['ate_iate_fig_pfad_jpeg']
                                  + '/' + ttt + '.jpeg')
                file_name_pdf = (p_dic['ate_iate_fig_pfad_pdf']
                                 + '/' + ttt + '.pdf')
                if post_dic['plots']:
                    mcf_sys.delete_file_if_exists(file_name_jpeg)
                    mcf_sys.delete_file_if_exists(file_name_pdf)
                    fig.savefig(file_name_jpeg, dpi=int_dic['dpi'])
                    fig.savefig(file_name_pdf, dpi=int_dic['dpi'])
                if post_dic['plots']:
                    plt.show()
                else:
                    plt.close()
                if imate == iterator[-1]:
                    figure_list.append(file_name_jpeg)
    ps.print_mcf(gen_dic, txt, summary=True)
    return figure_list


def k_means_of_x_iate(mcf_, results_prev):
    """Compute kmeans."""
    results = deepcopy(results_prev)
    gen_dic, post_dic = mcf_.gen_dict, mcf_.post_dict
    var_dic, var_x_type = mcf_.var_dict, mcf_.var_x_type
    if (post_dic['relative_to_first_group_only']
            or gen_dic['d_type'] == 'continuous'):
        iate_pot_name = results['iate_names_dic'][1]
    else:
        iate_pot_name = results['iate_names_dic'][0]
    names_unordered = [xn for xn in var_x_type.keys()
                       if var_x_type[xn] > 0]
    data_df = results['iate_data_df']
    if mcf_.lc_dict['uncenter_po']:
        y_pot = data_df[iate_pot_name['names_y_pot_uncenter']]
    else:
        y_pot = data_df[iate_pot_name['names_y_pot']]
    x_name = delete_x_with_catv(var_x_type.keys())
    x_dat = data_df[x_name]

    iate_name_all = iate_pot_name['names_iate']
    no_of_kmeans = len(iate_name_all) + 1 if post_dic['k_means_single'] else 1
    for val_idx in range(no_of_kmeans):
        if val_idx == 0:
            iate_names_cluster = iate_pot_name['names_iate'].copy()
            joint = True
        else:
            iate_names_cluster = [iate_pot_name['names_iate'][val_idx-1]]
            joint = False
        iate = data_df[iate_names_cluster]
        x_name = x_dat.columns.tolist()
        iate_np = iate.to_numpy()
        silhouette_avg_prev = -1
        txt = '\n' + '=' * 100 + '\nK-Means++ clustering '
        if joint:
            txt += '(all effects jointly)'
        else:
            txt += (f'({iate_names_cluster} only)')
        txt += '\n' + '-' * 100
        for cluster_no in post_dic['kmeans_no_of_groups']:
            cluster_lab_tmp = KMeans(
                n_clusters=cluster_no,
                n_init=post_dic['kmeans_replications'], init='k-means++',
                max_iter=post_dic['kmeans_max_tries'], algorithm='lloyd',
                random_state=42, tol=1e-5, verbose=0, copy_x=True
                ).fit_predict(iate_np)
            smallest_cluster_size = np.min(np.bincount(cluster_lab_tmp))
            n_min = len(cluster_lab_tmp)/100  # Clustersize should be > 1%
            if smallest_cluster_size > n_min:
                silhouette_avg = silhouette_score(iate_np, cluster_lab_tmp)
                cluster_too_small = False
            else:
                silhouette_avg = -100000
                cluster_too_small = True
            txt += (f'\nNumber of clusters: {cluster_no}   '
                    f'Average silhouette score: {silhouette_avg: 8.3f}')
            if cluster_too_small:
                txt += (f' Smallest cluster has only {smallest_cluster_size} '
                        'observations. Average silhouette score set to -100000.'
                        )
            if silhouette_avg > silhouette_avg_prev:
                cluster_lab_np = np.copy(cluster_lab_tmp)
                silhouette_avg_prev = np.copy(silhouette_avg)
        txt += ('\n\nBest value of average silhouette score:'
                f' {silhouette_avg_prev: 8.3f}')

        del iate_np
        # Reorder labels for better visible inspection of results
        iate_name = iate_names_cluster
        namesfirsty = iate_name[0:round(len(iate_name)/len(var_dic['y_name']))]
        cl_means = iate[namesfirsty].groupby(by=cluster_lab_np).mean(
            numeric_only=True)
        cl_means_np = cl_means.to_numpy()
        cl_means_np = np.mean(cl_means_np, axis=1)
        sort_ind = np.argsort(cl_means_np)
        cl_group = cluster_lab_np.copy()
        for cl_j, cl_old in enumerate(sort_ind):
            cl_group[cluster_lab_np == cl_old] = cl_j
        txt += ('\n' + '- ' * 50 +
                '\nEffects are ordered w.r.t. to the size of the effects'
                ' for the first outcome and first treatment.')
        cl_values, cl_obs = np.unique(cl_group, return_counts=True)
        txt += '\n' + '-' * 100 + '\nNumber of observations in the clusters'
        if joint:
            txt_report = '\n' * 2 + 'Number of observations in the clusters\n'
        txt += '\n' + '- ' * 50
        for idx, val in enumerate(cl_values):
            string = f'\nCluster {val:2}: {cl_obs[idx]:6} '
            txt += string
            if joint:
                txt_report += string
        if joint:
            report_dic = {'obs_cluster_str': txt_report}
        txt += '\n' + '-' * 100 + '\nEffects\n' + '- ' * 50
        if joint:
            txt_report += '\n' * 2 + 'Effects\n'
            iate_cluster_id_name = 'IATE_Cluster'
        else:
            iate_cluster_id_name = 'IATE_Cluster_' + iate_name[0]
        data_df[iate_cluster_id_name] = cl_group  # Add cluster as new variable
        cl_means = iate.groupby(by=cl_group).mean(numeric_only=True)
        if joint:
            report_dic['IATE_df'] = np.round(cl_means.transpose().copy(), 2)
        txt += '\n' + cl_means.transpose().to_string()
        txt += '\n' + '-' * 100 + '\nPotential outcomes\n' + '- ' * 50
        cl_means = y_pot.groupby(by=cl_group).mean(numeric_only=True)
        if joint:
            report_dic['PotOutcomes_df'] = np.round(cl_means.transpose(), 2)
        txt += '\n' + cl_means.transpose().to_string()
        txt += '\n' + '-' * 100 + '\nCovariates\n' + '- ' * 50

        names_unordered = [xn for xn in var_x_type.keys()
                           if var_x_type[xn] > 0]
        if names_unordered:  # List is not empty
            x_dummies = pd.get_dummies(x_dat, columns=names_unordered,
                                       dtype=int)
            x_km = pd.concat([x_dat[names_unordered], x_dummies], axis=1)
        else:
            x_km = x_dat
        cl_means = x_km.groupby(by=cl_group).mean(numeric_only=True)
        ps.print_mcf(gen_dic, txt, summary=True)
        all_names = cl_means.columns.copy()
        cl_means.columns = [name.replace('_prime', '') for name in all_names]
        all_names = cl_means.columns.copy()
        cl_means.columns = [name.replace('_prime_', '_D') for name in all_names]
        all_names = cl_means.columns.copy()
        cl_means.columns = [name.replace('.0', '') for name in all_names]
        if joint:
            report_dic['Features_df'] = np.round(cl_means.transpose(), 2)
        txt = cl_means.transpose().to_string() + '\n' + '-' * 100
        pd.set_option('display.max_rows', 1000, 'display.max_columns', 100)
        ps.print_mcf(gen_dic, txt, summary=True)
        pd.set_option('display.max_rows', None, 'display.max_columns', None)
        txt = '\nSaving cluster indicator from k-means clustering.\n'
        ps.print_mcf(gen_dic, txt, summary=False)
    results['iate_data_df'] = data_df
    return results, report_dic


def random_forest_of_iate(mcf_, results):
    """Analyse IATEs by Random Forest."""
    gen_dic, var_x_type = mcf_.gen_dict, mcf_.var_x_type
    if (mcf_.post_dict['relative_to_first_group_only']
            or gen_dic['d_type'] == 'continuous'):
        iate_pot_name = results['iate_names_dic'][1]
    else:
        iate_pot_name = results['iate_names_dic'][0]
    names_unordered = [xn for xn in var_x_type.keys()
                       if var_x_type[xn] > 0]
    x_name = delete_x_with_catv(var_x_type.keys())
    no_of_names = len(iate_pot_name['names_iate'])
    eva_points = eva_points_fct(no_of_names, len(mcf_.var_dict['y_name']))
    data_df = results['iate_data_df']
    x_dat = data_df[x_name]
    iate = data_df[iate_pot_name['names_iate']]
    x_name = x_dat.columns.tolist()
    dummy_group_names = []
    txt = '\n' + '=' * 100 + '\nRandom Forest Analysis of IATES\n' + 50 * '- '
    ps.print_mcf(gen_dic, txt, summary=False)
    txt = ''
    if names_unordered:  # List is not empty
        dummy_names = []
        replace_dict = dict(zip(mcf_gp.primes_list(1000),
                                list(range(1000))))
        for name in names_unordered:
            x_t_d = x_dat[name].replace(replace_dict)
            x_t_d = pd.get_dummies(x_t_d, prefix=name, dtype=int)
            this_dummy_names = x_t_d.columns.tolist()
            dummy_names.extend(this_dummy_names[:])
            this_dummy_names.append(name)
            dummy_group_names.append(this_dummy_names[:])
            x_dat = pd.concat([x_dat, x_t_d], axis=1)
        x_name.extend(dummy_names)
        txt += ('The following dummy variables have been created'
                f'{dummy_names}')
    x_train = x_dat.to_numpy(copy=True)
    txt += '\nFeatures used to build random forest'
    txt += f'\n{x_dat.describe()}\n'
    ps.print_mcf(gen_dic, txt, summary=False)
    txt = '=' * 100
    for idx, y_name in enumerate(iate_pot_name['names_iate']):
        if gen_dic['d_type'] == 'continuous' and idx not in eva_points:
            continue
        txt += f'\nPost estimation random forests for {y_name}'
        y_train = iate[y_name].to_numpy(copy=True)
        (_, _, _, _, _, _, _, txt_rf) = mcf_est_g.random_forest_scikit(
            x_train, y_train, None, x_name=x_name, y_name=y_name,
            boot=mcf_.cf_dict['boot'], n_min=2, no_features='sqrt',
            max_depth=None, workers=gen_dic['mp_parallel'], alpha=0,
            var_im_groups=dummy_group_names, max_leaf_nodes=None,
            pred_p_flag=False, pred_t_flag=True, pred_oob_flag=True,
            with_output=True, variable_importance=True,
            pred_uncertainty=False, pu_ci_level=mcf_.p_dict['ci_level'],
            pu_skew_sym=0.5, var_im_with_output=True)
        txt += txt_rf
    ps.print_mcf(gen_dic, txt, summary=True)


def eva_points_fct(no_of_names, no_of_y_name):
    """Get evaluation points."""
    no_of_names_y = round(no_of_names / no_of_y_name)
    eva_points_y = [round(no_of_names_y / 3), round(2 * no_of_names_y / 3)]
    eva_points = []
    for idx in range(no_of_y_name):
        eva_points_t = [i + no_of_names_y * idx for i in eva_points_y]
        eva_points.extend(eva_points_t)
    return eva_points


def delete_x_with_catv(names_with_catv):
    """Delete variables which end with catv from list."""
    return [x_name for x_name in names_with_catv
            if not x_name.endswith('catv')]
