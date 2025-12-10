"""
Created on Tue Jul 18 07:58:29 2023.

Provide functions for the evaluation of allocations.

# -*- coding: utf-8 -*-

@author: MLechner
"""
from copy import deepcopy
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mcf.mcf_estimation_generic_functions import bayes_factor, cramers_v
from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats_functions as mcf_ps
from mcf import mcf_variable_importance_functions as mcf_vi

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy


def evaluate_fct(optp_: 'OptimalPolicy',
                 data_df: pd.DataFrame,
                 allocation_df: pd.DataFrame,
                 d_ok: bool,
                 polscore_ok: bool,
                 polscore_desc_ok: bool,
                 desc_var: list[str] | None,
                 data_title: str
                 ) -> dict:
    """Evalute allocations."""
    var_cfg, gen_cfg = deepcopy(optp_.var_cfg), optp_.gen_cfg
    # len_scores = optp_.number_scores
    allocations = allocation_df.columns
    results_dic = {}
    population = ['All', 'Switchers'] if d_ok else ['All']
    value, index = value_to_index_dic_fct(allocation_df['random'].to_numpy())

    # Adjust variables if fairness transformed scores have been used
    if (not polscore_ok
        and (optp_.fair_cfg.solvefair_used or optp_.estriskcfg.estrisk_used)
            and polscore_desc_ok):
        # Since polscore_ok == False, scores have been transformed
        # Take polscore name back from last elements in polscore_desc_name
        var_cfg.polscore_name = (
            var_cfg.polscore_desc_name[-gen_cfg.no_of_treat:]
            )
        if not var_cfg.polscore_desc_name:
            polscore_desc_ok = False
        polscore_ok, adjust_score = True, True
    else:
        adjust_score = False

    if polscore_ok:
        scores = data_df[var_cfg.polscore_name].to_numpy()
    else:
        scores = None

    if polscore_desc_ok:
        scores_add = data_df[var_cfg.polscore_desc_name].to_numpy()
        no_of_vars = np.int16(np.around(scores_add.shape[1]
                                        / gen_cfg.no_of_treat)
                              )
        scores_add_idx = np.int16(
            np.reshape(np.arange(scores_add.shape[1]),
                       (no_of_vars, gen_cfg.no_of_treat)))
    else:
        scores_add = None

    if optp_.fair_cfg.solvefair_used:
        protected_var = var_cfg.protected_ord_name.copy()
        protected_var.extend(var_cfg.protected_unord_name)
        protected_df = data_df[protected_var]
    else:
        protected_df = None
    titel = True
    scores_, scores_add_ = scores, scores_add
    if polscore_ok and optp_.gen_cfg.method in ('policy_tree',
                                                'bps_classifier',
                                                ):
        alloc_name_ref_qini = ['best ATE', 'random']
        if d_ok:
            alloc_name_ref_qini.append('observed')

        match optp_.gen_cfg.method:
            case 'policy_tree':    alloc_name_qini = 'Policy Tree'
            case 'bps_classifier': alloc_name_qini = 'bps_classif_'
            case _:                alloc_name_qini = None

        qini_plots = True
        alloc_qini_np, score_qini_np, name_qini = [], [], []
        alloc_ref_qini_np, score_ref_qini_np, name_ref_qini = [], [], []
    else:
        qini_plots = False
        alloc_qini_np = score_qini_np = name_qini = alloc_name_qini = None
        alloc_ref_qini_np = score_ref_qini_np = name_ref_qini = None

    for pop in population:
        for alloc_name in allocations:
            alloc_np = np.int32(np.round(allocation_df[alloc_name].to_numpy()))
            if pop == 'Switchers':
                if alloc_name == 'observed':
                    continue
                switcher = alloc_np != np.int32(np.round(
                    allocation_df['observed'].to_numpy()))
                alloc_ = alloc_np[switcher]
                if polscore_ok:
                    scores_ = scores[switcher]
                if polscore_desc_ok:
                    scores_add_ = scores_add[switcher]
            else:
                alloc_, scores_, = alloc_np, scores
                scores_add_ = scores_add
            len_scores = len(scores_) if polscore_ok else len(scores_add_)
            indices_ = alloc_.copy()
            for idx, val in enumerate(value):
                indices_ = np.where(alloc_ == val, index[idx], indices_)
            if polscore_ok:
                score_sel = scores_[np.arange(len_scores), indices_]
                score_sel_new_df = pd.DataFrame(
                    score_sel, columns=(var_cfg.polscore_name[0]+'alloc',))
            else:
                score_sel_new_df = None
            if polscore_desc_ok:
                scores_add_sel = np.zeros((len(scores_add_), no_of_vars))
                score_all_sel_name = []
                for idx, val_idx in enumerate(scores_add_idx):
                    scores_add_k = scores_add_[:, val_idx]
                    scores_add_sel[:, idx] = scores_add_k[
                        np.arange(len_scores), indices_]
                    score_all_sel_name.append(
                        var_cfg.polscore_desc_name[val_idx[0]]+'alloc')
                score_add_df = pd.DataFrame(scores_add_sel,
                                            columns=score_all_sel_name)
            else:
                scores_add_sel = score_all_sel_name = score_add_df = None
            desc_new_var_df_ = score_add_df if polscore_desc_ok else None
            results_dic = evaluation_of_alloc(
                gen_cfg, results_dic, pop, alloc_name,
                score_sel_new_df, desc_new_var_df_, alloc_, titel=titel,
                protected_df=protected_df, transformed_scores=adjust_score,
                cont_min_values=optp_.fair_cfg.cont_min_values
                )
            titel = False

            if qini_plots and pop != 'Switchers':
                if alloc_name in alloc_name_ref_qini:
                    alloc_ref_qini_np.append(alloc_np)
                    score_ref_qini_np.append(score_sel)
                    name_ref_qini.append(alloc_name)
                elif alloc_name.startswith(alloc_name_qini):
                    alloc_qini_np.append(alloc_np)
                    score_qini_np.append(score_sel)
                    name_qini.append(alloc_name)

    if gen_cfg.with_output:
        mcf_ps.print_mcf(gen_cfg, '-' * 100, summary=True)
        if polscore_ok:
            mcf_ps.print_mcf(
                gen_cfg,
                'Standard error of mean outcome in brackets. Note that this '
                'standard error does only reflect the variability '
                '\nin the evaluation data for a given assignment rule.'
                '\nThe variability in the training data when learning the '
                'assignment rule is ignored.',
                summary=True
                )
            mcf_ps.print_mcf(gen_cfg, '-' * 100, summary=True)

        if optp_.fair_cfg.solvefair_used:
            txt_fair = txt_fairness_statistics(results_dic)
            txt = '\n' + '-' * 100 + '\n' + txt_fair + '\n' + '-' * 100
            mcf_ps.print_mcf(gen_cfg, txt, summary=True)

        if desc_var:
            data_new_df = data_df[desc_var].copy()
            for pop in population:
                for alloc_name in allocations:
                    alloc_np = np.int32(
                        np.round(allocation_df[alloc_name].to_numpy()))
                    if pop == 'Switchers':
                        if alloc_name == 'observed':
                            continue
                        switcher = alloc_np != np.int32(np.round(
                            allocation_df['observed'].to_numpy()))
                        alloc_ = alloc_np[switcher]
                        data_new_df_ = data_new_df[switcher].copy()
                    else:
                        alloc_, data_new_df_ = alloc_np, data_new_df.copy()
                    data_new_df_[alloc_name] = alloc_
                    txt = '\n' + '-' * 100 + '\nDescriptive statistics of '
                    txt += f'features for {alloc_name} '
                    if data_title:
                        txt += f'({data_title})'
                    if pop == 'Switchers':
                        txt += ' for switchers'
                    txt += '\n' + '-' * 100
                    summary = not alloc_name == 'random_rest'
                    mcf_ps.print_mcf(gen_cfg, txt, summary=summary)
                    mcf_ps.statistics_by_treatment(
                        gen_cfg,
                        data_new_df_,
                        [alloc_name],
                        desc_var,
                        only_next=False,
                        summary=summary,
                        median_yes=False,
                        std_yes=False,
                        balancing_yes=False,
                        data_train_dic=None
                        )

        if qini_plots:
            qini_dict = {'alloc_ref_list': alloc_ref_qini_np,
                         'score_ref_list': score_ref_qini_np,
                         'name_ref_list': name_ref_qini,
                         'alloc_list': alloc_qini_np,
                         'score_list': score_qini_np,
                         'name_list': name_qini,
                         'data_title': data_title
                         }
            qini_plots_fct(optp_, qini_dict)

    return results_dic


def txt_fairness_statistics(results_dic: dict) -> str:
    """Print the fairness statistics."""
    txt = ('Fairness corrections: Dependence of allocations on protected '
           'attributes using different measures')

    first_time = True
    length_var, length_alloc = 8, 46
    for key, item in results_dic.items():
        if 'correlation(allocation, protected)' not in item:
            continue
        if first_time:  # Add header
            first_time = False
            measures = ['Correlation (%)', 'Cramers V (%)', 'Bayes Factor']
            longest_measure = max(len(s) for s in measures)

            variables = item['correlation(allocation, protected)'].columns
            if len(variables) == 1:
                length_var = max(longest_measure, length_var)
            variables_str = ' '.join([a[:length_var].rjust(length_var)
                                      for a in variables[:]])
            measures_txt = '  '.join([s.rjust((length_var) * len(variables) - 1)
                                     for s in measures])
            txt += ('\n' + ' ' * (length_alloc + 1) + measures_txt
                    + '\n' + ' ' * length_alloc + variables_str * len(measures)
                    )
        txt += f'\n{key:{length_alloc}}'
        stat = (item['correlation(allocation, protected)'].iloc[0] * 100
                ).tolist()
        txt += f'{" ".join([str(round(x, 2)).rjust(length_var) for x in stat])}'
        stat = (item['Cramers V (allocation, protected)'].iloc[0] * 100
                ).tolist()
        txt += f'{" ".join([str(round(x, 2)).rjust(length_var) for x in stat])}'
        stat_ = item['Bayes factor (allocation, protected)'].iloc[0].tolist()
        stat = [min(s, 10000) if not np.isnan(s) else s for s in stat_]
        txt += f'{" ".join([str(round(x, 2)).rjust(length_var) for x in stat])}'
    txt += '\n' + '- ' * int(((length_alloc + 1) + len(measures_txt)) / 2)
    txt += ("\nNote: Cramér's V is a statistic to measure the association of "
            'two categorical variables. '
            '\n  It varies between 0 '
            '(no association) and  1 (perfect association). '
            '\nThe Bayes Factor (BF10) is a (nonnegativ) statistic to measure '
            'the association of two categorical variables. '
            '\n  A value much larger than 1 indicates strong evidence for '
            'dependence. '
            '\n  A value much smaller than 1, it provides evidence for '
            'independence.'
            '\n  A value close to 0 means an inclusive result. '
            '\n  The Bayes factor is truncated at 10000. '
            "\nCramér's V and the Bayes Factor are NOT computed for "
            'continuous variables.')

    return txt


def evaluation_of_alloc(gen_cfg: Any,
                        results_dic: dict,
                        pop: str,
                        alloc_name: str,
                        score_sel_new_df: pd.DataFrame,
                        desc_new_var_df: pd.DataFrame,
                        allocation: np.array,
                        titel: bool = False,
                        protected_df: pd.DataFrame = None,
                        transformed_scores: bool = False,
                        cont_min_values: int | float = 50,
                        ):
    """Compute and print results of evaluations."""
    results_new_dic = deepcopy(results_dic)
    local_dic = {}
    score_ok = score_sel_new_df is not None
    if desc_new_var_df is not None:
        desc_var = desc_new_var_df.columns
    else:
        desc_var = []

    if transformed_scores:
        if len(desc_var) > 1:
            desc_var_print = desc_var[:-1]
        else:
            desc_var_print = []
    else:
        desc_var_print = desc_var
    txt = ''

    if titel:
        scorevar = score_sel_new_df.columns if score_ok else None
        if score_ok:
            txt += ('\n' + '-' * 100 + '\n' + 'Mean of variables /'
                    ' treatment shares' + ' ' * 6
                    )
            if not transformed_scores:
                txt += f'Welfare measure used: {scorevar[0]:10}'
        else:
            txt += ('\n' + '-' * 100 + '\n' + 'Mean of variables /'
                    ' treatment shares')

        txt += '\n'
        if desc_new_var_df is not None:
            txt += 'Main outcome, additional outcomes, shares:        '
            if scorevar is not None:
                txt += f'{scorevar[0]:16}'
            for var in desc_var_print:
                txt += f' {var:10s}'
        txt += ''.join([f'{s:8d}' for s in gen_cfg.d_values])
        txt += '\n'
    name = pop + ' ' + alloc_name
    if score_ok:
        mean1, std1 = get_mean_std_welfare(score_sel_new_df,
                                           no_of_bootstraps=1000,
                                           seed=12345)
        txt += f'{name:46} {mean1:10.4f} ({std1:.4f})'
        local_dic[name] = mean1
    else:
        txt += f'{name:46}'
        mean1 = []
    if desc_new_var_df is not None:
        for var in desc_var:
            variance_avg = desc_new_var_df[var].mean()
            if var in desc_var_print:
                txt += f' {variance_avg:10.4f}'
            local_dic[var] = variance_avg
    local_dic['treatment share'] = get_treat_shares(allocation,
                                                    gen_cfg.d_values)
    local_dic['number of observations'] = len(allocation)
    treat_shares_s = [f' {s:>7.2%}' for s in local_dic['treatment share']]
    txt += ' ' * 7 + ''.join(treat_shares_s)
    if gen_cfg.with_output:
        mcf_ps.print_mcf(gen_cfg, txt, summary=True)

    if protected_df is not None and pop == 'All':
        corr, cramer, bayes = dependence_allocation_variables(allocation,
                                                              protected_df,
                                                              cont_min_values
                                                              )
        local_dic['correlation(allocation, protected)'] = corr
        local_dic['Cramers V (allocation, protected)'] = cramer
        local_dic['Bayes factor (allocation, protected)'] = bayes

    results_new_dic[name] = local_dic.copy()

    return results_new_dic


def dependence_allocation_variables(allocation: NDArray[Any],
                                    protected_df: pd.DataFrame,
                                    no_continuous: int | float = 50,
                                    ) -> tuple[pd.DataFrame, ...]:
    """Compute dependence measures for discrete variables."""
    prot_var = protected_df.columns
    prot_np = protected_df.to_numpy()
    data = np.concatenate((allocation.reshape(-1, 1), prot_np), axis=1)
    if allocation.std() > 1e-8:
        corr = np.corrcoef(data, rowvar=False)[0, 1:].reshape(-1, 1).T
    else:
        corr = np.full((1, prot_np.shape[1]), np.nan).reshape(-1, 1).T
    corr_df = pd.DataFrame(corr, columns=prot_var)

    cramer = np.empty(len(prot_var))
    bayes = np.empty_like(cramer)
    for idx, _ in enumerate(prot_var):
        if ((len(np.unique(prot_np[:, idx])) > no_continuous)
                or allocation.std() < 1e-8):
            cramer[idx] = np.nan
            bayes[idx] = np.nan
        else:
            prot_df = pd.Series(prot_np[:, idx])
            allocation_df = pd.Series(allocation)
            cramer[idx] = cramers_v(allocation_df, prot_df)
            bayes[idx] = bayes_factor(allocation_df, prot_df)
    cramer_df = pd.DataFrame(cramer.reshape(-1, 1).T, columns=prot_var)
    bayes_df = pd.DataFrame(bayes.reshape(-1, 1).T, columns=prot_var)

    return corr_df, cramer_df, bayes_df


def get_mean_std_welfare(score_df: pd.DataFrame,
                         no_of_bootstraps: int = 100,
                         seed: int = 12345
                         ) -> NDArray[Any]:
    """Compute mean and bootstrapped standard errors."""
    score_np = score_df.to_numpy()
    obs = len(score_np)
    mean = score_np.mean()
    rng = np.random.default_rng(seed=seed)
    idx_boot = rng.integers(low=0, high=obs, size=(obs, no_of_bootstraps))
    scores_boot = score_np[idx_boot]
    means_boot = scores_boot.mean(axis=0)
    std_mean = means_boot.std()

    return mean, std_mean


def get_treat_shares(allocation: NDArray[Any],
                     d_values: list | NDArray[Any]
                     ) -> NDArray[Any]:
    """Get treatment shares."""
    values_, treat_counts = np.unique(allocation, return_counts=True)
    treat_shares_ = treat_counts / len(allocation)
    treat_shares = np.zeros(len(d_values))
    jdx = 0
    if list(values_):
        for idx, vals in enumerate(d_values):
            if jdx < len(values_) and values_[jdx] == vals:
                treat_shares[idx] = treat_shares_[jdx]
                jdx += 1

    return treat_shares


def evaluate_multiple(optp_: 'OptimalPolicy',
                      allocations_dic: dict,
                      scores_df: pd.DataFrame
                      ) -> None:
    """Compute and print results for multiple allocation."""
    txt = ('\n' * 2 + '-' * 100 + '\n' + f'{"Allocation":12s}'
           + 'Shares:  '
           + ' '.join([f'{s:7d}' for s in optp_.gen_cfg.d_values])
           + '     Sum / Mean / Variance of Potential Outcomes')
    scores_np = scores_df.to_numpy()
    for alloc in allocations_dic:
        allocation = np.int32(np.round(allocations_dic[alloc].to_numpy()))
        if allocation.shape[1] > 1:
            allocation = allocation[:, 0]
        if (np.min(allocation) < min(optp_.gen_cfg.d_values)
                or np.max(allocation) < max(optp_.gen_cfg.d_values)):
            raise ValueError('Inconistent definition of treatment values')
        treat_shares = get_treat_shares(allocation, optp_.gen_cfg.d_values)
        index = np.int32(np.arange(len(optp_.gen_cfg.d_values)))
        indices_ = allocation.copy()
        idx_row = np.arange(len(scores_np))
        for idx, val in enumerate(np.int32(optp_.gen_cfg.d_values)):
            indices_ = np.where(allocation == val, index[idx], indices_)
        score_sel = scores_np[idx_row, indices_]
        mean, variance = np.mean(score_sel), np.var(score_sel)
        summ = np.sum(score_sel)
        txt += f'\n{alloc:20s}: '
        treat_shares_s = [f'{s:>8.2%}' for s in treat_shares]
        txt += ' ' + ''.join(treat_shares_s)
        txt += f'  {summ:12.2f}  {mean:8.4f}  {variance:8.4f}'
    txt += '\n' + '-' * 100
    mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=True)


def variable_importance(optp_: 'OptimalPolicy',
                        data_df: pd.DataFrame,
                        allocation_df: pd.DataFrame,
                        seed: int = 1234567,
                        data_title: str = ''
                        ) -> None:
    """Compute variable importance measures for various allocations."""
    names_unordered = optp_.var_cfg.vi_to_dummy_name
    for alloc_name in allocation_df.columns:
        if alloc_name in ('random', 'RANDOM'):
            continue
        txt = ('\n' * 2 + '-' * 100 + '\n'
               + f'Variable importance statistic for {alloc_name} '
               )
        if isinstance(data_title, str) and len(data_title) > 0:
            txt += f'({data_title})'
        txt += '\n' + '- ' * 50
        mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=True)

        vtd = optp_.var_cfg.vi_to_dummy_name
        vx = optp_.var_cfg.vi_x_name
        match (vtd, vx):
            case (None | [], _): x_name = vx
            case (_, None | []): x_name = vtd
            case _:              x_name = vtd + vx

        if x_name is None or len(x_name) == 1:
            continue
        x_df = data_df[x_name].copy()
        d_df = allocation_df[alloc_name]
        dummy_names = {}
        if names_unordered:
            for idx, name in enumerate(names_unordered):
                x_dummies = pd.get_dummies(x_df[name], columns=[name],
                                           dtype=int)
                x_dummies_names = [name + str(dummy)
                                   for dummy in x_dummies.columns]
                dummy_names[name] = x_dummies.columns = x_dummies_names
                x_dummies_s = x_dummies.reindex(sorted(x_dummies.columns),
                                                axis=1)
                if idx == 0:
                    x_all_dummies = x_dummies_s
                else:
                    x_all_dummies = pd.concat([x_all_dummies, x_dummies_s],
                                              axis=1, copy=True)
            names_wo = [name for name in x_df.columns
                        if name not in names_unordered]
            x_new_df = pd.concat((x_df[names_wo], x_all_dummies), axis=1,
                                 copy=True)
        else:
            x_new_df = x_df
        classif = RandomForestClassifier(
            n_estimators=1000,
            max_features='sqrt',
            bootstrap=True,
            oob_score=False,
            n_jobs=optp_.gen_cfg.mp_parallel,
            random_state=seed,
            verbose=False,
            min_samples_split=5)

        mcf_vi.print_variable_importance(
            classif, x_new_df, d_df, x_name, names_unordered, dummy_names,
            optp_.gen_cfg, summary=True, classification=True,
            )


def value_to_index_dic_fct(values_np: NDArray[Any]):
    """Array of indices of values in numpy array."""
    vals = np.int32(np.round(np.unique(values_np)))
    indices = np.int32(np.arange(len(vals)))
    return vals, indices


def get_random_allocation(optp_: 'OptimalPolicy',
                          no_obs: int,
                          seed: int
                          ) -> NDArray[Any]:
    """Obtain random allocation."""
    rng = np.random.default_rng(seed=seed)
    allocation_np = rng.choice(optp_.gen_cfg.d_values, size=no_obs,
                               p=optp_.rnd_cfg.shares
                               )
    return allocation_np


def get_best_ate_allocation(optp_: 'OptimalPolicy',
                            data_df: pd.DataFrame
                            ) -> NDArray[Any]:
    """Get allocation by sending everybody to treatment with highest ATE."""
    pol_score = data_df[optp_.var_cfg.polscore_name]
    avg_polscore = pol_score.mean(axis=0)
    best_treat_idx = avg_polscore.argmax()
    allocation_np = (np.ones(len(data_df))
                     * optp_.gen_cfg.d_values[best_treat_idx])

    return allocation_np


def qini_plots_fct(optp_: 'OptimalPolicy', qini_dict: dict) -> None:
    """Compute and show Quasi-Qini plots."""
    # Plot compare share of a share of alpha % of individuals is treated
    # optimally while the rest is treated as suggested by reference rule
    obs = len(qini_dict['score_ref_list'][0])
    obs_cum = np.arange(obs) + 1
    share_optimally_treated = obs_cum / obs * 100
    data_title = qini_dict['data_title']
    for idx_ref, name_ref in enumerate(qini_dict['name_ref_list']):
        welfare_reference = (qini_dict['score_ref_list'][idx_ref].mean()
                             * np.ones(obs))

        for idx, name in enumerate(qini_dict['name_list']):
            score_combined = qini_like_score(
                qini_dict['score_list'][idx],
                qini_dict['score_ref_list'][idx_ref])

            figs, axs = plt.subplots()
            axs.plot(share_optimally_treated,
                     score_combined,
                     label=name,
                     color='r'
                     )
            axs.plot(share_optimally_treated,
                     welfare_reference,
                     label=name_ref,
                     color='b'
                     )
            axs.set_ylabel('Average welfare')
            axs.set_xlabel('Share optimally treated '
                           '(reference rule applied to rest)'
                           )
            axs.legend(loc=optp_.int_cfg.legend_loc,
                       shadow=True,
                       fontsize=optp_.int_cfg.fontsize
                       )
            axs.set_title(
                f'Welfare of {name} rule \nvs. {name_ref} ({data_title} data)')
            name_title = (name_ref.replace(' ', '')
                          + name.replace(' ', '')
                          + data_title.replace(' ', '')
                          )
            file_jpeg = optp_.gen_cfg.outpath / (name_title + '.jpeg')
            file_pdf = optp_.gen_cfg.outpath / (name_title + '.pdf')
            file_csv = optp_.gen_cfg.outpath / (name_title + '.csv')
            mcf_sys.delete_file_if_exists(file_jpeg)
            mcf_sys.delete_file_if_exists(file_pdf)
            figs.savefig(file_jpeg, dpi=optp_.int_cfg.dpi)
            figs.savefig(file_pdf, dpi=optp_.int_cfg.dpi)
            plt.show()
            plt.close()
            # Save data for plot
            all_data_np = np.concatenate(
                (share_optimally_treated.reshape(-1, 1),
                 score_combined.reshape(-1, 1),
                 welfare_reference.reshape(-1, 1)),
                axis=1)
            all_data_names = ('Share', 'combined', 'reference',)
            all_data_plot_df = pd.DataFrame(data=all_data_np,
                                            columns=all_data_names)
            all_data_plot_df.to_csv(file_csv)


def qini_like_score(score: NDArray[Any],
                    score_ref: NDArray[Any]
                    ) -> NDArray[Any]:
    """Compute intertwined welfare measures for qini like measure."""
    gain = score - score_ref    # Used for sorting only
    obs = len(gain)
    sorted_indices = np.argsort(gain)[::-1]  # descending order
    score_sorted = score[sorted_indices]
    score_ref_sorted = score_ref[sorted_indices]
    score_combined = np.empty(obs)
    for idx in range(obs-1):
        score_combined[idx] = (score_sorted[:idx+1].sum()
                               + score_ref_sorted[idx+1:].sum()) / obs
    score_combined[-1] = score_sorted.mean()

    return score_combined
