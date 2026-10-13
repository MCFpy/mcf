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

from mcf import mcf_general_sys as mcf_sys
from mcf import mcf_print_stats as mcf_ps
from mcf import mcf_variable_importance as mcf_vi
from mcf.mcf_estimation_generic import bayes_factor, cramers_v

from mcf.optpolicy_data import var_available

if TYPE_CHECKING:
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.optpolicy_init import GenCfg


def evaluate_fct(optp_: 'OptimalPolicy',
                 data_df: pd.DataFrame,
                 allocation_df: pd.DataFrame, *,
                 d_ok: bool,
                 polscore_ok: bool,
                 polscore_desc_ok: bool,
                 polscore_desc_name: list[str] | None,
                 desc_var: list[str] | None,
                 data_title: str
                 ) -> dict:
    """Evalute allocations."""
    var_cfg, gen_cfg = deepcopy(optp_.var_cfg), optp_.gen_cfg
    var_cfg.polscore_desc_name = polscore_desc_name
    allocations = allocation_df.columns
    results_dic = {}
    population = ['All', 'Switchers'] if d_ok else ['All']
    value = np.asarray(gen_cfg.d_values, dtype=np.int32)
    index = np.arange(len(value), dtype=np.int32)
    costs = np.asarray(optp_.other_cfg.costs_of_treat, dtype=float)
    if costs.shape != (len(value),):
        raise ValueError('Number of treatment costs is inconsistent with treatment values.')

    # Adjust variables if fairness transformed scores have been used
    if (not polscore_ok
        and (optp_.fair_cfg.solvefair_used or optp_.estriskcfg.estrisk_used)
            and polscore_desc_ok
            ):
        # Since polscore_ok == False, scores have been transformed
        # Take polscore name back from last elements in polscore_desc_name
        var_cfg.polscore_name = var_cfg.polscore_desc_name[-gen_cfg.no_of_treat:]
        if not var_cfg.polscore_desc_name:
            polscore_desc_ok = False
        polscore_ok, adjust_score = True, True
    else:
        adjust_score = False

    if polscore_ok:
        scores = data_df[var_cfg.polscore_name].to_numpy(copy=True)
    else:
        scores = None

    if polscore_desc_ok:
        scores_add = data_df[var_cfg.polscore_desc_name].to_numpy(copy=True)
        no_of_vars = np.int16(np.around(scores_add.shape[1] / gen_cfg.no_of_treat))
        scores_add_idx = np.int16(np.reshape(np.arange(scores_add.shape[1]),
                                             (no_of_vars, gen_cfg.no_of_treat)
                                             )
                                  )
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
    if polscore_ok and optp_.gen_cfg.method in ('policy_tree', 'bps_classifier',):
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

    validated_allocations = {}
    for alloc_name in allocations:
        allocation_raw = allocation_df[alloc_name].to_numpy(copy=False)
        try:
            allocation_float = np.asarray(allocation_raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Allocation {alloc_name} must be numeric.') from exc

        allocation_rounded = np.rint(allocation_float)
        if (not np.all(np.isfinite(allocation_float))
                or not np.allclose(allocation_float, allocation_rounded, rtol=0, atol=1e-8)
                ):
            raise ValueError(f'Allocation {alloc_name} must contain finite integer values.')

        allocation_int = allocation_rounded.astype(np.int64)
        invalid_treatments = np.setdiff1d(np.unique(allocation_int), value)
        if invalid_treatments.size:
            raise ValueError('Inconsistent definition of treatment values. '
                             f'Unknown values in {alloc_name}: {invalid_treatments.tolist()}'
                             )

        validated_allocations[alloc_name] = allocation_int.astype(np.int32)

    for pop in population:
        for alloc_name in allocations:
            alloc_np = validated_allocations[alloc_name]
            if pop == 'Switchers':
                if alloc_name == 'observed':
                    continue
                switcher = alloc_np != np.int32(np.round(allocation_df['observed'].to_numpy()))
                alloc_ = alloc_np[switcher]
                if polscore_ok:
                    scores_ = scores[switcher]
                if polscore_desc_ok:
                    scores_add_ = scores_add[switcher]
            else:
                alloc_, scores_, = alloc_np, scores
                scores_add_ = scores_add
            len_scores = len(alloc_)
            indices_ = alloc_.copy()
            for idx, val in enumerate(value):
                indices_ = np.where(alloc_ == val, index[idx], indices_)
            if polscore_ok:
                score_sel = scores_[np.arange(len_scores), indices_]
                score_sel_new_df = pd.DataFrame(score_sel,
                                                columns=(var_cfg.polscore_name[0] + 'alloc',),
                                                )
                score_sel_net = score_sel - costs[indices_]
                score_sel_net_df = pd.DataFrame(score_sel_net,
                                                columns=(var_cfg.polscore_name[0] + 'alloc_net',),
                                                )
            else:
                score_sel_new_df = score_sel_net_df = None

            if polscore_desc_ok:
                scores_add_sel = np.zeros((len(scores_add_), no_of_vars))
                score_all_sel_name = []
                for idx, val_idx in enumerate(scores_add_idx):
                    scores_add_k = scores_add_[:, val_idx]
                    scores_add_sel[:, idx] = scores_add_k[
                        np.arange(len_scores), indices_]
                    score_all_sel_name.append(var_cfg.polscore_desc_name[val_idx[0]]+'alloc')
                score_add_df = pd.DataFrame(scores_add_sel, columns=score_all_sel_name)
            else:
                score_add_df = None
            desc_new_var_df_ = score_add_df if polscore_desc_ok else None
            results_dic = evaluation_of_alloc(gen_cfg=gen_cfg,
                                              results_dic=results_dic,
                                              pop=pop,
                                              alloc_name=alloc_name,
                                              score_sel_new_df=score_sel_new_df,
                                              score_sel_net_df=score_sel_net_df,
                                              desc_new_var_df=desc_new_var_df_,
                                              allocation=alloc_,
                                              titel=titel,
                                              protected_df=protected_df,
                                              transformed_scores=adjust_score,
                                              cont_min_values=optp_.fair_cfg.cont_min_values,
                                              )
            titel = False

            if qini_plots and pop != 'Switchers':
                if alloc_name in alloc_name_ref_qini:
                    alloc_ref_qini_np.append(alloc_np)
                    score_ref_qini_np.append(score_sel_net)
                    name_ref_qini.append(alloc_name)
                elif alloc_name.startswith(alloc_name_qini):
                    alloc_qini_np.append(alloc_np)
                    score_qini_np.append(score_sel_net)
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
                    alloc_np = validated_allocations[alloc_name]
                    if pop == 'Switchers':
                        if alloc_name == 'observed':
                            continue
                        switcher = alloc_np != np.int32(np.round(
                            allocation_df['observed'].to_numpy(copy=True))
                            )
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

                    if len(alloc_) == 0:
                        txt = ('Statistics by treatment status are not computed because the '
                               'selected population contains no observations.'
                               )
                        mcf_ps.print_mcf(gen_cfg, txt, summary=summary)
                        continue

                    if np.all(alloc_ == alloc_[0]):
                        txt = ('Statistics by treatment status are not computed because all '
                               'observations are allocated to the same treatment.'
                               )
                        mcf_ps.print_mcf(gen_cfg, txt, summary=summary)
                        continue
                    mcf_ps.statistics_by_treatment(gen_cfg,
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
    """Format allocation-fairness statistics with complete variable names."""
    txt = ('Fairness corrections: Dependence of allocations on protected '
           'attributes using different measures'
           )
    measures = (('Correlation (%)', 'correlation(allocation, protected)'),
                ('Cramers V (%)', 'Cramers V (allocation, protected)'),
                ('Bayes Factor', 'Bayes factor (allocation, protected)'),
                )
    tables_added = False
    for label, result_key in measures:
        rows: dict[str, pd.Series] = {}
        for allocation_name, item in results_dic.items():
            if result_key not in item:
                continue
            values = item[result_key].iloc[0].astype(float)
            if result_key == 'Bayes factor (allocation, protected)':
                values = values.clip(upper=10000)
            else:
                values = values * 100
            rows[allocation_name] = pd.Series([round(value, 2) for value in values.tolist()],
                                              index=values.index
                                              )
        if not rows:
            continue
        table = pd.DataFrame.from_dict(rows, orient='index')
        table.index.name = 'Allocation'
        txt += '\n\n' + label + '\n' + table.to_string()
        tables_added = True

    if not tables_added:
        return txt + '\nNo allocation-fairness statistics are available.'

    txt += ("\n\nNote: Cramér's V measures the association between two categorical variables."
            '\n  It ranges from 0 (no association) to 1 (perfect association).'
            '\n  The table displays 100 times this statistic, giving a scale from 0 to 100.'
            '\nThe Bayes factor (BF10) compares the dependence model with the independence model.'
            '\n  A value much larger than 1 provides evidence for dependence.'
            '\n  A value much smaller than 1 provides evidence for independence.'
            '\n  A value close to 1 provides little evidence in either direction.'
            '\n  Displayed Bayes factors are capped at 10000.'
            "\nCramér's V and the Bayes factor are not computed for protected variables treated"
            '\n  as continuous or for allocations without variation.'
            )
    return txt


def evaluation_of_alloc(*, gen_cfg: 'GenCfg',
                        results_dic: dict,
                        pop: str,
                        alloc_name: str,
                        score_sel_new_df: pd.DataFrame,
                        desc_new_var_df: pd.DataFrame,
                        allocation: np.array,
                        score_sel_net_df: pd.DataFrame,
                        titel: bool = False,
                        protected_df: pd.DataFrame = None,
                        transformed_scores: bool = False,
                        cont_min_values: int | float = 50,
                        ) -> dict:
    """Compute and print results of evaluations."""
    results_new_dic = results_dic.copy()
    local_dic = {}
    score_ok = score_sel_new_df is not None
    if desc_new_var_df is not None:
        desc_var = desc_new_var_df.columns
    else:
        desc_var = []

    if transformed_scores:
        desc_var_print = desc_var[:-1] if len(desc_var) > 1 else []
    else:
        desc_var_print = desc_var
    txt = ''

    if titel:
        scorevar = score_sel_new_df.columns if score_ok else None
        txt += '\n' + '-' * 100 + '\nMean of variables / treatment shares'

        if score_ok and not transformed_scores:
            txt += f'   Welfare measure used: {scorevar[0]:10}'

        txt += '\n'

        if score_ok:
            txt += f'{"Allocation":46} {"Gross welfare":19} {"Net welfare":19}'
        else:
            txt += f'{"Allocation":46}'

        if desc_new_var_df is not None:
            for var in desc_var_print:
                txt += f' {var:10s}'

        txt += ' ' * 7
        txt += ''.join([f'{value:8d}' for value in gen_cfg.d_values])
        txt += '\n'

    name = pop + ' ' + alloc_name

    if score_ok:
        mean1, std1 = get_mean_std_welfare(score_sel_new_df)
        mean_net, std_net = get_mean_std_welfare(score_sel_net_df)
        txt += f'{name:46} {mean1:10.4f} ({std1:.4f})'
        txt += f' {mean_net:10.4f} ({std_net:.4f})'
        local_dic[name] = mean1
        local_dic['welfare net of costs'] = mean_net
    else:
        txt += f'{name:46}'

    if desc_new_var_df is not None:
        for var in desc_var:
            variance_avg = desc_new_var_df[var].mean()
            if var in desc_var_print:
                txt += f' {variance_avg:{max(10, len(var))}.4f}'
            local_dic[var] = variance_avg
    local_dic['treatment share'] = get_treat_shares(allocation, gen_cfg.d_values)
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
                                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute numeric correlations and categorical dependence measures."""
    prot_var = protected_df.columns
    corr = np.full(len(prot_var), np.nan)
    cramer = np.full(len(prot_var), np.nan)
    bayes = np.full(len(prot_var), np.nan)
    allocation_varies = len(allocation) > 0 and allocation.std() > 1e-8

    if allocation_varies:
        allocation_series = pd.Series(allocation)
        for idx, name in enumerate(prot_var):
            protected = protected_df[name].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(protected.dtype):
                protected_np = protected.to_numpy(dtype=float, na_value=np.nan)
                if protected_np.std() > 0:
                    corr[idx] = np.corrcoef(allocation, protected_np)[0, 1]
            if protected.nunique(dropna=False) < no_continuous:
                cramer[idx] = cramers_v(allocation_series, protected)
                bayes[idx] = bayes_factor(allocation_series, protected)

    corr_df = pd.DataFrame(corr.reshape(1, -1), columns=prot_var)
    cramer_df = pd.DataFrame(cramer.reshape(1, -1), columns=prot_var)
    bayes_df = pd.DataFrame(bayes.reshape(1, -1), columns=prot_var)

    return corr_df, cramer_df, bayes_df


def get_mean_std_welfare(score_df: pd.DataFrame) -> tuple[float, float]:
    """Compute the mean and its exact conditional bootstrap standard error."""
    score_np = score_df.to_numpy(dtype=float, copy=False).reshape(-1)
    obs = score_np.size

    if obs == 0:
        return float('nan'), float('nan')

    mean = float(score_np.mean())
    std_mean = float(score_np.std(ddof=0) / np.sqrt(obs))

    return mean, std_mean


def get_treat_shares(allocation: NDArray[Any],
                     d_values: list | NDArray[Any]
                     ) -> NDArray[Any]:
    """Get treatment shares."""
    if len(allocation) == 0:
        return np.full(len(d_values), np.nan, dtype=float)
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
    """Compute and print results for multiple allocations."""
    txt = ('\n' * 2 + '-' * 100 + '\n' + f'{"Allocation":20s}'
           + 'Shares:  '
           + ' '.join([f'{s:7d}' for s in optp_.gen_cfg.d_values])
           + '\n' + ' ' * 20
           + 'Gross welfare: Sum / Mean / Variance'
           + '     Net welfare: Sum / Mean / Variance'
           )
    scores_np = scores_df.to_numpy(copy=True)
    costs = np.asarray(optp_.other_cfg.costs_of_treat, dtype=float)

    if costs.shape != (len(optp_.gen_cfg.d_values),):
        raise ValueError('Number of treatment costs is inconsistent with treatment values.')

    for alloc in allocations_dic:
        allocation_raw = allocations_dic[alloc].to_numpy(copy=False)

        if allocation_raw.ndim == 2:
            if allocation_raw.shape[1] != 1:
                raise ValueError(f'Allocation {alloc} must contain exactly one column.')
            allocation_raw = allocation_raw[:, 0]
        elif allocation_raw.ndim != 1:
            raise ValueError(f'Allocation {alloc} must be one-dimensional.')

        if len(allocation_raw) != len(scores_np):
            raise ValueError(f'Allocation {alloc} contains {len(allocation_raw)} observations, '
                             f'but the policy scores contain {len(scores_np)}.'
                             )
        try:
            allocation_float = np.asarray(allocation_raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Allocation {alloc} must be numeric.') from exc

        allocation_rounded = np.rint(allocation_float)

        if (not np.all(np.isfinite(allocation_float))
                or not np.allclose(allocation_float, allocation_rounded, rtol=0, atol=1e-8)
                ):
            raise ValueError(f'Allocation {alloc} must contain finite integer values.')

        allocation_int = allocation_rounded.astype(np.int64)
        invalid_treatments = np.setdiff1d(np.unique(allocation_int), optp_.gen_cfg.d_values)
        if invalid_treatments.size:
            raise ValueError('Inconsistent definition of treatment values. '
                             f'Unknown values: {invalid_treatments.tolist()}'
                             )
        allocation = allocation_int.astype(np.int32)
        treat_shares = get_treat_shares(allocation, optp_.gen_cfg.d_values)
        index = np.arange(len(optp_.gen_cfg.d_values), dtype=np.int32)
        indices_ = allocation.copy()
        idx_row = np.arange(len(scores_np))

        for idx, val in enumerate(np.asarray(optp_.gen_cfg.d_values, dtype=np.int32)):
            indices_ = np.where(allocation == val, index[idx], indices_)

        score_sel = scores_np[idx_row, indices_]
        score_sel_net = score_sel - costs[indices_]

        sum_gross = np.sum(score_sel)
        mean_gross = np.mean(score_sel)
        variance_gross = np.var(score_sel)

        sum_net = np.sum(score_sel_net)
        mean_net = np.mean(score_sel_net)
        variance_net = np.var(score_sel_net)

        txt += f'\n{alloc:20s}: '
        treat_shares_s = [f'{share:>8.2%}' for share in treat_shares]
        txt += ' ' + ''.join(treat_shares_s)
        txt += (f'  {sum_gross:12.2f}  {mean_gross:8.4f}  {variance_gross:8.4f}'
                f'     {sum_net:12.2f}  {mean_net:8.4f}  {variance_net:8.4f}'
                )
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
    x_name = [*(names_unordered or []), *(optp_.var_cfg.vi_x_name or [])]
    if not var_available(x_name, list(data_df.columns), needed='nice_to_have'):
        if x_name:
            missing = [name for name in x_name if name not in data_df.columns]
            txt = ('Variable importance is not computed for this evaluation. '
                   f'Missing features: {", ".join(missing)}.'
                   )
        else:
            txt = ('Variable importance is not computed for this evaluation because '
                   'no features were specified for variable importance.'
                   )
        mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=True)
        return

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
        d_df = allocation_df[alloc_name]
        if d_df.nunique(dropna=False) < 2:
            txt = ('Variable importance is not computed because all observations are '
                   'allocated to the same treatment.'
                   )
            mcf_ps.print_mcf(optp_.gen_cfg, txt, summary=True)
            continue

        x_df = data_df[x_name].copy()

        dummy_names = {}
        if names_unordered:
            for idx, name in enumerate(names_unordered):
                x_dummies = pd.get_dummies(x_df[name], columns=[name], dtype=int)
                x_dummies_names = [name + str(dummy) for dummy in x_dummies.columns]
                dummy_names[name] = x_dummies.columns = x_dummies_names
                x_dummies_s = x_dummies.reindex(sorted(x_dummies.columns), axis=1)
                if idx == 0:
                    x_all_dummies = x_dummies_s
                else:
                    x_all_dummies = pd.concat([x_all_dummies, x_dummies_s], axis=1)
            names_wo = [name for name in x_df.columns if name not in names_unordered]
            x_new_df = pd.concat((x_df[names_wo], x_all_dummies), axis=1)
        else:
            x_new_df = x_df
        classif = RandomForestClassifier(n_estimators=1000,
                                         max_features='sqrt',
                                         bootstrap=True,
                                         oob_score=False,
                                         n_jobs=optp_.gen_cfg.mp_parallel,
                                         random_state=seed,
                                         verbose=False,
                                         min_samples_split=5,
                                         )
        mcf_vi.print_variable_importance(classif, x_new_df, d_df,
                                         x_name=x_name, names_uo=names_unordered,
                                         unordered_dummy_names=dummy_names, gen_cfg=optp_.gen_cfg,
                                         summary=True, classification=True,
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
    allocation_np = rng.choice(optp_.gen_cfg.d_values, size=no_obs, p=optp_.rnd_cfg.shares)

    return allocation_np


def get_best_ate_allocation(optp_: 'OptimalPolicy',
                            data_df: pd.DataFrame,
                            polscore_name: list[str] | None = None,
                            ) -> NDArray[Any]:
    """Get allocation by sending everybody to treatment with highest ATE."""
    if polscore_name is None:
        polscore_name = optp_.var_cfg.polscore_name

    avg_polscore = data_df[polscore_name].mean(axis=0).to_numpy(dtype=float)

    costs = np.asarray(optp_.other_cfg.costs_of_treat, dtype=float)
    best_treat_idx = np.argmax(avg_polscore - costs)
    allocation_np = np.full(len(data_df), optp_.gen_cfg.d_values[best_treat_idx], dtype=np.int32)

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
        welfare_reference = qini_dict['score_ref_list'][idx_ref].mean() * np.ones(obs)

        for idx, name in enumerate(qini_dict['name_list']):
            score_combined = qini_like_score(
                qini_dict['score_list'][idx],
                qini_dict['score_ref_list'][idx_ref])

            figs, axs = plt.subplots()
            axs.plot(share_optimally_treated, score_combined, label=name, color='r')
            axs.plot(share_optimally_treated, welfare_reference, label=name_ref, color='b')
            axs.set_ylabel('Average welfare net of treatment costs')
            axs.set_xlabel('Share optimally treated (reference rule applied to rest)')
            axs.legend(loc=optp_.int_cfg.legend_loc, shadow=True, fontsize=optp_.int_cfg.fontsize)
            axs.set_title(f'Net welfare of {name} rule \nvs. {name_ref} ({data_title} data)')

            name_title = ('NetWelfare_'
                          + name_ref.replace(' ', '')
                          + name.replace(' ', '')
                          + data_title.replace(' ', '')
                          )
            invalid_chars = '<>:"/\\|?*'
            name_title = ''.join('_' if char in invalid_chars or ord(char) < 32 else char
                                 for char in name_title
                                 )
            file_jpeg = optp_.gen_cfg.outpath / (name_title + '.jpeg')
            file_pdf = optp_.gen_cfg.outpath / (name_title + '.pdf')
            file_csv = optp_.gen_cfg.outpath / (name_title + '.csv')
            mcf_sys.delete_file_if_exists(file_jpeg)
            mcf_sys.delete_file_if_exists(file_pdf)
            figs.savefig(file_jpeg, dpi=optp_.int_cfg.dpi)
            figs.savefig(file_pdf, dpi=optp_.int_cfg.dpi)
            if optp_.int_cfg.show_plots:
                plt.show()
            plt.close()
            # Save data for plot
            all_data_np = np.concatenate((share_optimally_treated.reshape(-1, 1),
                                          score_combined.reshape(-1, 1),
                                          welfare_reference.reshape(-1, 1)
                                          ),
                                         axis=1)
            all_data_names = ('Share', 'combined', 'reference',)
            all_data_plot_df = pd.DataFrame(data=all_data_np, columns=all_data_names)
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

    score_cumsum = np.cumsum(score_sorted)
    score_ref_cumsum = np.cumsum(score_ref_sorted)

    return (score_cumsum + score_ref_cumsum[-1] - score_ref_cumsum) / obs
