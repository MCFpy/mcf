"""Created on Fri Dec  8 13:42:19 2023.

Contains classes and methods needed for mcf and optimal policy reporting.

# -*- coding: utf-8 -*-

@author: MLechner
"""
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import t

if TYPE_CHECKING:
    from mcf.mcf_main import ModifiedCausalForest
    # from mcf.optpolicy_main import OptimalPolicy


def general(rep_o):
    """Provide basic information."""
    def basic_helper(obj_inst, titel, mcf=False):
        """Create method-specific output string."""
        help_txt = (
            '\n\n' + titel +
            f'\nAll outputs: {obj_inst.gen_dict["outpath"]}')
        if mcf:
            help_txt += (
                '\nSubdirectories with figures and data are named '
                '"ate_iate", "gate", "bgate", and "common support" and contain '
                'the content indicated by their name.')
        help_txt += (
            f'\nDetailed text output: {obj_inst.gen_dict["outfiletext"]}'
            f'\nSummary text output: {obj_inst.gen_dict["outfilesummary"]}'
            )
        return help_txt

    txt = (
        '\nWelcome to the mcf estimation and optimal policy package.'
        '\nThis report provides you '
        'with a summary of specifications and results. More '
        'detailed information can be found in the respective output '
        'files. Figures and data are provided in the output '
        'path in csv-format.'
        'This will partly allow you to recreate the figures on your own.'
        )

    pre_titel = 'Output path information for '
    if rep_o.mcf_o is not None:
        txt += basic_helper(rep_o.mcf_o, pre_titel + 'MCF ESTIMATION', mcf=True)
    if rep_o.sens_o is not None:
        txt += basic_helper(rep_o.sens_o, pre_titel + 'SENSITIVITY ANALYSIS')
    if rep_o.blind_o is not None:
        txt += basic_helper(rep_o.blind_o, pre_titel + 'BLINDED IATEs')
    if rep_o.opt_o is not None:
        txt += basic_helper(rep_o.opt_o, pre_titel + 'OPTIMAL POLICY ANALYSIS')
    txt += '\n' * 2 + 'BACKGROUND\n'
    if rep_o.mcf_o is not None:
        txt += ('\nESTIMATION OF EFFECTS'
                '\nThe MCF is a comprehensive causal machine learning '
                'estimator for the estimation of treatment effects at various '
                'levels of granularity, from the average effect at the '
                'population '
                'level to very fine grained effects at the (almost) individual '
                'level. Since most effects at the higher levels are obtained '
                'from lower level effects, all effects are internally '
                'consistent. Recently, the basic package has been extended to '
                'cover new average effects (like Balanced Group Average '
                'Treatment Effects, '
                'BGATEs, see Bearth and Lechner, 2025, and Quantile '
                'Individualized Average Treatment Effects, QIATE, '
                'see Kutz and Lechner, 2025) '
                'as well as for an optimal policy (decision) module. '
                '\nEffect estimation is implemented for identification by '
                'unconfoundedness and instrumental variables. '
                'While unconfoundedness estimation can deal with multiple '
                'treatments, instrumental variable estimation is (currently) '
                'restricted '
                'to binary instruments and binary treatments.'
                '\nThe basis of the MCF estimator is the causal forest '
                'suggested by Wager and Athey (2018). Their estimator has '
                'been changed in several dimensions which are described in '
                'Lechner (2018). The main changes relate to the objective '
                'function as well as to the aggreation of effects. '
                "Lechner and Mareckova (2025a) provide the MCF's asymptotic "
                'guarantees for the case of unconfoundedness. '
                'Based on asymptotic theory and a large simulation study, '
                'they also compare the MCF to other approaches like the '
                'Generalized Random Forest (GRF, Athey, Tibshirani, Wager, '
                '2019) and Double Machine Learning (DML, Chernozhukov, '
                'Chetverikov, Demirer, Duflo, Hansen, Newey, Robins, 2018, '
                'Knaus, 2022). In this comparison '
                'the MCF faired very well, in particular, but not only, for '
                'heterogeneity estimation. Some operational issues of the MCF '
                'are discussed in Bodory, Busshof, Lechner (2022) and '
                'Bodory, Mascolo, and Lechner (2024). There are '
                'several empirical studies in different fields using the MCF, '
                'like Cockx, Lechner, '
                'Boolens (2023), Boller, Lechner, Okasa (2025), or '
                'Audrino, Chassot, Huang, Knaus, Lechner, and Ortega (2024), '
                'for example.'
                '\nThe asymptotic guarantees (and some finite sample '
                'behaviour) of the instrumental variable version of the MCF is '
                'discussed in Lechner and Mareckova (2025b). '
                )
        txt += '\n' * 2 + 'References'
        txt += ('\n- Athey, S., J. Tibshirani, S. Wager (2019): Generalized '
                'Random Forests, The Annals of Statistics,  47, 1148-1178.'
                '\n- Athey, S., S. Wager (2019): Estimating Treatment Effects '
                'with Causal Forests: An Application,  Observational Studies, '
                '5, 21-35.'
                'Audrino, F., J. Chassot, C. Huang, M. Knaus, M. Lechner, '
                'and J.P. Ortega (2024): How does post-earnings announcement '
                "sentiment affect firms' dynamics? New evidence from causal "
                'machine learning, Journal of Financial Econometrics, 22, '
                '1184, 575-604.'
                '\n- Bearth, N., and M. Lechner (2025): Causal Machine '
                'Learning for Moderation Effects, Journal '
                'of Business & Economic Statistics. '
                '\n- Boller, D., M. Lechner, and G. Okasa (2025): '
                'The effect of sport in online dating: evidence from causal '
                'machine learning, Humanities and Social Sciences '
                'Communications, 2025,12:39.'
                '\n- Bodory, H., H. Busshoff, M. Lechner (2023): High '
                'Resolution Treatment Effects Estimation:  Uncovering Effect '
                'Heterogeneities with the Modified Causal Forest, Entropy, 24, '
                '1039. '
                '\n- Bodory, H., F. Mascolo, M. Lechner (2024): Enabling '
                'Decision Making with the Modified Causal Forest: Policy Trees '
                'for Treatment Assignment, Algorithm, 17, 318. '
                '\n- Chernozhukov, V., D. Chetverikov, M. Demirer, '
                'E. Duflo, C. Hansen, W. Newey, J. Robins (2018):  '
                'Double/debiased  machine learning for treatment and '
                'structural parameters, Econometrics Journal,  21, C1-C68.'
                '\n- Cockx, B., M. Lechner, J. Bollens (2023): Priority to '
                'unemployed immigrants? A causal machine  learning evaluation '
                'of training in Belgium, Labour Economics, 80, Article 102306.'
                '\n- Knaus, M. (2022): Double Machine Learning based Program '
                'Evaluation under Unconfoundedness,  Econometrics Journal. '
                '\n- Kutz, J, and M. Lechner (2025): Quantile '
                'Individualized Average Treatment Effects, arXiv.'
                '\n- Lechner, M. (2018): Modified Causal Forests for '
                'Estimating Heterogeneous Causal Effects, arXiv.'
                '\n- Lechner, M. (2023): Causal Machine Learning and its Use '
                'for Public Policy, Swiss Journal of  Economics & Statistics, '
                '159:8.'
                '\n- Lechner, M., J. Mareckova (2025a): Comprehensive Causal '
                'Machine Learning, arXiv.'
                '\n- Lechner, M., J. Mareckova (2025b): Comprehensive Causal '
                'Machine Learning with Instrumental Variables,  arXiv. '
                '\n- Wager, S., S. Athey (2018): Estimation and Inference of '
                'Heterogeneous Treatment Effects using  Random Forests, '
                'Journal of the American Statistical Association, 113:523, '
                '1228-1242.'
                )
    if rep_o.sens_o is not None:
        txt += ('\n\nSENSITIVITY' + '\nSensitivity analysis is currently '
                'experimental and not (yet) documented here.')
    if rep_o.blind_o is not None:
        txt += ('\n\nBLIND IATEs' + '\nThe blinding of IATEs is currently '
                'experimental and not (yet) documented here.')
    if rep_o.opt_o is not None:
        txt += ('\nThe optimal policy module offers three '
                '(basic) algorithms that can be used to exploit fine '
                'grained knowledge about effect heterogeneity to obtain '
                'decision rules. The current version is implemented '
                'for discrete treatments only.'
                '\nThere is also an option for different '
                'fairness adjustments, as well as an option to take into '
                'account estimation risk.'
                '\n\nThe BEST_POLICY_SCORE algorithm is based on assigning '
                'the treatment that has the highest impact at the '
                'unit (e.g., individual) level. If the treatment heterogeneity '
                'is known (not estimated), this will lead to the best possible '
                'result. This algorithm is computationally not burdensome. '
                'However, it will not be easy to understand how the implied '
                'rules depends on the features of the unit. Its statistical '
                'properties are also not clear (for estimated treatment '
                'heterogeneity) and there is a certain danger of overfitting, '
                'which could lead to an unsatisfactory out-of-training-sample '
                'performance.'
                '\n\nThe BPS_CLASSIFIER classifier algorithm runs a classifier '
                'for each of the allocations obtained by the '
                'BEST_POLICY_SCORE algorithm. One advantage of this '
                'approach compared to the BEST_POLICY_SCORE algorithm is that '
                'prediction of the allocation of (new) observations is fast '
                'because it does not require to recompute the policy score (as '
                'it is the case with the BEST_POLICY_SCORE algorithm). The '
                'specific classifier is selected among four different '
                'classifiers from scikit-learn, namely a simple neural '
                'network, two classification random forests with minimum leaf '
                'size of 2 and 5, and ADDABoost. The selection is made '
                'according to the out-of-sample performance of the Accuracy '
                'Score of scikit-learn.'
                '\n\nThe POLICY TREE algorithm builds optimal shallow decision '
                'trees. While these trees are unlikely to lead to gloablly '
                'optimal allocations, and are computationally much more '
                'expensive, they have the advantage that the decision rule '
                'is much easier to '
                'understand and that some statistical properties are known, at '
                'least for certain versions of such decision trees (e.g., '
                'Zhou, Athey, Wager, 2023). The basic algorithmic '
                'implementation follows '
                'the recursive algorithm suggested by Zhou, Athey, Wager '
                '(2023) with three (more substantial) deviations (=extensions).'
                '\nExtension 1: Since using One Hot Encoding for categorical '
                'variables may lead to rather extreme leaves for '
                'such variables with many different values when '
                'building (shallow) trees (splitting one value against the '
                'rest), a more sophisticated procedure is used that allows to '
                'have several values of the categorical variables on both '
                'sides of the split.'
                '\nExtension 2: Constraints are allowed for. They are handled '
                'in a sequential manner: First, an approximate '
                'treatment-specific cost vector is obtained and used to '
                'adjust the policy score accordingly. Second, trees that '
                'violate the constraints are removed '
                '(to some extent, optional).'
                '\nExtensions 3: There are several options implemented to '
                'reduce the computational burden, which are discussed below '
                'in the section showing the implementation of the policy '
                'score. '
                )

        txt += ('\n' * 2 + 'References'
                '\n-Zhou, Z., S. Athey, S. Wager (2023): Offline '
                'Multi-Action Policy Learning: Generalization and '
                'Optimization, Operations Research, INFORMS, 71(1), 148-183.')
    return txt


def sensitivity(sens_o, empty_lines_end):
    """Sensitivity part of description."""
    if sens_o.sens_dict["reference_population"] is None:
        ref_pop = 'the largest treatment'
    else:
        ref_pop = f'treatment {sens_o.sens_dict["reference_population"]}'
    txt = ('\nMETHOD'
           '\nThis sensitivity analysis is based on a placebo-like experiment: '
           '\n(1) Using a random forest classifier, probabilities into the '
           'different treatments are estimated. Predictions are based on '
           f'{sens_o.sens_dict["cv_k"]}-fold cross-fitting.'
           '\n(2) All observations that do not belong to '
           f'{ref_pop} are deleted.'
           '\n(3) The conditional treatment probabilities are used to '
           'simulate treatments on the remaining observations. These '
           'treatments should feature the same selectivity as the original '
           'data but with a zero treatment effect.'
           '\n(4) Training is performed on this reduced data and the usual '
           'effects are estimated. Ideally, they are close to zero.'
           '\n(5) If the results dictionary of the prediction method is '
           'passed to the sensitivity method, and if it '
           'includes estimated IATEs, '
           'the same data as in the estimation is used. In this case '
           'the estimated IATEs are compared to the placebo IATEs. This '
           'indicates in which regions of the IATEs violations may take place.')
    txt += '\n' * 2 + 'APPLICATION'
    txt += ('\nThe following scenarions are investigated: '
            f'{", ".join(sens_o.sens_dict["scenarios"])}')
    if sens_o.sens_dict['replications'] > 1:
        txt += ('\nSimulations are repeated '
                f'{sens_o.sens_dict["replications"]} times and results '
                'are averaged to reduce simulation noise.')
    txt += f'\nPath for all output files: \n   {sens_o.gen_dict["outpath"]}\n'

    if sens_o.sens_dict['cbgate']:
        txt += '\nCBGATE is investigated. '
        txt += 'Output is contained in the output files.'
    if sens_o.sens_dict['bgate']:
        txt += '\nBGATE is investigated. '
        txt += 'Output is contained in the output files.'
    if sens_o.sens_dict['iate']:
        txt += ('\nIATEs are investigated. If estimated IATEs are available, '
                'then plots comparing placebo IATEs with estimated IATEs are '
                'presented below. ')
        txt += 'Further statistics are contained in the output files.'
    if sens_o.sens_dict['iate_se']:
        txt += '\nStandard errors for IATEs are available. '
        txt += 'Output is contained in the output files.'

    txt += '\n' * 2 + 'IMPORTANT REMARK'
    txt += ('\nThis sensitivity analysis is experimental, rudimentary, '
            'and not yet fully tested (although it appears to work fine in '
            'tests so far). In the future, it will be expanded and improved.')
    if 'sens_txt_ate' in sens_o.report:
        txt += '\n\nRESULTS: ATE\n' + sens_o.report['sens_txt_ate']
    if sens_o.report['sens_plots_iate'] is not None:
        txt += '\n\nRESULTS: Plots of estimated and placebo IATEs\n'
    return txt + '\n' * empty_lines_end


def mcf_general(mcf_o, empty_lines_end, iv):
    """Collect general mcf information."""
    mtot = mcf_o.cf_dict['mtot']  # 1 standard
    prog_score = mcf_o.cf_dict['match_nn_prog_score']
    matching = 'Prognostic Score' if prog_score else 'Mahalanobis Distance'
    txt = '\nMETHOD\n'
    if iv:
        txt += ('The instrumental variable method is used. MCFs are run for '
                'the first stage and the reduced form.\n')

    if mtot == 1:
        txt += ('The standard MCF method used for forest building. '
                'Nearest neighbour matching is performed'
                f' using the {matching}.'
                )
    else:
        txt += ('A nonstandard method for training the MCF is used. '
                'ARE YOU SURE THIS WAS ON PURPOSE?'
                )
    txt += ('\nFeature selection is '
            f'{"" if mcf_o.fs_dict["yes"] else "not "}used.'
            )
    txt += (f'\nLocal centering is {"" if mcf_o.lc_dict["yes"] else "not "} '
            'used.')
    txt += ('\nCommon support is '
            f'{"" if mcf_o.cs_dict["type"] > 0 else "not "}enforced.'
            )

    txt += '\n\nVARIABLES'
    y_name = [y[:-3] if y[-3:] == '_lc' else y
              for y in mcf_o.var_dict["y_name"]]
    out = 'Outcome' if len(y_name) == 1 else 'Outcomes'
    txt += f'\n{out}: {", ".join(y_name)}'
    d_values = [str(val) for val in mcf_o.gen_dict["d_values"]]
    txt += (f'\nTreatment: {mcf_o.var_dict["d_name"][0]} (with values '
            f'{" ".join(d_values)})')
    if iv:
        txt += f'\nInstrument: {mcf_o.var_dict["iv_name"][0]}'
    conf = 'instrument confounders' if iv else 'confounders'
    if mcf_o.var_dict["x_name_ord"]:
        txt += (f'\nOrdered {conf}: '
                f' {", ".join(mcf_o.var_dict["x_name_ord"])}')
    if mcf_o.var_dict["x_name_unord"]:
        var = [y[:-6] if y.endswith('_prime') else y
               for y in mcf_o.var_dict["x_name_unord"]]
        txt += (f'\nUnordered (categorical) {conf}: {", ".join(var)}')
    if mcf_o.var_dict["z_name_cont"]:
        txt += ('\nContinuous heterogeneity variables: '
                f' {", ".join(mcf_o.var_dict["z_name_cont"])}')
    if mcf_o.var_dict["z_name_ord"]:
        txt += ('\nOrdered heterogeneity variables (few values, continuous '
                'variables are discretized): '
                f' {", ".join(mcf_o.var_dict["z_name_ord"])}')
    if mcf_o.var_dict["z_name_unord"]:
        var = [y[:-6] if y.endswith('_prime') else y
               for y in mcf_o.var_dict["z_name_unord"]]
        txt += ('\nUnordered heterogeneity variables: '
                f' {", ".join(var)}')
    if mcf_o.var_dict["x_name_balance_test"] and mcf_o.p_dict['bt_yes']:
        var = [y[:-6] if y.endswith('_prime') else y
               for y in mcf_o.var_dict["x_name_balance_test"]]
        txt += f'\nVariables to check balancing: {", ".join(var)}'
    if mcf_o.var_dict["x_name_balance_bgate"]:
        txt += ('\nVariables to balance the distribution for the BGATE: '
                f' {", ".join(mcf_o.var_dict["x_name_balance_bgate"])}')
    if mcf_o.gen_dict["weighted"]:
        txt += ('\n\nWeighting used. Weights are contained in'
                f' {mcf_o.var_dict["w_name"][0]}')
    if mcf_o.p_dict["cluster_std"]:
        txt += ('\n\nClustered standard errors used. Clusters are contained in'
                f' {mcf_o.var_dict["cluster_name"][0]}')
    if mcf_o.gen_dict["panel_data"]:
        txt += '\n\nPanel data used.'

    effects = add_effects(mcf_o.p_dict, mcf_o.gen_dict)
    txt += f'\n\nEFFECTS ESTIMATED\n{", ".join(effects)}'

    if mcf_o.var_dict["z_name_ord"] or mcf_o.var_dict["x_name_unord"]:
        txt += '\n' * 2 + 'NOTE on unordered variables: '
        txt += ('\nOne-hot-encoding (dummy variables) is not used as it '
                'is expected to perform poorly with trees: It may lead '
                'to splits of one category versus all other categories. '
                'Instead the approach used is analogous to the one discussed '
                'in Chapter 9.2.4 of Hastie, Tibshirani, Friedmann (2013), '
                'The Elements of Statistical Learning, 2nd edition.')

    return txt + '\n' * empty_lines_end


def mcf_iate_analyse(mcf_o: 'ModifiedCausalForest',
                     empty_lines_end: int,
                     iv: bool = False
                     ) -> tuple[str,
                                object | None,
                                str, str,
                                pd.DataFrame | None
                                ]:
    """Create text, table, and figures for ate estimation."""
    txt = ('\n' * 2 +
           'This section contains parts of the descriptive analysis '
           ' of the '
           )
    txt += 'LIATEs. ' if iv else 'IATEs. '
    txt += ('More detailed tables and figures are contained in '
            'the output files and directories. These additional results '
            'include variable importance plots from a regression random forest '
            'as well as and linear regression results. Both estimators use '
            'the estimated IATEs as dependent variables and the confounders '
            'and heterogeneity variables as features. '
            )
    if iv:
        txt += ('\nFigures relate to the distributions of the LIATEs as well '
                'as to the IATEs for the first stage and the reduced form. '
                )
    figure_note = '\n' * empty_lines_end
    knn_text = ''
    if (mcf_o.report.get('knn_table') is not None
            and mcf_o.report['knn_table'] is not None):
        knn_text = '\n' * 2 + 'K-MEANS CLUSTERING'
        knn_text += ('\nThe sample is divided using k-means clustering based '
                     'on the estimated '
                     )
        knn_text += 'LIATEs' if iv else 'IATEs'
        knn_text += ('. The number of clusters is determined by maximizing '
                     'the Average Silhouette Score on a given grid. The '
                     'following table shows the means of the '
                     )
        knn_text += 'LIATEs' if iv else 'IATEs'
        knn_text += (', potential outcomes, and the features in these '
                     'clusters, respectively.'
                     )
        if mcf_o.post_dict['kmeans_single']:
            knn_text += ('There are also results for clustering according to '
                         'single effects only. These results are contained in '
                         'the respective .txt-files.')
    if mcf_o.report.get('fig_iate') is not None:
        fig_iate = mcf_o.report['fig_iate']
    else:
        fig_iate = None
    if mcf_o.report.get('knn_table') is not None:
        knn_table = mcf_o.report['knn_table']
    else:
        knn_table = None
    knn_text = knn_text + '\n' * empty_lines_end
    return txt, fig_iate, figure_note, knn_text, knn_table


def mcf_iate_part1(mcf_o: 'ModifiedCausalForest',
                   empty_lines_end: int,
                   iv: bool = False
                   ) -> str:
    """Text basic descriptives of IATEs."""
    txt = ('\n' * 2 +
           'This section contains parts of the descriptive analysis '
           'of the '
           )
    txt += 'LIATEs. ' if iv else 'IATEs. '
    txt += ('Use the analyse method to obtain more '
            'descriptives of the '
            )
    txt += 'LIATEs, ' if iv else 'IATEs, '
    txt += 'like their distribution, and their relations to the features.'

    if mcf_o.gen_dict['iate_eff']:
        txt += ('\n' * 2 + 'METHODOLOGICAL NOTE'
                '\nIn order to increase the efficiency of the IATE estimation, '
                'a second set of IATEs is computed by reversing the role of '
                'the two samples used to build the forest and to populate it '
                'with the outcome information. The two IATEs are averaged '
                'to obtain a more precise estimator (which may be particulary '
                'useful when the IATEs, or the corresponding potential '
                'outcomes, are used as inputs for decision models). '
                '\nThe following descriptive analysis is based on the first '
                'round IATEs only.')
    txt += '\n' * 2 + 'RESULTS' + '\n' + mcf_o.report['iate_text']
    return txt + '\n' * empty_lines_end


def mcf_balance(empty_lines_end: int, iv: bool = False) -> str:
    """Text balancing tests."""
    txt = ('\n' * 2 + 'Balancing tests are experimental. '
           'See the output files for the results.')
    if iv:
        txt += 'Balancing tests are for 1st stage and reduced form only.'
    return txt + '\n' * empty_lines_end


def mcf_gate(mcf_o: 'ModifiedCausalForest',
             empty_lines_end: int,
             gate_type: str = 'gate',
             iv: bool = False
             ) -> tuple[list, str, str]:
    """Create text, table, and figures for gate estimation."""
    dic = mcf_o.p_dict
    txt = ('Note: Detailed tables and figures for additional effects are '
           'contained in the output files and output directories. ')
    if gate_type == 'gate' and dic['gatet']:
        txt += ('Similarly, figures and tables of the effects for the '
                'different treatment groups are contained in the output files '
                'and directories.')
    if iv:
        txt += 'Estimated effects are Group Averaged Local Complier Effects. '
        figure_list = mcf_o.report['fig_' + gate_type + '_local']
    else:
        figure_list = mcf_o.report['fig_' + gate_type]

    if gate_type == 'bgate':
        txt_bgate = ('\nVariables to balance the distribution for the BGATE: '
                     f' {", ".join(mcf_o.var_dict["x_name_balance_bgate"])}')
        return_tuple = (figure_list, txt + '\n' * empty_lines_end, txt_bgate)
    else:
        return_tuple = (figure_list, txt + '\n' * empty_lines_end)
    return return_tuple


def mcf_ate_proc(mcf_o: 'ModifiedCausalForest',
                 empty_lines_end: int,
                 iv: bool = False,
                 alloc: bool = False
                 ) -> tuple[str, list | tuple, list | tuple | None, int]:
    """Create text and table for ate estimation."""
    dic = mcf_o.p_dict
    if dic['se_boot_ate'] or dic['cluster_std']:
        txt = '\n' * 2 + 'METHOD'
        txt += (f'\nBootstrap standard error with {dic["se_boot_ate"]} '
                'replications. ')
    txt = '\n' * 2 + 'RESULT '
    tables_results_extra = None
    if iv:
        tables_results_local = tables_results_global = None
        txt_local = txt_global = None
        txt += '\n'
        if global_iv := 'mcf_pred_results_global' in mcf_o.report:
            txt_global = 'Local Average Treatment Effects for Global Compliers'
            tables_results_global = build_ate_table(mcf_o,
                                                    alloc=alloc,
                                                    iv_local=False,
                                                    iv_global=True
                                                    )

        if local_iv := 'mcf_pred_results_local' in mcf_o.report:
            txt_local = 'Averaged Local Complier Effects'
            tables_results_local = build_ate_table(mcf_o,
                                                   alloc=alloc,
                                                   iv_local=True,
                                                   iv_global=False
                                                   )
        if local_iv and global_iv:
            txt += '(1) ' + txt_global + ' (2) ' + txt_local
            tables_results = tables_results_global
            tables_results_extra = tables_results_local
        elif local_iv:
            txt += txt_local
            tables_results = tables_results_global
        else:
            tables_results = tables_results_local

    else:
        tables_results = build_ate_table(mcf_o,
                                         alloc=alloc,
                                         iv_local=False,
                                         iv_global=False
                                         )  # List:One Table per outcome
    table_note = results_table_note(dic['atet'], alloc)

    return (txt, tables_results, tables_results_extra,
            table_note + '\n' * empty_lines_end
            )


def mcf_welfare_alloc(mcf_o, empty_lines_end):
    """Create text and table for welfare estimation of different allocations."""
    return mcf_o.report['alloc_welfare_allocations'] + '\n' * empty_lines_end


def results_table_note(treatment_specific, alloc=False):
    """Create Note for results tables."""
    table_note = (' *, **, ***, **** denote significance at the 10%, '
                  '5%, 1%, 0.1% level. The results for the ')
    if alloc:
        table_note += 'average '
    else:
        table_note += 'potential '
    table_note += 'outcomes '
    if treatment_specific and not alloc:
        table_note += 'and the treatment specific effects'
    table_note += 'can be found in the output files.'
    return table_note


def build_ate_table(mcf_o: 'ModifiedCausalForest',
                    alloc: bool = False,
                    iv_local: bool = False,
                    iv_global: bool = False
                    ) -> list:
    """Build table for ATE estimation as DataFrame."""
    if alloc and 'alloc_mcf_pred_results' in mcf_o.report:
        dic = mcf_o.report['alloc_mcf_pred_results']
    elif iv_local:
        dic = mcf_o.report['mcf_pred_results_local']
    elif iv_global:
        dic = mcf_o.report['mcf_pred_results_global']
    else:
        dic = mcf_o.report['mcf_pred_results']
    ate, ate_se = dic['ate'], dic['ate_se']
    ate_effect_list = [str(row[0]) + ' vs ' + str(row[1])
                       for row in dic['ate_effect_list']]
    # 3D numpy arrays, outcome, treatment population, 3rd element compariosn
    table_list = []
    for o_idx in range(ate.shape[0]):
        ate_1d = np.round(ate[o_idx, 0, :].flatten().reshape(-1, 1), 3)
        ate_se_1d = np.round(ate_se[o_idx, 0, :].flatten().reshape(-1, 1), 3)
        t_val = np.round(np.abs(ate_1d / ate_se_1d), 2)
        p_val = np.round(t.sf(t_val, 1000000) * 2 * 100, 2)
        p_stars = [p_star_string(val/100) for val in p_val]
        col1_name = 'Difference' if alloc else 'Effect'
        table1_df = pd.DataFrame(
            data=np.concatenate((ate_1d, ate_se_1d, t_val, p_val), axis=1),
            index=ate_effect_list,
            columns=(col1_name, 'SE', 't-val', 'p_val (%)')
            )
        table2_df = pd.DataFrame(data=p_stars, index=ate_effect_list,
                                 columns=('Sig.', ))
        table_df = pd.concat((table1_df, table2_df,), axis=1)
        table_list.append(table_df)
    return table_list


def p_star_string(p_val):
    """Create stars for p values."""
    if p_val < 0.001:
        print_str = '****'
    elif p_val < 0.01:
        print_str = ' ***'
    elif p_val < 0.05:
        print_str = '  **'
    elif p_val < 0.1:
        print_str = '   *'
    else:
        print_str = '    '
    return print_str


def mcf_results(mcf_o: 'ModifiedCausalForest',
                empty_lines_end, iv=False, alloc=False):
    """Text some general remarks about the estimation process."""
    dic = mcf_o.p_dict
    txt = '\n' * 2 + 'GENERAL REMARKS'
    if alloc:
        txt += ('\n'
                'Mean values and their standard error are computed for '
                'various user-provided treatment '
                'allocations . These values are based on aggregating the '
                'respective estimated potential outcomes. '
                )
    else:
        txt += ('\nThe following results for the different parameters are all '
                'based on the same causal forests (CF). The combination of the '
                'CF with the potentially new data provided leads to weight '
                'matrices. These matrices may be large requiring some '
                'computational optimisations, such as processing them in '
                'batches and saving them in a sparse matrix format. One '
                'advantage of this approach is that aggregated effects '
                )
        txt += '(LATE, LGATE, LBGATE) ' if iv else '(ATE, GATE, BGATE) '
        txt += 'can be computed by aggregation of the weights used for the '
    if alloc:
        txt += 'Thus a high internal consistency is preserved.'
    else:
        if iv:
            txt += ('LIATE. Thus a high internal consistency is preserved in '
                    'the sense that LIATEs will aggregate to LGATEs, which in '
                    'turn will aggregate to LATEs.'
                    )
        else:
            txt += ('IATE. Thus a high internal consistency is preserved in '
                    'the sense that IATEs will aggregate to GATEs, which in '
                    'turn will aggregate to ATEs.')

    if dic["max_weight_share"] > 0:
        txt += '\n' * 2 + 'ESTIMATION'
        txt += ('\nWeights of individual training observations are truncated '
                f'at {dic["max_weight_share"]:4.2%}. ')
        if alloc:
            txt += ('No truncation of weights for value differences.')
        else:
            txt += 'Aggregation of '
            txt += ('LIATEs to LATE and GATEs' if iv
                    else 'IATEs to ATE and GATEs')
            txt += ' may not be exact due to weight truncation.'
        if dic['choice_based_sampling']:
            txt += ('\nTreatment based choice based sampling is used with '
                    f'probabilities {", ".join(dic["choice_based_probs"])}.'
                    )

    txt += '\n' * 2 + 'INFERENCE'
    txt += '\nInference is based on using the weight matrix. '
    if not dic['cond_var']:
        txt += 'The unconditional variance of weight x outcome is used.'
    else:
        txt += 'Nonparametric regressions are based on '
        if dic['knn']:
            txt += 'k-nearest neighbours. '
        else:
            txt += 'Nadaraya-Watson kernel regression.'
    if dic['cluster_std']:
        txt += ('\nStandard errors are clustered. '
                f'{mcf_o.var_dict["cluster_name"]} is the cluster variable.'
                'Clustering is implemented by the block-bootstrap.')
    if not (iv or alloc):
        txt += '\n' * 2 + 'NOTE'
        txt += ('\nTreatment effects for specific treatment groups (so-called '
                'treatment effects on the treated or non-treated) can only be '
                'provided if the data provided for prediction contains a '
                'treatment variable (which is not required for the other '
                'effects).'
                )

    return txt + '\n' * empty_lines_end


def mcf_prediction(mcf_o, empty_lines_end):
    """Create the general text for predictions."""
    txt = (f'Training uses {mcf_o.gen_dict["mp_parallel"]} CPU '
           f'{"cores" if mcf_o.gen_dict["mp_parallel"] > 1 else "core"}. ')
    if mcf_o.int_dict["cuda"] and mcf_o.gen_dict['iate_eff']:
        txt += '\nGPU may be used for predicting IATEs.'
    return txt + '\n' * empty_lines_end


def mcf_forest(mcf_o, empty_lines_end, iv=False):
    """Text the forest method."""
    if iv:
        dic_1st, rep_1st = mcf_o.cf_dict, mcf_o.report['cf_1st']
        dic_redf, rep_redf = mcf_o.cf_dict, mcf_o.report['cf_redf']
        dic, rep = dic_redf, rep_redf
    else:
        dic, rep = mcf_o.cf_dict, mcf_o.report['cf']

    txt = '\n' * 2 + 'METHOD and tuning parameters'
    if iv:
        txt += ('\nThe method used for forest building for the 1st stage is '
                f'{dic_1st["estimator_str"]}. {dic_redf["estimator_str"]} '
                'is used for the reduced form. ')
    else:
        txt += f'\n{dic["estimator_str"]} is used for forest building.'

    if dic['compare_only_to_zero']:
        txt += ('MSE is only computed for IATEs comparing all treatments to '
                'the first (control) treatment.')
    if iv:
        txt += (f'\nThe causal forest consists of {dic_1st["boot"]}, '
                f'{dic_redf["boot"]} (1st stage, reduced form) trees.'
                f'\nThe minimum leaf size is {rep_1st["n_min"]}, '
                f'{rep_redf["n_min"]}.'
                '\nThe number of variables considered for each split is '
                f'{rep_1st["m_split"]}, {rep_redf["m_split"]}.'
                '\nThe share of data used in the subsamples for forest '
                f'building is {rep_1st["f_share_build"]:4.0%}, '
                f'{rep_redf["f_share_build"]:4.0%}.'
                '\nThe share of the data used in the subsamples for forest '
                f'evaluation (outcomes) is {rep_1st["f_share_fill"]:4.0%}, '
                f'{rep_redf["f_share_fill"]:4.0%}.'
                f'\nAlpha regularity is set to {rep_1st["alpha"]:3.0%}, '
                f'{rep_redf["alpha"]:3.0%}. \n\n{rep_1st["y_name_tree"]}, '
                f'{rep_redf["y_name_tree"]} are the outcome variable used for '
                'splitting ')
    else:
        txt += (f'\nThe causal forest consists of {dic["boot"]} trees.'
                f'\nThe minimum leaf size is {rep["n_min"]}.'
                '\nThe number of variables considered for each split is '
                f'{rep["m_split"]}.'
                '\nThe share of data used in the subsamples for forest '
                f'building is {rep["f_share_build"]:4.0%}.'
                '\nThe share of the data used in the subsamples for forest '
                f'evaluation (outcomes) is {rep["f_share_fill"]:4.0%}.'
                f'\nAlpha regularity is set to {rep["alpha"]:3.0%}.'
                f'\n\n{rep["y_name_tree"]} is the outcome variable used for '
                'splitting ')
    if mcf_o.lc_dict["yes"]:
        txt += '(locally centered).'
    var = rep["Features"].replace('_prime', '')
    txt += f'\nThe features used for splitting are {var}.'

    txt += '\n' * 2 + 'CHARACTERISTICS OF THE TRAINED CAUSAL FORESTS'
    if iv:
        txt += (f'\nEach tree has on average {rep_1st["leaves_mean"]:5.2f}, '
                f'{rep_redf["leaves_mean"]:5.2f} (1st stage, reduced form) '
                'leaves. '
                '\nEach leaf contains on average '
                f'{rep_1st["obs_leaf_mean"]:4.1f}, '
                f'{rep_redf["obs_leaf_mean"]:4.1f}  observations. The median # '
                'of observations per leaf is '
                f'{rep_1st["obs_leaf_med"]:2.0f}, '
                f'{rep_redf["obs_leaf_med"]:2.0f}. '
                f'\nThe smallest leaves have {rep_1st["obs_leaf_min"]:2.0f}, '
                f'{rep_redf["obs_leaf_min"]:2.0f} observations.'
                f'\nThe largest leaf has {rep_1st["obs_leaf_max"]:3.0f}, '
                f'{rep_redf["obs_leaf_max"]:3.0f} observations.'
                f'\n{rep_1st["share_leaf_merged"]:5.2%}, '
                f'{rep_redf["share_leaf_merged"]:5.2%} of the leaves were'
                ' merged when populating the forest with outcomes. Note that '
                'the data used populating the leaves are from the honesty '
                'sample for the reduced form, while the 1st stage is using the '
                'same data as was used to build the trees (thus, it is not '
                'honest).'
                )
    else:
        txt += (f'\nEach tree has on average {rep["leaves_mean"]:5.2f} leaves. '
                f'\nEach leaf contains on average {rep["obs_leaf_mean"]:4.1f} '
                'observations. The median # of observations per leaf is '
                f'{rep["obs_leaf_med"]:2.0f}. '
                f'\nThe smallest leaves have {rep["obs_leaf_min"]:2.0f} '
                'observations.'
                f'\nThe largest leaf has {rep["obs_leaf_max"]:3.0f} '
                'observations.'
                )
        txt += (f'\n{rep["share_leaf_merged"]:5.2%} of the leaves were merged '
                'when populating the forest with outcomes from the honesty '
                'sample.'
                )

    if dic['folds'] > 1 or mcf_o.gen_dict['iate_eff']:
        txt += '\n' * 2 + 'NOTE'
        if dic['folds'] > 1 and mcf_o.gen_dict['iate_eff']:
            txt += 'S'
        if dic['folds'] > 1:
            txt += ('\nTo reduce computational demands, data is randomly '
                    f'splitted in {dic["folds"]} folds. In each fold, forests '
                    'and effects are estimated. Subsequently, effects are '
                    'averaged over the {dic["folds"]} folds.')
        if mcf_o.gen_dict['iate_eff']:
            txt += ('\nFor the estimation of the "efficient" IATEs, the role '
                    'of the samples used for building the forest and '
                    'populating it are reversed. Subsequently, the two sets of '
                    'estimates for the IATEs are averaged.')
    return txt + '\n' * empty_lines_end


def mcf_local_center(mcf_o, empty_lines_end, iv=False):
    """Text the local centering results."""
    txt = '\nMETHOD'
    txt += ('\nLocal centering is based on training a regression to predict '
            'the outcome variable conditional on the features (without '
            'the treatment). The regression method is selected among various '
            'versions of Random Forests, Support Vector Machines, Boosting '
            'methods, and Neural Networks of scikit-learn. The best method is '
            'selected by minimizing their out-of-sample Mean Squared Error ')
    if mcf_o.lc_dict["cs_cv"]:
        txt += f'using {mcf_o.lc_dict["cs_cv_k"]}-fold cross-validation. '
    else:
        txt += (f'using a test sample ({mcf_o.lc_dict["cs_share"]:5.2%} of the '
                'training data). ')
    txt += ('The full set of results of the method selection step are '
            f'contained in {mcf_o.gen_dict["outfiletext"]}. '
            '\nThe respective out-of-sample predictions are '
            'subtracted from the observed outcome in the training data used to '
            'build the forest. ')
    if mcf_o.lc_dict["cs_cv"]:
        txt += ('These out-of-sample predictions are generated by '
                f'{mcf_o.lc_dict["cs_cv_k"]}-fold cross-validation.')
    else:
        txt += (f'\n{mcf_o.lc_dict["cs_share"]:5.2%} of the training data is '
                ' used for local centering only.')
    txt += '\n\nRESULTS'
    if iv:
        txt += ('\nOut-of-sample fit (1st stage)'
                f' for {mcf_o.report["lc_r2_1st"]} ')
        txt += ('\nOut-of-sample fit (reduced form)'
                f' for {mcf_o.report["lc_r2_redform"]} ')
    else:
        txt += f'\nOut-of-sample fit for {mcf_o.report["lc_r2"]} '
    return txt + '\n' * empty_lines_end


def mcf_common_support(mcf_o, empty_lines_end, train=True, iv=False,
                       alloc=False):
    """Text the common support results."""
    txt = ''
    if train:
        txt += '\nMETHOD'
        txt += ('\nThe common support analysis is based on checking the '
                'overlap in the out-of-sample predictions of the propensity ')
        if iv:
            txt += 'scores (PS) for the different values of the instrument. '
        else:
            txt += 'scores (PS) for the different treatment arms. '
        txt += ('PSs are '
                'estimated by random forest classifiers. Overlap is '
                'operationalized by computing cut-offs probabilities of '
                'the PSs (ignoring the first '
                )
        txt += 'instrument value' if iv else 'treatment arm'
        txt += ', because probabilities add to 1 over all '
        txt += 'instrument values' if iv else 'treatment arms'
        txt += ('). These cut-offs are subsequently also applied to the data '
                'used for predicting the effects.')
        if mcf_o.cs_dict['type'] == 1:
            quant = mcf_o.cs_dict["quantil"]
            string = "min / max" if quant == 1 else (str(quant*100)
                                                     + "% quantile")
            txt += f'\nOverlap is determined by the {string} rule.'
        else:
            min_p = mcf_o.cs_dict['min_p']
            txt += (f'\nYou set the upper bound of the PS to {1-min_p} and the'
                    f' lower bound to {min_p}')
        if mcf_o.cs_dict["adjust_limits"] > 0 and mcf_o.cs_dict['type'] == 1:
            txt += ('\nCut-offs for PS are widened by'
                    f' {mcf_o.cs_dict["adjust_limits"]}.')
        if mcf_o.lc_dict['cs_cv']:
            txt += ('\nOut-of-sample predictions are generated by '
                    f'{mcf_o.lc_dict["cs_cv_k"]}-fold cross-validation.')
        else:
            txt += (f'\n{mcf_o.lc_dict["cs_share"]:5.2%} of the training data '
                    ' is used for common support estimation only.')
        share_del = mcf_o.report['cs_t_share_deleted']
        obs_keep = mcf_o.report['cs_t_obs_remain']
        txt += '\n' * 2 + 'RESULTS'
    else:
        if alloc and 'alloc_cs_p_share_deleted' in mcf_o.report:
            share_del = mcf_o.report['alloc_cs_p_share_deleted']
            obs_keep = mcf_o.report['alloc_cs_p_obs_remain']
        else:
            share_del = mcf_o.report['cs_p_share_deleted']
            obs_keep = mcf_o.report['cs_p_obs_remain']
    txt += (f'\nShare of observations deleted: {share_del:5.2%}'
            f'\nNumber of observations remaining: {obs_keep:5}')
    if share_del > 0.1:
        txt += ('\nWARNING: Check output files whether the distribution of the '
                'features changed due to the deletion of part of the data.')
    return txt + '\n' * empty_lines_end


def mcf_feature_selection(mcf_o, empty_lines_end, iv=False):
    """Text the feature selection result."""
    txt = '\nMETHOD'
    txt += ('\nA Random Forest is estimated for the propensity score '
            '(classifier) and the outcome (classifier or regression, '
            'specification is without the treatment). If by deleting a '
            'specific variable the values of BOTH objective functions '
            ' (evaluated with out-of-bag data) are reduced by '
            f'less than {mcf_o.fs_dict["rf_threshold"]:5.2%}, '
            'then the variable is removed. Care is taken for variables that '
            'are highly correlated with other variables, or dummies, or '
            'variables that should not be removed for other reasons '
            '(computing heterogeneity or checking balancing).')
    if iv:
        txt += '\nFeature selection is based on the 1st stage only.'
    other = mcf_o.fs_dict['other_sample']
    txt += '\nFeature selection is performed '
    if other:
        txt += ('on an independent random sample '
                f'({mcf_o.fs_dict["other_sample_share"]:5.2%} of the training '
                'data). The training data is reduced accordingly.')
    else:
        txt += 'on the same data as the other parts of training.'
    txt = '\n' * 2
    if mcf_o.report['fs_vars_deleted']:
        txt += ('The following variables were deleted: '
                f'{" ".join(mcf_o.report["fs_vars_deleted"])}')
    else:
        txt += 'Feature selection did not delete any variables.'
    return txt + '\n' * empty_lines_end


def mcf_descriptives(mcf_o, empty_lines_end):
    """Provide basic descriptive information about training data."""
    screen = mcf_o.dc_dict['screen_covariates']
    percorr = mcf_o.dc_dict['check_perfectcorr']
    clean = mcf_o.dc_dict['clean_data']
    dummy = mcf_o.dc_dict['min_dummy_obs']
    txt = ''
    if screen or percorr or clean or dummy:
        txt += '\nMETHOD'
        if screen:
            txt += '\nVariables without variation are removed.'
        if percorr:
            txt += ('\nVariables that are perfectly correlated with other '
                    'variables are removed.')
        if dummy:
            txt += ('\nDummy variables with less than '
                    f'{mcf_o.dc_dict["min_dummy_obs"]} observations in the '
                    'smaller group are removed.')
        if clean:
            txt += ('\nRows with any missing values for variables needed for'
                    ' training are removed.')
    txt += '\n\n' * 2 + 'RESULTS'
    if mcf_o.report['removed_vars']:
        txt += ('The following control variables were removed from the data: '
                f'{" ".join(mcf_o.report["removed_vars"])}')
    else:
        txt += '\nNo relevant variables were removed.'
    txt += ('\nSample size of training data: '
            f'{mcf_o.report["training_obs"][0]} ')
    if mcf_o.report["training_obs"][1] > 0:
        del_obs = mcf_o.report["training_obs"][1]
        txt += f'\n    - Deleted number of observations: {del_obs}'
        remain_obs = mcf_o.report["training_obs"][0] - del_obs
        txt += f'\nRemaining number of observations: {remain_obs}'
    else:
        txt += '   (no observations removed).'
    return txt + '\n' * empty_lines_end


def mcf_training(mcf_o, empty_lines_end):
    """Text basic training information."""
    txt = (f'Training uses {mcf_o.gen_dict["mp_parallel"]} CPU '
           f'{"cores" if mcf_o.gen_dict["mp_parallel"] > 1 else "core"}.')
    if mcf_o.int_dict["cuda"] and mcf_o.cf_dict['match_nn_prog_score']:
        txt += '\nGPU is likely to be used for Mahalanobis matching.'
    return txt + '\n' * empty_lines_end


def add_effects(p_dict, gen_dict):
    """Build list with effects to be shown."""
    effects = ['Average Treatment Effect (ATE)']
    if p_dict['atet']:
        effects.append('ATE for the treatment groups (ATET)')
    if p_dict['gate']:
        effects.append('Group Average Treatment Effect (GATE)')
    if p_dict['gatet']:
        effects.append('GATE for the treatment groups (GATET)')
    if p_dict['bgate']:
        effects.append('Balanced GATE (BGATE)')
    if p_dict['cbgate']:
        effects.append('Fully Balanced (CAUSAL) GATE (CBGATE)')
    if p_dict['iate']:
        effects.append('Individualized Average Treatment Effect (IATE)')
    if gen_dict['iate_eff']:
        effects.append('Efficient IATE')
    return effects
