"""Created on Thu Mai 11 11:05:15 2023.

Contains the functions needed for feature selection.
@author: MLechner
-*- coding: utf-8 -*-
"""
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

from mcf.mcf_general import to_numpy_big_data
from mcf.mcf_print_stats import print_mcf

if TYPE_CHECKING:
    from mcf.mcf_init import VarCfg as VarCfgMcf
    from mcf.mcf_init import GenCfg as GenCfgMcf
    from mcfoptpolicy_init import VarCfg as VarCfgOptpol
    from mcfoptpolicy_init import GenCfg as GenCfgOptpol
    from mcf.optpolicy_main import OptimalPolicy, OptimalPolicyVersion

type BoolLike = bool | None
type IntLike = int | None
type FloatLike = float | None
type NumberLike = int | float | None


@dataclass(slots=True, kw_only=True)
class FsCfg:
    """Feature-selection configuration (canonical form).

    Notes
    -----
    - rel_vi_threshold, rel_vi_threshold_d, rel_vi_threshold_y are stored as a fraction in [0, 1].
    - rel_vi_threshold is only use for optimal policy.
    - rel_vi_threshold_d and rel_vi_threshold_y are only used for mcf.
    - other_sample_share is in [0, 0.5], but forced to 0 if yes=False or other_sample=False.
    """

    yes: bool = False
    rel_vi_threshold: float = 0.0     # use for optimal policy
    rel_vi_threshold_y: float = 0.0
    rel_vi_threshold_d: float = 0.0
    rel_vi_keep_if: str = 'y_or_d_relevant'
    other_sample: bool = True
    other_sample_share: float = 0.33

    def __post_init__(self) -> None:
        """Keep values valid even if fields are modified after init."""
        if not 0.0 <= self.rel_vi_threshold <= 1.0:
            self.rel_vi_threshold = 0.0

        if not 0.0 <= self.other_sample_share <= 0.5:
            self.other_sample_share = 0.33

        if (not self.other_sample) or (not self.yes):
            self.other_sample_share = 0.0

    @classmethod
    def from_args(cls, *,
                  rel_vi_threshold: NumberLike = None,
                  rel_vi_threshold_y: NumberLike = None,
                  rel_vi_threshold_d: NumberLike = None,
                  rel_vi_keep_if: str = 'y_or_d_relevant',
                  other_sample: BoolLike = None,
                  other_sample_share: FloatLike = None,
                  yes: BoolLike = None,
                  ) -> 'FsCfg':
        """Get input and normalize parameters."""
        # yes only if explicitly True
        yes_b = yes is True

        # rel_vi_threshold: fraction (0-1)
        default = 0.0
        rel_vi_frac = check_threshold(rel_vi_threshold, default)   # Used for optimal policy only
        rel_vi_frac_y = check_threshold(rel_vi_threshold_y, default)
        rel_vi_frac_d = check_threshold(rel_vi_threshold_d, default)

        if rel_vi_keep_if in ('y_relevant', 'y_or_d_relevant', 'y_and_d_relevant'):
            keep_if = rel_vi_keep_if      # NOT used for optimal policy
        else:
            keep_if = 'y_or_d_relevant'

        # other_sample defaults to True unless explicitly False
        other_b = other_sample is not False

        # share in [0, 0.5], else default 0.33
        if other_sample_share is None or not 0.0 <= other_sample_share <= 0.5:
            share = 0.33
        else:
            share = float(other_sample_share)

        # disable share if not using other sample or FS not active
        if (not other_b) or (not yes_b):
            share = 0.0

        return cls(yes=yes_b,
                   rel_vi_threshold=rel_vi_frac,
                   rel_vi_threshold_y=rel_vi_frac_y,
                   rel_vi_threshold_d=rel_vi_frac_d,
                   other_sample=other_b,
                   other_sample_share=share,
                   rel_vi_keep_if=keep_if,
                   )


def check_threshold(rel_vi_threshold: NumberLike=0.0, default: float=0.0) -> float:
    """Check values of threshold variable."""
    if rel_vi_threshold is None or rel_vi_threshold <= 0 or rel_vi_threshold > 100:
        return default

    if 1 < rel_vi_threshold <= 100:
        return float(rel_vi_threshold / 100)

    return float(rel_vi_threshold)


def feature_selection_optpol(optp_: Union['OptimalPolicy', 'OptimalPolicyVersion'],
                             data_df: pd.DataFrame,
                             ) -> pd.DataFrame:
    """Perform feature selection for optimal policy."""
    (data_optp_df, _, var_cfg, var_x_type, var_x_values
     ) = feature_selection(data_df,
                           gen_cfg=copy(optp_.gen_cfg),
                           fs_cfg=copy(optp_.fs_cfg),
                           var_cfg=copy(optp_.var_cfg),
                           var_x_type=copy(optp_.var_x_type),
                           var_x_values=copy(optp_.var_x_values),
                           boot = 1000,
                           versions = False,
                           replication = False,
                           obs_bigdata = 1_000_000,
                           mcf_estimation = False,
                           )
    # Adjust optpolicy instance
    optp_.var_cfg = var_cfg
    optp_.var_x_type = var_x_type
    optp_.var_x_values = var_x_values

    return data_optp_df


def feature_selection(data_df: pd.DataFrame, *,
                      gen_cfg: Union['GenCfgMcf', 'GenCfgOptpol'],
                      fs_cfg: 'FsCfg',
                      var_cfg: Union['VarCfgMcf', 'VarCfgOptpol'],
                      var_x_type:  dict[str, int] | None = None,
                      var_x_values: dict | None = None,
                      boot: IntLike = None,
                      versions: bool = False,
                      replication: IntLike = None,
                      obs_bigdata: NumberLike = 1_000_000,
                      mcf_estimation: bool = True,
                      ) -> tuple[pd.DataFrame,
                                 list[str],
                                 Union['VarCfgMcf', 'VarCfgOptpol'],
                                 dict, dict
                                 ]:
    """
    Feature selection using scikit-learn.

    Step 1: Set-aside a random sample for feature selection analysis (option)
    Step 2: Estimate classifier forest for treatment
    Step 3: Estimate regression (>=10 values) or classifier (<10 values) forest
            for outcome
            Estimation is 75%, prediction on 25% random sample
    Step 4: Permutate variable groups; create variable importance measure based
            on r2 or accuary; compute relative loss compared to baseline.
    Step 5: Deselect variable if loss is less than threshold.
    Step 6: Do not deselect variable if on list of protected variables.
    Repeat steps 2-6 until no variable is deselected.
    """
    welcome_print(gen_cfg)
    max_workers = 1 if replication else gen_cfg.mp_parallel

    # Only main treatments
    if mcf_estimation:
        d_name = [var_cfg.d_name[0]] if versions else var_cfg.d_name
    else:
        d_name = None

    # Set aside sample for feature selection
    if fs_cfg.other_sample:
        data_mcf_df, data_fs_df = train_test_split(data_df,
                                                   test_size=fs_cfg.other_sample_share,
                                                   random_state=42,
                                                   )
    else:
        data_mcf_df, data_fs_df = data_df.copy(), data_df.copy()
    print_basic_sample_info(gen_cfg, data_fs_df, data_mcf_df)

    # Deal with unordered variables --> create dummies if such variables exist
    (data_fs_df, x_names, x_names_org, names_unordered, unordered_dummy_names
     ) = unordered_variables_to_dummy(var_x_type=var_x_type, x_df=data_fs_df)

    # Test sample is used for variable importance calculations
    if mcf_estimation:
        (y_train_df, y_test_df, d_train_df, d_test_df, x_train_df, x_test_df
         ) = train_test_split(data_fs_df[var_cfg.y_tree_name],
                              data_fs_df[d_name],
                              data_fs_df[x_names],
                              test_size=0.25,
                              random_state=42,
                              )
    else:  # Optimal policy
        (scores_train_df, scores_test_df, x_train_df, x_test_df
         ) = train_test_split(data_fs_df[var_cfg.polscore_name],
                              data_fs_df[x_names],
                              test_size=0.25,
                              random_state=42,
                              )
        # Substract first colum (PO of treatment 0) to focus on effect differences
        scores_train_df = scores_train_df.iloc[:, 1:].sub(scores_train_df.iloc[:, 0], axis=0)
        scores_test_df = scores_test_df.iloc[:, 1:].sub(scores_test_df.iloc[:, 0], axis=0)

    # Define variables that are not allowed to be deleted
    if mcf_estimation:
        forbidden_to_delete_vars = (var_cfg.x_name_always_in
                                    + var_cfg.x_name_remain
                                    + var_cfg.z_name
                                    + var_cfg.z_name_cont
                                    + var_cfg.x_name_balance_bgate
                                    )
    else:
        forbidden_to_delete_vars = []   # all variables can be removed for optimal policy

    x_names_to_delete = []
    max_iter = len(x_names_org) - 2   # At least two variables will not be removed
    vi_information = None
    for iter_i in range(max_iter):
        # Interate until no more variables are removed

        # Adjust datasets to current number of active variables captured by x_names
        x_train_df, x_test_df = x_train_df[x_names], x_test_df[x_names]

        # Baseline estimate for the active number of variables
        if mcf_estimation:
            if fs_cfg.rel_vi_keep_if == 'y_relevant':
                score_d_full = d_np = d_rf_obj = None
            else:
                score_d_full, d_np, d_rf_obj = randomforests_estimate(y_train_df=d_train_df,
                                                                      x_train_df=x_train_df,
                                                                      y_test_df=d_test_df,
                                                                      x_test_df=x_test_df,
                                                                      classifier=True,
                                                                      boot=boot,
                                                                      max_workers=max_workers,
                                                                      obs_bigdata=obs_bigdata,
                                                                      )
            y_as_classifier = int(y_train_df.nunique().iloc[0]) < 10
            score_y_full, y_np, y_rf_obj = randomforests_estimate(y_train_df=y_train_df,
                                                                  x_train_df=x_train_df,
                                                                  y_test_df=y_test_df,
                                                                  x_test_df=x_test_df,
                                                                  classifier=y_as_classifier,
                                                                  boot=boot,
                                                                  max_workers=max_workers,
                                                                  obs_bigdata=obs_bigdata,
                                                                  )
        else:
            no_of_scores = scores_train_df.shape[1]
            score_y_full = np.zeros(no_of_scores)
            y_np = np.zeros_like(scores_test_df)
            y_rf_obj = [None] * no_of_scores
            y_as_classifier = (scores_train_df.nunique() < 10).tolist()
            for i in range(no_of_scores):
                (score_y_full[i], y_np[:, i], y_rf_obj[i]
                 ) = randomforests_estimate(y_train_df=scores_train_df.iloc[:, i],
                                            x_train_df=x_train_df,
                                            y_test_df=scores_test_df.iloc[:, i],
                                            x_test_df=x_test_df,
                                            classifier=y_as_classifier[i],
                                            boot=boot,
                                            max_workers=max_workers,
                                            obs_bigdata=obs_bigdata,
                                            )
        # Compute variable importance for all variables (dummies as group)
        if gen_cfg.with_output:
            if mcf_estimation:
                if fs_cfg.rel_vi_keep_if == 'y_relevant':
                    index = ['score_w/o_x_y',]
                    index_rel = ['rel_diff_y_%',]
                else:
                    index = ['score_w/o_x_y', 'score_w/o_x_d']
                    index_rel = ['rel_diff_y_%', 'rel_diff_d_%']
            else:
                index = [iate + '_w/o_x' for iate in scores_train_df.columns]
                index_rel = [riate + '_diff_%' for riate in scores_train_df.columns]
            vi_information = pd.DataFrame(columns=x_names_org, index=index+index_rel)

        x_name_deleted_iter_i = None
        for name in x_names_org:
            if name in forbidden_to_delete_vars or name in x_names_to_delete:
                continue
            x_all_rnd_df = randomize_one_variable(name=name,
                                                  names_unordered=names_unordered,
                                                  unordered_dummy_names=unordered_dummy_names,
                                                  x_df=x_test_df,
                                                  )
            if mcf_estimation:
                if fs_cfg.rel_vi_keep_if == 'y_relevant':
                    d_score = d_rel_diff = None
                else:
                    d_score, d_rel_diff = score_fct(x_dat_df=x_all_rnd_df,
                                                    y_np=d_np,
                                                    score_full=score_d_full,
                                                    rf_obj=d_rf_obj,
                                                    classifier=True,
                                                    obs_bigdata=obs_bigdata,
                                                    )
                y_score, y_rel_diff = score_fct(x_dat_df=x_all_rnd_df,
                                                y_np=y_np,
                                                score_full=score_y_full,
                                                rf_obj=y_rf_obj,
                                                classifier=y_as_classifier,
                                                obs_bigdata=obs_bigdata,
                                                )
                if gen_cfg.with_output:
                    if fs_cfg.rel_vi_keep_if == 'y_relevant':
                        vi_information[name] = [y_score, y_rel_diff * 100]
                    else:
                        vi_information[name] = [y_score, d_score,
                                                y_rel_diff * 100, d_rel_diff * 100
                                                ]
                delete_name = delete_variable_fct(y_rel_diff=y_rel_diff,
                                                  d_rel_diff=d_rel_diff,
                                                  threshold_y=fs_cfg.rel_vi_threshold_y,
                                                  threshold_d=fs_cfg.rel_vi_threshold_d,
                                                  keep_if=fs_cfg.rel_vi_keep_if,
                                                  )
                if delete_name:   # Variable found that fulfills criteria and thus should be removed
                    x_names_to_delete.append(name)
                    x_name_deleted_iter_i = name
                    # break out of inner loop which loops all currently active variables
                    break
            else:
                y_score = np.zeros(no_of_scores)
                y_rel_diff = np.zeros(no_of_scores)
                for i in range(no_of_scores):
                    y_score[i], y_rel_diff[i] = score_fct(x_dat_df=x_all_rnd_df,
                                                          y_np=y_np[:, i],
                                                          score_full=score_y_full[i],
                                                          rf_obj=y_rf_obj[i],
                                                          classifier=y_as_classifier[i],
                                                          obs_bigdata=obs_bigdata,
                                                          )
                if gen_cfg.with_output:
                    vi_information[name] = y_score.tolist() + (y_rel_diff * 100).tolist()

                if np.all(y_rel_diff <= fs_cfg.rel_vi_threshold):
                    # Variable found that fulfills criteria and thus should be removed
                    x_names_to_delete.append(name)
                    x_name_deleted_iter_i = name
                    # break out of inner loop which loops all currently active variables
                    break

        print_results_of_feature_selection(gen_cfg,
                                           score_y_full,
                                           score_d_full if mcf_estimation else None,
                                           vi_information,
                                           iteration=iter_i,
                                           name_deleted=x_name_deleted_iter_i,
                                           threshold=fs_cfg.rel_vi_threshold,
                                           threshold_y=fs_cfg.rel_vi_threshold_y,
                                           threshold_d=fs_cfg.rel_vi_threshold_d,
                                           keep_if=fs_cfg.rel_vi_keep_if,
                                           mcf_estimation=mcf_estimation,
                                           summary=False,
                                           )
        # If no variable left for removal, jump out of outer loop
        if x_name_deleted_iter_i is None:
            break

        # If variables to delete found, adjust x_names
        x_names = delete_variable(x_names, x_name_deleted_iter_i, unordered_dummy_names)

    # End of feature selection iteration
    if x_names_to_delete:
        names_to_remove = [name for name in x_names_to_delete
                           if name not in forbidden_to_delete_vars
                           ]
        if names_to_remove:
            for name_weg in names_to_remove:
                var_x_type.pop(name_weg)
                var_x_values.pop(name_weg)
                var_cfg.x_name.remove(name_weg)
        print_deleted(gen_cfg, names_to_remove, var_cfg.x_name, summary=True)
    else:
        print_no_deleted(gen_cfg)
        names_to_remove = []

    return data_mcf_df, names_to_remove, var_cfg, var_x_type, var_x_values


def delete_variable_fct(*,
                        y_rel_diff: float,
                        d_rel_diff: float,
                        threshold_y: float,
                        threshold_d: float | None,
                        keep_if: str,
                        ) -> bool | None:
    """Check if variable importance statistics is fullfilling criteria to delete variable."""
    if keep_if == 'y_relevant':
        return y_rel_diff <= threshold_y

    if keep_if == 'y_or_d_relevant' and threshold_d is not None:
        return y_rel_diff <= threshold_y and d_rel_diff <= threshold_d

    if keep_if == 'y_and_d_relevant' and threshold_d is not None:
        return y_rel_diff <= threshold_y or d_rel_diff <= threshold_d

    return None


def unordered_variables_to_dummy(*,
                                 var_x_type: dict[str, int] | None = None,
                                 x_df: pd.DataFrame,
                                 ) -> tuple[pd.DataFrame,
                                            list[str],
                                            list[str],
                                            list[str],
                                            dict,
                                            ]:
    """Transform unordered variables to dummies and keep records."""
    all_names = var_x_type.keys()
    # Remove all covariates that are discretized for GATE estimation
    x_cat_names = [name + 'catv' for name in all_names if name + 'catv' in all_names]
    x_names = [name for name in all_names if name not in x_cat_names]
    x_names_org = x_names.copy()
    names_unordered = []
    for x_name in x_names:
        var = var_x_type[x_name]
        if ((isinstance(var, (int, float)) and var > 0)
            or (isinstance(var, str) and var_x_type[x_name] in ('disc', 'unord'))
                ):
            names_unordered.append(x_name)
    if names_unordered:  # List is not empty
        unordered_dummy_names = {}  # Dict contains dummy name correspondence
        for idx, name in enumerate(names_unordered):
            x_dummies = pd.get_dummies(x_df[name], columns=[name], dtype=int)
            x_dummies_names = [name + str(ddd) for ddd in x_dummies.columns]
            unordered_dummy_names[name] = x_dummies.columns = x_dummies_names
            if idx == 0:
                x_all_dummies = x_dummies
            else:
                x_all_dummies = pd.concat([x_all_dummies, x_dummies], axis=1)
        data_fs_df = pd.concat([x_df, x_all_dummies], axis=1)
        # Remove names of unordered variables
        x_names = [name for name in x_names if name not in names_unordered]
        # Add their dummy names
        x_names.extend(x_all_dummies.columns)
    else:
        data_fs_df = x_df
        unordered_dummy_names = None

    return data_fs_df, x_names, x_names_org, names_unordered, unordered_dummy_names


def randomize_one_variable(*,
                           name: str,
                           names_unordered: list[str] | None,
                           unordered_dummy_names: list[str] | None,
                           x_df: pd.DataFrame,
                           ) -> pd.DataFrame:
    """Reshufle one variable to make its content uninformative."""
    names_to_shuffle = unordered_dummy_names[name] if name in names_unordered else name
    x_all_rnd_df = x_df.copy().reset_index(drop=True)
    x_rnd_df = x_df[names_to_shuffle].sample(frac=1, random_state=42)
    x_all_rnd_df[names_to_shuffle] = x_rnd_df.reset_index(drop=True)

    return x_all_rnd_df


def score_fct(*,
              x_dat_df: pd.DataFrame,
              y_np: NDArray,
              score_full: float,
              rf_obj: RandomForestClassifier | RandomForestRegressor,
              classifier: bool = False,
              obs_bigdata: int = 1_000_000,
              ) -> tuple[float, float]:
    """Compute absolute and relative score."""
    y_pred = rf_obj.predict(to_numpy_big_data(x_dat_df, obs_bigdata))
    score = score_for_y(y_np, y_pred, classifier)
    rel_score_diff = (score_full - score) / score_full

    return score, rel_score_diff


def randomforests_estimate(*,
                           y_train_df: pd.DataFrame,
                           x_train_df: pd.DataFrame,
                           y_test_df: pd.DataFrame,
                           x_test_df: pd.DataFrame,
                           classifier: bool = False,
                           boot: int = 1_000,
                           max_workers: int = 1,
                           obs_bigdata: int = 1_000_000,
                           ) -> tuple[float,
                                      NDArray,
                                      RandomForestClassifier | RandomForestRegressor
                                      ]:
    """Estimate with random forest classifier or regressor."""
    params = {'n_estimators': boot, 'max_features': 'sqrt', 'bootstrap': True,
              'oob_score': False, 'n_jobs': max_workers, 'random_state': 42, 'verbose': False,
              }
    rf_obj = RandomForestClassifier(**params) if classifier else RandomForestRegressor(**params)
    rf_obj.fit(to_numpy_big_data(x_train_df, obs_bigdata),
               to_numpy_big_data(y_train_df, obs_bigdata).ravel(),
               )
    y_pred = rf_obj.predict(x_test_df.to_numpy())
    y_test = y_test_df.to_numpy().ravel()
    score = score_for_y(y_test, y_pred, classifier)

    return score, y_test, rf_obj


def score_for_y(y_true: NDArray, y_pred: NDArray, classifier: bool) -> float:
    """Compute score dependending on type of y."""
    if classifier:
        return accuracy_score(y_true, y_pred, normalize=True)

    return r2_score(y_true, y_pred)


def check_correlation(*,
                      gen_cfg: Union['GenCfgMcf', 'GenCfgOptpol'],
                      data_df: pd.DataFrame,
                      names_to_remove: list[str],
                      summary: bool = False,
                      ) -> list[str]:
    """Check correlations of removed variables and adjust list."""
    # Check if two vars to remove are highly correlated.
    # Use full data for this exercise.
    # No longer used as it is irrelevant when deleting variables sequentially
    weg_corr = data_df[names_to_remove].corr()
    names_weg, do_not_remove = names_to_remove.copy(), []
    if gen_cfg.with_output:
        with pd.option_context('display.max_rows', None,
                               'display.expand_frame_repr', True,
                               'display.width', 120,
                               'chop_threshold', 1e-13
                               ):
            print_mcf(gen_cfg,
                      '\nCorrelation of variables to be deleted\n',
                      weg_corr,
                      summary=summary,
                      )
    for idx, name_weg in enumerate(names_weg[:-1]):
        for name_weg2 in names_weg[idx+1:]:
            if np.abs(weg_corr.loc[name_weg, name_weg2]) > 0.5:
                do_not_remove.append(name_weg)

    names_to_remove = [name for name in names_weg if name not in do_not_remove]

    return names_to_remove


def delete_variable(names, name_to_be_deleted, unordered_dummy_names):
    """Delete variable from list of variables (including unordered case)."""
    if name_to_be_deleted in names:
        names.remove(name_to_be_deleted)
        return names

    names_remove = unordered_dummy_names[name_to_be_deleted]
    names_red = [x for x in names if x not in names_remove]

    return names_red


def print_deleted(gen_cfg: Union['GenCfgMcf', 'GenCfgOptpol'],
                  names_to_remove: list[str],
                  x_name: list[str],
                  summary: bool = True,
                  ) -> None:
    """Print deleted variables."""
    if gen_cfg.with_output:
        print_mcf(gen_cfg,
                  '\nFeature selection used!'
                  '\nVariables deleted: ' + ' '.join(names_to_remove)
                  + '\nVariables kept:    ' + ' '.join(x_name)
                  + '\n' + '-' * 100,
                  summary=summary,
                  )


def print_no_deleted(gen_cfg: Union['GenCfgMcf', 'GenCfgOptpol'],  summary: bool = True) -> None:
    """Print nothing deleted."""
    if gen_cfg.with_output:
        print_mcf(gen_cfg,
                  '\nNo variables removed in feature selection' + '\n' + '-' * 100,
                  summary=summary,
                  )


def print_basic_sample_info(gen_cfg: Union['GenCfgMcf', 'GenCfgOptpol'],
                            data_fs_df: pd.DataFrame,
                            data_mcf_df: pd.DataFrame,
                            summary: bool = False,
                            ) -> None:
    """Print basic info of test and training samples."""
    if gen_cfg.with_output:
        print_mcf(gen_cfg,
                  f'\nSample used for feature selection {len(data_fs_df)} '
                  f'\nSample used for mcf estimation {len(data_mcf_df)}',
                  summary=summary,
                  )

def welcome_print(gen_cfg: Union['GenCfgMcf', 'GenCfgOptpol'], summary: bool = False) -> None:
    """Print module name."""
    if gen_cfg.with_output:
        print_mcf(gen_cfg, '\nFeature selection', summary=summary)


def print_results_of_feature_selection(gen_cfg: Union['GenCfgMcf', 'GenCfgOptpol'],
                                       score_y_full: float | NDArray,
                                       score_d_full: float | None,
                                       vi_information: pd.DataFrame,
                                       *,
                                       iteration: int,
                                       name_deleted: IntLike = None,
                                       threshold: float = 0.0,
                                       threshold_y: float = 0.0,
                                       threshold_d: float | None = 0.0,
                                       keep_if: str = 'y_or_d_relevant',
                                       mcf_estimation: bool = True,
                                       summary=True,
                                       ) -> None:
    """Print results of feature selection."""
    if gen_cfg.with_output:
        txt = ('-' * 100
               + f'\nIntermediate results of Feature Selection based on iteration {iteration} '
               + '\n' + '- ' * 50
               )
        txt += '\nR2 used as score for regression and Accuracy Score used for classifiers. '
        if mcf_estimation:
            txt += f'Method used for selecting variables: {keep_if} '
            txt += f'\nScore for y based on all features: {score_y_full:6.3f}. '
            txt += f'Threshold for outcome estimation in %: {threshold_y:4.2%}. '
            if score_d_full is not None:
                txt += f'Score for d based on all features: {score_d_full:6.3f}. '
                txt += f'Threshold for propensity score in %: {threshold_d:4.2%}.'
            txt += '\n'
        else:
            score_y_str = ','.join([f'{x:6.3f}' for x in score_y_full])
            txt += f'\nScore for y based on all features: {score_y_str} '
            txt += f'Threshold in %: {threshold:4.2%}\n'

        print_mcf(gen_cfg, txt, summary=summary)

        with pd.option_context('display.max_rows', None,
                               'display.expand_frame_repr', True,
                               'chop_threshold', 1e-13,
                               ):
            vi = vi_information.transpose().dropna()
            if mcf_estimation:
                if score_d_full is None:
                    vi_sort = vi.sort_values(by=['rel_diff_y_%'], ascending=False)
                else:
                    vi_sort = vi.sort_values(by=['rel_diff_y_%', 'rel_diff_d_%'], ascending=False)

                print_mcf(gen_cfg, vi_sort, summary=summary)

            else:
                print_mcf(gen_cfg, vi, summary=summary)

        if name_deleted is None:
            txt_out = '\nNo variable deleted in this round. Feature selection ends.\n' + '- ' * 50
        else:
            txt_out = (f'\nVariable to be deleted in this iteration: {name_deleted}'
                       + '\n' + '- ' * 50
                       + '\nDue to early stopping, this table may not include all active variables.'
                       + '\n' + '- ' * 50
                       )
        print_mcf(gen_cfg, txt_out, summary=summary)
