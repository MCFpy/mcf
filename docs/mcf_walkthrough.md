# MCF Walkthrough

## Getting started

### Variables

The program requires the provision of at least three distinct lists of variables. You have to define a treatment, at least one outcome, and at least either ordered or unordered features.

### Compulsory variable arguments

| Variable name                            | Description           |
| ---------------------------------------- | --------------------- |
| [var_d_name](./mcf_api.md#var_d_name)             | Treatment.            |
| [var_y_name](./mcf_api.md#var_y_name)             | Outcome(s).           |
| [var_x_name_ord](./mcf_api.md#var_x_name_ord)     | Ordered feature(s).   |
| [var_x_name_unord](./mcf_api.md#var_x_name_unord) | Unordered feature(s). |

For a more detailed analysis, further variables can be specified optionally.

### Optional variable arguments

| Variable name                                                | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [var_id_name](./mcf_api.md#var_id_name)                               | Individual identifier.                                       |
| [var_cluster_name](./mcf_api.md#var_cluster_name)                     | Cluster identifier.                                          |
| [var_w_name](./mcf_api.md#var_w_name)                                 | Weights assigned to each observation.                        |
| [var_y_tree_name](./mcf_api.md#var_y_tree_name)                       | Outcome used to build trees. If not specified, the first outcome in [y_name](./mcf_api.md#var_y_name) is selected for building trees. |
| [var_x_name_always_in_ord](./mcf_api.md#var_x_name_always_in_ord)     | Ordered feature(s) always used in splitting decision.        |
| [var_x_name_always_in_unord](./mcf_api.md#var_x_name_always_in_unord) | Unordered feature(s) always used in splitting decision.      |
| [var_x_name_remain_ord](./mcf_api.md#var_x_name_remain_ord)           | Ordered feature(s) excluded from feature selection.          |
| [var_x_name_remain_unord](./mcf_api.md#var_x_name_remain_unord)       | Unordered feature(s) excluded from feature selection.        |
| [var_x_balance_name_ord](./mcf_api.md#var_x_balance_name_ord)         | Ordered feature(s) for balancing tests.                      |
| [var_x_balance_name_unord](./mcf_api.md#var_x_balance_name_unord)     | Unordered feature(s) for balancing tests.                    |
| [var_z_name_list](./mcf_api.md#var_z_name_list)                       | Ordered GATE (group average treatment effect) variable(s) with many values. |
| [var_z_name_ord](./mcf_api.md#var_z_name_ord)             | Ordered GATE variable(s) with few values.                    |
| [var_z_name_unord](./mcf_api.md#var_z_name_unord)         | Unordered GATE variable(s).                                  |


## Data cleaning

This program offers several options for data cleaning to improve the estimation quality. Below, we provide details on these options and present a table with details on the corresponding keyword arguments for data cleaning.

### Missing values and variables

In the standard setting, the program drops observations with missing values. Additionally, the data analysis drops variables that are not explicitly defined as variable arguments, even if they are included in the data set. Both of these properties can be set by the keyword argument [dc_clean_data](./mcf_api.md#dc_clean_data).

### No variation in features

Features without variation in their values are not informative for the data analysis. In the default setting, such features will be dropped. The keyword argument [dc_screen_covariates](./mcf_api.md#dc_screen_covariates) can be called to change this default.

### Perfect correlation of features

By default, the program deletes perfectly correlated features. The flag for the correlation check is [dc_check_perfectcorr](./mcf_api.md#dc_check_perfectcorr).

### Low variation in binary features

The keyword argument [dc_min_dummy_obs](./mcf_api.md#dc_min_dummy_obs) provides the option to delete binary features with a low number of 1's or 0's. The user can pass the minimum number of observations for a binary category to this keyword argument, so that dummies not fulfilling this condition will be removed.

### Reweighting

The estimated effects are computed as differences in weighted outcomes. The program offers several options to reweight the forest weights. The user can specify different sampling weights for each treatment state and each observation.

The user can specify sampling weights for each observation of the data. This option can be selected with [gen_weighted](./mcf_api.md#gen_weighted) and the sampling weights must be defined in [var_w_name](./mcf_api.md#var_w_name).

### Keyword arguments for data cleaning and sampling weights

| Keyword                                                    | Details                                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [dc_clean_data](./mcf_api.md#dc_clean_data)             | If True, all missing and unnecessary variables are removed. The default is True. |
| [dc_screen_covariates](./mcf_api.md#dc_screen_covariates)         | If True, covariates are screened. The default is True. |
| [dc_check_perfectcorr](./mcf_api.md#dc_check_perfectcorr)         | If **dc_screen_covariates** is True, variables that are perfectly correlated with others will be deleted. |
| [dc_min_dummy_obs](./mcf_api.md#dc_min_dummy_obs)                 | If **dc_screen_covariates** is True, dummy variables with less than **dc_min_dummy_obs** will be deleted. The default is 10.|
| [gen_weighted](./mcf_api.md#gen_weighted)                           | If *True*, sampling weights for each observation $i$ specified in  [var_w_name](./mcf_api.md#var_w_name) will be used. The default is *False*. |

## Common support

### General remarks

Estimation methods adjusting for differences in features require common support in all treatment arms. The mcf drops observations off support.

### Implementation

Common support checks and corrections are done before any estimation. The estimated probabilities are based on the random forest classifier. If common support (together with local centering) is based on cross-validation, all training data will be used.

The support checks are based on the estimations of propensity scores. You may specify a quantile in [cs_quantil](./mcf_api.md#cs_quantil). Denoting by $q$ the quantile chosen, the program drops observations with propensities scores smaller than the largest $q$ or larger than the smallest ($1-q$) quantile of the treatment groups. Alternatively, you may specify the support threshold of the propensity scores in [cs_min_p](./mcf_api.md#cs_min_p). If a support check is conducted, the program removes all observations with at least one treatment state off support.

The argument [cs_max_del_train](./mcf_api.md#cs_max_del_train) defines a threshold for the share of observations off support in the training data set. If this threshold is exceeded, the program terminates because of too large imbalances in the features across treatment states. In such a case, a new and more balanced input data set is required to run the program.

### Input arguments for common support

| Argument                                       | Description                                                  |
| ---------------------------------------------- | ------------------------------------------------------------ |
| [cs_type](./mcf_api.md#cs_type)     | Specifies type of common support adjustment. If set to 0, there is no common support adjustment. If set to 1 or 2, the support check is based on the estimated classification regression forests. For 1, the min-max rules for the estimated probabilities in the treatment subsamples are deployed. For 2, the minimum and maximum probabilities for all observations are deployed. All observations off support are removed. Note that out-of-bag predictions are used to avoid overfitting (which leads to a too large reduction in observations). |



## Feature selection

### General remarks

When the number of irrelevant features increases, random forests become more likely to pick splits based on those features, which will generally lead to a deterioration of the quality of the estimates. Also computational speed decreases with the number of features. Therefore, it should be advantageous to remove such features prior to estimation. Here, the program deselects features, which are irrelevant in predicting both outcome and treatment. Details on the selection and the corresponding keyword arguments are indicated below.


### Method

If [fs_yes](./mcf_api.md#fs_yes) is True, the program builds a random forest classifier or random regression forest. By default, the data deployed in the feature selection is different from the remaining mcf computations. The default can be overwritten by changing  [fs_other_sample](./mcf_api.md#fs_other_sample). If the importance statistic for a specific feature is too low -- as specified in [fs_rf_threshold](./mcf_api.md#fs_rf_threshold) -- this feature is (in principle) deleted. Irrelevance is conceptualised as follows: The values of a variable of interest are randomly permutated. If the reduction in the loss emanating from no permutation is not sizeable, the variable is classified as irrelevant. The defaut is 1 (measured in percent).

Of note, an irrelevant feature is not dropped if
- the correlation between two variables to be deleted is bigger than 0.5,
- the variable is required for the estimation of the GATEs, AMGATEs, and BGATEs,
- the variable is specified in [var_x_name_remain_ord](./mcf_api.md#var_x_name_remain_ord) or [var_x_name_remain_unord](./mcf_api.md#var_x_name_remain_unord).

### Keyword arguments for feature selection

| Keyword                                                    | Details                                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [fs_yes](./mcf_api.md#fs_yes)                               | If True, feature selection is active. Default is False. |
| [fs_other_sample](./mcf_api.md#fs_other_sample)| If True, random sample from training data is used, which will not be used for the causal forest. If False, the same data is used for feature selection and the causal forest. The default is True. |
| [fs_other_sample_share](./mcf_api.md#fs_other_sample_share) | If [fs_other_sample](./mcf_api.md#fs_other_sample) is set to True, [fs_other_sample_share](./mcf_api.md#fs_other_sample_share) determines the sample share for feature selection. The default is 0.33. |
|[fs_rf_threshold](./mcf_api.md#fs_rf_threshold)| Specifies the threshold for feature selection as relative loss of variable importance (in percent). The default is 1. |


## Forest growing

### Idea

Random Forests are an ensemble of decorrelated trees. A regression tree is a non-parametric estimator that splits the data into non-overlapping regions and takes the average of the dependent variable in these strata as prediction for observations sharing the same or similar values of the covariates. The key issue with this approach is that a discrete, non-overlapping data split may be inefficient (no information from neighboring cells are used) and the potential curse of dimensionality may make it difficult to fit a stable split (‘tree’) that has overall good performance. Furthermore, when the number of covariates increases, there are many possible splits of the data, and the computing time needed to form a tree may increase exponentially if all possible splits are considered at each knot. Random Forests solve these problems to some extent by building many decorrelated trees and averaging their predictions. This is achieved by using different random samples of the data for each tree (generated by bootstrapping or subsampling) as well as random subsets of covariates for each splitting decision in an individual leaf of a developing tree. Note that the mcf differs from  the causal forest of [Wager and Athey (2018)](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839) with respect to the splitting criterion when growing the forest. Setting [cf_mce_vart](./mcf_api.md#cf_mce_vart) to 2, you may switch to the splitting rule of [Wager and Athey (2018)](https://www.tandfonline.com/doi/abs/10.1080/01621459.2017.1319839). Abstracting from the difference in the splitting criterion, the regression forest may seem very much related to the mcf. However, note that the underlying prediction tasks are fundamentally different. The mcf aims to predict causal effects, for which there is no data, and provides (asymptotically) valid inference. To impute the missing data, the mcf requires a causal model. To provide valid inference, the mcf borrows the concept of honesty introduced by [Athey and Imbens (2016)](https://www.pnas.org/doi/10.1073/pnas.1510489113). For a textbook-like discussion refer to [Bodory, Busshoff and Lechner (2022)](https://www.pnas.org/doi/10.1073/pnas.1510489113).

### Implementation

The number of trees forming the forest is given by the argument [cf_boot](./mcf_api.md#cf_boot). To make valid counterfactual predictions,  tree-growing and within-leaf predictions are performed on different random samples of the data, the so-called honesty.

|Argument | Description |
|---------|-------------|
|[cf_boot](./mcf_api.md#cf_boot)| Number of trees (default is 1000). |


As a tree is grown the algorithm greedily chooses the split, which leads to the best possible reduction of the objective function, which is specified in [cf_mce_vart](./mcf_api.md#cf_mce_vart). To this end, the following objective criteria are implemented: (i) the outcome mean squared error (MSE), (ii) the outcome MSE and mean correlated errors (MCE), (iii) the variance of the effect, and (iv) the criterion randomly switches between outcome MSE and MCE and penalty functions which are defined under [cf_p_diff_penalty](./mcf_api.md#cf_p_diff_penalty). The outcome MSE is estimated as the sum of mean squared errors of the outcome regression in each treatment. The MCE depends on correlations between treatment states. For this reason, before building the trees, for each observation in each treatment state, the program finds a close ‘neighbor’ in every other treatment state and saves its outcome to then estimate the MCE. How the program matches is governed by the argument [cf_match_nn_prog_score](./mcf_api.md#cf_match_nn_prog_score). The program matches either by outcome scores (one per treatment) or on all covariates by Mahalanobis matching. If there are many covariates, it is advisable to match on outcome scores due to the curse of dimensionality. When performing Mahalanobis matching, a practical issue may be that the required inverse of the covariance matrix is unstable. For this reason the program allows to only use the main diagonal to invert the covariance matrix. This is regulated via the argument [cf_nn_main_diag_only](./mcf_api.md#cf_nn_main_diag_only). Likewise, the program allows for a modification of the splitting rule by adding a penalty to the objective function specified in [cf_mce_vart](./mcf_api.md#cf_mce_vart). The idea for deploying a penalty based upon the propensity score is to increase treatment homogeneity within new splits in order to reduce selection bias. Which specific penalty function is used is passed over to the program via the argument [cf_p_diff_penalty](./mcf_api.md#cf_p_diff_penalty). Note that from [cf_mce_vart](./mcf_api.md#cf_mce_vart) only option (iv) cannot work without the penalty. More details on choosing the minimum number of observations in a leaf are given below in section [Parameter tuning](#Parameter-tuning). Once the forest is settled for the training data, the splits obtained in the training data are transferred to all data subsamples (by treatment state) in the held-out data. Finally, the mean of the outcomes in the respective leaf is the prediction.

|Argument | Description |
|---------|-------------|
|[cf_mce_vart](./mcf_api.md#cf_mce_vart) | Determines the splitting rule when growing trees. |
|[cf_match_nn_prog_score](./mcf_api.md#cf_match_nn_prog_score) | Specifies the matching procedure in the MCE computation. If set to False, Mahalanobis matching is deployed. If set to True, prognostic scores are deployed. Default is True. |
|[cf_nn_main_diag_only](./mcf_api.md#cf_nn_main_diag_only) | Relevant if [cf_match_nn_prog_score](./mcf_api.md#cf_match_nn_prog_score) is set to False. If set to True, only the main diagonal is used. If False, the inverse of the covariance matrix is used. Default is False. |
|[cf_p_diff_penalty](./mcf_api.md#cf_p_diff_penalty) | Determines the penalty function. |

### Parameter tuning

The program allows for a grid search over tree tuning parameters: (i) the number of variables drawn at each split, (ii) the alpha-regularity, and (iii) the minimum leaf size. In practical terms, for all possible combinations, a forest is estimated fixing a random seed. Remark: The finer the grid-search, the more forests are effectively estimated, which slows down computation time. To identify the best values from the grid-search, the program implements the *out-of-bag estimation* of the chosen objective. The forest, which performs best with respect to its out-of-bag value of its objective function is taken for further computations.   

|Argument | Description |
|---------|-------------|
|[cf_n_min_grid](./mcf_api.md#cf_n_min_grid) | Determines number of grid values. Default is 1. For the default of 1, **n_min**= 0.5(**n_min_min**+**n_min_max**).|
|[cf_n_min_min](./mcf_api.md#cf_n_min_min) | Determines smallest minimum leaf size; specify an integer larger than 2. The default is round(max((n_d_subsam**0.4) / 10, 1.5) * # of treatments|
|[cf_n_min_max](./mcf_api.md#cf_n_min_max)| Determines largest minimum leaf size. The default is round(max((n_d_subsam**0.5) / 10, 2) * # of treatments, where n_d_subsam denotes the number of observations in the smallest treatment arm. All values are multiplied by the number of treatments.|
|[cf_n_min_treat](./mcf_api.md#cf_n_min_treat) | Determines minimum number of observations per treatment in leaf. The default is (n_min_min + n_min_max) / 2 / # of treatments / 10. Minimum is 1.|
|[cf_alpha_reg_grid](./mcf_api.md#cf_alpha_reg_grid) | Number of grid values. Default is 1.|
|[cf_alpha_reg_max](./mcf_api.md#cf_alpha_reg_max)  | Maximum alpha. May take values between 0 and 0.5. Default is 0.15.|
|[cf_alpha_reg_min](./mcf_api.md#cf_alpha_reg_min)  | Minimum alpha. May take values between 0 and 0.4. Default is 0.05.|
|[cf_m_share_min](./mcf_api.md#cf_m_share_min) | Minimum share of variables to be included in tree growing. Viable range is from 0 to 1 excluding the bounds. Default is 0.1.|
|[cf_m_share_max](./mcf_api.md#cf_m_share_max) | Maximum share of variables to be included in tree growing. Viable range is from 0 to 1 excluding the bounds. Default is 0.6.|
|[cf_m_grid](./mcf_api.md#cf_m_grid) | Number of grid values which are logarithmically spaced between the upper and lower bounds.|
|[cf_m_random_poisson](./mcf_api.md#cf_m_random_poisson) | If True the number of randomly selected variables is stochastic and obtained as a random draw from the Poisson distribution with expectation $m-1$, where $m$ denotes the number of variables used for splitting. |

### Remarks on computational speed

The smaller the minimum leaf size, the longer is the computation time, as the tree is grown deeper. This increase in computation time can be substantial for large data.


## Local Centering

### Method

Local centering is a form of residualization, which can improve the performance of forest estimators. This performance improvement can be achieved by regressing out the impact of the features on the outcome.

Formally, a conditionally centered outcome $\tilde{Y}_i$ can be defined as:


$$
\tilde{Y}_i = Y_i - \hat{y}^{-i}(X_i).
$$

$Y_i$ indicates the outcome for observation $i$. The term $\hat{y}^{-i}(X_i)$ is an estimate of the conditional outcome expectation $\mathbb{E}[Y_i \vert X_i=x_i]$, given the observed values $x_i$ of the feature vector $X_i$, computed without using the observation $i$.

Local centering is activated if [l_centering](./mcf_api.md#l_centering) is *True*.

### Implementation

The local centering procedure applies the *RandomForestRegressor* method of the *sklearn.ensemble* module to compute the predicted outcomes $\hat{y}^{-i}(X_i)$ for each observation $i$ non-parametrically. To turn the procedure off, overrule the default [lc_yes](./mcf_api.md#lc_yes) and set it to False. The predicted outcomes are computed in distinct subsets by cross-validation, where the number of folds can be specified in [lc_cs_cv_k](./mcf_api.md#lc_cs_cv_k). Finally, the centered outcomes are obtained by subtracting the predicted from the observed outcomes.

Alternatively, two separate data sets can be generated for running the local centering procedure with [lc_cs_cv](./mcf_api.md#lc_cs_cv). In this case, the size of the first data set can be defined in  [lc_cs_share](./mcf_api.md#lc_cs_share) and it is used for training a Random Forest, again by applying the *RandomForestRegressor* method. The predicted and centered outcomes $\hat{y}^{-i}(X_i)$ and $\tilde{Y}_i$, respectively, are computed in the second data set. Finally, this second data set is divided into mutually exclusive data sets for feature selection (optionally), tree building, and effect estimation.

### Keyword arguments for local centering

| Argument                                                     | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [lc_yes](./mcf_api.md#lc_yes)                       | *True* activates local centering. The default is *True*.    |
| [lc_cs_cv](./mcf_api.md#lc_cs_cv) |  Specifies number of folds for cross validation. The default is 5. |
| [lc_cs_share](./mcf_api.md#lc_cs_share)           | Specifies share of data used for conditonal outcome estimation. Viable range is from 0.1 to 0.9. The default is 0.25. |
| [lc_cs_cv_k](./mcf_api.md#lc_cs_cv_k)             | Specifies number of folds for cross validation. The default is 5. |


## Estimation  

### Average effects

The program computes three types of average treatment effects, which differ in their aggregation level and are discussed in depth by [Lechner (2018)](https://arxiv.org/abs/1812.09487). The effects are the individualized average treatment effect (IATE), the group average treatment effect (GATE), and the average treatment effect (ATE). They can be defined in the following way:


$$IATE(m,l;x) = \mathbb{E} \big[ Y^m-Y^l \big\vert X=x \big]$$

$$GATE(m,l;z,\Delta) = \mathbb{E} \big[ Y^m-Y^l \big\vert Z=z, D\in \Delta \big]$$

$$ATE(m,l;\Delta) = \mathbb{E} \big[ Y^m-Y^l \big\vert D\in \Delta \big]$$


The potential outcomes are denoted by $Y^d$, where $Y$ stands for the outcome and $\Delta$ comprises different treatment states $d$. By default, the program will consider all treatments, as well as treatment $m$ if ATET and/or GATET are set to *True*. Group-specific features are indicated by $Z$, and the feature vector $X$ comprises all features used by the program (and not deselected by Feature Selection, if activated).

The IATEs measure the mean impact of treatment $m$ compared to treatment $l$ for units with features $x$. The IATEs represent the estimands at the finest aggregation level available. On the other extreme, the ATEs represent the population averages. If Δ relates the population with $D=m$, then this is the average treatment effect on the treated (ATET) for treatment $m$. The ATE and ATET are the classical parameters investigated in many econometric causal studies. The group average treatment effect (GATE) parameters are in-between those two extremes with respect to their aggregation levels. The IATEs and the GATEs are special cases of the so-called conditional average treatment effects (CATEs).

In case of different distributions of $X$ in the estimation and prediction samples, the average treatment effects are based on the distribution of $X$ in the prediction sample.

#### IATEs

The program estimates IATEs for each observation in the prediction sample without the need of specifying any input arguments. Predicted IATEs and their standard errors will be saved to a CSV file and descriptive statistics are printed to the text file containing the results.

#### GATEs

By default, the program smooths the distribution of the GATEs for continuous features. A smoothing procedure evaluates the effects at a local neighborhood around a pre-defined number of evaluation points. The flag [p_gates_smooth](./mcf_api.md#p_gates_smooth) activates this procedure. The level of discretization depends on the number of evaluation points, which can be defined in [p_gates_smooth_no_evalu_points](./mcf_api.md#p_gates_smooth_no_evalu_points). The local neighborhood is based on an Epanechnikov kernel estimation using Silverman's bandwidth rule. The keyword argument [p_gates_smooth_bandwidth](./mcf_api.md#p_gates_smooth_bandwidth) specifies a multiplier for Silverman's bandwidth rule. In addition, it discretizes continuous features and computes the GATEs for those discrete approximations.

#### ATEs

ATEs are computed without the need of specifying any input arguments.

### AMGATEs and BGATEs

The average marginal group effect (AMGATE) is an integrated pseudo-causal derivative of the treatment effect with respect to a heterogeneity variable of interest, i.e.

$$AMGATE(m,l;x)    = \mathbb{E} \big[ MGATE(m,l;x) \big],$$

where

$$MGATE(m,l;x) = \frac{\mathbb{E} \big[ IATE(m,l;x) \big\vert X^p=x^{pU}, X^{-p}=x^{-p} \big]}{x^{pU}-x^{pL}}\\ - \frac{\mathbb{E} \big[ IATE(m,l;x) \big\vert X^p=x^{pL}, X^{-p}=x^{-p} \big]}{x^{pU}-x^{pL}}.$$

Here, $p$ is a single feature of $X$ and $X^{-p}$ denotes the remaining features of $X$ without $p$. The values of $x^{pU}$ and $x^{pL}$ are chosen to be larger and smaller than $x^p$, respectively, while insuring that the support of $x^p$ is respected.

The AMGATE hence overcomes the causal attribution problem related to simple GATE differences, where other relevant variables may confound effect heterogeneity. For a discussion of the full set of assumptions required for a causal interpretation refer to Bearth and Lechner (2023).  

The Balanced Group Average Treatment Effects (BGATEs) relax the AMGATEs in that sense that only a subset of the variables in the computation of the pseudo-derivative are held constant. Hence, both AMGATEs and the plain-vanilla difference of GATEs are limiting cases of the BGATEs.

Algorithmically, the BGATEs and the AMGATEs as the limiting case are implemented as follows:
1. Draw a random sample from the prediction data.
2. Keep the heterogeneity and balancing variables.
3. Replicate the data from step 2 $n_z$ times, where $n_z$ denotes the cardinality of the heterogeneity variable of interest. In each of the $n_z$ folds set $z$ to a specific value.
4. Draw the nearest neighbours of each observation in the prediction data in terms of the balancing variables and heterogeneity variable. If there is a tie, the algorithm choses one randomly.
4. Form a new sample with all selected neighbours.
5. Compute GATEs and their standard errors.

To turn on the AMGATE, set [p_amgate](./mcf_api.md#p_amgate) to True. To turn on the BGATE, set [p_bgate](./mcf_api.md#p_bgate) to True.

### Computing effects for the treated

The effects for the treated are computed if the input arguments [p_atet](./mcf_api.md#p_atet) or [p_gatet](./mcf_api.md#p_gatet) are set to *True*.

### Stabilizing estimates of effects by truncating weights

To obtain stable estimates, the program provides the option to truncate estimated forest weights to an upper threshold. After truncation, the program renormalizes the weights for estimation. Because of the renormalization step, the final weights can be slightly above the threshold defined in [p_max_weight_share](./mcf_api.md#p_max_weight_share).

### Evaluation of effect heterogeneity

To see if the estimated treatment effects are heterogeneous in their features, the program presents both, statistics on the treatment effects and on their deviations from the ATE.

### Input arguments for estimations of treatment effects

| Arguments                                                    | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [p_gates_smooth](./mcf_api.md#p_gates_smooth)                     | Flag for smoothing the distribution of the estimated GATEs for continuous features. The default is True. |
| [p_gates_smooth_no_evalu_points](./mcf_api.md#p_gates_smooth_no_evalu_points) | Number of evaluation points for GATEs. The default is 50.    |
| [p_gates_smooth_bandwidth](./mcf_api.md#p_gates_smooth_bandwidth) | Multiplier for Silverman's bandwidth rule for GATEs. The default is 1. |
| [p_gmate_no_evalu_points](./mcf_api.md#p_gmate_no_evalu_points) | Number of evaluation points for marginal treatment effects. The default is 50. |
| [p_gmate_sample_share](./mcf_api.md#p_gmate_sample_share)         | Number in the interval $(0,1]$ determining the size of $N_{SS}$ for the computation of AMTEs. Note that $N_{SS}$ also depends on the number of evaluation points. |
| [p_atet](./mcf_api.md#p_atet)                           | If *True*, average treatment effects for subpopulations defined by treatment status are computed. This only works if at least one GATE feature is specified. The default is *False*. |
| [p_gatet](./mcf_api.md#p_gatet)                         | If *True*, group average treatment effects for subpopulations defined by treatment status are computed. The default is *False*. |
| [p_max_weight_share](./mcf_api.md#p_max_weight_share)           | Maximum value of the weights. The default is 0.05.           |
|[p_gates_minus_previous](./mcf_api.md#p_gates_minus_previous)|If set to True, GATES will be compared to GATEs computed at the previous evaluation point. GATE estimation is a bit slower as it is not optimized for multiprocessing. No plots are shown. Default is False.|



## Inference

### General remarks

The program offers three ways of conducting inference. The default is a weights-based inference procedure, which is particularly useful for gaining information on the precision of estimators that have a representation as weighted averages of the outcomes, see [Lechner (2019)](https://arxiv.org/abs/1812.09487). The second inference procedure provided by the program simply estimates the variance of treatment effect estimates as the sum of the variance of weighted outcomes. Finally, a bootstrap algorithm can be applied to obtain inference.

### Methods

One way for conducting inference for treatment effects is to estimate the variance of the treatment effect estimator based on a variance decomposition into the expectation of the conditional variance and the variance of the conditional expectation, given the weights. This variance decomposition takes heteroscedasticity in the weights into account. The conditional means $\mu_{Y \vert \hat{W}} (\hat{w}_i)$ and variances $\sigma^2_{Y \vert \hat{W}} (\hat{w}_i)$ are estimated non-parametrically, either by the Nadaraya-Watson kernel estimator  or by the *k-Nearest Neighbor* (*k-NN*) estimator (default).

Another way to obtain inference is to compute the variance of a treatment effect estimator as the sum of the variances of the weighted outcomes in the respective treatment states. A drawback of this inference method is that it implicitly assumes homoscedasticity in the weights for each treatment state.

Alternatively, the standard bootstrap can be applied to compute standard errors. Our algorithm bootstraps the equally weighted weights $\hat{w}_i$ and then renormalizes $\hat{w}_i$.   

Note that because of the weighting representation, inference can also readily be used to account for clustering, which is a common feature in economics data.

## Post-estimation diagnostics

After estimating treatment effects and inference, the program conducts a  heterogeneity analysis. It evaluates the correlation of treatment effects and presents graphical representations of the effects and their densities. In addition, it investigates heterogeneity for different clusters with respect to treatment effects, potential outcomes, and features. Finally, the program computes feature importance statistics of the features that are used for estimation.

### Starting the diagnostics

To start the post-estimation diagnostics, the flag [post_est_stats](./mcf_api.md#post_est_stats) has to be activated. The input argument [post_relative_to_first_group_only](./mcf_api.md#post_relative_to_first_group_only) specifies the reference group. If [post_relative_to_first_group_only](./mcf_api.md#post_relative_to_first_group_only) is *True*, the comparison group comprises the units assigned to the first treatment state only. Otherwise, all possible treatment combinations are compared among each other. The confidence levels are specified by [p_ci_level](./mcf_api.md#p_ci_level).

### Correlation analysis

The binary correlation analysis computes dependencies among the IATEs, as well as between the IATEs and the potential (weighted) outcomes, and between the IATEs and the features. The flag [post_bin_corr_yes](./mcf_api.md#post_bin_corr_yes) activates the correlation analysis. The correlation coefficients between the IATEs and the features are only displayed if their absolute values are above a threshold specified by the argument [post_bin_corr_threshold](./mcf_api.md#post_bin_corr_threshold).


### *k-Means* clustering

To analyze heterogeneity in different groups (clusters), the program applies *k-Means* clustering if [post_kmeans_yes](./mcf_api.md#post_kmeans_yes) is set to *True*. It uses the *k-means++* algorithm of the *KMeans* method provided by Python's sklearn.cluster module. Clusters are formed on the IATEs only. For these clusters, descriptive statistics of the IATEs, the potential outcomes, and the features are displayed. Cluster memberships are saved to the output file, which facilitates further in-depth analysis of the respective clusters.

You can define the number of clusters by specifying the input argument [post_kmeans_no_of_groups](./mcf_api.md#post_kmeans_no_of_groups), which can be a list or tuple specifying the number of clusters. The final number of clusters is chosen via silhouette analysis. To guard against getting stuck at local extrema, the number of replications with different random start centers can be defined in [post_kmeans_replications](./mcf_api.md#post_kmeans_replications). The argument [post_kmeans_max_tries](./mcf_api.md#post_kmeans_max_tries) sets the maximum number of iterations in each replication to achieve convergence.


### Post-estimation feature importance

The post-estimation feature importance procedure runs if the flag  [post_random_forest_vi](./mcf_api.md#post_random_forest_vi) is activated. The procedure builds a predictive random forest to learn major features influencing the IATEs. The feature importance statistics are presented in percentage points of the coefficient of determination *R<sup>2</sup>* lost when permuting single features. The *R<sup>2</sup>* statistics are obtained by the *score* method provided by the *RandomForestRegressor* object of Python's sklearn.ensemble module.

### Input arguments for post-estimation diagnostics

| Arguments                                                    | Descriptions                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [post_plots](./mcf_api.md#post_plots)                         | Printing plots. The default is *True*.                       |
| [post_kmeans_yes](./mcf_api.md#post_kmeans_yes)               | AIf True, the program uses k-means clustering to analyse patterns in the estimated effects. The default is True.      |
| [post_kmeans_no_of_groups](./mcf_api.md#post_kmeans_no_of_groups) | If **post_kmeans_yes** is True, **post_kmeans_no_of_groups** determines number of clusters. Information is passed over in the form  of an integer list or tuple. If not otherwise specified, the default is a list of 5 values: $[a, b, c, d, e]$, where depending on $n$, c takes values from 5 to 10. If c is smaller than 7, $a=c-2$, $b=c-1$, $d=c+1$, $e=c+2$ else $a=c-4$, $b=c-2$, $d=c+2$, $e=c+4$. |
| [post_kmeans_replications](./mcf_api.md#post_kmeans_replications) | If **post_kmeans_yes** is True, **post_kmeans_replications** regulates the number of replications for the k-means clustering algorithm. The default is 10. |
| [post_kmeans_max_tries](./mcf_api.md#post_kmeans_max_tries)   | If **post_kmeans_yes** is True, **post_kmeans_max_tries** sets the maximum number of iterations in each replication to archive convergence. Default is 1000. |


## Balancing Tests  


### General remarks

Treatment effects may be subject to selection bias if the distribution of the confounding features differs across treatment states. This program runs non-parametric balancing tests to check the statistical equality of the distribution of features after adjustment by the modified causal forest. Treatment specific statistics will only be printed for those variables used to check the balancing of the sample. This feature has to be considered experimental as it needs further investigation on how these balancing statistics are related to the bias of the estimation.  

### Balancing tests based on weights

The program runs balancing tests of the features specified in [var_x_balance_name_ord](./mcf_api.md#var_x_balance_name_ord) and [var_x_balance_name_unord](./mcf_api.md#var_x_balance_name_unord) if the [p_bt_yes](./mcf_api.md#p_bt_yes) flag is activated.

The tests are based on estimations of ATEs by replacing the outcomes with user-specified features. For multiple treatments, the results consider all possible treatment combinations. Features with ATEs not significantly different from zero can be regarded as balanced across treatment states.


### Input arguments for balancing tests

| Arguments                                    | Description                                                 |
| -------------------------------------------- | ----------------------------------------------------------- |
| [p_bt_yes](./mcf_api.md#p_bt_yes) | Flag for activating balancing tests. The default is False. |
