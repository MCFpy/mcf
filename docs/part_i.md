# MCF Walkthrough


## Getting started

### Structure of directory

To specify the path to your project folder, you can set your working directory in [outpfad](./core_6.md#outpfad). The data directory can be passed to [datpfad](./core_6.md#datpfad).

The data file has to be stored in *csv* format and its name has to be assigned to [indata](./core_6.md#indata) without adding the file extension *.csv*. Optionally, a separate data file for estimating and predicting treatment effects can be specified in [preddata](./core_6.md#preddata). While [indata](./core_6.md#indata) needs to contain all variables, it is ok if [preddata](./core_6.md#preddata) contains the features only. If [preddata](./core_6.md#preddata) is not specified, [indata](./core_6.md#indata) is used instead.

The program creates the folder *out* and stores it in your default working directory or in the directory specified in [outpfad](./core_6.md#outpfad). This *out* folder contains the results.  The text document summarizes descriptive statistics and results on estimation and inference. If a folder *out* already exists, folder *out1* is created, if *out1* exists *out2* and so on. The sub-folders *fig_csv*, *fig_jpeg*, and *fig_pdf* comprise *csv* data files and *jpeg* and *pdf* plots of the results.  The csv file ending with the name *X_IATE.csv* in the directory *cs_ate_iate* stores predicted effects and standard errors for all observations. User-specific names for these files can be assigned with [forest_files](./core_6.md#forest_files). Finally, the folder *\_tempmcf_*, filled with temporary data files, will be temporarily stored in *out* and deleted after program execution.

### Variables

The program requires the provision of at least three distinct lists of variables. You have to define a treatment, at least one outcome, and at least either ordered or unordered features.

### Compulsory variable arguments

| Variable name                            | Description           |
| ---------------------------------------- | --------------------- |
| [d_name](./core_6.md#d_name)             | Treatment.            |
| [y_name](./core_6.md#y_name)             | Outcome(s).           |
| [x_name_ord](./core_6.md#x_name_ord)     | Ordered feature(s).   |
| [x_name_unord](./core_6.md#x_name_unord) | Unordered feature(s). |

For a more detailed analysis, further variables can be specified optionally.

### Optional variable arguments

| Variable name                                                | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [id_name](./core_6.md#id_name)                               | Individual identifier.                                       |
| [cluster_name](./core_6.md#cluster_name)                     | Cluster identifier.                                          |
| [w_name](./core_6.md#w_name)                                 | Weights assigned to each observation.                        |
| [y_tree_name](./core_6.md#y_tree_name)                       | Outcome used to build trees. If not specified, the first outcome in [y_name](./core_6.md#y_name) is selected for building trees. |
| [x_name_always_in_ord](./core_6.md#x_name_always_in_ord)     | Ordered feature(s) always used in splitting decision.        |
| [x_name_always_in_unord](./core_6.md#x_name_always_in_unord) | Unordered feature(s) always used in splitting decision.      |
| [x_name_remain_ord](./core_6.md#x_name_remain_ord)           | Ordered feature(s) excluded from feature selection.          |
| [x_name_remain_unord](./core_6.md#x_name_remain_unord)       | Unordered feature(s) excluded from feature selection.        |
| [x_balance_name_ord](./core_6.md#x_balance_name_ord)         | Ordered feature(s) for balancing tests.                      |
| [x_balance_name_unord](./core_6.md#x_balance_name_unord)     | Unordered feature(s) for balancing tests.                    |
| [z_name_list](./core_6.md#z_name_list)                       | Ordered GATE (group average treatment effect) variable(s) with many values. |
| [z_name_split_ord](./core_6.md#z_name_split_ord)             | Ordered GATE variable(s) with few values.                    |
| [z_name_split_unord](./core_6.md#z_name_split_unord)         | Unordered GATE variable(s).                                  |
| [z_name_mgate](./core_6.md#z_name_mgate)                     | GATE variable(s) for estimating marginal effect(s) evaluated at the median. Must be included in [x_name_ord](./core_6.md#x_name_ord) or [x_name_unord](./core_6.md#x_name_unord). |
| [z_name_amgate](./core_6.md#z_name_amgate)                   | GATE variable(s) for estimating average marginal effect(s). Must be included in [x_name_ord](./core_6.md#x_name_ord) or [x_name_unord](./core_6.md#x_name_unord). |



## Data cleaning

This program offers several options for data cleaning to improve the estimation quality. Below, we provide details on these options and present a table with details on the corresponding keyword arguments for data cleaning.

### Missing values and variables

In the standard setting, the program drops observations with missing values. Additionally, the data analysis drops variables that are not explicitly defined as variable arguments, even if they are included in the data set. Both of these properties can be set by the keyword argument [clean_data_flag](./core_6.md#clean_data_flag).

### No variation in features

Features without variation in their values are not informative for the data analysis. In the default setting, such features will be dropped. The keyword argument [screen_covariates](./core_6.md#screen_covariates) can be called to change this default.

### Perfect correlation of features

By default, the program deletes perfectly correlated features. The flag for the correlation check is [check_perfectcorr](./core_6.md#check_perfectcorr).

### Low variation in binary features

The keyword argument [min_dummy_obs](./core_6.md#min_dummy_obs) provides the option to delete binary features with a low number of 1's or 0's. The user can pass the minimum number of observations for a binary category to this keyword argument, so that dummies not fulfilling this condition will be removed.

### Reweighting

The estimated effects are computed as differences in weighted outcomes. The program offers several options to reweight the forest weights. The user can specify different sampling weights for each treatment state and each observation.

The flag [choice_based_sampling](./core_6.md#choice_based_sampling) activates the option for unequal treatment weights. The new weights have to be assigned to [choice_based_weights](./core_6.md#choice_based_weights).

The user can further specify sampling weights for each observation of the data. This option can be selected with [weighted](./core_6.md#weighted) and the sampling weights must be defined in [w_name](./core_6.md#w_name).

### Keyword arguments for data cleaning and sampling weights

| Keyword                                                    | Details                                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [clean_data_flag](./core_6.md#clean_data_flag)             | Removing observations with missing values and keeping only variables in the data set that are defined as variable arguments. The default is *True*. |
| [screen_covariates](./core_6.md#screen_covariates)         | Dropping features without variation. The default is *True*.  |
| [check_perfectcorr](./core_6.md#check_perfectcorr)         | Removing perfectly correlated features. The default is *True*. |
| [min_dummy_obs](./core_6.md#min_dummy_obs)                 | Threshold in observations per category determining if dummies are dropped due to low variation. The default is 10. |
| [choice_based_sampling](./core_6.md#choice_based_sampling) | If *True*, assignment of treatment-specific probabilities is activated. The default is *False*. |
| [choice_based_weights](./core_6.md#choice_based_weights)   | List with one entry (=weight) for each treatment. This only runs if [choice_based_sampling](./core_6.md#choice_based_sampling) is activated. |
| [weighted](./core_6.md#weighted)                           | If *True*, sampling weights for each observation $i$ specified in  [w_name](./core_6.md#w_name) will be used. The default is *False*. |

## Common support

### General remarks

Estimation methods adjusting for differences in features require common support in all treatment arms. This program thus drops observations considered to be off support.

### Implementation

Common support checks and corrections are done before any estimation, and they relate to all data files for training, prediction, and feature selection. 

First, the program uses the training data to build one forest per treatment by applying the *RandomForestRegressor* method of the *sklearn.ensemble* module. To avoid overfitting, it then predicts in the prediction data set specified by [preddata](./core_6.md#preddata) the  propensity scores $P(D = m \vert X)$ for all mutually exclusive treatments $m = 1,\cdots,M$, where $M$ denotes the number of treatments.

The program allows one to choose between no, data-driven, and user-defined support conditions, which can be selected by [support_check](./core_6.md#support_check). The support checks are based on the estimated propensity scores discussed above. For data-driven support, you have to specify a quantile in [support_quantil](./core_6.md#support_quantil). Denoting by $q$ the quantile chosen, the program drops observations with propensities scores smaller than the largest $q$ or larger than the smallest ($1-q$) quantile of the treatment groups. Contrary to this condition, user-defined support directly specifies the support threshold of the propensity scores in [support_min_p](./core_6.md#support_min_p). If a support check is conducted, the program removes all observations with at least one treatment state off support.

The argument [support_max_del_train](./core_6.md#support_max_del_train) defines a threshold for the share of observations off support in the training data set. If this threshold is exceeded, the program terminates because of too large imbalances in the features across treatment states. In such a case, a new and more balanced input data set is required to run the program.

### Input arguments for common support

| Argument                                       | Description                                                  |
| ---------------------------------------------- | ------------------------------------------------------------ |
| [support_check](./core_6.md#support_check)     | Three options (0,1,2). The option 0 does not check for common support. The option 1 carries out a data-driven support check by using min-max decision rules for probabilities in each treatment state specified by [support_quantil](./core_6.md#support_quantil). The option 2 starts a support check by enforcing minimum and maximum probabilities defined by [support_min_p](./core_6.md#support_min_p). The default is 1. |
| [support_quantil](./core_6.md#support_quantil) | Float in the $(0.5,1]$ interval. Observations are dropped if propensities scores are smaller than the largest $q$ or larger than the smallest ($1-q$) quantile of the propensity score distribution. The default of $q$ is 1. |
| [support_min_p](./core_6.md#support_min_p)     | Float in the $(0,0.5)$ interval. Observations are deleted if propensity scores are smaller than or equal to this argument. The default is $\min(0.025,r)$, where $r$ is the ratio between the share of the smallest treatment group in the training data and the number of treatments. |
| [support_max_del_train](./core_6.md#support_max_del_train) | Float in the (0,1) interval. Specifies the threshold for the share of observations off support in the training data set. The default is 0.5. |


## Feature selection

### General remarks

When the number of irrelevant features increases, random forests become more likely to pick splits based on those features, which will generally lead to a deterioration of the quality of the estimates. Also computational speed decreases with the number of features. Therefore, it should be advantageous to remove such features prior to starting the main estimation. Here, the program selects features based on feature importance statistics. Details on the selection and the corresponding keyword arguments are indicated below.

### Keyword arguments for feature selection

| Keyword                                                    | Details                                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [fs_yes](./core_6.md#fs_yes)                               | True if feature selection is activated. The default is *False*. |
| [fs_other_sample](./core_6.md#fs_other_sample)             | Creating separate data for feature selection. The default is *True*. |
| [fs_other_sample_share](./core_6.md#fs_other_sample_share) | Share of data used for feature selection. The default is 0.2. |
| [fs_rf_threshold](./core_6.md#fs_rf_threshold)             | Threshold for loss of feature importance in percent. The default is 0. |

### Method

After activating feature selection, the program builds a modified causal forest to compute feature importance statistics. If the importance statistic for a specific feature is lower than [fs_rf_threshold](./core_6.md#fs_rf_threshold), this feature is (in principle) deleted. The program further offers an option to run the feature selection in a separate random subset of the data.

The feature selection procedure deselects unimportant features. To this end, it computes feature importance statistics (FIS) for the features, where higher FIS indicate higher importance. However, deselecting features based on small single FIS may result in removing relevant features. For example, if two features are highly correlated, both of their single FIS are close to zero, because each feature on its own adds barley new information to the model, given the other features. To avoid this trap of dropping such potentially relevant features in settings with strong correlations, the program forms groups of features ordered by their FIS and computes new feature importance statistics for each group (FISG). It also builds aggregated groups ordered by their FISG and calculates feature importance statistics for each aggregated group (FISaG). Then, only those features are regarded as unimportant, whose values of FIS, FISG, and FISaG are jointly below the user-defined importance threshold [fs_rf_threshold](./core_6.md#fs_rf_threshold). These features are not considered for further analysis.

Note that the program does not deselect features that are a priori considered to be very important, as defined by the input arguments [x_name_always_in_unord](./core_6.md#x_name_always_in_ord), [x_name_always_in_ord](./core_6.md#x_name_always_in_unord), [x_name_remain_ord](./core_6.md#x_name_remain_ord), [x_name_remain_unord](./core_6.md#x_name_remain_unord), [z_name_list](./core_6.md#z_name_list), [z_name_split_ord](./core_6.md#z_name_split_ord), [z_name_split_unord](./core_6.md#z_name_split_unord), [z_name_mgate](./core_6.md#z_name_mgate), and [z_name_amgate](./core_6.md#z_name_amgate).

The [Technical Appendix](./techn_app.md) presents information on the tuning parameters for the forest built for feature selection. It also gives details on the computation of the feature importance statistics and the construction of the groups.

## Forest growing

### Idea

Random Forests are an ensemble of decorrelated trees. A regression tree is a non-parametric estimator that splits the data into non-overlapping regions and takes the average of the dependent variable in these strata as prediction for observations sharing the same or similar values of the covariates. The key issue with this approach is that a discrete, non-overlapping data split may be inefficient (no information from neighboring cells are used) and the potential curse of dimensionality may make it difficult to fit a stable split (‘tree’) that has overall good performance. Furthermore, when the number of covariates increases, there are many possible splits of the data, and the computing time needed to form a tree may increase exponentially if all possible splits are considered at each knot. Random Forests solve these problems to some extent by building many decorrelated trees and averaging their predictions. This is achieved by using different random samples of the data for each tree (generated by bootstrapping or subsampling) as well as random subsets of covariates for each splitting decision in an individual leaf of a developing tree.

### Implementation

In the MCF program, the forest growing is implemented as follows. If [train_mcf](./core_6.md#train_mcf) is set to its default value True, a forest is estimated. The number of trees forming the forest is given by the argument [boot](./core_6.md#boot). To make valid counterfactual predictions,  tree-growing and within-leaf predictions are performed on different random samples of the data, the so-called honesty. The [share_forest_sample](./core_6.md#share_forest_sample) governs the share of the data that is used for predicting the causal effects. The share of the data used to build the forest is thus 1 - [share_forest_sample](./core_6.md#share_forest_sample).

|Argument | Description |
|---------|-------------|
|[boot](./core_6.md#boot)| Number of trees (default is 1000). |
|[share_forest_sample](./core_6.md#share_forest_sample)|Share of data used for predicting $y$ given forests, other data is used for building the forest (default is 0.5).|

As a tree is grown the algorithm greedily chooses the split, which leads to the best possible reduction of objective function, which is specified in [mce_vart](./core_6.md#mce_vart). To this end, the following objective criteria are implemented: (i) the outcome mean squared error (MSE), (ii) the outcome MSE and mean correlated errors (MCE), (iii) the variance of the effect, and (iv) the criterion randomly switches between outcome MSE and MCE and penalty functions which are defined under [p_diff_penalty](./core_6.md#p_diff_penalty). The outcome MSE is estimated as the sum of mean squared errors of the outcome regression in each treatment. The MCE depends on correlations between treatment states. For this reason, before building the trees, for each observation in each treatment state, the program finds a close ‘neighbor’ in every other treatment state and saves its outcome to then estimate the MCE. How the program matches is governed by the argument [match_nn_prog_score](./core_6.md#match_nn_prog_score). The program matches either by outcome scores (one per treatment) or on all covariates by Mahalanobis matching. If there are many covariates, it is advisable to match on outcome scores due to the curse of dimensionality. When performing Mahalanobis matching, a practical issue may be that the required inverse of the covariance matrix is unstable. For this reason the program allows to only use the main diagonal to invert the covariance matrix. This is regulated via the argument [nn_main_diag_only](./core_6.md#nn_main_diag_only). Likewise, the program allows for a modification of the splitting rule by adding a penalty the objective function specified in [mce_vart](./core_6.md#mce_vart). The idea for deploying a penalty based upon the propensity score is to increase treatment homogeneity within new split in order to reduce selection bias. Which specific penalty function is used, is passed over to the program via the argument [p_diff_penalty](./core_6.md#p_diff_penalty). Note that from mce_vart](./core_6.md#mce_vart) only option (iv) cannot work without the penalty. More details on choosing the minimum number of observations in a leaf are given below in section [Parameter tuning](#Parameter tuning). If [predict_mcf](./core_6.md#predict_mcf) is set to its default value, True, the program estimates the effects. In particular, once the forest is settled for the training data, the splits obtained in the training data are transferred to all data subsamples (by treatment state) in the held-out data. Finally, the mean of the outcomes in the respective leaf is the prediction.

|Argument | Description |
|---------|-------------|
|[mce_vart](./core_6.md#mce_vart) | Determines the splitting rule when growing trees. |
|[match_nn_prog_score](./core_6.md#match_nn_prog_score) | Computing prognostic scores to find close neighbors for MCE. |
|[nn_main_diag_only](./core_6.md#nn_main_diag_only) | Use main diagonal of covariance matrix only for Mahalanobis matching (only relevant if |[match_nn_prog_score](./core_6.md#match_nn_prog_score) is False). |
|[p_diff_penalty](./core_6.md#p_diff_penalty) | Determines penalty function. |

### Parameter tuning

The program allows for a grid search over tree tuning parameters: (i) the number of variables drawn at each split, (ii) the alpha-regularity, and (iii) the minimum leaf size. In practical terms, for all combinations of  a forest is estimated fixing a random seed. Remark: The finer the grid-search, the more forests are effectively estimated, which slows down computation time. To identify the best values from the grid-search, the program implements the *out-of-bag estimation* of the chosen objective. The forest, which performs best with respect to its out-of-bag value of its objective function is taken for further computations.   

|Argument | Description |
|---------|-------------|
|[n_min_grid](./core_6.md#n_min_grid) | Number of grid values for the minimum leaf size (default is 1, for which the minimum leaf size is governed by [n_min_min](./core_6.md#n_min_min)).|
|[n_min_min](./core_6.md#n_min_min) | Smallest minimum leaf size for grid-search,  (default is -1, for which the leaf size is computed as $\max(n^{0.4}/10, 5)$, where $n$ is twice the number of observations in the smallest treatment group).|
|[n_min_max](./core_6.md#n_min_max)| Largest minimum leaf size for grid-search (default is -1, for which the leaf size is computed as $\max(\sqrt{n}/5, 5)$, where $n$ is twice the number of observations in the smallest treatment group).|
|[alpha_reg_grid](./core_6.md#alpha_reg_grid) | Number of grid values for the  alpha-regularity parameter (default is 1).|
|[alpha_reg_max](./core_6.md#alpha_reg_max)  | Largest value for the  alpha-regularity parameter (default is 0.1).|
|[alpha_reg_min](./core_6.md#alpha_reg_min)  | Smallest value for the  alpha-regularity parameter (default is 0.1).|
|[m_min_share](./core_6.md#m_min_share) | Minimum share of variables used for splitting (default is -1, for which this share is computed as $0.1*q$, where $q$ denotes the number of variables).|
|[m_max_share](./core_6.md#m_max_share) | Maximum share of variables used for next split (default is -1, for which this share is computed as $0.66*q$, where $q$ denotes the number of variables).|
|[m_grid](./core_6.md#m_grid) | Number of grid values which are logarithmically spaced (default is 2).|
|[m_random_poisson](./core_6.md#m_random_poisson) | If True the number of randomly selected variables is stochastic and obtained as $1 + P$, where $P$ is a random draw from the Poisson distribution with expectation $m-1$, where $m$ denotes the number of variables used for splitting. |

### Remarks on computational speed

The smaller the minimum leaf size, the longer is the computation time, as the tree is grown deeper. This increase in computation time can be substantial for large data. 


## Local Centering

### Method

Local centering is a form of residualization, which can improve the performance of forest estimators. This performance improvement can be achieved by regressing out the impact of the features on the outcome.

Formally, a conditionally centered outcome $\tilde{Y}_i$ can be defined as:

<img src="https://render.githubusercontent.com/render/math?math=\tilde{Y}_i = Y_i - \hat{y}^{-i}(X_i)">

$$
\tilde{Y}_i = Y_i - \hat{y}^{-i}(X_i).
$$

$Y_i$ indicates the outcome for observation $i$. The term $\hat{y}^{-i}(X_i)$ is an estimate of the conditional outcome expectation $\mathbb{E}[Y_i \vert X_i=x_i]$, given the observed values $x_i$ of the feature vector $X_i$, computed without using the observation $i$.

Local centering is activated if [l_centering](./core_6.md#l_centering) is *True*.

### Implementation

The local centering procedure applies the *RandomForestRegressor* method of the *sklearn.ensemble* module to compute the predicted outcomes $\hat{y}^{-i}(X_i)$ for each observation $i$ non-parametrically. These predictions are computed in distinct subsets by cross-validation, where the number of folds can be specified in [l_centering_cv_k](./core_6.md#l_centering_cv_k). Finally, the centered outcomes are obtained by subtracting the predicted from the observed outcomes.

Alternatively, two separate data sets can be generated for running the local centering procedure with  [l_centering_new_sample](./core_6.md#l_centering_new_sample). In this case, the size of the first data set can be defined in  [l_centering_share](./core_6.md#l_centering_share) and it is used for training a Random Forest, again by applying the *RandomForestRegressor* method. The predicted and centered outcomes $\hat{y}^{-i}(X_i)$ and $\tilde{Y}_i$, respectively, are computed in the second data set. Finally, this second data set is divided into mutually exclusive data sets for feature selection (optionally), tree building, and effect estimation.

### Keyword arguments for local centering

| Argument                                                     | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [l_centering](./core_6.md#l_centering)                       | *True* activates local centering. The default is *True*.    |
| [l_centering_new_sample](./core_6.md#l_centering_new_sample) | Generating separate samples for tree building and estimation of $\mathbb{E}[Y_i \vert X_i=x_i]$. The default is *False*. |
| [l_centering_share](./core_6.md#l_centering_share)           | Share of data used for estimating $\mathbb{E}[Y_i \vert X_i=x_i]$. If [l_centering_new_sample](./core_6.md#l_centering_new_sample) is *True*, the default is 0.25. |
| [l_centering_cv_k](./core_6.md#l_centering_cv_k)             | Number of folds used in cross-validation. The default is 5.  |


## Estimation  

### Average effects

The program computes three types of average treatment effects, which differ in their aggregation level and are discussed in depth by [Lechner (2018)](https://arxiv.org/abs/1812.09487). The effects are the individualized average treatment effect (IATE), the group average treatment effect (GATE), and the average treatment effect (ATE). They can be defined in the following way:

\begin{aligned}
IATE(m,l;x) &= \mathbb{E} \big[ Y^m-Y^l \big\vert X=x \big] \\
GATE(m,l;z,\Delta) &= \mathbb{E} \big[ Y^m-Y^l \big\vert Z=z, D\in \Delta \big] \\
ATE(m,l;\Delta)    &= \mathbb{E} \big[ Y^m-Y^l \big\vert D\in \Delta \big] \\
\end{aligned}


The potential outcomes are denoted by $Y^d$, where $Y$ stands for the outcome and $\Delta$ comprises different treatment states $d$. By default, the program will consider all treatments, as well as treatment $m$ if ATET and/or GATET are set to *True*. Group-specific features are indicated by $Z$, and the feature vector $X$ comprises all features used by the program (and not deselected by Feature Selection, if activated).

The IATEs measure the mean impact of treatment $m$ compared to treatment $l$ for units with features $x$. The IATEs represent the estimands at the finest aggregation level available. On the other extreme, the ATEs represent the population averages. If Δ relates the population with $D=m$, then this is the average treatment effect on the treated (ATET) for treatment $m$. The ATE and ATET are the classical parameters investigated in many econometric causal studies. The group average treatment effect (GATE) parameters are in-between those two extremes with respect to their aggregation levels. The IATEs and the GATEs are special cases of the so-called conditional average treatment effects (CATEs).

In case of different distributions of $X$ in the estimation and prediction samples, the average treatment effects are based on the distribution of $X$ in the prediction sample.

#### IATEs

The program estimates IATEs for each observation in the prediction sample without the need of specifying any input arguments. Predicted IATEs and their standard errors will be saved to a CSV file and descriptive statistics are printed to the text file containing the results.

#### GATEs

By default, the program smooths the distribution of the GATEs for continuous features. A smoothing procedure evaluates the effects at a local neighborhood around a pre-defined number of evaluation points. The flag [smooth_gates](./core_6.md#smooth_gates) activates this procedure. The level of discretization depends on the number of evaluation points, which can be defined in [smooth_gates_no_evaluation_points](./core_6.md#smooth_gates_no_evaluation_points). The local neighborhood is based on an Epanechnikov kernel estimation using Silverman's bandwidth rule. The keyword argument [smooth_gates_bandwidth](./core_6.md#smooth_gates_bandwidth) specifies a multiplier for Silverman's bandwidth rule. In addition, it discretizes continuous features and computes the GATEs for those discrete approximations.

#### ATEs

ATEs are computed without the need of specifying any input arguments.

### Marginal effects

The program further calculates two forms of marginal effects. The difference to the GATEs show above is that they keep the distribution of the other features fixed, while the GATEs implicitly also capture changes in the distribution of other features if correlated with the feature of interest. The two forms of marginal effects are the marginal treatment effect evaluated at fixed reference points (MTE) and the average marginal treatment effect (AMTE). The marginal effects can be approximated by a discrete version of the definition of a derivative as:


$$MTE(m,l;x) = \frac{\mathbb{E} \big[ IATE(m,l;x) \big\vert X^p=x^{pU}, X^{-p}=x^{-p} \big]}{x^{pU}-x^{pL}}\\ - \frac{\mathbb{E} \big[ IATE(m,l;x) \big\vert X^p=x^{pL}, X^{-p}=x^{-p} \big]}{x^{pU}-x^{pL}}$$

$$AMTE(m,l;x)    = \mathbb{E} \big[ MTE(m,l;x) \big]$$

Here, $p$ is a single feature of $X$ and $X^{-p}$ denotes the remaining features of $X$ without $p$. The values of $x^{pU}$ and $x^{pL}$ are chosen to be larger and smaller than $x^p$, respectively, while insuring that the support of $x^p$ is respected.

#### MTEs and AMTEs

MTEs, evaluated at the median of ordered features or at the mode of unordered features, can be obtained by specifying [z_name_mgate](./core_6.md#z_name_mgate). To compute AMTEs, [z_name_amgate](./core_6.md#z_name_amgate) has to be defined.

Marginal effects for continuous features of $Z$ are evaluated at a pre-defined number of equally spaced quantiles of evaluation points. The number of quantiles can be specified by [gmate_no_evaluation_points](./core_6.md#gmate_no_evaluation_points).

Computing AMTEs is much more time consuming than calculating MTEs. The reason is that an MTE for a specific category $z \in Z$ evaluated at single reference points of the remaining features requires the computation of a single IATE only. To compute an AMTE for the same $z \in Z$, it is necessary to estimate a GATE, which aggregates the IATEs for all observations assigned to that category, evaluated at the same observed values of the other features. To save computation time, the AMTEs are based on random subsets of the data of size $N_{SS}$, which is determined by [gmate_sample_share](./core_6.md#gmate_sample_share). Details on $N_{SS}$ are shown in the [Technical Appendix](./techn_app.md).

### Computing effects for the treated

The effects for the treated are computed if the input arguments [atet_flag](./core_6.md#atet_flag) or [gatet_flag](./core_6.md#gatet_flag) are set to *True*.

### Stabilizing estimates of effects by truncating weights

To obtain stable estimates, the program provides the option to truncate estimated forest weights to an upper threshold given by [max_weight_shares](./core_6.md#max_weight_shares). After truncation, the program renormalizes the weights for estimation. Because of the renormalization step, the final weights can be slightly above the threshold defined in [max_weight_shares](./core_6.md#max_weight_shares).

### Evaluation of effect heterogeneity

To see if the estimated treatment effects are heterogeneous in their features, the program presents both, statistics on the treatment effects and on their deviations from the ATE.

### Input arguments for estimations of treatment effects

| Arguments                                                    | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [smooth_gates](./core_6.md#smooth_gates)                     | Flag for smoothing the distribution of the estimated GATEs for continuous features. The default is True. |
| [smooth_gates_no_evaluation_points](./core_6.md#smooth_gates_no_evaluation_points) | Number of evaluation points for GATEs. The default is 50.    |
| [smooth_gates_bandwidth](./core_6.md#smooth_gates_bandwidth) | Multiplier for Silverman's bandwidth rule for GATEs. The default is 1. |
| [gmate_no_evaluation_points](./core_6.md#gmate_no_evaluation_points) | Number of evaluation points for marginal treatment effects. The default is 50. |
| [gmate_sample_share](./core_6.md#gmate_sample_share)         | Number in the interval $(0,1]$ determining the size of $N_{SS}$ for the computation of AMTEs. Note that $N_{SS}$ also depends on the number of evaluation points, see [Technical Appendix](./techn_app.md). |
| [atet_flag](./core_6.md#atet_flag)                           | If *True*, average treatment effects for subpopulations defined by treatment status are computed. This only works if at least one GATE feature is specified. The default is *False*. |
| [gatet_flag](./core_6.md#gatet_flag)                         | If *True*, group average treatment effects for subpopulations defined by treatment status are computed. The default is *False*. |
| [max_weight_shares](./core_6.md#max_weight_shares)           | Maximum value of the weights. The default is 0.05.           |

### Plotting the treatment effects


The program generates graphical representations of the treatment effects. You can customize the font size of the legends, the number of pixels of the figures, and the level of the confidence bounds. The corresponding input arguments are [_fontsize](./core_6.md#_fontsize), [_dpi](./core_6.md#_dpi), and [ci_level](./core_6.md#ci_level), respectively. For line plots of GATEs, MTEs, and AMTEs, you can fill the area between lines if the number of evaluation points exceeds a threshold specified in [_no_filled_plot](./core_6.md#_no_filled_plot).

### Input arguments for graphical representations of treatment effects

| Arguments                                    | Descriptions                                                 |
| -------------------------------------------- | ------------------------------------------------------------ |
| [_fontsize](./core_6.md#_fontsize)             | Font size for legends in plots, ranging from 1 (very small) to 7 (very large). The default is 2. |
| [_dpi](./core_6.md#_dpi)                       | Number of pixels for plots. The default is 500.              |
| [ci_level](./core_6.md#ci_level)             | Number in the $ (0,1)$ interval determining the confidence level shown. The default is 0.9. |
| [_no_filled_plot](./core_6.md#_no_filled_plot) | Number of evaluation points to fill the area between lines. The default is 20. This works for line plots for GATEs, MTEs, and AMTEs. |

## Inference

### General remarks

The program offers three ways of conducting inference. The default is a weights-based inference procedure, which is particularly useful for gaining information on the precision of estimators that have a representation as weighted averages of the outcomes, see [Lechner (2018)](https://arxiv.org/abs/1812.09487). The [Technical Appendix](./techn_app.md#Technical Appendix) shows the tuning parameters for this method. The second inference procedure provided by the program simply estimates the variance of treatment effect estimates as the sum of the variance of weighted outcomes. Finally, a bootstrap algorithm can be applied to obtain inference.

### Methods

One way for conducting inference for treatment effects is to estimate the variance of the treatment effect estimator based on a variance decomposition into the expectation of the conditional variance and the variance of the conditional expectation, given the weights. This variance decomposition takes heteroscedasticity in the weights into account. The conditional means $\mu_{Y \vert \hat{W}} (\hat{w}_i)$ and variances $\sigma^2_{Y \vert \hat{W}} (\hat{w}_i)$ are estimated non-parametrically, either by the Nadaraya-Watson kernel estimator  or by the *k-Nearest Neighbor* (*k-NN*) estimator (default).

Another way to obtain inference is to compute the variance of a treatment effect estimator as the sum of the variances of the weighted outcomes in the respective treatment states. A drawback of this inference method is that it implicitly assumes homoscedasticity in the weights for each treatment state.

Alternatively, the standard bootstrap can be applied to compute standard errors. Our algorithm bootstraps the equally weighted weights $\hat{w}_i$ and then renormalizes $\hat{w}_i$.   

Note that because of the weighting representation, inference can also readily be used to account for clustering, which is a common feature in economics data.

Details on parameters of the inference procedures, as well as on options for clustered data, are shown in the [Technical Appendix](./techn_app.md).



## Post-estimation diagnostics

After estimating treatment effects and inference, the program conducts a  heterogeneity analysis. It evaluates the correlation of treatment effects and presents graphical representations of the effects and their densities. In addition, it investigates heterogeneity for different clusters with respect to treatment effects, potential outcomes, and features. Finally, the program computes feature importance statistics of the features that are used for estimation.

### Starting the diagnostics

To start the post-estimation diagnostics, the flag [post_est_stats](./core_6.md#post_est_stats) has to be activated. The input argument [relative_to_first_group_only](./core_6.md#relative_to_first_group_only) specifies the reference group. If [relative_to_first_group_only](./core_6.md#relative_to_first_group_only) is *True*, the comparison group comprises the units assigned to the first treatment state only. Otherwise, all possible treatment combinations are compared among each other. The confidence levels are specified by [ci_level](./core_6.md#ci_level).

### Correlation analysis

The binary correlation analysis computes dependencies among the IATEs, as well as between the IATEs and the potential (weighted) outcomes, and between the IATEs and the features. The flag [bin_corr_yes](./core_6.md#bin_corr_yes) activates the correlation analysis. The correlation coefficients between the IATEs and the features are only displayed if their absolute values are above a threshold specified by the argument [bin_corr_threshold](./core_6.md#bin_corr_threshold).

### Plots

Plots can be printed with [post_plots](./core_6.md#post_plots). This generates sorted effects plots of both, the IATEs and their deviations from the ATE, estimated for the observations in the prediction sample defined in [preddata](./core_6.md#preddata). The plots of the effects and the confidence bounds are based on moving averages of the IATEs and their standard errors. The number of neighbors *k* for the moving averages is computed by *k-NN* estimation, with input arguments [knn_min_k](./core_6.md#knn_min_k) and [knn_const](./core_6.md#knn_const), as discussed in the [Technical Appendix](./techn_app.md). The program further creates kernel density plots for the IATEs, using the Epanechnikov kernel and Silverman's bandwidth rule as defaults.


### *k-Means* clustering

To analyze heterogeneity in different groups (clusters), the program applies *k-Means* clustering if [post_kmeans_yes](./core_6.md#post_kmeans_yes) is set to *True*. It uses the *k-means++* algorithm of the *KMeans* method provided by Python's sklearn.cluster module. Clusters are formed on the IATEs only. For these clusters, descriptive statistics of the IATEs, the potential outcomes, and the features are displayed. Cluster memberships are saved to the output file, which facilitates further in-depth analysis of the respective clusters. 

You can define the number of clusters by specifying the input argument [post_kmeans_no_of_groups](./core_6.md#post_kmeans_no_of_groups), which can be a list or tuple specifying the number of clusters. The final number of clusters is chosen via silhouette analysis. To guard against getting stuck at local extrema, the number of replications with different random start centers can be defined in [post_kmeans_replications](./core_6.md#post_kmeans_replications). The argument [post_kmeans_max_tries](./core_6.md#post_kmeans_max_tries) sets the maximum number of iterations in each replication to achieve convergence.


### Post-estimation feature importance

The post-estimation feature importance procedure runs if the flag  [post_random_forest_vi](./core_6.md#post_random_forest_vi) is activated. The procedure builds a predictive random forest to learn major features influencing the IATEs. The feature importance statistics are presented in percentage points of the coefficient of determination *R<sup>2</sup>* lost when randomizing single features. The *R<sup>2</sup>* statistics are obtained by the *score* method provided by the *RandomForestRegressor* object of Python's sklearn.ensemble module.

### Input arguments for post-estimation diagnostics

| Arguments                                                    | Descriptions                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [post_plots](./core_6.md#post_plots)                         | Printing plots. The default is *True*.                       |
| [post_kmeans_yes](./core_6.md#post_kmeans_yes)               | Applying *k-Means* clustering. The default is *True*.        |
| [post_kmeans_no_of_groups](./core_6.md#post_kmeans_no_of_groups) | Number of clusters for *k-Means* clustering. The default is a function of the sample size and ranges from 5 to 10, see [Technical Appendix](./techn_app.md). |
| [post_kmeans_replications](./core_6.md#post_kmeans_replications) | Number of replications with different random start centers for *k-Means* clustering. The default is 10. |
| [post_kmeans_max_tries](./core_6.md#post_kmeans_max_tries)   | Maximum number of iterations in each replication for *k-Means* clustering. The default is 1000. |


## Balancing Tests  


### General remarks

Treatment effects may be subject to selection bias if the distribution of the confounding features differs across treatment states. This program runs non-parametric balancing tests to check the statistical equality of the distribution of features after adjustment by the modified causal forest. Treatment speciﬁc statistics will only be printed for those variables used to check the balancing of the sample. This feature has to be considered experimental as it needs further investigation on how these balancing statistics are related to the bias of the estimation.  

### Balancing tests based on weights

The program runs balancing tests of the features specified in [x_balance_name_ord](./core_6.md#x_balance_name_ord) and [x_balance_name_unord](./core_6.md#x_balance_name_unord) if the [balancing_test](./core_6.md#balancing_test) flag is activated.

The tests are based on estimations of ATEs by replacing the outcomes with user-specified features. For multiple treatments, the results consider all possible treatment combinations. Features with ATEs not significantly different from zero can be regarded as balanced across treatment states.


### Input arguments for balancing tests

| Arguments                                    | Description                                                 |
| -------------------------------------------- | ----------------------------------------------------------- |
| [balancing_test](./core_6.md#balancing_test) | Flag for activating balancing tests. The default is *True*. |

