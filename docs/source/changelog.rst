Changelog
=======================
.. 
    Conventions:

    1. Add a horizontal rule ----- before adding a new entry
    2. Refer to the mcf as a package in bold, i.e. **mcf**
    3. Nest parameters of functions/methods in double backticks, e.g. ``foo``
    4. Cross-reference classes, their methods and properties:
        - Refer to classes using :py:class:`~module.ClassName`, e.g. :py:class:`~mcf_functions.ModifiedCausalForest`
        - Refer to methods using :py:meth:`~module.ClassName.method_name`, e.g. :py:meth:`~mcf_functions.ModifiedCausalForest.train` 
        - Refer to class properties using :py:attr:`~module.ClassName.property_name`, e.g. :py:attr:`~mcf_functions.ModifiedCausalForest.blind_dict`
    5. Nested lists: You need to separate the lists with a blank line. Otherwise, the parent will be displayed as bold.

        - Wrong (will be bold):
            - A
            - B 

        - Right:

            - A
            - B

    The following should be removed from this file and just be added to the internal documentation:
    You can cross-reference classes/methods/properties also with a custom link text using e.g. 
    :py:class:`Custom link text <module.ClassName>` 

    Note the absence of the tilde '~' in this case. 


Version 0.7.2
====================

Documentation
-------------

- The user guide contains a new section on Computational Speed and Resources for effect estimation. This new section summarizes some considerations about computation and resource use.

  a) It consists of the (speed and resource relevant) content that is already in section 1.2 of the Algorithmic Reference.
  b) It now also contains the information on how to reduce the demand on RAM using the parameters `_int_iate_chunk_size` and `_int_weight_as_sparse_splits`.

  Finally, it contains the following considerations for large data sets:

  "When datasets are large, the computational burden (incl. demands on RAM) may increase rapidly. First of all, it is important to remember that the mcf estimation consists of two steps:
  
  1. Train the forest with the training data (Y, D, X).
  2. Predict the effects with the prediction data (needs X only, or D and X if e.g. treatment effects on the treated are estimated).

  The precision of the results is (almost) entirely determined by the training data, while the prediction data mainly defines the population for which the ATE and other effects are computed.

  mcf deals as follows with large training data: When the training data becomes larger than `cf_chunks_maxsize`, the data is randomly split and for each split a new forest is estimated. In the prediction part, effects are estimated for each forest and subsequently averaged.

  mcf deals as follows with large prediction data: The critical part when computing the effects is the weight matrix. Its size is `N_Tf x N_P`, where `N_P` is the number of observations in the prediction data and `N_Tf` is the number of observations used for the forest `f` estimated. The weight matrix is estimated for each forest (to save memory it is deleted from memory and stored on disk). Although the weight matrix is (as default) using a sparse data format, it can still be very large and it can be very time-consuming to compute.

  Reducing computation and demand on memory with minimal performance loss:
  Tests for very large data (1 million and more) have shown that indeed the prediction part becomes the bottleneck, while the training part computes reasonably fast. Therefore, one way to speed up the mcf and reduce the demand on RAM is to reduce the size of the prediction data (e.g., take a x% random sample). For this approach, tests have shown, for example, that with 1 million training observations, the effect estimates (and standard errors) are very similar if 1 million or only 100,000 prediction observations are used.

  New keywords (`_int_max_obs_training`, `_int_max_obs_prediction`, `_int_max_obs_kmeans`, `_int_max_obs_post_rel_graphs`) allow setting these parameters accordingly.

Example Programs
----------------

- `mcf_bgate` (name change): This program was called `min_parameters_mcf_bgate` in previous versions.
- All example programs have been renamed so that they either start with `mcf_`, `optpolicy_`, or `mcf_optpolicy_` to better indicate their purpose.
- `mcf_opt_combined` now includes a cross-fitting version that uses the data more effectively at the cost of additional computing costs. Additional information has been added to the file and is also reflected in the updated documentation.
- Small improvements to some other example programs.

All Classes
-----------

- `os` module substituted by `pathlib` module for better platform interoperability.
- New public attribute:
  - `version`: String
    - Version of the mcf module used to create the instance.

ModifiedCausalForest Class
--------------------------

- Fixing bug in variance estimation of BGATE (variance accounts for duplicates in matching).
- Minor bug fixes.
- Several smaller changes to increase robustness and speed when using very large data.

New Keywords
~~~~~~~~~~~~

- `_int_iate_chunk_size`: Integer or None, optional. Number of IATEs that are estimated in a ray worker. Default is the number of prediction observations divided by workers. If the program crashes in IATE 2/2 because of excess memory consumption, reduce this number. In the previous version, the value of `_int_iate_chunk_size` was implicitly set to 1.
- The following new keywords define upper limits for sample size. If the actual number is larger than the prespecified number, then the respective data will be randomly reduced to the specified upper limit:
  - `_int_max_obs_training`: Integer or None, optional. Reducing observations for training increases MSE and thus should be avoided. Default is infinity.
  - `_int_max_obs_prediction`: Integer or None, optional. Reducing observations for prediction does not much affect MSE. It may reduce detectable heterogeneity but may also dramatically reduce computation time. Default is 250,000.
  - `_int_max_obs_kmeans`: Integer or None, optional. Reducing observations for analyzing IATEs does not much affect MSE. It may reduce detectable heterogeneity but also reduces computation time. Default is 200,000.
  - `_int_max_obs_post_rel_graphs`: Integer or None, optional. Figures showing the relation of IATEs and features (in-built non-parametric regression is computationally intensive). Default is 50,000.
  - `_int_obs_bigdata`: Integer or None, optional. If the number of training observations is larger than this number, the following happens during training:
    1. Number of workers is halved in local centering.
    2. Ray is explicitly shut down.
    3. The number of workers used is reduced to 75% of default.
    4. The data type for some numpy arrays is reduced from float64 to float32. Default is 1,000,000.

New Features
~~~~~~~~~~~~

- New figures showing the univariate relations of IATE to single features. Depending on the type of features, these are box or scatter plots (with nonlinear smoother).

Change of Default Values
~~~~~~~~~~~~~~~~~~~~~~~~~

- Default value of `lc_cs_cv_k` becomes dependent on the size of the training sample (`N`):
  - `N < 100,000`: 5
  - `100,000 <= N < 250,000`: 4
  - `250,000 <= N < 500,000`: 3
  - `N >= 500,000`: 2.
- Default value of `_int_weight_as_sparse_splits` is increased to `(Rows of prediction data * rows of Fill_y data) / (number of training splits * 25,000 * 25,000)`. This should lead to some speed-up in larger data sets (at the expense of needing some more memory).
- The base value in the formula of `cf_chunks_maxsize` has been increased from 75,000 to 90,000, leading to somewhat deeper forests at the expense of some additional memory consumption.
- The default value for the size of the subsamples drawn in the data part used to be the forest has a new lower bound. It cannot be smaller than the square root of the number of training observations used for finding the splits.

Change of Keywords
~~~~~~~~~~~~~~~~~~

- `var_x_balance_name_ord` -> `var_x_name_balance_test_ord`
- `var_x_balance_name_unord` -> `var_x_name_balance_test_unord`
- `var_bgate_name` -> `var_x_name_balance_bgate`

OptimalPolicy Class
-------------------

- Minor bug fixes.
- Improved readability of output.
- More statistics describing the respective allocations:
  - An additional reference allocation has been added: It shows the allocation when every unit is allocated to the treatment which is best on average for the data used to evaluate the allocation.
  - A standard error for the mean of the main welfare measure is printed. This standard error reflects the variability in the evaluation data for a given assignment rule. The variability in the training data when learning the assignment rule is not captured.
  - New Qini-like plots are added. These plots compare the optimal allocation to a reference allocation (3 allocations are used as such reference allocations, if available: (i) observed, (ii) random, (iii) the treatment with the highest ATE is allocated to everybody). They show the mean welfare when an increasing share of observations (starting with those who gain most from the optimal allocation compared to the reference allocation) is allocated using the optimal allocation rule.

New Keywords
~~~~~~~~~~~~

- `_int_dpi`: Integer (or None), optional. DPI in plots. Default (or None) is 500. Internal variable, change default only if you know what you do.
- `_int_fontsize`: Integer (or None), optional. Font for legends, from 1 (very small) to 7 (very large). Default (or None) is 2. Internal variable, change default only if you know what you do.

Change of Default Values
~~~~~~~~~~~~~~~~~~~~~~~~~

- The default value `pt_eva_cat_mult` has been changed to 2.

Change of Keywords
~~~~~~~~~~~~~~~~~~

To increase the consistency between the mcf and the optimal policy module:

- `_int_parallel_processing` and `_int_how_many_parallel` are deprecated. Instead, the following keyword is used (as in mcf):
  - `gen_mp_parallel`: Integer (or None), optional. Number of parallel processes (>0). 0, 1: no parallel computations. Default is to use 80% of logical cores (reduce if memory problems!).


Version 0.7.1
-------------

- Bug in optimal policy module for policy variables with more than 20 unordered values.

Version 0.7.0
-------------

Documentation
~~~~~~~~~~~~~~

- New section added with published (!) papers using the mcf. We will try to update this section with every release. Please feel free to inform us about your publications when they use the mcf.
- New script with example on how to use the fairness correction in optimal policy: fairness_optpolicy.py. **This method is experimental.** A detailed description will be added in the next release.

Changes concerning all classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Speed increase by optimizing numba functions
- Adjustments required by new Numpy version 2.0

Changes concerning the class :py:class:`~mcf_functions.ModifiedCausalForest`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Small bug fixes.
- Improved output.
- A new, additional penalty function has been introduced based on the MSE of the propensity score computed in the split (causal) leaves. This penalty function favors splits that reduce selection bias. One advantage of this new penalty function is that it can be computed with the out-of-bag observations when tuning the forest (which was not possible with the existing penalty function). This change also required the introduction of a new keyword (cf_penalty_type; see below for details).
- The method blinder_iates (reducing dependence of IATEs on protected variables) is deprecated and removed from the documentation. It will be fully removed in future versions. Use the method fairscores of the OptimalPolicy class instead. It is computationally more efficient and works better at removing the influence of protected variables on scores.
- Change in k-means clustering of IATEs: If a cluster is smaller than required by post_kmeans_min_size_share, it will be merged with the cluster that has the closest centroid.
- Additional tool added to describe IATEs(x) with the analyse() method: Shallow regression trees are trained in standard and honest form. Figures and out-of-sample accuracy measures (R-squared) of how they fit the IATEs are provided.

- **Name change of keywords**

    - ``post_k_means_single`` -> ``post_kmeans_single``

New keywords
+++++++++++++

- **cf_tune_all**

    - Tune all parameters. If True, all *_grid keywords will be set to 3. User specified values are respected if larger than 3. Default is False.

- **cf_penalty_type**

    - Type of penalty function. 'mse_d':  MSE of treatment prediction in daughter leaf (new in 0.7.0).  'diff_d': Penalty as squared leaf difference (as in Lechner, 2018). Note that 'mse_d' that can also be used for tuning,  while (due to its computation), this is not possible for 'diff_d'. Default (or None) is 'mse_d'.

- **post_kmeans_min_size_share**

    - Smallest share of cluster size allowed in % (0-33). Default (None) is 1.

- **post_tree**

    - Regression trees (honest and standard) of Depth 2 to 5 are estimated to describe IATES(x). Default (or None) is True.

Changes concerning the class :py:class:`~optpolicy_functions.OptimalPolicy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The method fairscores has been improved and expanded (for details, see the future paper by Bearth, Lechner, Mareckova, Muny, 2024). However, fairness adjustments are still experimental.
- Change in content of keyword:
    - ``fair_type`` now captures 3 methods to perform score adjustments:
        - 'Mean': Mean dependence of the policy score on protected variables is removed.
        - 'MeanVar': Mean dependence and heteroscedasticity are removed.
        - 'Quantiled': Removing dependence via an empirical version of the approach by Strack and Yang (2024).
        - Default (or None) is 'Quantiled'.

New Keywords
++++++++++++

- **fair_consistency_test**: Boolean. Test for internal consistency. The fairness corrections are applied independently to every policy score (which usually is a potential outcome or an IATE(x) for each treatment relative to some base treatment, i.e., comparing 1-0, 2-0, 3-0, etc.). Thus, the IATE for the 2-1 comparison can be computed as IATE(2-0) - IATE(1-0). This test compares two ways to compute a fair score for the 2-1 (and all other comparisons), which should give similar results:

  - **a)** Difference of two fair (!) scores
  - **b)** Difference of corresponding scores, subsequently made fair.

  Note: Depending on the number of treatments, this test may be computationally more expensive than the original fairness corrections. Fairness adjustments are experimental. The default is False.

- **fair_protected_disc_method**, **fair_material_disc_method**: String
  Parameters for discretization of features (necessary for 'Quantilized'). Method on how to perform the discretization for materially relevant and protected variables.

  - **NoDiscretization**: Variables are not changed. If one of the features has more different values than `fair_protected_disc_method` / `fair_material_disc_method`, all protected / materially relevant features will formally be treated as continuous. The latter may become unreliable if their dimension is not small.
  - **EqualCell**: Attempts to create equal cells for each variable. May be useful for a very small number of variables with few different values.
  - **Kmeans**: Use Kmeans clustering algorithm to form homogeneous cells.

  Fairness adjustments are experimental. The default (or None) is **Kmeans**.

- **fair_protected_max_groups**, **fair_material_max_groups**: String.
  Level of discretization of variables (only if needed). Number of groups of values of features that are materially relevant / protected. This keyword is currently only necessary for 'Quantilized'. Its meaning depends on `fair_protected_disc_method`, `fair_material_disc_method`:

  - **EqualCell**: If more than 1 variable is included among the protected features, this restriction is applied to each variable.
  - **Kmeans**: This is the number of clusters used by Kmeans.

  Fairness adjustments are experimental. The default (or None) is 5.

Changes concerning the class :py:class:`~mcf_functions.McfOptPolReport`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **mcf_blind** is removed, because the method `blinder_iates` is deprecated.

Version 0.6.0
-------------

General
~~~~~~~

- Data are no longer provided as *.csv files. Instead they are generated directly by the new function example_data(*) (which has to be loaded from mcf.example_data_functions.py). These changes are reflected in the various parts of the documentation. The function itself is documented in the API. This leads to changes in all example programmes provided (and the related documentation).
- Programmes have been simplified as intermediate results are no longer saved. 

Changes concerning all classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Variable names are case insensitive in the package**

    - So far this has been achieved by converting all names to uppercase. This is now changed by converting names to lowercase using the casefold() methods which is more robust than the upper() and lower() methods.
    
- **New value error**

    - If variables with only two different values are passed as 'unordered' a value error is raised. These variables should appear in the category of 'ordered' variables.  

Changes concerning all methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Location of the output files**

    - All methods return the location of the output files on the computer as last return (the reporting method is an exception as it returns the full file name of the pdf file, not just the location).

Changes concerning the class :py:class:`~mcf_functions.ModifiedCausalForest`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bug fixes
+++++++++

    - Local centering using classifiers is disabled (implementation was incorrect for discrete outcomes with less than 10 values).
    - Data used to build common support plots are now properly created as DataFrames (instead of lists) and stored in csv files (as before).

Change of default values
+++++++++++++++++++++++++++

    - **p_ci_level**: The default significance levels used for the width of computing confidence intervals is changed from 90% to the more conventional 95%.
    - **_int_cuda**: As the gains in speed (on respective servers that run cuda) are currently slow, this experimental feature defaults to False.


Additional features and new keywords
+++++++++++++++++++++++++++++++++++++++++

- **New keyword: post_k_means_single**

    - If True, clustering is also with respect to all single effects. Default is False.
    - Setting **post_k_means_single** to True allows k-means clustering of IATEs also with respect to the single IATEs (in addition to jointly clustering on all relevant IATEs)

- **New keyword: cf_compare_only_to_zero**

    - If True, the computation of the MSE (and MCE) ignores all elements not related to the first treatment. 
    - When setting **cf_compare_only_to_zero** to True, the computation of the MSE (and MCE) ignores all elements not related to the first treatment (which usually is the control group). This speeds up computation and may be attractive when interest is only in the comparisons of each treatment to the control group and not among each other. This may also be attractive for optimal policy analysis based on using potential outcomes normalized by the potential outcome of the control group (i.e., IATEs of treatments vs. control group). Default is False.

- **New keyword: lc_estimator**

    - The estimation method used for local centering can be specified.
    - Possible choices are scikit-learn's regression methods: 'RandomForest', 'RandomForestNminl5','RandomForestNminls5', 'SupportVectorMachine', 'SupportVectorMachineC2', 'SupportVectorMachineC4', 'AdaBoost', 'AdaBoost100', 'AdaBoost200', 'GradBoost', 'GradBoostDepth6',  'GradBoostDepth12',  'LASSO',  'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger', 'Mean'.
    - If set to 'automatic', the estimator with the lowest out-of-sample mean squared error (MSE) is selected.Whether this selection is based on cross-validation  or a test sample is governed by the keyword lc_cs_cv. 'Mean' is included for the cases when none of the  methods have out-of-sample explanatory power. The default is 'RandomForest'.

Changes in the implementation of train method :py:meth:`~mcf_functions.ModifiedCausalForest.train`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  **Building the forest**

    - If variables randomly selected for splitting do not show any variation in the leaf considered for splitting, then up to 3 additional random draws are tried using variables not yet used  for splitting. If there is still no variation, then all remaining  variables will be tried for this potential split. This increases computation time somewhat, but leads to smaller leaves.

Changes in the implementation of train method :py:meth:`~mcf_functions.ModifiedCausalForest.predict`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **Clustering predictions with Kmeans algorithm**

    - When the smallest cluster is smaller than 1% of the sample, this case is now  discouraged when determining the optimal number of clusters with scikit-learn's silhouette_score.

Changes concerning the class :py:class:`~optpolicy_functions.OptimalPolicy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bug fixes
+++++++++

    - Bug removed when reporting results for policy trees (when treatment state was available in evaluation data).
    - Maximum number of iterations (1000) for automatic cost search added to avoid that the cost-search algorithm does not converge.

Removed features 
+++++++++++++++++++++

    - 'policy tree old' has been removed from the list of available methods (keyword: gen_method).

Additional features 
++++++++++++++++++++

-  **New method: fairscores(*args, *keyws)**

    - This fairness method is experimental. It is a preview of what  will be discussed in the paper by Bearth, Lechner, Mareckova, and   Muny (2024): Explainable Optimal Policy with Protected Variables.  The main idea is to adjust the policy scores in a way such that the resulting optimal allocation will not depend on the protected  variables.

       - The following keywords are new and related to this adjustment:

          - **fair_regression_method** : String (or None), optional. Regression method to adjust scores w.r.t. protected variables. Available methods are 'RandomForest', 'RandomForestNminl5', 'RandomForestNminls5', 'SupportVectorMachine',    'SupportVectorMachineC2', 'SupportVectorMachineC4', 'AdaBoost', 'AdaBoost100', 'AdaBoost200', 'GradBoost', 'GradBoostDepth6', 'GradBoostDepth12', 'LASSO', 'NeuralNet', 'NeuralNetLarge', 'NeuralNetLarger', 'Mean'. If 'automatic', an optimal methods will be chosen based on 5-fold cross-validation in the training data. If a method is specified it will be used for all scores and all adjustments. If 'automatic', every policy score might be adjusted with a different method. 'Mean' is included for cases in which regression methods have no explanatory power. Default is 'RandomForest'.
          - **fair_type** : String (or None), optional. Method to chose the type of correction for the policy scores. 'Mean':  Mean dependence of the policy score on protected var's is removed by residualisation. 'MeanVar':  Mean dependence and heteroscedasticity is removed by residualisation and rescaling. Default (or None) is 'MeanVar'.
          - **var_protected_ord_name** : List or tuple of strings (nor None), optional. Names of ordered variables for which their influence will be removed on the policy scores.
          - **var_protected_unord_name** : List or tuple of strings (nor None),optional. Names of unordered variables for which their influence will be removed on the policy scores.

-  **Solve method has an additional return (2nd position)**

    - **result_dic** : Dictionary that contains additional information about the trained allocation rule. Currently, the only entry is a dictionary decribing the terminal leaves of the policy tree (or None if the policy has been selected as allocation method).

- **Solve method has a new algorithm named 'bps_classifier'**

    - The **bps_classifier** classifier algorithm runs a classifier for each of the allocations obtained by the 'best_policy_score' algorithm. One advantage compared of this approach compared to the     'best_policy_score' algorithm is that the prediction of the allocation for new observations is fast as it does not require to recompute the policy score (as it is case with the 'best_policy_score' algorithm). The classifier is selected among four different classifiers offered by  sci-kit learn, namely a simple neural network, two classification random forests with minimum leaf size of 2 and 5, and ADDABoost. The selection is a made according to the out-of-sample performance on scikit-learns Accuracy Score.

- Some additional explanations to the output of the policy tree (including a warning if there are more than 30 features for the policy trees) have been added.

Changes concerning the class :py:class:`~mcf_functions.McfOptPolReport`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- The structure of the policy tree is added to the pdf file.


Version 0.5.1
-------------

General
~~~~~~~

- Updated link to new website on PyPI. 

Version 0.5.0
-------------

General
~~~~~~~

- In general, most changes lead to more efficient code.
- A new reporting tool is introduced that produces a pdf file that should be more informative about estimation and results. The existing output via figures, (*.csv) and (*.txt) files continue to exist. They contain more detailed information than the new pdf files.

Changes concerning the class :py:class:`~mcf_functions.ModifiedCausalForest`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Bug fixes**

    - OOB values were not available for tuning forests.

- **Performance improvements**

    - Several parts have been optimized that led to limited speed increases and reduced memory consumption.
    - Some GPU capabilities have been added (based on Pytorch tensors; therefore Pytorch needs to be installed in addition even if the GPU is not used). Currently, GPU (if available) is used only to speed up Mahalanobis matching prior to training the causal forest (note that the default is NOT to use Mahalanobis matching, but to use matching based on the prognostic score instead; partly on computational grounds).

- **Name change of keywords**

    - ``gen_replication`` --> ``_int_replication``
    - ``p_amgate`` --> ``p_cbgate``
    - ``p_gmate_no_evalu_points`` --> ``p_gates_no_evalu_points``
    - ``p_gmate_sample_share`` --> ``p_bgate_sample_share``

- **New keyword**

    - ``_int_cuda`` : Boolean (or None). Use CUDA based GPU if available on hardware. Default is True.

- **Sensitivity analysis**

    - The method :py:meth:`~mcf_functions.ModifiedCausalForest.sensitivity` has the new keyword ``results``. Here the standard output dictionary from the :meth:`~mcf_functions.ModifiedCausalForest.predict` method is expected. If this dictionary contains estimated IATEs, the same data as in the :meth:`~mcf_functions.ModifiedCausalForest.predict` method will be used, IATEs are computed under the no effect (basic) scenario and these IATEs are compared to the IATEs contained in the results dictionary. 
    - If the dictionary does not contain estimated IATEs, passing it has no consequence.
    - If the results dictionary is passed, and it contains IATEs, then the (new) default value for the keyword ``sens_iate`` is True (and False otherwise)
          
Changes concerning the class :py:class:`~optpolicy_functions.OptimalPolicy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Bug fixes**

    - Single variables can be passed as strings without leading to errors.

- **General performance improvements**

    - Several parts have been optimized that led to limited increases and reduced memory consumption.

- **Change of names of keywords**

    (to use the same names as in the :py:class:`~mcf_functions.ModifiedCausalForest` class)

    - ``var_x_ord_name`` --> ``var_x_name_ord``
    - ``var_x_unord_name`` --> ``var_x_name_unord``

- **Change of default values**

    - The default of ``pt_enforce_restriction`` is set to False.
    - The previous default of ``pt_min_leaf_size`` is now multiplied by the smallest allowed treatment if (and only if) treatment shares are restricted.

- **Method for policy trees**

    - "policy tree eff" becomes the standard method for policy trees and is renamed as "policy tree".

- **Change of default value for ``gen_variable_importance``**

    - Change of default value** for ``gen_variable_importance``. New default is True.

- **Changes to speed up the computation of policy trees**

    - New keyword: ``_int_xtr_parallel`` Parallelize to a larger degree to make sure all CPUs are busy for most of the time. Only used for "policy tree" and only used if ``_int_parallel_processing`` > 1 (or None). Default is True.

- **New option to build a new optimal policy trees**  

    There is the new option to build a new optimal policy trees based on the data in each leaf of the (first) optimal policy tree. Although this second tree will also be optimal, the combined tree is no longer optimal. The advantage is a huge speed increase, i.e. a 3+1 tree computes much, much faster than a 4+0 tree, etc. This increased capabilities require a change in keywords:

    - Deleted keyword: ``pt_depth_tree``
    - New keywords

        - ``pt_depth_tree_1``   Depth of 1st optimal tree. Default is 3.
        - ``pt_depth_tree_2``   Depth of 2nd optimal tree. This tree is build within the strata obtained from the leaves of the first tree. If set to 0, a second tree is not build. Default is 1. Using both defaults leads to a (not optimal) total tree of level of 4.

New class :py:class:`~mcf_functions.McfOptPolReport`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. versionadded:: 0.5.0
        Reporting tools for the :class:`~mcf_functions.ModifiedCausalForest` and
        :class:`~optpolicy_functions.OptimalPolicy` classes

- This new class provides informative reports about the main specification choices and most important results of the ModifiedCausalForest and OptimalPolicy estimations. The report is saved in pdf-format.The reporting capabilities in this version are still basic but will be continously extended in the future (if users see them as a useful addition to the package).
- Method: the :py:meth:`~reporting.McfOptPolReport.report` method takes the instance of the ModifiedCausalForest and the OptimalPolicy classes as input (after they were used in running the different methods of both classes). It creates the report on a pdf file, which is saved in a user provided location. 

-----

Version 0.4.3
-------------

Changes concerning the class :py:class:`~mcf_functions.ModifiedCausalForest`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bug fixes
+++++++++

- Minor bug fixes:

    - Weight computation (turned off and sparse weight matrix)
    - KeyError in Gate estimation
    - Corrected sample split when using feature selection

New
+++

- Leaf size adjustments:

  Sometimes, the mcf leads to fairly big leaves due to insufficient observations in each treatment arm. The following changes in default settings and minor code corrections have been implemented. They somewhat reduce leaf sizes, but necessarily lead to more cases, where the data used to populate the leaves will have to ignore more leaves as they cannot be populated with outcomes from all treatment arms.

  In this case, if the problem can be solved be redoing the last split (i.e. using the parent leave instead of the final child leaves), then these two leaves are merged.

  If this does not solve the problem (either because one of the children is split further, or because there are still treatment arms missing in the merged leave), then this leave is not used in the computation of the weights.

  - Default for ``cf_n_min_treat`` changed to `(n_min_min + n_min_max) / 2 / # of treatments / 10`. Minimum is 1.
  - Defaults for ``cf_n_min_min`` and ``cf_n_min_max`` changed to:
    - `n_min_min = round(max((n_d_subsam**0.4) / 10, 1.5) * # of treatments)`
    - `n_min_max = round(max((n_d_subsam**0.5) / 10, 2) * # of treatments)`
  - Default values for tuning parameters are taken into account when observations are used only for feature selection, common support, or local centering.

- Improved computational performance:

  - Speed-up for categorical (unordered) variables due to memorization. This requires some additional memory, but the gains could be substantial.
  - Improved internal computation and storage of estimated forests lead to speed and precision gains (instead of using lists of lists, we now use a list of dictionaries of optimized numpy arrays to save the trees). Since the precision of the new method is higher (by at the same time needing less RAM), this might lead to smallish changes in the results.

- **Experimental**: The method :py:meth:`~mcf_functions.ModifiedCausalForest.sensitivity` has been added. It contains some simulation-based tools to check how well the mcf works in removing selection bias and how sensitive the results are with respect to potentially missing confounding covariates (i.e., those related to treatment and potential outcome) added in the future.

  - Note: This section is currently experimental and thus not yet fully documented and tested. A paper by Armendariz-Pacheco, Lechner, Mareckova and Odermatt (2024) will discuss and investigate the different methods in detail. So far, please note that all methods are simulation based.

  - The sensitivity checks consist of the following steps:

    1. Estimate all treatment probabilities.

    2. Remove all observations from treatment states other than one (largest treatment or user-determined).

    3. Use estimated probabilities to simulate treated observations, respecting the original treatment shares (pseudo-treatments).

    4. Estimate the effects of pseudo-treatments. The true effects are known to be zero, so the deviation from 0 is used as a measure of result sensitivity.

    Steps 3 and 4 may be repeated, and results averaged to reduce simulation noise.

  - In this experimental version, the method depends on the following new keywords:

    - ``sens_amgate``: Boolean (or None), optional. Compute AMGATEs for sensitivity analysis. Default is False.
    - ``sens_bgate``: Boolean (or None), optional. Compute BGATEs for sensitivity analysis. Default is False.
    - ``sens_gate``: Boolean (or None), optional. Compute GATEs for sensitivity analysis. Default is False.
    - ``sens_iate``: Boolean (or None), optional. Compute IATEs for sensitivity analysis. Default is False.
    - ``sens_iate_se``: Boolean (or None), optional. Compute standard errors of IATEs for sensitivity analysis. Default is False.
    - ``sens_scenarios``: List or tuple of strings, optional. Different scenarios considered. Default is ('basic',). 'basic': Use estimated treatment probabilities for simulations. No confounding.
    - ``sens_cv_k``: Integer (or None), optional. Data to be used for any cross-validation: Number of folds in cross-validation. Default (or None) is 5.
    - ``sens_replications``: Integer (or None), optional. Number of replications for simulating placebo treatments. Default is 2.
    - ``sens_reference_population``: Integer or float (or None). Defines the treatment status of the reference population used by the sensitivity analysis. Default is to use the treatment with most observed observations.

Changes concerning the class :py:class:`~optpolicy_functions.OptimalPolicy`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- No changes.

-----

Version 0.4.2
-------------

Bug fixes
~~~~~~~~~

- Minor bug fixes for :py:class:`~mcf_functions.ModifiedCausalForest` (mainly redundant elements in return of prediction and analysis method deleted).

New
~~~

General
+++++++

- Output files for text, data and figures: So far, whenever a directory existed that has already been used for output, a new directory is created to avoid accidentally overwriting results. However, there is a new keyword for both the :py:class:`~mcf_functions.ModifiedCausalForest` and the :py:class:`~optpolicy_functions.OptimalPolicy` class:

    - ``_int_output_no_new_dir``: Boolean. Do not create a new directory for outputs when the path already exists. Default is False.

Changes concerning the class :py:class:`~mcf_functions.ModifiedCausalForest`
+++++++++++++++++++++++++++++++++++++++++++++++++++++

- Mild improvements of output when categorical variables are involved.
- Data used for common support are saved in csv files.
- New keyword ``_int_del_forest``: Boolean. Delete forests from instance. If True, less memory is needed, but the trained instance of the class cannot be reused when calling predict with the same instance again, i.e. the forest has to be retrained. Default is False.
- New keyword ``_int_keep_w0``: Boolean. Keep all zero weights when computing standard errors (slows down computation). Default is False.
- New keyword ``p_ate_no_se_only``: Boolean (or None). Computes only the ATE without standard errors. Default is False.
- New default value for ``gen_iate_eff``: The second round IATE estimation is no longer performed by default (i.e. the new default is False).
- There is a new experimental features to both the mcf estimation (of IATEs) as well as the optimal policy module. It allows to partially blind the decision with respect to certain variables. The accompanying discussion paper by Nora Bearth, Fabian Muny, Michael Lechner, and Jana Marackova ('Partially Blind Optimal Policy Analysis') is currently written. If you desire more information, please email one of the authors. 

        - New method :py:meth:`~mcf_functions.ModifiedCausalForest.blinder_iates`: Compute 'standard' IATEs as well as IATEs that are to a certain extent blinder than the standard ones. Available keywords:

            - ``blind_var_x_protected_name`` : List of strings (or None). Names of protected variables. Names that are explicitly denote as blind_var_x_unrestricted_name or as blind_var_x_policy_name and used to compute IATEs will be automatically added to this list. Default is None.
            - ``blind_var_x_policy_name`` : List of strings (or None). Names of decision variables. Default is None.
            - ``blind_var_x_unrestricted_name`` : List of strings (or None). Names of unrestricted variables. Default is None.
            - ``blind_weights_of_blind`` : Tuple of float (or None). Weights to compute weighted means of blinded and unblinded IATEs. Between 0 and 1. 1 implies all weight goes to fully blinded IATE. Default is None.
            - ``blind_obs_ref_data`` : Integer (or None), optional. Number of observations to be used for blinding. Runtime of programme is almost linear in this parameter. Default is 50.
            - ``blind_seed`` : Integer, optional. Seed for the random selection of the reference data. Default is 123456.

Changes concerning the class :py:class:`~optpolicy_functions.OptimalPolicy`
++++++++++++++++++++++++++++++++++++++++++++++

- General keyword change in the :py:class:`~optpolicy_functions.OptimalPolicy` class. All keywords that started with `int_` now start with `_int_` (in order to use the same conventions as in the :py:class:`~mcf_functions.ModifiedCausalForest` class).

- New keywords:

    - ``_pt_select_values_cat``: Approximation method for larger categorical variables. Since we search among optimal trees, for categorical variables variables we need to check for all possible combinations of the different values that lead to binary splits. This number could indeed be huge. Therefore, we compare only pt_no_of_evalupoints * 2 different combinations. Method 1 (pt_select_values_cat == True) does this by randomly drawing values from the particular categorical variable and forming groups only using those values. Method 2 (pt_select_values_cat==False) sorts the values of the categorical variables according to a values of the policy score as one would do for a standard random forest. If this set is still too large, a random sample of the entailed combinations is drawn.  Method 1 is only available for the method 'policy tree eff'. The default is False.
    - ``_pt_enforce_restriction``: Boolean (or None). Enforces the imposed restriction (to some extent) during the computation of the policy tree. This can be very time consuming. Default is True.
    - ``_pt_eva_cat_mult``: Integer (or None). Changes the number of the evaluation points (pt_no_of_evalupoints) for the unordered (categorical) variables to: pt_eva_cat_mult * pt_no_of_evalupoints (available only for the method 'policy tree eff'). Default is 1.
    - ``_gen_variable_importance``: Boolean. Compute variable importance statistics based on random forest classifiers. Default is False.
    - ``_var_vi_x_name``: List of strings or None, optional. Names of variables for which variable importance is computed. Default is None.
    - ``_var_vi_to_dummy_name``: List of strings or None, optional. Names of variables for which variable importance is computed. These variables will be broken up into dummies. Default is None.

The optimal policy module currently has three methods (:py:meth:`~optpolicy_functions.OptimalPolicy.best_policy_score`, :py:meth:`~optpolicy_functions.OptimalPolicy.policy tree`, :py:meth:`~optpolicy_functions.OptimalPolicypolicy tree eff`):

- :py:meth:`~optpolicy_functions.OptimalPolicypolicy tree eff` (NEW in 0.4.2) is very similar to 'policy tree'. It uses different approximation rules and uses slightly different coding.  In many cases it should be faster than 'policy tree'.  Default (or None) is 'best_policy_score'.
- :py:meth:`~optpolicy_functions.OptimalPolicy.best_policy_score` conducts Black-Box allocations, which are obtained by using the scores directly (potentially subject to restrictions). When the Black-Box allocations are used for allocation of data not used for training, the respective scores must be available.
- The implemented :py:meth:`~optpolicy_functions.OptimalPolicy.policy tree`'s are optimal trees, i.e. all possible trees are checked if they lead to a better performance. If restrictions are specified, then this is incorporated into treatment specific cost parameters. Many ideas of the implementation follow Zhou, Athey, Wager (2022). If the provided policy scores fulfil their conditions (i.e., they use a doubly robust double machine learning like score), then they also provide attractive theoretical properties.

- New method :py:meth:`~optpolicy_functions.OptimalPolicy.evaluate_multiple`: Evaluate several allocations simultaneously.  Parameters:

    - ``allocations_dic`` : Dictionary. Contains DataFrame's with specific allocations.
    - ``data_df`` : DataFrame. Data with the relevant information about potential outcomes which will be used to evaluate the allocations.

-----

Version 0.4.1
-------------

Bug fixes
~~~~~~~~~

- Bug fix for AMGATE and Balanced GATE (BGATE)
- Minor bug fixes in Forest and Optimal Policy module

New
~~~

- We provide the change_log.py script, which provides extensive information on past changes and upcoming changes.
- We provide example data and example files on how to use :py:class:`~mcf_functions.ModifiedCausalForest` and :py:class:`~optpolicy_functions.OptimalPolicy` in various ways.

    - The following data files are provided. The names are self-explanatory. The number denotes the sample size, x are features, y is outcome, d is treatment, and ps denotes policy scores.:

        - data_x_1000.csv
        - data_x_4000.csv
        - data_x_ps_1_1000.csv
        - data_x_ps_2_1000.csv
        - data_y_d_x_1000.csv
        - data_y_d_x_4000.csv

    - The following example programmes are provided:

        - all_parameters_mcf.py, all_parameters_optpolicy.py: Contains an explanation of all available parameters / keywords for the :py:class:`~mcf_functions.ModifiedCausalForest` and :py:class:`~optpolicy_functions.OptimalPolicy` classes.
        - min_parameters_mcf.py, min_parameters_optpolicy.py: Contains the minimum specifications to run the methods of the :py:class:`~mcf_functions.ModifiedCausalForest` and :py:class:`~optpolicy_functions.OptimalPolicy` classes.
        - training_prediction_data_same_mcf.py: One suggestion on how to proceed when data to train and fill the forest are the same as those used to compute the effects.
        - mcf_and_optpol_combined.py: One suggestion on how to combine mcf and optimal policy estimation in a simple split sample approach.

-----

Version 0.4.0
-------------

Both the mcf module and the optimal policy module have undergone major revisions. The goal was to increase scalability and reduce internal complexity of the modules. The entire package now runs on Python 3.11, which is also recommended and tested. Note that all keywords changed compared to prior versions. Refer to the APIs for an updated list. For details on the updated worfklow, consult the respective tutorials.

What's New
~~~~~~~~~~

Changes concerning the class :py:class:`~mcf_functions.ModifiedCausalForest`:
++++++++++++++++++++++++++++++++++++++++++++++++++++++

- Update in the feature selection algorithm.
- Update in the common support estimation.
- Updates related to GATE estimation:
  - Wald tests are no longer provided,
  - MGATEs are no longer estimated.
  - AMGATEs will be conducted for the same heterogeneity variables as the GATEs.
  - New parameter ``p_iate_m_ate`` to compute difference of the IATEs and the ATE. The default is False.
- New parameter ``p_iate_eff``.
- Introduction of the BGATEs.
- Sample reductions for computational speed ups, need to be user-defined. Related options are removed from the mcf:

    - ``_int_red_split_sample``
    - ``_int_red_split_sample_pred_share``
    - ``_int_smaller_sample``
    - ``_int_red_training``
    - ``_int_red_training_share``
    - ``_int_red_prediction``
    - ``_int_red_prediction_share``
    - ``_int_red_largest_group_train``
    - ``_int_red_largest_group_train_share``

- Improved scalability by splitting training data into chunks and taking averages.
- Unified data concept to deal with common support and local centering.

Name Changes and Default Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- All keywords are changed. Please refer to the :doc:`python_api`.

-----

Version 0.3.3
-------------

What's New
~~~~~~~~~~

- Now runs also on Python 3.10.x.
- Renaming of output: Marginal effects became Moderated effects.
- Speed and memory improvements:

    - Weight matrix computed in smaller chunks for large data
    - There is also a parameter that comes along this change (which should usually not be changed by the user)
    - ``_weight_as_sparse_splits``  Default value is round(Rows of prediction data * rows of Fill_y data / (20'000 * 20'000))
    
- Additional and improved statistics for balancing tests.

Bug fixes
~~~~~~~~~

- Correction of prognostic score nearest neighbour matching when local centering was activated.

Name Changes and Default Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Name changes:

    - ``m_share_min`` --> ``m_min_share``
    - ``m_share_max`` --> ``m_max_share``
    - ``nw_kern_flag`` --> ``nw_kern``
    - ``atet_flag`` --> ``atet``
    - ``gatet_flag`` --> ``gatet``
    - ``iate_flag`` --> ``iate``
    - ``iate_se_flag`` --> ``iate_se``
    - ``iate_eff_flag`` --> ``iate_eff``
    - ``iate_cv_flag`` --> ``iate_cv``
    - ``cond_var_flag`` --> ``cond_var``
    - ``knn_flag`` --> ``knn``
    - ``clean_data_flag`` --> ``clean_data``

- Default values

    - ``alpha_reg_min`` = 0.05
    - ``alpha_reg_max`` = 0.15
    - If ``alpha_reg_grid`` = 1 (default): ``alpha`` = (``alpha_reg_min`` + ``alpha_reg_ax``)/2
    - ``m_share_min`` = 0.1
    - ``m_share_max`` = 0.6
    - ``m_grid`` = 1
    - number of variables used for splitting = share * total # of variable
    - If ``m_grid`` ==1: ``m_share`` = (``m_share_min`` + ``m_share_max``)/2
    - ``n_min_min`` = ``n_d`` ** 0.4/6; at least 4
    - ``n_min_max`` = sqrt(``n_d``)/6, at least ^4 where n_d denotes the number of observations in the smallest treatment arm
    - If ``n_min_grid`` == 1: ``n_min``=(``n_min_min`` + ``n_min_max``)/2
    - ``n_min_treat`` = ``n_min_min`` + ``n_min_max``)/2 / # of treatments / 4. Minimum is 2.

-----

Version 0.3.2
-------------

What's New
~~~~~~~~~~

- In estimation use cross-fitting to compute the IATEs. To enable cross-fitting set iate_cv to True. The default is False. The default number of folds is 5 and can be overwritten via the input argument iate_cv_folds. The estimates are stored in the  iate_cv_file.csv. Further information on estimation and descriptives are stored in the iate_cv_file.txt.
- Compare GATE(x) to GATE(x-1), where x is the current evaluation point and x-1 the previous one by setting GATE_MINUS_PREVIOUS to True. The default is False.
- Set n_min_treat to regulate the minimum number of observations in the treatment leaves.
- Experimental support for Dask. The default for multiprocessing is Ray. You may deploy Dask by setting _RAY_OR_DASK ='dask'. Note that with Dask the call of the programme needs to proteced by setting `__name__ == '__main__'`

Bug fixes
~~~~~~~~~

- Minor bug when GATEs were printed is fixed.
- Updated labels in sorted effects plots.

Name Changes and Default Updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``effiate_flag`` = ``iate_eff_flag``
- ``smooth_gates`` = ``gates_smooth``
- ``smooth_gates_bandwidth`` = ``gates_smooth_bandwidth``
- ``smooth_gates_no_evaluation_points`` = ``gates_smooth_no_evaluation_points``
- ``relative_to_first_group_only`` = ``post_relative_to_first_group_only``
- ``bin_corr_yes`` = ``post_bin_corr_yes``
- ``bin_corr_threshold`` = ``post_bin_corr_threshold``
- Increase in the default for sampling share
- New defaults for feature selection
  - ``fs_other_sample_share`` = 0.33
  - ``fs_rf_threshold`` = 0.0001
- Defaults for ``n_min_min`` increased to n**0.4/10, at least 3; -1: n**0.4/5 - where n is the number of observations in the smallest treatment arm.
- Number of parallel processes set to ``mp_parallel`` = 80% of logical cores.
- ``subsample_factor_eval`` = True, where True means 2 * subsample size used for tree.

Version 0.3.1
-------------

What's New
~~~~~~~~~~

- New experimental feature: A new module is provided (optpolicy_with_mcf) that combines mcf estimations of IATEs with optimal policies (black-box and policy trees). It also provides out-of-sample evaluations of the allocations. For more details refer to Cox, Lechner, Bollens (2022) and user_evaluate_optpolicy_with_mcf.py.

Bug fixes
~~~~~~~~~

- csv files for GATE tables can also deal with general treatment definitions
- ``_mp_with_ray`` no longer an input argument
- names_pot_iate is an additional return from the estimator. It is a 2-tuple with the list of potentially outcomes.
- ``return_iate_sp`` is a new parameter to algorithm to predict and return effects despite ``with_output`` being set to False.

-----

Version 0.3.0
-------------

What's New
~~~~~~~~~~

- The mcf supports an object-oriented interface: new class :py:class:`~mcf_functions.ModifiedCausalForest` and methods (:py:meth:`~mcf_functions.ModifiedCausalForest.predict`, :py:meth:`~mcf_functions.ModifiedCausalForest.train` and :py:meth:`~mcf_functions.ModifiedCausalForest.train_predict`).
- Delivery of potential outcome estimates for which local centering is reversed by setting ``l_centering_undo_iate`` to True; default is True.
- Readily available tables for GATEs, AMGATEs, and MGATEs. Generated tables summarize all estimated causal effects. Tables are stored in respective folders.
- The optimal policy function is generalized to encompass also stochastic treatment allocations.

Bug fixes
~~~~~~~~~

- Training and prediction are done in separate runs.
- Issue in optimal policy learning for unobserved treatment was resolved.

-----

Version 0.2.6
-------------

Bug fixes
~~~~~~~~~

- Bug fix in general_purpose.py

-----

Version 0.2.5 (yanked)
----------------------

Bug fixes
~~~~~~~~~

- Bug fix in bootstrap of optimal policy module.

What's New
~~~~~~~~~~

- Change in output directory structure.
- Name change of file with predicted IATE (ends <foo>_IATE.csv)
- default value of ``l_centering_replication`` changed from False to True.
- More efficient estimation of IATE, referred to as EffIATE

-----

Version 0.2.4
-------------

Bug fixes
~~~~~~~~~

- Bug fix for cases when outcome had no variation when splitting.

What's New
~~~~~~~~~~

- File with IATEs also contains indicator of specific cluster in k-means clustering.
- Option for guaranteed replicability of results. sklearn.ensemble.RandomForestRegressor does not necessarily replicable results (due to threading). A new keyword argument (l_centering_replication, default is False) is added. Setting this argument to True slows down local centering a but but removes that problem

-----

Version 0.2.3
-------------

Bug fixes
~~~~~~~~~

- Missing information in init.py.

-----

Version 0.2.2
-------------

Bug fixes
~~~~~~~~~

- Bug fix in plotting GATEs.

What's New
~~~~~~~~~~

- ATEs are saved in csv file (same as data for figures and other effects).

-----

Version 0.2.1
-------------

Bug fixes
~~~~~~~~~

- Bug fix in MGATE estimation, which led to program aborting.

-----

Version 0.2.0
-------------

Bug fixes
~~~~~~~~~

- Bug fix for policy trees under restrictions.
- Bug fix for GATE estimation (when weighting was used).

What's New
~~~~~~~~~~

- Main function changed from `ModifiedCausalForest()` to `modified_causal_forest()`.
- Complete seeding of random number generator.
- Keyword modifications:

    - ``stop_empty`` removed as parameter,
    - ``descriptive_stats`` becomes ``_descriptive_stats``,
    - ``dpi`` becomes ``_dpi``,
    - ``fontsize`` becomes ``_fontsize``,
    - ``mp_vim_type`` becomes ``_mp_vim_type``,
    - ``mp_weights_tree_batch`` becomes ``_mp_weights_tree_batch``,
    - ``mp_weights_type`` becomes ``_mp_weights_type``,
    - ``mp_with_ray`` becomes ``_mp_with_ray``,
    - ``no_filled_plot`` becomes ``_no_filled_plot``,
    - ``show_plots`` becomes ``_show_plots``,
    - ``verbose`` becomes ``_verbose``,
    - ``weight_as_sparse`` becomes ``_weight_as_sparse``,
    - ``support_adjust_limits`` new keyword for common support.

- Experimental version of continuous treatment. Newly introduced keywords here

    - ``d_type``
    - ``ct_grid_nn``
    - ``ct_grid_w``
    - ``ct_grid_dr``

- The optimal policy function contains new rules based on 'black box' approaches, i.e., using the potential outcomes directly to obtain optimal allocations.
- The optimal policy function allows to describe allocations with respect to other policy variables than the ones used for determining the allocation.
- Plots:

    - improved plots
    - new overlapping plots for common support analysis

-----

Version 0.1.4
-------------

Bug fixes
~~~~~~~~~

- Bug fix for predicting from previously trained and saved forests.
- Bug fix in ``mcf_init_function`` when there are missing values.

What's New
~~~~~~~~~~

- ``_mp_ray_shutdown`` new defaults. If object size is smaller 100,000, the default is False and else True.

-----

Version 0.1.3
-------------

Bug fixes
~~~~~~~~~

- Minor bug fixes, which led to unstable performance.

What's New
~~~~~~~~~~

- ``subsample_factor`` is split into ``subsample_factor_eval`` and ``subsample_factor_forest``.
- New default value for ``stop_empty``.
- Optimal policy module computes the policy tree also sequentially. For this purpose, the `optpoltree` API has changed slightly. Renamed input arguments are

    - ``ft_yes``
    - ``ft_depth``
    - ``ft_min_leaf_size``
    - ``ft_no_of_evalupoints``
    - ``ft_yes``

- the new input arguments for the sequential tree are:

    - ``st_yes``
    - ``st_depth``
    - ``st_min_leaf_size``

-----

Version 0.1.2
-------------

Bug fixes
~~~~~~~~~

- Common support with very few observations is turned off.
- Minor fix of MSE computation for multiple treatments.  

What's New  
~~~~~~~~~~

- New default values for  

    - ``alpha_reg_grid``
    - ``alpha_reg_max``
    - ``alpha_reg_min``
    - ``knn_flag``
    - ``l_centering``
    - ``mp_parallel``
    - ``p_diff_penalty``
    - ``random_thresholds``
    - ``se_boot_ate``
    - ``se_boot_gate``
    - ``se_boot_iate``
    - ``stop_empty``

- Consistent use of a new random number generator.
- Ray is initialized once.
- Ray can be fine-tuned via

    - ``_mp_ray_del``
    - ``_mp_ray_shutdown``
    - ``mp_ray_objstore_multiplier`` becomes ``_mp_ray_objstore_multiplier``

- New options to deal with larger data sets:

    - ``reduce_split_sample``: split sample in a part used for estimation and predicting the effects for given x; large prediction sample may increase running time.
    - ``reduce_training``: take a random sample from training data.
    - ``reduce_prediction``: take a random sample from prediction data.
    - ``reduce_largest_group_train``: reduce the largest group in the training data; this should be less costly in terms of precision than taking random samples.

- Optional IATEs via ``iate_flag`` and optional standard errors via ``iate_se_flag``.
- `ModifiedCausalForest()` now also returns potential outcomes and their variances.
- ``mp_with_ray`` is a new input argument to `‌optpoltree()`;  Ray can be used for multiprocessing when calling `‌optpoltree()`.
- Block-bootstrap on :math:`w_i \times y_i` is the new clustered standard errors default. This is slower but likely to be more accurate  than the aggregation within-clusters deployed before.

-----

Version 0.1.1
-------------

Bug fixes
~~~~~~~~~

- Minor bug fixes concerning ``with_output``, ``smaller_sample``, (A,AM)GATE/IATE-ATE plots, and the sampling weights.

What's New
~~~~~~~~~~

- Optional tree-specific subsampling for evaluation sample (subsample variables got new names).
- k-Means cluster indicator for the IATEs saved in file with IATE predictions.
- Evaluation points of GATE figures are included in the output csv-file.
- Exception raised if choice based sampling is activated and there is no treatment information in predictions file.
- New defaults for ``random_thresholds``; by default the value is set to 20 percent of the square-root of the number of training observations.
- Stabilizing `ray` by deleting references to object store and tasks
- The function `ModifiedCausalForest()` returns now ATE, standard error (SE) of the ATE, GATE, SE of the GATE, IATE, SE of the IATE, and the name of the file with the predictions.

-----

Version 0.1.0
-------------

Bug fixes
~~~~~~~~~~

- Bug fix for dealing with missings.
- Bug fixes for problems computing treatment effects for treatment populations.
- Bug fixes for the use of panel data and clustering.

What's New
~~~~~~~~~~

- ``post_kmeans_no_of_groups`` can now be a list or tuple with multiple values for the number of clusters; the optimal value is chosen through silhouette analysis.
- Detection of numerical variables added; raises an exception for non-numerical inputs.
- All variables used are shown in initial treatment-specific statistics to detect common support issues.
- Improved statistics for common support analysis.

Experimental
~~~~~~~~~~~~

- Optimal Policy Tool building policy trees included bases on estimated IATEs (allowing implicitly for constraints and programme costs).
