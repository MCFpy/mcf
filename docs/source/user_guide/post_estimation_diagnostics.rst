Post-estimation diagnostics
===========================

The class :py:class:`~mcf_main.ModifiedCausalForest` provides you with several diagnostic tools to analyse the estimated :math:`\text{IATE's}`. They cover

- descriptive statistics
- a correlation analysis
- :math:`k`-means clustering 
- a feature importance analysis

To conduct *any* post-estimation diagnostics, the parameter ``post_est_stats`` of the class :py:class:`~mcf_main.ModifiedCausalForest` needs to be set to True. Once you have estimated your :math:`\text{IATE's}` using the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method, you can conduct the post-estimation diagnostics with the :py:meth:`~mcf_main.ModifiedCausalForest.analyse` method:

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.reporting import McfOptPolReport
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    
    my_mcf = ModifiedCausalForest(
            var_y_name="outcome",
            var_d_name="treat",
            var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
            var_x_name_unord=["x_unord0"],
            # Enable post-estimation diagnostics
            post_est_stats=True
        )
    
    my_mcf.train(training_df)
    results = my_mcf.predict(prediction_df)
    
    post_estimation_diagnostics = my_mcf.analyse(results)

The easiest way to to inspect the results of the post-estimation diagnostics, is to read the PDF-report that can be generated using the class :py:class:`~reporting.McfOptPolReport`:

.. code-block:: python

    mcf_report = McfOptPolReport(mcf=my_mcf, outputfile='Modified-Causal-Forest_Report')
    mcf_report.report()

You can additionally specify the reference group for the :math:`\text{IATE's}` with the parameter ``post_relative_to_first_group_only``. If ``post_relative_to_first_group_only`` is True, the comparison group will be the first treatment state. This is the default. If False, all possible treatment combinations are compared with each other. The confidence level in the post-estimation diagnostics is specified through the parameter ``p_ci_level``.


Descriptive statistics
----------------------

With ``post_est_stats`` set to True, the distribution of the estimated :math:`\text{IATE's}` will be presented. The produced plots are also available in the output folder that is produced by the **mcf** package. You can find the location of this folder by accessing the `"outpath"` entry of the `gen_dict` attribute of your Modified Causal Forest:

.. code-block:: python

    my_mcf.gen_cfg.outpath

You can also specify this path through the ``gen_outpath`` parameter of the class :py:meth:`~mcf_main.ModifiedCausalForest`. The output folder will contain the jpeg/pdf-files of the plots as well as csv-files of the underlying data in the subfolder `ate_iate`.


Correlation analysis
--------------------

The correlation analysis estimates the dependencies between the different :math:`\text{IATE's}`, between the :math:`\text{IATE's}` and the potential outcomes, and between the :math:`\text{IATE's}` and the features. You can activate the correlation analysis by setting the parameter ``post_bin_corr_yes`` to True. Note that the correlation coefficients are only displayed if their absolute values exceed the threshold specified by the parameter ``post_bin_corr_threshold``.


:math:`k`-means clustering
------------------

To analyze heterogeneity in different groups (clusters), you can conduct :math:`k`-means clustering by setting the parameter ``post_kmeans_yes`` to *True*. The **mcf** package uses the *k-means++* algorithm from scikit-learn to build clusters based on the :math:`\text{IATE's}`. 

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.reporting import McfOptPolReport
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    my_mcf = ModifiedCausalForest(
            var_y_name="outcome",
            var_d_name="treat",
            var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
            var_x_name_unord=["x_unord0"],
            post_est_stats=True,
            # Perform k-means clustering
            post_kmeans_yes=True
        )
    
    my_mcf.train(training_df)
    results = my_mcf.predict(prediction_df)
    
    post_estimation_diagnostics = my_mcf.analyse(results)

The report obtained through the class :py:class:`~reporting.McfOptPolReport` will contain descriptive statistics of the :math:`\text{IATE's}`, the potential outcomes and the features for each cluster. 

.. code-block:: python

    mcf_report = McfOptPolReport(mcf=my_mcf, outputfile='Modified-Causal-Forest_Report')
    mcf_report.report()

If you wish to analyse the clusters yourself, you can access the cluster membership of each observation through the *"iate_data_df"* entry of the dictionary returned by the :py:meth:`~mcf_main.ModifiedCausalForest.analyse` method. The cluster membership is stored in the column *IATE_Cluster* of the DataFrame.

.. code-block:: python

    post_estimation_diagnostics["iate_data_df"]


You can define a range for the number of clusters through the parameter ``post_kmeans_no_of_groups``. The final number of clusters is chosen via silhouette analysis. To guard against getting stuck at local extrema, the number of replications with different random start centers can be defined through the parameter ``post_kmeans_replications``. The parameter ``post_kmeans_max_tries`` sets the maximum number of iterations in each replication to achieve convergence.


Feature importance
----------------------------------

If you are interested in learning which of your features have a lot of predictive power for the estimated :math:`\text{IATE's}` you can activate the feature importance procedure by setting the parameter ``post_random_forest_vi`` to True. This procedure will build a predictive random forest to determine which features influence the :math:`\text{IATE's}` most. The feature importance statistics are presented in percentage points of the coefficient of determination, :math:`R^2`, that is lost when the respective feature is randomly permuted. The :math:`R^2` statistics are obtained through the *RandomForestRegressor* provided by scikit-learn.


Parameter overview
------------------

Below is an overview of the above mentioned parameters related to post-estimation diagnostics in the class :py:class:`~mcf_main.ModifiedCausalForest`:  

+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                             | Description                                                                                                                                                                                                                                                           |
+=======================================+=======================================================================================================================================================================================================================================================================+
| ``post_est_stats``                    | If True, post-estimation diagnostics are conducted. Default: True.                                                                                                                                                                                                    |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_relative_to_first_group_only`` | If True, post-estimation diagnostics will only be conducted for :math:`\text{IATE's}` relative to the first treatment state. If False, the diagnostics cover the :math:`\text{IATE's}` of all possible treatment combinations. Default: True.                         |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``p_ci_level``                        | Confidence level for plots, including the post-estimation diagnostic plots. Default: 0.9.                                                                                                                                                                             |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_bin_corr_yes``                 | If True, the binary correlation analysis is conducted. Default: True.                                                                                                                                                                                                 |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_bin_corr_threshold``           | If ``post_bin_corr_yes`` is True, correlations are only displayed if their absolute value is at least ``post_bin_corr_threshold``. Default: 0.1.                                                                                                                      |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_kmeans_yes``                   | If True, :math:`k`-means clustering is conducted to build clusters based on the :math:`\text{IATE's}`. Default: True.                                                                                                                                                 |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_kmeans_no_of_groups``          | Only relevant if ``post_kmeans_yes`` is True. Determines the number of clusters for :math:`k`-means clustering. Should be specified as a list of values. Default: See the :py:class:`API <mcf_main.ModifiedCausalForest>`.                                       |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_kmeans_max_tries``             | Only relevant if ``post_kmeans_yes`` is True. Determines the maximum number of iterations to achieve convergence in each :math:`k`-means clustering replication. Default: 1000.                                                                                       |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_kmeans_replications``          | Only relevant if ``post_kmeans_yes`` is True. Determines the number of replications for :math:`k`-means clustering. Default: 10.                                                                                                                                      |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_kmeans_min_size_share``        | Smallest share of cluster size allowed in %. Default (None) is 1.                                                                                                                                                                                                     |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_random_forest_vi``             | If True, the feature importance analysis is conduced. Default: True.                                                                                                                                                                                                  |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_plots``                        | If True, post-estimation diagnostic plots are printed during runtime. Default: True.                                                                                                                                                                                  |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``post_tree``                         | Regression trees (honest and standard) of Depth 2 to 5 are estimated to describe IATES(x). Default (or None) is True.                                                                                                                                                 |
+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Please consult the :py:class:`API <mcf_main.ModifiedCausalForest>` for more details.


Example
-------

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.reporting import McfOptPolReport
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    my_mcf = ModifiedCausalForest(
            var_y_name="outcome",
            var_d_name="treat",
            var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
            var_x_name_unord=["x_unord0"],
            p_ci_level=0.95,
            # Parameters for post-estimation diagnostics
            post_est_stats=True,
            post_relative_to_first_group_only=True,
            post_bin_corr_yes=True,
            post_bin_corr_threshold=0.1,
            post_kmeans_yes=True,
            post_kmeans_no_of_groups=[3, 4, 5, 6, 7],
            post_kmeans_max_tries=1000,
            post_kmeans_replications=10,
            post_random_forest_vi=True,
            post_plots=True,
            post_kmeans_min_size_share=1,
            post_tree=True
        )
    
    my_mcf.train(training_df)
    results = my_mcf.predict(prediction_df)
    
    # Compute the post-estimation diagnostics
    post_estimation_diagnostics = my_mcf.analyse(results)
    
    # Access cluster memberships (column 'IATE_Cluster')
    post_estimation_diagnostics["iate_data_df"]
    
    # Produce a PDF-report with the results, including post-estimation diagnostics
    mcf_report = McfOptPolReport(mcf=my_mcf, outputfile='Modified-Causal-Forest_Report')
    mcf_report.report()
