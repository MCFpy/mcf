Estimation of treatment effects
===============================

Types of treatment effects
---------------------------

The Modified Causal Forest estimates three types of treatment effects, which differ in their aggregation level and are discussed in depth by `Lechner (2018) <https://doi.org/10.48550/arXiv.1812.09487>`_. These effects are the average treatment effect (:math:`\textrm{ATE}`), the group average treatment effect (:math:`\textrm{GATE}`), and the individualized average treatment effect (:math:`\textrm{IATE}`). [1]_

Let us consider a discrete, multi-valued treatment :math:`D`. The potential outcome of treatment state :math:`d` is denoted by :math:`Y^d`. The covariates that are needed to correct for selection bias are denoted by :math:`X`. :math:`Z \subset X` is a vector of features that defines the effect heterogeneity of interest. :math:`Z` can contain continuous and discrete variables. Often these are variables with
relatively "few values" that define population groups (e.g. age, gender, etc.). The effects of interest are then defined as:

.. math::

    \textrm{ATE}(m,l;\Delta) &:= \mathbb{E} \big[ Y^m-Y^l \big\vert D\in \Delta \big]

    \textrm{GATE}(m,l;z,\Delta) &:= \mathbb{E} \big[ Y^m-Y^l \big\vert Z=z, D\in \Delta \big]

    \textrm{IATE}(m,l;x) &:= \mathbb{E} \big[ Y^m-Y^l \big\vert X=x \big]

If :math:`\Delta = \{m\}` then :math:`\textrm{ATE}(m,l;\Delta)` is better known as the average treatment effect on the treated (:math:`\textrm{ATET}`) for the individuals that received treatment :math:`m`. :math:`\textrm{ATE's}` measure the average impact of treatment :math:`m` compared to treatment :math:`l` either for the entire population, or in case of an :math:`\textrm{ATET}`, for the units that actually received a specific treatment.

Whereas :math:`\textrm{ATE's}` are population averages, :math:`\textrm{IATE's}` are average effects at the finest possible aggregation level. They measure the average impact of treatment :math:`m` compared to treatment :math:`l` for units with features :math:`X = x`. :math:`\textrm{GATE's}` lie somewhere in-between these two extremes. They measure the average impact of treatment :math:`m` compared to treatment :math:`l` for units in group :math:`Z = z`. :math:`\textrm{GATE's}` and :math:`\textrm{IATES's}` are special cases of the so-called conditional average treatment effects (:math:`\textrm{CATE's}`).

The following sections will show you how to estimate these different types of treatment effects with the **mcf** package.

-----------------

.. [1] A recent paper by `Bearth & Lechner (2024) <https://browse.arxiv.org/abs/2401.08290>`_ introduced the Balanced Group Average Treatment Effect (:math:`\textrm{BGATE}`). Click :doc:`here </algorithm_reference/bgates_cbgates>` to learn more about estimating :math:`\textrm{BGATE's}` with the Modified Causal Forest.

Estimating ATE's / IATE's 
----------------------------------

The :math:`\textrm{ATE's}` as well as the :math:`\textrm{IATE's}` are estimated by default through the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method of the class :py:class:`~mcf_main.ModifiedCausalForest`. See :doc:`../getting_started` for a quick example on how to access these estimates.

Another way to access the estimated :math:`\textrm{ATE's}` is through the output folder that the **mcf** package generates once a Modified Causal Forest is initialized. You can find the location of this folder by accessing the `"outpath"` entry of the `gen_dict` attribute of your Modified Causal Forest:

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_main import ModifiedCausalForest
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"]
    )
    my_mcf.gen_dict["outpath"]

You can also specify this path through the ``gen_outpath`` parameter of the class :py:meth:`~mcf_main.ModifiedCausalForest`. The output folder will contain csv-files with the estimated :math:`\textrm{ATE's}` in the subfolder `ate_iate`.

You can control whether :math:`\textrm{IATE's}` and their standard errors are estimated by setting the parameters ``p_iate`` and ``p_iate_se`` of the class :py:class:`~mcf_main.ModifiedCausalForest` to True or False:

+---------------+-----------------------------------------------------------------------+
| Parameter     | Description                                                           |
+---------------+-----------------------------------------------------------------------+
| ``p_iate``    | If True, IATE's will be estimated. Default: True.                     |
+---------------+-----------------------------------------------------------------------+
| ``p_iate_se`` | If True, standard errors of IATE's will be estimated. Default: False. |
+---------------+-----------------------------------------------------------------------+

Example
~~~~~~~

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_main import ModifiedCausalForest
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Estimate IATE's but not their standard errors
        p_iate = True,
        p_iate_se = False
    )


Estimating ATET's
----------------------------------

The average treatment effects for the treated are estimated by the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method if the parameter ``p_atet`` of the class :py:class:`~mcf_main.ModifiedCausalForest` is set to True:

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_main import ModifiedCausalForest
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Estimating ATET's
        p_atet = True
    )
    
    my_mcf.train(training_df)
    results = my_mcf.predict(prediction_df)

The :math:`\textrm{ATET's}` are, similar to the :math:`\textrm{ATE's}`, stored in the `"ate"` entry of the dictionary returned by the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method. This entry will then contain both the estimated :math:`\textrm{ATET's}` as well as the :math:`\textrm{ATE's}`. The output that is printed to the console during prediction will present you a table with all estimated :math:`\textrm{ATE's}` and :math:`\textrm{ATET's}`, which should give you a good idea of the structure of the `"ate"` entry in the result dictionary.

.. code-block:: python

    results["ate"]

The standard errors of the estimates are stored in the `"ate_se"` entry of the same dictionary. The structure of the `"ate_se"` entry is analogous to the `"ate"` entry. 

.. code-block:: python

    results["ate_se"]

Another way to access the estimated :math:`\textrm{ATET's}` is through the output folder that the **mcf** package generates once a Modified Causal Forest is initialized. You can find the location of this folder by accessing the `"outpath"` entry of the `gen_dict` attribute of your Modified Causal Forest:

.. code-block:: python

    my_mcf.gen_dict["outpath"]

You can also specify this path through the ``gen_outpath`` parameter of the class :py:meth:`~mcf_main.ModifiedCausalForest`. The output folder will contain csv-files with the estimated :math:`\textrm{ATET's}` in the subfolder `ate_iate`.

Estimating GATE's
-----------------

Group average treatment effects are estimated by the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method if you define heterogeneity variables through the parameters ``var_z_name_cont``, ``var_z_name_ord`` or ``var_z_name_unord`` in your :py:class:`~mcf_main.ModifiedCausalForest`. For every feature in the vector of heterogeneity variables :math:`Z`, a :math:`\textrm{GATE}` will be estimated separately. Please refer to the table further below or the :py:class:`API <mcf_main.ModifiedCausalForest>` for more details on how to specify your heterogeneity variables with the above mentioned parameters.

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_main import ModifiedCausalForest
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        # Define binary variables as ordered for faster performance
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Specify the unordered heterogeneity variable 'female' for GATE estimation
        var_z_name_unord=["x_unord0"]
    )
    my_mcf.train(training_df)
    results = my_mcf.predict(training_df)

You can access the estimated :math:`\textrm{GATE's}` and their standard errors through their corresponding entries in the dictionary that is returned by the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method:

.. code-block:: python

    results["gate_names_values"] # Describes the structure of the 'gate' entry
    results["gate"] # Estimated GATE's
    results["gate_se"] # Standard errors of the estimated GATE's

A simpler way to inspect the estimated :math:`\textrm{GATE's}` is through the output folder that the **mcf** package generates once a Modified Causal Forest is initialized. You can find the location of this folder by accessing the `"outpath"` entry of the `gen_dict` attribute of your Modified Causal Forest:

.. code-block:: python

    my_mcf.gen_dict["outpath"]

You can also specify this path through the ``gen_outpath`` parameter of the class :py:meth:`~mcf_main.ModifiedCausalForest`. The output folder will contain both csv-files with the results as well as plots of the estimated :math:`\textrm{GATE's}` in the subfolder `gate`.

To estimate the :math:`\textrm{GATE's}` for subpopulations defined by treatment status (:math:`\textrm{GATET's}`), you can set the parameter ``p_gatet`` of the class :py:class:`~mcf_main.ModifiedCausalForest` to True. These estimates can be accessed in the same manner as regular :math:`\textrm{GATE's}`.

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        var_z_name_unord=["x_unord0"],
        # Estimate the GATE's for var_z_name_unord by treatment status
        p_gatet = True
    )

For a continuous heterogeneity variable, the Modified Causal Forest will by default
smooth the distribution of the variable. The smoothing procedure evaluates the effects at a local neighborhood around a pre-defined number of evaluation points. The number of evaluation points can be specified through the parameter ``p_gates_smooth_no_evalu_points`` of the class :py:class:`~mcf_main.ModifiedCausalForest`. The local neighborhood is based on an Epanechnikov kernel estimation using Silverman's bandwidth rule. The multiplier for Silverman's bandwidth rule can be chosen through the parameter ``p_gates_smooth_bandwidth``. 

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Specify the continuous heterogeneity variable for GATE estimation
        var_z_name_cont=["x_ord0"],
        # Smoothing the distribution of the continuous variable for GATE estimation
        p_gates_smooth = True,
        # The number of evaluation points is set to 40
        p_gates_smooth_no_evalu_points = 40
    )

Instead of smoothing continuous heterogeneity variables, you can also discretize them and estimate GATE's for the resulting categories. This can be done by setting the parameter ``p_gates_smooth`` of the class :py:class:`~mcf_main.ModifiedCausalForest` to False. The maximum number of categories for discretizing continuous variables can be specified through the parameter ``p_max_cats_z_vars``.

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Specify the continuous heterogeneity variable for GATE estimation
        var_z_name_cont=["x_ord0"],
        # Discretizing the continuous variable for GATE estimation
        p_gates_smooth = False,
        # The maximum number of categories for discretizing is set to 5
        p_max_cats_z_vars = 5
    )


Below you find a list of the discussed parameters that are relevant for the estimation of :math:`\textrm{GATE's}`. Please consult the :py:class:`API <mcf_main.ModifiedCausalForest>` for more details or additional parameters on :math:`\textrm{GATE}` estimation.

.. dropdown:: Commonly used parameters to estimate :math:`\ \textrm{GATE's}`

    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Parameter                         | Description                                                                                                                                                              |
    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_cont``               | Ordered feature(s) with many values used for :math:`\textrm{GATE}` estimation.                                                                                           |
    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_ord``                | Ordered feature(s) with few values used for :math:`\textrm{GATE}` estimation.                                                                                            |
    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_unord``              | Unordered feature(s) used for :math:`\textrm{GATE}` estimation.                                                                                                          |
    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``p_gatet``                       | If True, :math:`\textrm{GATE's}` are also computed by treatment status (:math:`\textrm{GATET's}`). Default: False.                                                       |
    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``p_gates_smooth``                | If True, a smoothing procedure is applied to estimate :math:`\textrm{GATE's}` for continuous variables in :math:`Z`. Default: True.                                      |
    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``p_gates_smooth_no_evalu_points``| If ``p_gates_smooth`` is True, this defines the number of evaluation points. Default: 50.                                                                                |
    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``p_gates_smooth_bandwidth``      | If ``p_gates_smooth`` is True, this defines the multiplier for Silverman's bandwidth rule. Default: 1.                                                                   |
    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | ``p_max_cats_z_vars``             | If ``p_gates_smooth`` is False, this defines the maximum number categorizes when discretizing continuous heterogeneity variables in :math:`Z`. Default: :math:`N^{0.3}`. |
    +-----------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


Stabilizing estimates by truncating weights
------------------------------------------------------

The Modified Causal Forest uses weighted averages to estimate treatment effects. If the weights of some observations are very large, they can lead to unstable estimates. To obtain more stable estimates, the **mcf** package provides the option to truncate forest weights to an upper threshold through the parameter ``p_max_weight_share`` of the class :py:class:`~mcf_main.ModifiedCausalForest`. By default, ``p_max_weight_share`` is set to 0.05. After truncation, the program renormalizes the weights for estimation. Because of the renormalization step, the final weights can be slightly above the threshold defined in ``p_max_weight_share``.

Example
~~~~~~~

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Truncate weights to an upper threshold of 0.01
        p_max_weight_share = 0.01
    )


Increase in Efficiency
----------------------

ATEs, GATEs, QIATEs, and IATEs can be estimated more efficiently by training the algortihm twice. In this procedure, the data used to build the forest and the data used to populate the leaves with outcome values switch roles. The predictions from both forests are then averaged. The same approach is applied to the variance. However, because the averaged variance is likely to overestimate the true variance, the resulting inference is conservative. 

Note that computation time is approximately doubled. Also, the dictionary returned by the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method provides results for the standard effects only (i.e., without efficiency optimization).
