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

If :math:`\Delta = \{m\}` then :math:`\textrm{ATE}(m,l;\Delta)` is better known as the average treatment effect on the treated (:math:`\textrm{ATET}`) for the individuals that received treatment :math:`m`. :math:`\textrm{ATEs}` measure the average impact of treatment :math:`m` compared to treatment :math:`l` either for the entire population, or in case of an :math:`\textrm{ATET}`, for the units that actually received a specific treatment.

.. [1] The paper by `Bearth & Lechner (2024) <https://browse.arxiv.org/abs/2401.08290>`_ introduced the Balanced Group Average Treatment Effect (:math:`\textrm{BGATE}`). Click :doc:`here </algorithm_reference/bgates_cbgates>` to learn more about estimating :math:`\textrm{BGATEs}` with the Modified Causal Forest.

Estimating ATEs and IATEs
----------------------------------

The :math:`\textrm{ATEs}` as well as the :math:`\textrm{IATEs}` are estimated by default through the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method of the class :py:class:`~mcf_main.ModifiedCausalForest`. See :doc:`../getting_started` for a quick example on how to access these estimates.

Another way to access the estimated :math:`\textrm{ATEs}` is through the output folder that the **mcf** package generates once a Modified Causal Forest is initialized. You can find the location of this folder by accessing the ``outpath`` attribute of ``gen_cfg`` of your Modified Causal Forest:

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"]
    )
    my_mcf.gen_cfg.outpath

You can also specify this path through the ``gen_outpath`` parameter of the class :py:class:`~mcf_main.ModifiedCausalForest`. The output folder will contain csv-files with the estimated :math:`\textrm{ATEs}` in the subfolder `ate_iate`.

You can control whether :math:`\textrm{IATEs}` and their standard errors are estimated by setting the parameters ``p_iate`` and ``p_iate_se`` of the class :py:class:`~mcf_main.ModifiedCausalForest` to True or False:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - ``p_iate``
     - If True, IATEs will be estimated. Default: True.
   * - ``p_iate_se``
     - If True, standard errors of IATEs will be estimated. Default: False.


Example
~~~~~~~

The following block illustrates a configuration; call ``train()`` and
``predict()`` as in :doc:`/getting_started` to obtain estimates.

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Estimate IATEs but not their standard errors
        p_iate = True,
        p_iate_se = False
    )


Estimating ATETs
----------------------------------

The average treatment effects for the treated are estimated by the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method if the parameter ``p_atet`` of the class :py:class:`~mcf_main.ModifiedCausalForest` is set to True:

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Estimating ATETs
        p_atet = True
    )

    my_mcf.train(training_df)
    results = my_mcf.predict(prediction_df)

The :math:`\textrm{ATETs}` are, similar to the :math:`\textrm{ATEs}`, stored in the `"ate"` entry of the dictionary returned by the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method. This entry will then contain both the estimated :math:`\textrm{ATETs}` as well as the :math:`\textrm{ATEs}`. The output that is printed to the console during prediction will present you a table with all estimated :math:`\textrm{ATEs}` and :math:`\textrm{ATETs}`, which should give you a good idea of the structure of the `"ate"` entry in the result dictionary.

.. code-block:: python

    results["ate"]

The standard errors of the estimates are stored in the `"ate_se"` entry of the same dictionary. The structure of the `"ate_se"` entry is analogous to the `"ate"` entry.

.. code-block:: python

    results["ate_se"]

Another way to access the estimated :math:`\textrm{ATETs}` is through the output folder that the **mcf** package generates once a Modified Causal Forest is initialized. You can find the location of this folder by accessing the ``outpath`` attribute of ``gen_cfg`` of your Modified Causal Forest:

.. code-block:: python

    my_mcf.gen_cfg.outpath

You can also specify this path through the ``gen_outpath`` parameter of the class :py:class:`~mcf_main.ModifiedCausalForest`. The output folder will contain csv-files with the estimated :math:`\textrm{ATETs}` in the subfolder `ate_iate`.

Estimating GATEs
-----------------

Group average treatment effects are estimated by the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method if you define heterogeneity variables through the parameters ``var_z_name_cont``, ``var_z_name_ord`` or ``var_z_name_unord`` in your :py:class:`~mcf_main.ModifiedCausalForest`. For every feature in the vector of heterogeneity variables :math:`Z`, a :math:`\textrm{GATE}` will be estimated separately. Please refer to the table further below or the :py:class:`API <mcf_main.ModifiedCausalForest>` for more details on how to specify your heterogeneity variables with the above mentioned parameters.

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        # Define binary variables as ordered for faster performance
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Specify the unordered heterogeneity variable for GATE estimation
        var_z_name_unord=["x_unord0"]
    )
    my_mcf.train(training_df)
    results = my_mcf.predict(prediction_df)

You can access the estimated :math:`\textrm{GATEs}` and their standard errors through their corresponding entries in the dictionary that is returned by the :py:meth:`~mcf_main.ModifiedCausalForest.predict` method:

.. code-block:: python

    results["gate_names_values"] # Describes the structure of the 'gate' entry
    results["gate"] # Estimated GATEs
    results["gate_se"] # Standard errors of the estimated GATEs

A simpler way to inspect the estimated :math:`\textrm{GATEs}` is through the output folder that the **mcf** package generates once a Modified Causal Forest is initialized. You can find the location of this folder by accessing the ``outpath`` attribute of ``gen_cfg`` of your Modified Causal Forest:

.. code-block:: python

    my_mcf.gen_cfg.outpath

You can also specify this path through the ``gen_outpath`` parameter of the class :py:class:`~mcf_main.ModifiedCausalForest`. The output folder will contain both csv-files with the results as well as plots of the estimated :math:`\textrm{GATEs}` in the subfolder `gate`.

To estimate the :math:`\textrm{GATEs}` for subpopulations defined by treatment status (:math:`\textrm{GATETs}`), you can set the parameter ``p_gatet`` of the class :py:class:`~mcf_main.ModifiedCausalForest` to True. These estimates can be accessed in the same manner as regular :math:`\textrm{GATEs}`.

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        var_z_name_unord=["x_unord0"],
        # Estimate the GATEs for var_z_name_unord by treatment status
        p_gatet = True
    )

For continuous heterogeneity variables, ``p_gates_smooth=True`` adds smoothed
estimates alongside estimates for discretized groups. Smoothing uses an
Epanechnikov kernel; ``p_gates_smooth_bandwidth`` scales the bandwidth and
``p_gates_smooth_no_evalu_points`` limits the evaluation points.

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Specify the continuous heterogeneity variable for GATE estimation
        var_z_name_cont=["x_cont0"],
        # Smoothing the distribution of the continuous variable for GATE estimation
        p_gates_smooth = True,
        # The number of evaluation points is set to 40
        p_gates_smooth_no_evalu_points = 40
    )

Set ``p_gates_smooth=False`` to disable the additional smoothed estimates.
``p_max_cats_z_vars`` controls discretization during training; its default
depends on the effective training size per chunk.

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1"],
        # Specify the continuous heterogeneity variable for GATE estimation
        var_z_name_cont=["x_cont0"],
        # Discretizing the continuous variable for GATE estimation
        p_gates_smooth = False,
        # The maximum number of categories for discretizing is set to 5
        p_max_cats_z_vars = 5
    )


Below you find a list of the discussed parameters that are relevant for the estimation of :math:`\textrm{GATEs}`. Please consult the :py:class:`API <mcf_main.ModifiedCausalForest>` for more details or additional parameters on :math:`\textrm{GATE}` estimation.

.. dropdown:: Commonly used parameters to estimate :math:`\ \textrm{GATEs}`

    .. list-table::
       :widths: 30 70
       :header-rows: 1

       * - Parameter
         - Description
       * - ``var_z_name_cont``
         - Ordered feature(s) with many values used for :math:`\textrm{GATE}` estimation.
       * - ``var_z_name_ord``
         - Ordered feature(s) with few values used for :math:`\textrm{GATE}` estimation.
       * - ``var_z_name_unord``
         - Unordered feature(s) used for :math:`\textrm{GATE}` estimation.
       * - ``p_gatet``
         - If True, :math:`\textrm{GATEs}` are also computed by treatment status (:math:`\textrm{GATETs}`). Default: False.
       * - ``p_gates_smooth``
         - If True, a smoothing procedure is applied to estimate :math:`\textrm{GATEs}` for continuous variables in :math:`Z`. Default: True.
       * - ``p_gates_smooth_no_evalu_points``
         - If ``p_gates_smooth`` is True, this defines the number of evaluation points. Default: 50.
       * - ``p_gates_smooth_bandwidth``
         - If ``p_gates_smooth`` is True, this defines the multiplier for Silverman's bandwidth rule. Default: 1.
       * - ``p_max_cats_z_vars``
         - Maximum number of discretization groups, determined during training. See the API for the sample-dependent default.



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


Balanced and quantile effects
-----------------------------

BGATEs balance selected covariates across heterogeneous groups; CBGATEs balance
all remaining features. Their common reference population and implementation are
explained in :doc:`../algorithm_reference/bgates_cbgates`. Balancing alone does
not establish a causal moderation interpretation.

QIATEs summarize quantiles of conditional mean treatment effects across the
prediction population. They require discrete treatments, no treatment versions,
``p_ba=False``, and ``_int_low_memory_predict=False``. See
:doc:`../algorithm_reference/QIATEs` for quantile levels, contrasts and an example.

Increase in Efficiency
----------------------

The ``gen_*_eff`` options exchange the construction and filling samples and
average point estimates from two rounds. The variance calculation differs by
estimand:

- ATEs and group effects use the average of the two variance estimates, described
  as conservative inference.
- ``gen_iate_eff`` uses half of the regular-round variance for requested IATE
  inference; it does not estimate a separate additional-round variance. This is
  an implementation approximation.
- ``gen_qiate_eff`` averages QIATE estimates and requested variances from two
  rounds. Each round ranks its own effects; reported IATE averaging is controlled
  separately.

The returned dictionary contains the estimates for the resolved settings, including
efficiency adjustments. ``iate_eff`` identifies the resolved IATE setting.
Enabling efficient GATEs, IATEs or QIATEs also enables efficient ATEs during
initialization. Two-round estimation increases computation.

Obtaining New Results Without Retraining the Forest
---------------------------------------------------

``predict()`` accepts ``new_keywords`` only after training
and before the first prediction. This restriction also applies to an empty
update dictionary. For alternative specifications, copy the trained object
before its first prediction:

.. code-block:: python

    from copy import deepcopy

    # Fragment: trained_mcf has been trained but has not predicted yet.
    alternative_mcf = deepcopy(trained_mcf)
    alternative_results = alternative_mcf.predict(
        prediction_df, new_keywords={'p_iate_se': True}
    )

Updates persist on the instance. Repeated prediction without updates remains
possible when the trained forest is retained. Some flags can only be turned off
through an update; see the method-specific accepted keywords in
:py:meth:`~mcf_main.ModifiedCausalForest.predict`.

``analyse()`` has a different state requirement: updates require an instance
already used for prediction. They change the analysis configuration, not the
supplied effect estimates. See :py:meth:`~mcf_main.ModifiedCausalForest.analyse`.
