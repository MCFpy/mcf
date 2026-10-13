Experimental features
=====================

This section describes advanced and experimental features. 
Consult the method-specific API before combining them. Please open an issue `here <https://github.com/MCFpy/mcf/issues>`__ if you encounter any problems or have any questions.

Balancing Tests
---------------

Treatment effects may be subject to selection bias if the distribution of the confounding features differs across treatment arms. The class :py:class:`~mcf_main.ModifiedCausalForest` provides the option to conduct balancing tests to assess whether the feature distributions are equal across treatment arms after adjustment by the Modified Causal Forest. The balancing tests are based on the estimation of average treatment effects (:math:`\text{ATE's}`) with user-specified features as outcomes. If the features are balanced across treatment arms, the estimated :math:`\text{ATE's}` should be close to zero.

The Modified Causal Forest runs balancing tests for the features specified in the parameters ``var_x_name_balance_test_ord`` and ``var_x_name_balance_test_unord`` if the parameter ``p_bt_yes`` is set to True. See also the table below.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Purpose
   * - ``p_bt_yes``
     - Enable balancing tests.
   * - ``var_x_name_balance_test_ord``
     - Ordered variables used as balancing-test outcomes.
   * - ``var_x_name_balance_test_unord``
     - Unordered variables used as balancing-test outcomes.

Please consult the :py:class:`API <mcf_main.ModifiedCausalForest>` for more details.

The results of the balancing tests are part of the txt-file in the output folder that the **mcf** package generates. You can find the location of this folder by accessing the `"outpath"` entry of the ``gen_cfg`` attribute of your Modified Causal Forest:

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord="x_cont0"
        )

    my_mcf.gen_cfg.outpath

Example
~~~~~~~

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    params = dict(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        var_x_name_unord=["x_unord0"],
        # Parameters for balancing tests:
        p_bt_yes=True,
        var_x_name_balance_test_ord=["x_cont0", "x_cont1", "x_ord1"],
        var_x_name_balance_test_unord=["x_unord0"]
    )

    my_mcf = ModifiedCausalForest(**params)
    my_mcf.train(training_df)
    results = my_mcf.predict(prediction_df)

    # sensitivity() needs a new, untrained instance with the same settings.
    my_mcf_sens = ModifiedCausalForest(**params)
    sensitivity_results = my_mcf_sens.sensitivity(
        training_df, prediction_df, results=results
    )


Sensitivity checks
------------------

:py:meth:`~mcf_main.ModifiedCausalForest.sensitivity` provides a simulation-based
diagnostic. The implemented ``basic`` scenario estimates treatment probabilities,
selects a reference treatment population and simulates placebo treatments.
Effects are known to be zero in this scenario, so estimated deviations from zero
help assess the procedure in that simulated setting. Replications reduce
simulation noise. This does not identify or bound arbitrary unobserved confounding.

The following fragment uses an existing **mcf** object and data:

.. code-block:: python

    sensitivity_results = my_mcf.sensitivity(
        training_df, prediction_df, results=results
    )

``results`` is required; pass ``None`` if no prior results are to be used. When
supplied results contain an IATE dataframe and name metadata, those rows take
precedence over ``prediction_df``. Feature selection is not supported by this
sensitivity procedure. See the method API for compatible settings.

GPU Support
-----------

Set ``_int_cuda=True`` to request GPU processing. This requires CUDA-capable
PyTorch and may fall back to the CPU when CUDA is unavailable. The IATE GPU path
also requires low-memory prediction and bias adjustment to be disabled, and fewer
than 16 selected IATE workers. Availability does not guarantee a speed improvement;
see the :py:class:`API <mcf_main.ModifiedCausalForest>` for current restrictions.
