Data cleaning
=============

All supplied columns must be numeric and contain no missing values.
Otherwise, training or prediction stops with an error identifying the
affected columns. This also applies to unused columns, so remove them
before passing your data to **mcf**. The check runs before the cleaning step,
regardless of the setting of ``dc_clean_data``.

For discrete treatments, **mcf** internally converts the treatment categories
to consecutive integers starting at zero. If treatment information is
required for prediction, use the same treatment labels as in the training
data. For policy evaluation, treatment codes must instead correspond to
the order of the policy scores specified in ``var_polscore_name``.

Unordered features require at least three distinct training values. Put binary
features in ``var_x_name_ord``. Heterogeneity variables in ``var_z_name_*`` are added to the
corresponding forest-feature lists automatically.

Parameter overview
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Description
   * - ``dc_clean_data``
     - Drops missing observations and unused variables at the cleaning stage; prior validation still applies. Default: True.
   * - ``dc_screen_covariates``
     - Screens features, including those without variation. Default: True.
   * - ``dc_check_perfectcorr``
     - Screens near-duplicates using absolute Pearson correlation above 0.999. Protected variables may remain. Default: True.
   * - ``dc_min_dummy_obs``
     - Minimum count for either value of a binary feature, not only literal 0/1. Use 1 to disable this filter; 0 restores the default of 10.

See the :py:class:`API <mcf_main.ModifiedCausalForest>` for details.

Example
--------

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
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        # Parameters for data cleaning:
        dc_clean_data=True,
        dc_screen_covariates=True,
        dc_check_perfectcorr=False,
        dc_min_dummy_obs=100
    )
