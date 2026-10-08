CBGATEs and BGATEs
==================

Balanced Group Average Treatment Effects (BGATEs) compare group effects using a
common distribution of selected covariates. For a heterogeneity variable :math:`Z`
and balancing variables :math:`W`, define

.. math::

    \operatorname{BGATE}(m,l;z,F_W)
    = \int E[Y^m-Y^l\mid Z=z,W=w]\,dF_W(w).

:math:`F_W` describes the distribution of the balancing variables in the
prediction data remaining after data preparation and common-support
checks. The same distribution is used for all groups, so their effects
are compared using a common covariate distribution.

CBGATEs balance all other features. A causal moderation interpretation requires
additional assumptions, not balancing alone; see `Bearth and Lechner, Causal
Machine Learning for Moderation Effects <https://arxiv.org/abs/2401.08290v3>`_.

Implementation
--------------

- BGATEs use selected balancing covariates and nearest-neighbor reconstruction
  at the heterogeneity evaluation values. Set ``p_bgate=True`` and select
  ``var_x_name_balance_bgate``.
- CBGATEs duplicate reference covariate rows and set the heterogeneity variable
  at each evaluation value. This balances all remaining features.

Both methods use the previously trained forest. ``p_bgate_sample_share``
controls the size of the reference sample, which is reused at each
evaluation value of the heterogeneity variable. Minimum sample-size
requirements can increase the number of observations used and the
computation time.

See ``p_bgate_sample_share`` in the
:py:class:`API <mcf_main.ModifiedCausalForest>` for the exact sampling rule.


Parameters
----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Purpose
   * - ``p_bgate``, ``p_cbgate``
     - Enable BGATE and CBGATE estimation, respectively.
   * - ``var_x_name_balance_bgate``
     - Variables used for BGATE balancing. Defaults to the other heterogeneity variables.
   * - ``p_gates_no_evalu_points``
     - Number of evaluation points when no stored evaluation values are available.
   * - ``p_bgate_sample_share``
     - Controls reference-sample subsampling for BGATE and CBGATE estimation.

See the :py:class:`API <mcf_main.ModifiedCausalForest>` for defaults
and detailed parameter descriptions.


Examples
~~~~~~~~~

.. code-block:: python

   from mcf.example_data import example_data
   from mcf.mcf_main import ModifiedCausalForest

   # Generate example data using the built-in function `example_data()`
   training_df, prediction_df, name_dict = example_data()

   my_mcf = ModifiedCausalForest(
       var_y_name="outcome",
       var_d_name="treat",
       var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
       var_z_name_cont=["x_cont0", "x_cont1"],
       # Variables to balance the GATEs on
       var_x_name_balance_bgate=["x_cont0", "x_cont1"],
       # Estimate a balanced GATE in selected features
       p_bgate=True,
       # Random samples to speed up the programme
       p_bgate_sample_share = None
   )

   my_mcf.train(training_df)
   results = my_mcf.predict(prediction_df)


.. code-block:: python

   my_mcf = ModifiedCausalForest(
       var_y_name="outcome",
       var_d_name="treat",
       var_x_name_ord=["x_cont0", "x_ord1"],
       var_x_name_unord=["x_unord0"],
       var_z_name_cont=["x_cont0"],
       # Estimate a GATE that is balanced in all other features
       p_cbgate=True
   )

   my_mcf.train(training_df)
   results = my_mcf.predict(prediction_df)
