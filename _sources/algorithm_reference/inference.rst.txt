Inference
=========

Weights-based inference
-----------------------

The default inference uses the representation of effect estimates as weighted
averages of outcomes. It combines the expectation of conditional variance with
the variance of conditional expectation, conditioning on forest weights.
Conditional moments are estimated by k-nearest neighbors (default) or
Nadaraya-Watson kernel regression.

With ``p_cond_var=False``, variance is estimated directly from weighted outcomes
rather than conditional moments. See `Lechner and Mareckova, Comprehensive
Causal Machine Learning <https://arxiv.org/abs/2405.10198v2>`_ for the theoretical
context and its assumptions.

Bootstrap inference
-------------------

The weight-based bootstrap resamples observations or clusters and renormalizes
weights. It does not retrain an entire forest in every replication.
``p_se_boot_ate``, ``p_se_boot_gate``, ``p_se_boot_iate`` and ``p_se_boot_qiate``
control the corresponding inference.

Use ``p_cluster_std`` with ``var_cluster_name`` for clustered inference; see
:doc:`../user_guide/weights_clusters`. When efficiency improvements are enabled through ``gen_ate_eff``,
``gen_gate_eff``, ``gen_iate_eff`` or ``gen_qiate_eff``, the variance
calculation depends on the type of treatment effect being estimated.
See :doc:`../user_guide/estimation` for details.

Parameters
----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Purpose
   * - ``p_cond_var``
     - Use conditional moments (default True).
   * - ``p_knn``
     - Use k-nearest neighbors (True) or Nadaraya-Watson (False).
   * - ``p_se_boot_ate``, ``p_se_boot_gate``
     - Weight-based bootstrap for average and group effects.
   * - ``p_se_boot_iate``, ``p_se_boot_qiate``
     - Weight-based bootstrap for individual and quantile effects.
   * - ``p_ci_level``
     - Pointwise confidence level used in effect plots.

See the :py:class:`API <mcf_main.ModifiedCausalForest>` for details.

Example
~~~~~~~

.. code-block:: python

   from mcf.example_data import example_data
   from mcf.mcf_main import ModifiedCausalForest

   # Generate example data using the built-in function `example_data()`
   training_df, prediction_df, name_dict = example_data()

   my_mcf = ModifiedCausalForest(
       var_y_name="outcome",
       var_d_name="treat",
       var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
       # Bootstrap of standard errors for ATE
       p_se_boot_ate=None,
       # Conditional mean & variances are used
       p_cond_var=True,
       # Specifies k-NN method
       p_knn=True
   )

   my_mcf.train(training_df)
   results = my_mcf.predict(prediction_df)
