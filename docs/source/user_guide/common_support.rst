.. _common-support:

Common support
==============

Common support requires overlap across treatment groups. The
:py:class:`~mcf_main.ModifiedCausalForest` checks estimated treatment
probabilities before effect estimation and removes observations outside the
selected support. This changes the population to which the results apply.

- ``cs_type=0`` disables the adjustment.
- ``cs_type=1`` (default) intersects treatment-group probability ranges.
- ``cs_type=2`` retains a row only when every treatment probability lies in
  ``[cs_min_p, 1 - cs_min_p]``. Both endpoints are included.

Probability models use classification forests. With ``lc_cs_cv=True``, construction
predictions are cross-fitted; filling and prediction observations use the average
of the fitted fold models. Alternatively, ``lc_cs_cv=False`` reserves a separate
sample for fitting these models.

Advanced options
----------------

For type 1, let :math:`L_{gj}` and :math:`U_{gj}` be the
:math:`(1-q)` and :math:`q` quantiles of probability :math:`p_j(X)` within
observed treatment group :math:`g`, where ``cs_quantil=q``. For :math:`q=1`,
use the group minima and maxima. With ``cs_adjust_limits=a``, the bounds are

.. math::

    \mathrm{lower}_j &= \max_g \operatorname{clip}((1-a)L_{gj},0,1), \\
    \mathrm{upper}_j &= \min_g \operatorname{clip}((1+a)U_{gj},0,1).

For example, ``cs_quantil=0.95`` uses the 5th and 95th percentiles of each
estimated treatment probability, calculated separately within each observed
treatment group. These bounds are widened using ``cs_adjust_limits``.
The common-support interval then runs from the largest lower bound to
the smallest upper bound across treatment groups.

With ``cs_type=1``, this check uses the estimated probabilities of all
treatments except the first. With ``cs_type=2``, every estimated treatment
probability must lie between ``cs_min_p`` and ``1 - cs_min_p``, including
the endpoints.

``cs_max_del_train`` sets the maximum proportion of observations that
may be removed by the common-support adjustment in the training data.
This limit is checked separately for the sample used to construct the
trees and the sample used to fill their leaves with outcomes.

For example, a value of 0.5 stops estimation if more than 50% of either
sample is removed. In that case, inspect the data and the overlap between
treatment groups before changing the common-support settings.

This limit does not apply to prediction data. Prediction observations
outside the estimated common support are still removed, but their removal
share is not restricted by ``cs_max_del_train``.

Parameter overview
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Purpose
   * - ``cs_type``
     - No adjustment (0), intersected group ranges (1), or fixed bounds (2).
   * - ``cs_min_p``
     - Lower bound for type 2; the upper bound is one minus this value. Default: 0.01.
   * - ``cs_quantil``
     - Type-1 quantile setting. Default: 1 (minima/maxima).
   * - ``cs_adjust_limits``
     - Relaxes type-1 ranges before intersection.
   * - ``cs_max_del_train``
     - Maximum training deletion share, checked in each sample. Default: 0.5.

Inspecting retained observations
--------------------------------

The following fragment uses an initialized ``my_mcf`` and prepared training and
prediction data. It learns support rules without building a forest:

.. code-block:: python

    my_mcf.train(training_df, exit_after_commonsupport=True)
    support = my_mcf.predict(prediction_df, exit_after_commonsupport=True)
    retained_df = support['inputdata_on_support']

``inputdata_on_support`` contains the observations remaining after data
preparation and the common-support adjustment. When output is enabled,
the returned probability tables include an ``on_support`` column indicating
which observations passed the common-support check. Observations removed
during earlier data preparation are not included in these tables.

When you also estimate treatment effects, ``iate_data_df`` contains the
remaining prediction observations together with their estimated effects
and potential outcomes. This dataframe is available only when IATE
estimation and dataframe output are enabled.

See :py:meth:`~mcf_main.ModifiedCausalForest.train` and
:py:meth:`~mcf_main.ModifiedCausalForest.predict` for the returned table names.
Examples
------------------

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()


    my_mcf = ModifiedCausalForest(
            var_y_name="outcome",
            var_d_name="treat",
            var_x_name_ord=["x_cont0", "x_cont1"],
            # Turn common support adjustments off:
            cs_type=0)


    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        # Use automatic common support adjustments
        cs_type=1,
        # Offset the cut-off limits: Multiply the upper cut-off by 1.1 and the
        # lower cut-off by 0.9:
        cs_adjust_limits=0.1,
        # Raise an exception if more than 25% of the observations are dropped:
        cs_max_del_train=0.25)


    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        # Use common support adjustments and specify cut-off probabilities manually:
        cs_type=2,
        cs_min_p=0.05)
