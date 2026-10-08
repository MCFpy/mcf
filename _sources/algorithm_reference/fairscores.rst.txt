Fairness Extensions in Optimal Policy
===========================================

The :py:class:`~optpolicy_main.OptimalPolicy` class in **mcf** includes experimental capabilities for fairness adjustments, available through its ``solvefair`` method.
These adjustments aim to improve fairness in policy allocations with respect to specified protected variables, following `Bearth, Lechner, Mareckova, and Muny (2025), Fairness-Aware and Interpretable Policy Learning <https://arxiv.org/abs/2509.12119v1>`_.

The approach preprocesses scores, decision variables, or both. Joint independence
of all decision inputs from protected features would imply independence of a rule
formed from those inputs. Adjusting each coordinate separately does not generally
establish joint independence. Materially relevant variables permit the specified
conditional distinctions and enter the adjustment.

The following CDF illustration applies to a continuous scalar variable under
appropriate distributional conditions. Discrete values and estimated empirical
CDFs do not provide an unconditional exact-independence guarantee.

To transform a variable :math:`X` with respect to a protected feature :math:`S`, the method proceeds in two steps:

1. **Conditioning:** Compute the cumulative distribution function (CDF) of
   :math:`X` given :math:`S`,

   .. math::

      u = F_{X \mid S}(X \mid S)

   so that :math:`u` is uniformly distributed within each level of :math:`S`.

2. **Marginal Mapping:** Apply the inverse marginal CDF of :math:`X`,

   .. math::

      \tilde{X} = F_X^{-1}(u)

   to restore the original marginal distribution.

Under those conditions, the transformed variable preserves its marginal
distribution and removes its dependence on the protected feature.

The adjustment can be applied to the decision variables, the policy scores, or both.
Note that the fairness adjustment implemented here may slightly differ from the
`R implementation <https://github.com/fmuny/fairpolicytree>`_
used in Bearth, Lechner, Mareckova, and Muny (2025).

Implementation
--------------

``fair_adjust_target`` selects ``'scores'``, ``'xvariables'`` or
``'scores_xvariables'``. Decision-variable-only adjustment is not supported for
``best_policy_score``. ``fair_type='Mean'`` and ``'MeanVar'`` adjust moments;
``'Quantiled'`` performs quantile adjustment. Discrete decision variables always
use quantile adjustment, as do all decision variables in policy trees.

Use ``solvefair()`` to learn a fairness-adjusted policy rule. When calling
``allocate()``, any required adjusted decision variables are created
automatically if missing. Existing adjusted variables are reused; set
``fair_adjust_decision_vars=True`` to recalculate them.
For ``best_policy_score``, supply either the adjusted policy scores or
the original data needed to calculate those adjustments.

Include the protected variables in the evaluation data to assess fairness.
``evaluate()`` uses the policy scores currently selected by the policy
object, which may be fairness- or risk-adjusted. If these scores are
unavailable, it uses a complete set of scores from
``var_polscore_desc_name``, if available. These may be the original,
unadjusted scores.

Example
-------
To use the fairness adjustments, configure the :py:class:`~optpolicy_main.OptimalPolicy` class with the appropriate parameters and call the ``solvefair`` instead of the ``solve`` method to build the decision rule.

.. code-block:: python

    import os
    from mcf.example_data import example_data
    from mcf.optpolicy_main import OptimalPolicy

    # Generate data with simulated potential outcomes as demonstration scores
    training_df, prediction_df, name_dict = example_data(
        obs_y_d_x_iate=1000,
        obs_x_iate=1000,
        no_features=5,
        no_treatments=2,
        seed=12345,
        type_of_heterogeneity='WagerAthey',
        descr_stats=True
    )

    # Initializing a class instance.
    myoptp = OptimalPolicy(
        gen_method='policy_tree',
        var_polscore_name=('y_pot0', 'y_pot1'),
        var_protected_name_ord=('x_ord0',),
        var_x_name_ord=('x_cont0',),
        pt_depth_tree_1=2,
        pt_depth_tree_2=0,
        gen_outpath=os.getcwd() + '/out'
    )

    # Solve: get dictionary with allocation and other infos
    alloc_train_dict = myoptp.solvefair(
        training_df.copy(),
        data_title='training'
    )

    # Extract only the allocation_df key from the dictionary before passing it to evaluate().
    alloc_train_df_only = alloc_train_dict['allocation_df']

    # Pass DataFrame to evaluate
    results_eva_train = myoptp.evaluate(
        alloc_train_df_only,
        training_df,
        data_title='training'
    )

    # Allocate for prediction
    alloc_pred_dict = myoptp.allocate(
        prediction_df.copy(),
        data_title='prediction'
    )
    alloc_pred_df_only = alloc_pred_dict['allocation_df']

    # Evaluate predictions
    results_eva_pred = myoptp.evaluate(
        alloc_pred_df_only,
        prediction_df,
        data_title='prediction'
    )


Note
----
These features are experimental and may require further testing and validation. For more details or additional parameters, please consult the :py:class:`API <optpolicy_main.OptimalPolicy>` documentation.
