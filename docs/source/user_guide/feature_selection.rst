Feature selection
=================

Removing irrelevant features can improve prediction and reduce computation.
Set ``fs_yes`` to enable sequential feature selection. At each step, the
procedure removes at most one eligible original feature, refits the auxiliary
models and reassesses importance. It stops when none qualifies or at most two
original features remain.

Modified Causal Forest
----------------------

Auxiliary forests predict the selected tree outcome and treatment. Importance is
the loss in held-out prediction performance after permuting a feature. Regression
uses R-squared; classification uses probability MSE by default.

``fs_rel_vi_threshold_y`` and ``fs_rel_vi_threshold_d`` control removal thresholds.
The default ``fs_rel_vi_keep_if='y_or_d_relevant'`` keeps a feature relevant to
either target, so both must be unimportant for removal. ``'y_and_d_relevant'``
can remove a feature when either criterion is weak; ``'y_relevant'`` uses only
the outcome criterion. ``fs_mse_for_classifier`` selects probability MSE;
explicit ``None`` selects accuracy.

Always-in, remain, grouping and BGATE-balancing variables are protected by **mcf**.
Use ``var_x_name_remain_ord`` or ``var_x_name_remain_unord`` to protect required
confounders. Predictive relevance is not a test of causal identification.

Optimal Policy
--------------

Selection uses each policy-score difference relative to the first score. A
feature qualifies for removal only when its importance is at or below
``fs_rel_vi_threshold`` for every difference. Continuous targets use held-out
R-squared; targets with few distinct values use classifier accuracy.

Selection applies to tree and classifier learning. The best-score method
ignores it; ``solvefair()`` and the versioned policy-tree path do not support it.
The protected-variable rules of the **mcf** should not be assumed to apply to policy learning.

Parameter overview
------------------

``fs_other_sample`` reserves observations for feature selection and excludes
them from subsequent fitting. ``fs_other_sample_share`` controls the share.
For policy learning, evaluate training allocations on the dataframe returned by
``solve()``. The class-specific APIs describe thresholds and defaults:
:py:class:`ModifiedCausalForest <mcf_main.ModifiedCausalForest>` and
:py:class:`OptimalPolicy <optpolicy_main.OptimalPolicy>`.

Example
--------

.. code-block:: python

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest

    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()

    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        # Parameters for feature selection:
        fs_yes=True,
        fs_other_sample=True,
        fs_other_sample_share=0.1,
        fs_rel_vi_keep_if='y_or_d_relevant'
    )


    my_mcf.train(training_df)
    print(my_mcf.var_cfg.x_name_ord)
    print(my_mcf.var_cfg.x_name_unord)

The output records the sequential selection and retained features.
