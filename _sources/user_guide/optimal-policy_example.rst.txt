Learning an optimal policy
==========================

Different types of policy rules
-------------------------------

The :py:class:`~optpolicy_main.OptimalPolicy` class offers three methods:

- ``policy_tree`` learns an interpretable rule from decision variables.
- ``best_policy_score`` assigns treatments from supplied scores, accounting for
  costs and requested restrictions where applicable.
- ``bps_classifier`` learns a classifier from best-score allocations.

The policy-tree approach follows `Zhou, Athey, and Wager (2022)
<https://doi.org/10.1287/opre.2022.2271>`_, with implementation differences in
scores, constraints and multi-valued features.

The following sections demonstrate how to implement these methods for policy learning. The :doc:`Algorithm reference <../algorithm_reference/optimal-policy_algorithm>` provides more details on the computational algorithms.

Policy Trees
------------

Generating data
~~~~~~~~~~~~~~~

The examples below use simulated true potential outcomes (``y_pot0`` etc.) as
policy scores to demonstrate the policy methods. For learning from estimated
MCF potential outcomes, follow :doc:`../getting_started` or the combined public
example. The code blocks in each example are intended to be read in sequence.

.. code-block:: python

    import os
    from mcf.example_data import example_data
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.reporting import McfOptPolReport

    # Generate the data.
    training_df, prediction_df, name_dict=example_data(
        obs_y_d_x_iate=1000,
        obs_x_iate=1000,
        no_treatments=3
        )

Estimating an optimal policy tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we initialize an instance by calling the :py:class:`~optpolicy_main.OptimalPolicy` class to estimate an optimal policy tree with maximum depth two (at most four leaves).

.. code-block:: python

    # Initializing a class instance.
    myoptp=OptimalPolicy(
        gen_method='policy_tree',
        var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
        var_x_name_ord=('x_cont0', 'x_ord0'),
        pt_depth_tree_1=2,
        pt_depth_tree_2=0,
        gen_outpath=os.getcwd() + '/out'
        )

Estimating sequentially optimal policy trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, we can create an instance to estimate sequentially optimal policy trees. This may lead to a reduction in runtime by, for instance, creating two sequentially optimal trees of depth 2+1 instead of a single tree of depth 3. The two-stage search is sequential and need not match the optimum of a single search at the combined depth.

.. code-block:: python

    # Initializing a class instance.
    myoptp=OptimalPolicy(
        gen_method='policy_tree',
        var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
        var_x_name_ord=('x_cont0', 'x_ord0'),
        pt_depth_tree_1=2,
        pt_depth_tree_2=1,
        gen_outpath=os.getcwd() + '/out'
        )

Estimating a constrained optimal policy tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following alternative configuration uses treatment-share targets. Policy trees
use calibrated costs to encourage these shares, but the fitted tree may exceed
them. Inspect achieved shares in both training and new data.

.. code-block:: python

    # Initializing a class instance.
    myoptp=OptimalPolicy(
        gen_method='policy_tree',
        var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
        var_x_name_ord=('x_cont0', 'x_ord0'),
        pt_depth_tree_1=2,
        pt_depth_tree_2=0,
        other_max_shares=(0.2, 0.8, 0),
        gen_outpath=os.getcwd() + '/out'
        )

Solve, allocate, and evaluate methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After initializing a class instance, we use it to solve for an optimal allocation rule, to allocate units to treatment states, and to evaluate the allocations with potential outcome data.

.. code-block:: python

    # Solve, allocate, and evaluate methods.
    solve_dict, train_df=myoptp.solve(
        training_df,
        data_title='training'
        )

    results_eva_train=myoptp.evaluate(
        solve_dict['allocation_df'],
        train_df,
        data_title='training'
        )

    allocation_result=myoptp.allocate(
        prediction_df,
        data_title='prediction'
        )

    results_eva_pred=myoptp.evaluate(
        allocation_result['allocation_df'],
        prediction_df,
        data_title='prediction'
        )


``solve()`` returns a dictionary and retained training data; cleaning and
feature-selection holdouts can remove rows. ``allocate()`` returns a dictionary.
``evaluate()`` requires allocation and evaluation rows in exactly the same order
and pairs them by position, not by identifier. Resetting an index does not repair
a mismatch.

Allocation codes 0, 1, ... follow ``var_polscore_name`` order. Observed treatment
codes used for evaluation must follow that same order. For tree and classifier
assignment, new data need decision variables; welfare evaluation also needs
scores. Best-score allocation needs scores for the new observations, not just
features.

For the constrained example, inspect the allocation shares before calling
``evaluate()``, which may add reference-allocation columns:

.. code-block:: python

    for rule in solve_dict['allocation_df']:
        print(rule, solve_dict['allocation_df'][rule].value_counts(normalize=True))

Inference for different allocations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method ``predict_different_allocations`` enables the prediction of average potential outcomes and variances for different allocations. See :py:meth:`~mcf_main.ModifiedCausalForest.predict_different_allocations` for its data requirements and supported settings.

The combined example in :doc:`../user_guide` illustrates this method. Train the forest first and provide evaluation rows and allocations in matching positional order.


Estimate a policy tree under uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method ``estrisk_adjust`` allows accounting for estimation error in the policy scores. Generally, the idea implemented follows the paper
*Policy Learning With Confidence* by Chernozhukov, Lee, Rosen, and Sun `(arXiv, 2025) <https://arxiv.org/abs/2502.04501>`_.

The following fragment assumes ``score_names`` and ``score_se_names`` identify
matching treatment columns in your data. Use a fresh instance for each
specification and pass the returned adjusted dataframe to policy learning:

.. code-block:: python

    risk_policy = OptimalPolicy(
        gen_method='policy_tree',
        var_polscore_name=score_names,
        var_polscore_se_name=score_se_names,
        var_x_name_ord=['x_cont0', 'x_ord0'],
        pt_depth_tree_1=2, pt_depth_tree_2=0,
        estrisk_value=1
    )
    risk_results = risk_policy.estrisk_adjust(training_df)
    risk_fit, risk_train_df = risk_policy.solve(risk_results['data_estrisk_df'])
    risk_evaluation = risk_policy.evaluate(risk_fit['allocation_df'], risk_train_df)

Evaluation uses current adjusted scores when all are supplied. Otherwise it may
fall back to an available descriptive-score block. Identify the score block when
comparing welfare across adjustments; see the ``evaluate()`` API.

Estimate a fair optimal policy tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another alternative is to estimate a fair policy tree. This uses the ``solvefair``
method to build the decision rule. ``solvefair()`` returns a dictionary only.
The generated data in this example are complete; evaluation must still use
exactly the allocated rows. Supply protected variables for fairness diagnostics.
Adjustment targets and achieved fairness are distinct; see
:doc:`../algorithm_reference/fairscores`.

.. code-block:: python

    # Initializing a class instance.
    myoptp_fair=OptimalPolicy(
        gen_method='policy_tree',
        var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
        var_protected_name_ord=('x_ord0',),
        var_x_name_ord=('x_cont0',),
        pt_depth_tree_1=2,
        pt_depth_tree_2=0,
        gen_outpath=os.getcwd() + '/out'
        )

    # Solve, allocate, and evaluate methods.
    alloc_train_fair_dict=myoptp_fair.solvefair(
        training_df.copy(),
        data_title='training'
        )

    results_eva_train=myoptp_fair.evaluate(
        alloc_train_fair_dict['allocation_df'],
        training_df.copy(),
        data_title='training'
        )

    alloc_pred_fair_dict=myoptp_fair.allocate(
        prediction_df.copy(),
        data_title='prediction'
        )

    results_eva_pred=myoptp_fair.evaluate(
        alloc_pred_fair_dict['allocation_df'],
        prediction_df.copy(),
        data_title='prediction'
        )


The method ``winners_losers`` compares winners and losers between two allocations. It uses the k-means algorithm to cluster individuals who exhibit similar gains and losses across the two user-provided allocations. Each resulting group is described by the policy scores as well as the decision, protected, and materially relevant variables.


Reporting
~~~~~~~~~

Finally, the code creates a PDF report. Please note that the program saves by default information like summary statistics and leaf information for the policy tree in a folder in the current working directory.

.. code-block:: python

    # Generate the PDF report.
    my_report=McfOptPolReport(
        optpol=myoptp,
        outputpath=myoptp.gen_cfg.outpath,
        outputfile='Report_OptP_' + 'policy_tree'
        )
    my_report.report()

Best Policy Scores
------------------

The following code demonstrates how to obtain a policy rule based on the best-score method.

Estimating a policy rule using the best-score method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    from mcf.example_data import example_data
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.reporting import McfOptPolReport

.. code-block:: python

    # Creating data.
    training_df, prediction_df, name_dict=example_data(
        obs_y_d_x_iate=1000,
        obs_x_iate=1000,
        no_treatments=3
        )

.. code-block:: python

    # Initializing a class instance.
    myoptp=OptimalPolicy(
        gen_method='best_policy_score',
        var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
        var_x_name_ord=('x_cont0', 'x_ord0'),
        gen_outpath=os.getcwd() + '/out'
        )

.. code-block:: python

    # Solve, allocate, and evaluate methods.
    solve_dict, train_df=myoptp.solve(
        training_df,
        data_title='training'
        )

    results_eva_train=myoptp.evaluate(
        solve_dict['allocation_df'],
        train_df,
        data_title='training'
        )

    allocation_result=myoptp.allocate(
        prediction_df,
        data_title='prediction'
        )

    results_eva_pred=myoptp.evaluate(
        allocation_result['allocation_df'],
        prediction_df,
        data_title='prediction'
        )

.. code-block:: python

    # Generate a PDF report.
    my_report=McfOptPolReport(
        optpol=myoptp,
        outputpath=myoptp.gen_cfg.outpath,
        outputfile='Report_OptP_' + 'best_policy_score'
        )
    my_report.report()

With restrictions, best-score results can contain both restricted rules and the
unrestricted ``bb`` allocation. Select the intended column. A classifier learns
from its training labels; its predictions need not satisfy their treatment shares.
See :doc:`../algorithm_reference/optimal-policy_algorithm` for capacities and costs.
