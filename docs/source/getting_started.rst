.. _getting-started:

Getting started
===============

This guide shows how to estimate heterogeneous treatment effects, learn a policy
tree from those estimates, and inspect the results.

Example data
------------

The :py:func:`~example_data.example_data` function generates synthetic data and
returns variable names in ``name_dict``. The sample sizes, number of features,
number of treatments and type of heterogeneity can be chosen through its arguments.

.. code-block:: python

    from pathlib import Path

    from mcf.example_data import example_data
    from mcf.mcf_main import ModifiedCausalForest
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.reporting import McfOptPolReport

    train_mcf_df, prediction_df, name_dict = example_data(
        obs_y_d_x_iate=3000, obs_x_iate=3000
    )
    prediction_df = prediction_df.sample(frac=1, random_state=42)
    split = len(prediction_df) // 2
    pred_mcf_train_pt_df = prediction_df.iloc[:split].copy()
    evaluate_pt_df = prediction_df.iloc[split:].copy()
    out = Path.cwd() / 'mcf_tutorial'

The forest, policy learning and final evaluation use separate samples.

Estimating heterogeneous treatment effects
------------------------------------------

Specify the outcome, treatment and ordered or unordered features when creating a
:py:class:`~mcf_main.ModifiedCausalForest`. Binary features belong in the ordered
list. Use numeric, nonmissing input data; see :doc:`user_guide/data_cleaning`.

.. code-block:: python

    mcf = ModifiedCausalForest(
        var_y_name='outcome', var_d_name='treat',
        var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
        var_x_name_unord=['x_unord0'],
        gen_outpath=out / 'effects', _int_show_plots=False
    )
    training_results = mcf.train(train_mcf_df)
    results = mcf.predict(pred_mcf_train_pt_df)

``train()`` returns the processed construction and filling samples as ``tree_df``
and ``fill_y_df``, along with common-support information and ``path_output``.

Frequently used parameters
--------------------------

- ``cf_boot`` controls the number of trees.
- ``p_atet`` requests average effects by observed treatment status.
- ``var_z_name_cont``, ``var_z_name_ord`` and ``var_z_name_unord`` identify
  heterogeneity variables for group effects.
- ``p_gatet`` requests group effects by treatment status.
- ``var_y_tree_name`` selects the outcome used for constructing the forest.
- ``var_id_name`` identifies individual observations.

See :doc:`user_guide/estimation` and the :py:class:`API <mcf_main.ModifiedCausalForest>`
for the available settings.

Accessing results
-----------------

``predict()`` returns one dictionary. Unrequested treatment effect families and standard
errors are ``None``, and a key's presence does not mean that an estimate was computed.
Use the returned labels to identify treatment comparisons:

.. code-block:: python

    print(results['ate_effect_list'])
    print(results['ate'])
    print(results['ate_se'])

For example, for treatments 0, 1 and 2, the comparisons are 1 versus 0, 2 versus 0 and 2 versus 1.

The returned dataframe contains the predicted potential outcomes and the estimated IATEs.

.. code-block:: python

    data_train_pt = results['iate_data_df'].copy()
    estimate_names = results['iate_names_dic'][0]
    iate_df = data_train_pt[estimate_names['names_iate']]
    score_names = estimate_names['names_y_pot_uncenter']
    potential_outcomes_df = data_train_pt[score_names]

Here, uncentered potential outcomes are available because the example uses the
default local centering and one outcome. ``iate_names_dic`` is a tuple: element 0
covers all computed comparisons; element 1 restricts effect names to comparisons
with the first treatment.

The ``iate`` array stored in the results dictionary has shape ``(N, Y, C, 2)`` – retained observations, outcomes,
treatment comparisons, and effect type. Its last axis contains IATEs at index 0
and IATE-minus-ATE at index 1; the latter is ``NaN`` when not requested.
``iate_data_df`` can be ``None`` when ``p_iate`` is disabled or dataframe return
is disabled through the output settings. See ``_int_with_output`` and
``_int_return_iate_sp`` in the API.


Post-estimation
---------------

Use :py:meth:`~mcf_main.ModifiedCausalForest.analyse` for descriptive analysis
of estimated IATEs:

.. code-block:: python

    diagnostics = mcf.analyse(results)

This requires the keyword argument ``post_est_stats``, enabled output, and an IATE dataframe. With
k-means clustering enabled it returns augmented results; otherwise it returns
``None`` while other enabled analyses still run. These diagnostics describe
estimated heterogeneity; see :doc:`user_guide/post_estimation_diagnostics`.

Learning an optimal policy rule
-------------------------------

The policy scores below are the estimated potential outcomes from ``predict()``.

.. code-block:: python

    policy = OptimalPolicy(
        gen_method='policy_tree', var_d_name='treat',
        var_polscore_name=score_names,
        var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
        pt_depth_tree_1=2, pt_depth_tree_2=0,
        gen_outpath=out / 'policy', _int_show_plots=False
    )
    fit, data_train_pt = policy.solve(data_train_pt)
    train_evaluation = policy.evaluate(fit['allocation_df'], data_train_pt)

A depth-two tree has at most four leaves. Setting the second depth to zero
turns off the second tree stage. Choose the complexity according to the application;
see :doc:`user_guide/optimal-policy_example`.

``solve()`` returns a results dictionary and the retained training dataframe.
Evaluate its allocations on that returned dataframe: evaluation matches rows
by position, not by index or identifier. Allocation codes follow the order of
``var_polscore_name``. Observed treatment codes must use the same order.

Apply the rule to the separate evaluation sample:

.. code-block:: python

    test_results = mcf.predict(evaluate_pt_df)
    oos_df = test_results['iate_data_df'].copy()
    allocation = policy.allocate(oos_df)
    evaluation = policy.evaluate(allocation['allocation_df'], oos_df)


Accessing and customizing output location
------------------------------------------

Use ``gen_outpath`` to choose each object's output directory. The resolved **mcf**
path is available as ``mcf.gen_cfg.outpath`` or ``results['path_output']``.
Enabled outputs include detailed and condensed text files, result tables and
figures. A PDF requires an explicit report call and has its own output directory:

.. code-block:: python

    report = McfOptPolReport(
        mcf=mcf, optpol=policy, outputpath=out, outputfile='Tutorial'
    )
    pdf_path = report.report()
    print(pdf_path)

Without an explicit ``outputpath``, the PDF is saved in the ``output``
subdirectory of the current working directory. The report summarizes
selected estimation, analysis and policy-evaluation results saved by the
objects passed to it.

Next steps
----------

- The :doc:`user_guide` explains additional features and links to example scripts,
  including ``mcf_optpol_combined.py`` for cross-fitting.
- The :doc:`python_api` documents arguments and return values.
- The :doc:`algorithm_reference` explains the methods.
