Learning an optimal policy
==========================

Different types of policy rules
-------------------------------

The Modified Causal Forest refines treatment assignment mechanisms by defining the objective of an assignment rule. Its :py:class:`~optpolicy_main.OptimalPolicy` class offers two different methods to compute assignment algorithms, namely the policy tree and the best-scores methods. The policy tree method is based on the work of `Zhou, Athey, and Wager (2022) <https://doi.org/10.1287/opre.2022.2271>`_ , modifying their approach with respect to policy score calculation, definition of constraints, and handling of multi-valued features. In contrast, the best-score method simply assigns units to the treatment with the highest individual potential outcome. The **mcf** algorithm compares the two methods with random treatment allocations to demonstrate the increase in reward.

The following sections demonstrate how to implement these methods for policy learning. The :doc:`Algorithm reference <../algorithm_reference/optimal-policy_algorithm>` provides more details on the computational algorithms.

Policy Trees
------------

Generating data
~~~~~~~~~~~~~~~

The code below creates artificial example data for training and prediction. 

.. code-block:: python

    import os
    from mcf.example_data_functions import example_data
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.reporting import McfOptPolReport

    # Generate the data.
    training_df, prediction_df, name_dict = example_data(
        obs_y_d_x_iate=1000, obs_x_iate=1000, no_treatments=3)

Estimating an optimal policy tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we initialize an instance by calling the :py:class:`~optpolicy_main.OptimalPolicy` class to estimate an optimal policy tree of depth two.

.. code-block:: python

    # Initializing a class instance.
    myoptp = OptimalPolicy(gen_method='policy_tree',
                           var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
                           var_x_name_ord=('x_cont0', 'x_ord0'),
                           pt_depth_tree_1=2,
                           pt_depth_tree_2=0,
                           gen_outpath=os.getcwd() + '/out')

Estimating sequentially optimal policy trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, we can create an instance to estimate sequentially optimal policy trees. This may lead to a reduction in runtime by, for instance, creating two sequentially optimal trees of depth 2+1 instead of a single optimal tree of depth 3. 

.. code-block:: python

    # Initializing a class instance.
    myoptp = OptimalPolicy(gen_method='policy_tree',
                           var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
                           var_x_name_ord=('x_cont0', 'x_ord0'),
                           pt_depth_tree_1=2,
                           pt_depth_tree_2=1,
                           gen_outpath=os.getcwd() + '/out')

Estimating a constrained optimal policy tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code block creates a constrained optimal policy tree, considering a constraint with respect to the treatment shares.

.. code-block:: python

    # Initializing a class instance.
    myoptp = OptimalPolicy(gen_method='policy_tree',
                           var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
                           var_x_name_ord=('x_cont0', 'x_ord0'),
                           pt_depth_tree_1=2,
                           pt_depth_tree_2=0,
                           other_max_shares=(0.2, 0.8, 0),
                           gen_outpath=os.getcwd() + '/out')

Solve, allocate, and evaluate methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After initializing a class instance, we use it to solve for an optimal allocation rule, to allocate units to treatment states, and to evaluate the allocations with potential outcome data. 

.. code-block:: python

    # Solve, allocate, and evaluate methods.
    alloc_train_df = myoptp.solve(
        training_df, 
        data_title='training')
    
    results_eva_train = myoptp.evaluate(
        alloc_train_df['allocation_df'] ,
        training_df,
        data_title='training'
        )
    
    alloc_pred_df = myoptp.allocate(
        prediction_df,
        data_title='prediction'
        )
    
    results_eva_pred = myoptp.evaluate(
        alloc_pred_df['allocation_df'], 
        prediction_df,
        data_title='prediction')


Inference for different allocations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method ``predict_different_allocations`` enables the prediction of average potential outcomes and variances for different allocations. Details of this method are provided in the working paper by Busshoff and Lechner (mimeo, 2025).

Example scripts in the User Guide illustrate the Python code for this method. Note that the ``train`` method of the ``ModifiedCausalForest`` class must be executed beforehand.


Estimate a policy tree under uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The method ``estrisk_adjust`` allows accounting for estimation error in the policy scores. Generally, the idea implemented follows the paper *Policy Learning With Confidence* by Chernozhukov, Lee, Rosen, and Sun (arXiv, 2025). However, since several approximations are used in the algorithm, the method will not have the direct confidence-level-related interpretations suggested by these authors. To use ``estrisk_adjust``, it is necessary to provide the standard errors of the policy scores. ``estrisk_adjust`` adjusts the policy scores for estimation error by subtracting multiples of the standard errors from these scores. Once the scores are adjusted, standard procedures can be used to obtain optimal decisions. An example script demonstrating the use of this method, including the keywords for providing standard errors of the scores and their multiples, is provided in the User Guide.


Estimate a fair optimal policy tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another alternative is to estimate a fair policy tree. This uses the ``solvefair`` 
method to build the decision rule.

.. code-block:: python

# Initializing a class instance.
myoptp_fair = OptimalPolicy(
    gen_method='policy_tree',
    var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
    var_protected_name_ord=('x_ord0'),
    var_x_name_ord=('x_cont0'),
    pt_depth_tree_1=2,
    pt_depth_tree_2=0,
    gen_outpath=os.getcwd() + '/out'
)

# Solve, allocate, and evaluate methods.
alloc_train_fair_dict = myoptp_fair.solvefair(
    training_df.copy(),
    data_title='training'
    )

results_eva_train = myoptp_fair.evaluate(
    alloc_train_fair_dict['allocation_df'],
    training_df.copy(),
    data_title='training'
    )

alloc_pred_fair_dict = myoptp_fair.allocate(
    prediction_df.copy(),
    data_title='prediction'
    )

results_eva_pred = myoptp_fair.evaluate(
    alloc_pred_fair_dict['allocation_df'],
    prediction_df.copy(),
    data_title='prediction'
    )

The method ``winners_losers`` compares winners and losers between two allocations. It uses the k-means algorithm to cluster individuals who exhibit similar gains and losses across the two user-provided allocations. Each resulting group is described by the policy scores as well as the decision, protected, and materially relevant variables.


Reporting
~~~~~~~~~

Finally, the code creates a PDF report. Please note that the program saves by default information like summary statistics and leaf information for the policy tree in a folder in the current working directory. 

.. code-block:: python

    # Creating a PDF report.
    my_report = McfOptPolReport(
        optpol=myoptp, outputfile='Report_OptP_' + 'policy_tree')
    my_report.report()

Best Policy Scores
------------------

The following code demonstrates how to obtain a policy rule based on the best-score method. 
                       
Estimating a policy rule using the best-score method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    import os
    from mcf.example_data_functions import example_data
    from mcf.optpolicy_main import OptimalPolicy
    from mcf.reporting import McfOptPolReport

.. code-block:: python

    # Creating data.
      training_df, prediction_df, name_dict = example_data(
          obs_y_d_x_iate=1000, obs_x_iate=1000, no_treatments=3)

.. code-block:: python

    # Initializing a class instance.
    myoptp = OptimalPolicy(gen_method='best_policy_score',
                          var_polscore_name=('y_pot0', 'y_pot1', 'y_pot2'),
                          var_x_name_ord=('x_cont0', 'x_ord0'),
                          gen_outpath=os.getcwd() + '/out')

.. code-block:: python

    # Solve, allocate, and evaluate methods.
    alloc_train_df, _, _ = myoptp.solve(training_df, data_title='training')
    results_eva_train, _ = myoptp.evaluate(alloc_train_df, training_df,
                                           data_title='training')
    alloc_pred_df, _ = myoptp.allocate(prediction_df,
                                       data_title='prediction')
    results_eva_pred, _ = myoptp.evaluate(alloc_pred_df, prediction_df,
                                      data_title='prediction')

.. code-block:: python

    # Creating a PDF report.
    my_report = McfOptPolReport(
        optpol=myoptp, outputfile='Report_OptP_' + 'best_policy_score')
    my_report.report()
