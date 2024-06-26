.. _getting-started:

Getting started
=======================

This guide will walk you through using the **mcf** package to

- estimate heterogeneous treatment effects using the Modified Causal Forest
- learn an optimal policy rule based on a Policy Tree


Example data
^^^^^^^^^^^^^^^^

First, we'll use the example_data function which generates synthetic datasets for training and prediction. It creates training (train_df) and prediction (pred_df) DataFrames with a specified number of observations, features, and treatments, allowing for various heterogeneity types ('linear', 'nonlinear', 'quadratic', 'WagerAthey'). 
By default, it produces 1000 observations for both training and prediction, with 20 features and 3 treatments. The function also returns name_dict, a dictionary containing the names of variable groups. For more details, visit the :doc:`python_api`.

.. code-block:: python

    from mcf.example_data_functions import example_data
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
Estimating heterogeneous treatment effects
------------------------------------------

To estimate a Modified Causal Forest, we use the :py:class:`~mcf_functions.ModifiedCausalForest` class of the **mcf** package. To create an instance of the :py:class:`~mcf_functions.ModifiedCausalForest` class, we need to specify the name of

- at least one outcome variable through the ``var_y_name`` parameter
- the treatment variable through the ``var_d_name`` parameter
- ordered features through ``var_x_name_ord`` and/or unordered features through ``var_x_name_unord``

as follows:

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_functions import ModifiedCausalForest
    from mcf.optpolicy_functions import OptimalPolicy
    from mcf.reporting import McfOptPolReport
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    # Create an instance of the Modified Causal Forest model
    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",  # Outcome variable
        var_d_name="treat",    # Treatment variable
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],  # Ordered covariates
        var_x_name_unord=["x_unord0"],  # Unordered covariate
        _int_show_plots=False  # Disable plots for faster performance
    )


Accessing and customizing output location
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **mcf** package generates a number of standard outputs for your convenience. After initializing a Modified Causal Forest, the package will create an output folder where these results will be stored. You can find the location of this folder by accessing the `"outpath"` entry of the `gen_dict` attribute of your Modified Causal Forest:

.. code-block:: python

    my_mcf.gen_dict["outpath"]

You can also specify the location of this folder manually using the ``gen_outpath`` parameter of the class :py:class:`~mcf_functions.ModifiedCausalForest`.

Below you find a selected list of optional parameters that are often used to initialize a Modified Causal Forest. For a more detailed description of these parameters, please refer to the documentation of :py:class:`~mcf_functions.ModifiedCausalForest`.

.. dropdown:: Commonly used optional parameters

    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | Parameter                        | Description                                                                                                       |
    +==================================+===================================================================================================================+
    | ``cf_boot``                      | Number of Causal Trees. Default: 1000.                                                                            |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``p_atet``                       | If True, :math:`\textrm{ATE's}` are also computed by treatment status (:math:`\textrm{ATET's}`). Default: False.  |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_list``              | Ordered feature(s) with many values used for :math:`\textrm{GATE}` estimation.                                    |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_ord``               | Ordered feature(s) with few values used for :math:`\textrm{GATE}` estimation.                                     |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_unord``             | Unordered feature(s) used for :math:`\textrm{GATE}` estimation.                                                   |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``p_gatet``                      | If True, :math:`\textrm{GATE's}` are also computed by treatment status (:math:`\textrm{GATET's}`). Default: False.|
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_x_name_always_in_ord``     | Ordered feature(s) always used in splitting decision.                                                             |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_x_name_always_in_unord``   | Unordered feature(s) always used in splitting decision.                                                           |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_y_tree_name``              | Outcome used to build trees. If not specified, the first outcome in ``y_name`` is selected for building trees.    |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+
    | ``var_id_name``                  | Individual identifier.                                                                                            |
    +----------------------------------+-------------------------------------------------------------------------------------------------------------------+


Training a Modified Causal Forest
-----------------------------------

Next we will train the Modified Causal Forest on the *train_mcf_df* data using the :py:meth:`~mcf_functions.ModifiedCausalForest.train` method:

.. code-block:: python

    my_mcf.train(training_df)

Now we are ready to estimate heterogeneous treatment effects on the *pred_mcf_train_pt_df* data using the :py:meth:`~mcf_functions.ModifiedCausalForest.predict` method.

.. code-block:: python

    results, _ = my_mcf.predict(prediction_df)


Accessing results
~~~~~~~~~~~~~~~~~

The easiest way to get an overview of your results is to read the PDF-report that can be generated using the class :py:class:`~reporting.McfOptPolReport`:

.. code-block:: python

    mcf_report = McfOptPolReport(mcf=my_mcf, outputfile='Modified-Causal-Forest_Report')
    mcf_report.report()

Next, we describe ways to access the results programmatically:

The :py:meth:`~mcf_functions.ModifiedCausalForest.predict` method returns a dictionary containing the estimation results. To gain an overview, have a look at the keys of the dictionary:

.. code-block:: python

    print(results.keys())

By default the average treatment effects (:math:`\textrm{ATE's}`) as well as the individualized average treatment effects (:math:`\textrm{IATE's}`) are estimated. If these terms do not sound familiar, click :doc:`here <user_guide/estimation>` to learn more about the different kinds of heterogeneous treatment effects.

In the multiple treatment setting there is more than one average treatment effect to consider. The following entry of the results dictionary lists the estimated treatment contrasts:

.. code-block:: python

    results["ate effect_list"]

An entry *[1, 0]* for instance specifies the treatment contrast between treatment level 1 and treatment level 0. These contrasts are aligned with the estimated :math:`\textrm{ATE's}` and their standard errors, which you can access using:

.. code-block:: python

    results["ate"]
    results["ate_se"]

The estimated :math:`\textrm{IATE's}`, together with the predicted potential outcomes, are stored as a Pandas DataFrame in the following entry of the results dictionary:

.. code-block:: python

    results["iate_data_df"]

Please refer to the documentation of the :py:meth:`~mcf_functions.ModifiedCausalForest.predict` method for a more detailed description of the contents of the results dictionary.


Post-estimation
-----------------

You can use the :py:meth:`~mcf_functions.ModifiedCausalForest.analyse` method to investigate a number of post-estimation plots. These plots are also exported to the previously created output folder:

.. code-block:: python

    my_mcf.analyse(results)

    
Learning an optimal policy rule
-------------------------------

Let's explore how to learn an optimal policy rule using the :py:class:`~optpolicy_functions.OptimalPolicy` class of the **mcf** package. To get started we need a Pandas DataFrame that holds the estimated potential outcomes (also called policy scores), the treatment variable and the features on which we want to base the decision tree.

As you may recall, we estimated the potential outcomes in the previous section. They are stored as columns in the *"iate_data_df"* entry of the results dictionary:

.. code-block:: python

    print(results["iate_data_df"].head())

The column names are explained in the `iate_names_dic` entry of the results dictionary. The uncentered potential outcomes are stored in columns with the suffix *_un_lc_pot*.

.. code-block:: python

    print(results["iate_names_dic"])

Now that we understand this, we are ready to build an Optimal Policy Tree. To do so, we need to create an instance of class :py:class:`~optpolicy_functions.OptimalPolicy` where we set the ``gen_method`` parameter to "policy tree" and provide the names of

- the treatment through the ``var_d_name`` parameter
- the potential outcomes through the ``var_polscore_name`` parameter
- ordered and/or unordered features used to build the policy tree using the ``var_x_name_ord`` and ``var_x_name_unord`` parameter respectively

as follows:

.. code-block:: python

    # Create an instance of the OptimalPolicy class:
    my_optimal_policy = OptimalPolicy(
        var_d_name="treat",
        var_polscore_name=['y_pot0', 'y_pot1', 'y_pot2'],
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        var_x_name_unord=["x_unord0"],
        gen_method="best_policy_score", 
        pt_depth_tree_1=2
        )


Note that the ``pt_depth_tree_1`` parameter specifies the depth of the (first) policy tree. For demonstration purposes we set it to 2. In practice, you should choose a larger value which will increase the computational burden. See the :doc:`User guide <user_guide/optimal-policy_example>` and the :doc:`Algorithm reference <algorithm_reference/optimal-policy_algorithm>` for more detailed explanations.

Accessing results
~~~~~~~~~~~~~~~~~

After initializing an Optimal Policy Tree, the **mcf** package will automatically create an output folder. This folder will contain a number of standard outputs for your convenience. You can find the location of this folder in your console output. Alternatively, you can manually specify the folder location using the ``gen_outpath`` parameter.


Fit an Optimal Policy Tree
----------------------------

To find the Optimal Policy Tree, we use the :py:meth:`~optpolicy_functions.OptimalPolicy.solve` method, where we need to supply the pandas DataFrame holding the potential outcomes, treatment variable and the features:

.. code-block:: python

    train_pt_df = results["iate_data_df"]
    alloc_train_df, _, _ = my_optimal_policy.solve(training_df, data_title='training')

The returned DataFrame contains the optimal allocation rule for the training data.

.. code-block:: python

    print(alloc_train_df)

Next, we can use the :py:meth:`~optpolicy_functions.OptimalPolicy.evaluate` method to evaluate this allocation rule. This will return a dictionary holding the results of the evaluation. As a side-effect, the DataFrame with the optimal allocation is augmented with columns that contain the observed treatment and a random allocation of treatments.

.. code-block:: python

    results_eva_train, _ = my_optimal_policy.evaluate(alloc_train_df, training_df,
                                           data_title='training')

    print(results_eva_train)

Overview of results
~~~~~~~~~~~~~~~~~~~~~

A great way to get an overview of the results is to read the PDF-report that can be generated using the class :py:class:`~reporting.McfOptPolReport`:

.. code-block:: python

    policy_tree_report = McfOptPolReport(
        optpol = my_policy_tree,
        outputfile = 'Optimal-Policy_Report'
        )
    policy_tree_report.report()

Additionally, you can access the results programmatically. The `report` attribute of your optimal policy object is a dictionary containing the results. Here's how you can access a specific element:

.. code-block:: python

    dictionary_of_results = my_optimal_policy.report
    print(dictionary_of_results.keys())
    evaluation_list = dictionary_of_results['evalu_list']
    print("Evaluation List: ", evaluation_list)

Finally, it is straightforward to apply our Optimal Policy Tree to new data. To do so, we simply apply the :py:meth:`~optpolicy_functions.OptimalPolicy.allocate` method
to the DataFrame holding the potential outcomes, treatment variable and the features for the data that was held out for evaluation:

.. code-block:: python

    alloc_pred_df, _ = my_optimal_policy.allocate(prediction_df, data_title='prediction')

To evaluate this allocation rule, again apply the :py:meth:`~optpolicy_functions.OptimalPolicy.allocate` method similar to above.

.. code-block:: python

    results_eva_pred, _ = my_optimal_policy.evaluate(alloc_pred_df, prediction_df,
                                      data_title='prediction')

    print(results_eva_pred)

Next steps
----------

The following are great sources to learn even more about the **mcf** package:

- The :doc:`user_guide` offers explanations on additional features of the mcf package and provides several example scripts.
- Check out the :doc:`python_api` for details on interacting with the mcf package.
- The :doc:`algorithm_reference` provides a technical description of the methods used in the package.
