Getting started
=======================

In the following we will show how the **mcf** package can be used to

- estimate heterogeneous treatment effects using the Modified Causal Forest
- learn an optimal policy rule based on a Policy Tree


Simulating data
---------------

Let us first create some synthetic data. We will consider a setting with three treatment levels (0, 1, 2):

.. code-block:: python

    import numpy as np
    import pandas as pd
    from mcf import ModifiedCausalForest
    from mcf import OptimalPolicy 

    def simulate_data(n):
        """
        Simulate data with treatment 'd', outcome 'y', an unordered control
        variable 'female' and two ordered controls 'x1', 'x2'.

        Parameters:
        - n (int): Number of observations in the simulated data.

        Returns:
        pd.DataFrame: Simulated data in a Pandas DataFrame.

        """
        d = np.random.choice([0, 1, 2], n, replace=True)
        female = np.random.choice([0, 1], n, replace=True)
        x_ordered = np.random.normal(size=(n, 2))
        y = (x_ordered[:, 0] +
            x_ordered[:, 1] * (d == 1) +
            x_ordered[:, 1] * (d == 2) +
            0.5 * female +
            np.random.normal(size=n))

        data = {"y": y, "d": d, "female": female}

        for i in range(x_ordered.shape[1]):
            data["x" + str(i + 1)] = x_ordered[:, i]

        return pd.DataFrame(data)

    df = simulate_data(1000)

To estimate both the Modified Causal Forest and the Optimal Policy Tree, we will consider a simple sample splitting approach where we divide the simulated data into three equally sized parts:

1. *train_mcf_df*: Used to train the Modified Causal Forest.
2. *pred_mcf_train_pt_df*: Used to the predict the heterogeneous treatment effects and to train the Optimal Policy Tree.
3. *evaluate_pt_df*: Used to evaluate the Optimal Policy Tree.

.. code-block:: python

    indices = np.array_split(df.index, 3)
    train_mcf_df, pred_mcf_train_pt_df, evaluate_pt_df = (df.iloc[ind] for ind in indices)


Estimating heterogeneous treatment effects
------------------------------------------

To estimate a Modified Causal Forest, we use the :py:class:`~mcf_mini.ModifiedCausalForest` class of the **mcf** package. To create an instance of the :py:class:`~mcf_mini.ModifiedCausalForest` class, we need to specify the name of

- at least one outcome variable through the ``var_y_name`` parameter
- the treatment variable through the ``var_d_name`` parameter
- ordered features through ``var_x_name_ord`` and/or unordered features through ``var_x_name_unord``

as follows:

.. code-block:: python

    my_mcf = ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord=["x1", "x2"],
        var_x_name_unord=["female"],
        _int_show_plots=False # Suppress the display of diagnostic plots during estimation
    )

The **mcf** package generates a number of standard outputs for your convenience. After initializing a Modified Causal Forest, the package will create an output folder - as indicated in the console output - where these results will subsequently be stored. You can also manually specify this folder using the ``gen_outpath`` parameter.

.. dropdown:: Commonly used optional parameters 

    Below you find a selected list of optional parameters that are often used to initialize a Modified Causal Forest. For a more detailed description of these parameters, please refer to the documentation of :py:class:`~mcf_mini.ModifiedCausalForest`.

    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | Parameter                        | Description                                                                                                      |
    +==================================+==================================================================================================================+
    | ``cf_boot``                      | Number of Causal Trees. Default is 1000.                                                                         |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``p_atet``                       | If True, :math:`\textrm{ATET's}` are estimated. The default is False.                                            |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``p_gatet``                      | If True, :math:`\textrm{GATE's}` and :math:`\textrm{GATET's}` are estimated. The default is False.               |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``var_w_name``                   | Weights assigned to each observation.                                                                            |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``var_cluster_name``             | Cluster identifier.                                                                                              |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``var_id_name``                  | Individual identifier.                                                                                           |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_list``              | Ordered feature(s) with many values used for :math:`\textrm{GATE}` estimation.                                   |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_ord``               | Ordered feature(s) with few values used for :math:`\textrm{GATE}` estimation.                                    |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``var_z_name_unord``             | Unordered feature(s) used for :math:`\textrm{GATE}` estimation.                                                  |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``var_x_name_always_in_ord``     | Ordered feature(s) always used in splitting decision.                                                            |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``var_x_name_always_in_unord``   | Unordered feature(s) always used in splitting decision.                                                          |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+
    | ``var_y_tree_name``              | Outcome used to build trees. If not specified, the first outcome in ``y_name`` is selected for building trees.   |
    +----------------------------------+------------------------------------------------------------------------------------------------------------------+


Training a Modified Causal Forest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next we will train the Modified Causal Forest on the *train_mcf_df* data using the :py:meth:`~mcf_mini.ModifiedCausalForest.train` method:

.. code-block:: python

    my_mcf.train(train_mcf_df)

Now we are ready to estimate heterogeneous treatment effects on the *pred_mcf_train_pt_df* data using the :py:meth:`~mcf_mini.ModifiedCausalForest.predict` method.

.. code-block:: python

    results = my_mcf.predict(pred_mcf_train_pt_df)


Results
~~~~~~~

The :py:meth:`~mcf_mini.ModifiedCausalForest.predict` method returns a dictionary containing the estimation results. To gain an overview, have a look at the keys of the dictionary:

.. code-block:: python

    print(results.keys())

By default the average treatment effects (:math:`\textrm{ATE}`) as well as the individualized average treatment effects (:math:`\textrm{IATE}`) are estimated. If these terms do not sound familiar, click here to learn more about the different kinds of heterogeneous treatment effects.

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

Please refer to the documentation of the :py:meth:`~mcf_mini.ModifiedCausalForest.predict` method for a detailed description of the contents of this dictionary.


Post-estimation
~~~~~~~~~~~~~~~

You can also use the :py:meth:`~mcf_mini.ModifiedCausalForest.analyse` method to investigate a number of post-estimation plots. These plots are also exported to the previously created output folder:

.. code-block:: python

    my_mcf.analyse(results)

Finally, for out-of-sample evaluation, simply apply the :py:meth:`~mcf_mini.ModifiedCausalForest.predict` method to the data held out for evaluation:

.. code-block:: python

    oos_results = my_mcf.predict(evaluate_pt_df)

    
Learning an optimal policy rule
-------------------------------

To learn an optimal policy rule, we can use the OptimalPolicy class of the **mcf** package. To get started we need a Pandas DataFrame that holds the estimated potential outcomes (also called policy scores), the treatment variable and the features on which we want to base the decision tree. We can use

.. code-block:: python

    results["iate_data_df"]


To build an optimal policy tree, we then need to create an instance of class
OptimalPolicy where we set ``gen_method`` to 'policy tree' and provide the names
of 

- the treatment
- the potential outcome
- ordered and/or unordered features

using the following parameters:

.. code-block:: python

    my_opt_policy_tree = OptimalPolicy(
        var_d_name="d", 
        var_polscore_name=["Y_LC0_un_lc_pot", "Y_LC1_un_lc_pot", "Y_LC2_un_lc_pot"],
        var_x_ord_name=["x1", "x2"],
        var_x_unord_name=["female"],
        gen_method='policy tree',
        pt_depth=2 # Depth of the policy tree
        )


The **mcf** package generates a number of standard outputs for your convenience. After initializing a Modified Causal Forest, the package will create a folder - as indicated in the console output - where these outputs will subsequently be stored. You can also manually specify this folder using the ``gen_outpath`` parameter.


Next steps
----------

The following are great sources to learn even more about the **mcf** package:

- The :doc:`user_guide` offers explanations on additional features of the **mcf** package.
- Check out the :doc:`python_api` for details on interacting with the **mcf** package.
- The :doc:`algorithm_reference` provides a technical description of the methods used in the package.