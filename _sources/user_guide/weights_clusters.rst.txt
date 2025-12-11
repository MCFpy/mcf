Sampling weights and clustering 
===============================

Sampling weights
----------------

You can provide sampling weights for each observation in your data set. To estimate a Modified Causal Forest with sampling weights, you need to set the ``gen_weighted`` parameter to ``True`` and provide the name of the variable containing the sampling weights in the ``var_w_name`` parameter.

Clustering
----------

If your data set contains clusters, you can provide the name of the variable containing the cluster identifier through the ``var_cluster_name`` parameter.

In case your data has a panel structure, your data set is also clustered, namely at the level of the individual. In this case you can provide the name of the variable containing the individual identifier through the ``var_cluster_name`` parameter.

The clusters are by default used to draw the random samples when growing the forest. You can control this behaviour through the ``gen_panel_in_rf`` parameter. To compute clustered standard errors, you need to set the ``gen_panel_data`` parameter to True.

Parameter overview
------------------

The following table summarizes the parameters related to sampling weights and clustering in the class :py:class:`~mcf_main.ModifiedCausalForest`:

+----------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| Parameter            | Description                                                                                                                                 |
+======================+=============================================================================================================================================+
| ``var_w_name``       | Name of the variable holding the sampling weight of each observation.                                                                       |
+----------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| ``gen_weighted``     | If True, sampling weights from ``var_w_name`` will be used. Default: False.                                                                 |
+----------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| ``var_cluster_name`` | Name of the variable holding the cluster identifier.                                                                                        |
+----------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| ``gen_panel_data``   | If True, clustered standard errors based on ``var_cluster_name`` are computed. Default: False.                                              |
+----------------------+---------------------------------------------------------------------------------------------------------------------------------------------+
| ``gen_panel_in_rf``  | If True, clusters are used to draw the random samples when building the forest. Default: True. Only relevant if ``gen_panel_data`` is True. |
+----------------------+---------------------------------------------------------------------------------------------------------------------------------------------+

Please consult the :py:class:`API <mcf_main.ModifiedCausalForest>` for more details.

Examples
--------

.. code-block:: python

    from mcf.example_data_functions import example_data
    from mcf.mcf_main import ModifiedCausalForest
    
    # Generate example data using the built-in function `example_data()`
    training_df, prediction_df, name_dict = example_data()
    
    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        # Parameters for sampling weights:
        var_w_name="weight",
        gen_weighted=True
    )
    
    
    my_mcf = ModifiedCausalForest(
        var_y_name="outcome",
        var_d_name="treat",
        var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
        # Parameters for clustering:
        var_cluster_name="cluster",
        gen_panel_data=True,
        gen_panel_in_rf=True
    )
        
