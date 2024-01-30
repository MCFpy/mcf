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

The following table summarizes the parameters related to sampling weights and clustering in the class :py:class:`~mcf_functions.ModifiedCausalForest`:

+----------------------+----------------------------------------------------------------------------------------------------+
| Parameter            | Description                                                                                        |
+======================+====================================================================================================+
| ``var_w_name``       | Name of the variable holding the sampling weight of each observation.                              |
+----------------------+----------------------------------------------------------------------------------------------------+
| ``gen_weighted``     | If True, sampling weights from ``var_w_name`` will be used. Default: False.                        |
+----------------------+----------------------------------------------------------------------------------------------------+
| ``var_cluster_name`` | Name of the variable holding the cluster identifier.                                               |
+----------------------+----------------------------------------------------------------------------------------------------+
| ``gen_panel_in_rf``  | If True, clusters are used to draw the random samples when building the forest. Default: True.     |
+----------------------+----------------------------------------------------------------------------------------------------+
| ``gen_panel_data``   | If True, clustered standard errors are computed. Default: False.                                   |
+----------------------+----------------------------------------------------------------------------------------------------+

Please consult the :py:class:`API <mcf_functions.ModifiedCausalForest>` for more details.

Examples
--------

.. code-block:: python

    from mcf import ModifiedCausalForest

    ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord=["x1", "x2"],
        # Parameters for sampling weights:
        var_w_name="sampling_weight",
        gen_weighted=True
    )

    ModifiedCausalForest(
        var_y_name="y",
        var_d_name="d",
        var_x_name_ord=["x1", "x2"],
        # Parameters for clustering:
        var_cluster_name="cluster_id",
        gen_panel_in_rf=True,
        gen_panel_data=True
    )
