Sampling weights and clustering
===============================

Sampling weights
----------------

Set ``gen_weighted`` and provide ``var_w_name`` to use sampling weights.
The weight column is required in both training and prediction data. These weights
are distinct from the forest weights used to estimate potential outcomes.

Clustering
----------

Set ``p_cluster_std`` and supply ``var_cluster_name`` for cluster-robust
standard errors. ``gen_panel_data`` also enables clustered inference and
requires a cluster identifier. A row identifier (``var_id_name``) identifies an
observation; a cluster identifier groups related observations, such as repeated
measurements of the same person.

With panel mode enabled, ``gen_panel_in_rf`` samples clusters within each
tree's construction subsample. The initial construction/filling split and forest
chunking still operate on rows, so clusters do not necessarily remain intact
through every split. Bootstrap defaults depend on clustering; see
:doc:`../algorithm_reference/inference`.

Parameter overview
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Purpose
   * - ``var_w_name``, ``gen_weighted``
     - Sampling-weight column and activation.
   * - ``var_id_name``
     - Observation identifier.
   * - ``var_cluster_name``, ``p_cluster_std``
     - Cluster identifier and cluster-robust inference.
   * - ``gen_panel_data``
     - Enables panel mode and cluster-robust inference.
   * - ``gen_panel_in_rf``
     - Cluster sampling within trees when panel mode is enabled.

Example
-------

This configuration fragment assumes your data contain ``row_id`` and
``person_id``, it does not create these identifiers.

.. code-block:: python

    from mcf.mcf_main import ModifiedCausalForest

    my_mcf = ModifiedCausalForest(
        var_y_name='outcome', var_d_name='treat',
        var_x_name_ord=['age', 'income'],
        var_id_name='row_id', var_cluster_name='person_id',
        p_cluster_std=True
    )
    
For sampling weights, additionally set ``gen_weighted`` and
``var_w_name='weight'`` and supply that column in both datasets.
See the :py:class:`API <mcf_main.ModifiedCausalForest>` for supported combinations.
