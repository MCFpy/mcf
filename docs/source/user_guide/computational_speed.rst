.. _computational-speed:

Computational Speed and Resources for Effect Estimation
=======================================================

Training and prediction have different resource demands. Adjust settings with
the estimand and the retained population in mind; reducing computation can also
change the statistical result.

Parallel Processing
-------------------

For parallel processing with Ray, use Python 3.12 on Windows, or Python 3.13 on Linux and macOS. Joblib is installed as a dependency of
**mcf** and also supports Python 3.14. See :ref:`installation-guide`.

By default, ``ModifiedCausalForest`` uses Joblib when Ray is unavailable
or when the adjusted training sample is sufficiently small on Windows.
Otherwise, it uses Ray. See ``_int_mp_backend`` in the
:py:class:`API <mcf_main.ModifiedCausalForest>` for the exact selection rule.

``gen_mp_parallel`` controls the number of parallel processes. Reducing
this number can lower memory use. If parallel processing causes problems,
create a new instance with ``_int_mp_backend='joblib'`` to use Joblib,
or ``_int_mp_backend='sequential'`` to run sequentially. You can also set
``gen_mp_parallel=1`` to use a single process. When explicitly selecting
Joblib on Windows, the initial number of workers must not exceed 63.

Forest Tuning via Grid Search
-----------------------------

Grid search considers three parameters: candidate-variable counts, alpha balance and minimum leaf
sizes. The forest evaluates combinations using its out-of-bag objective.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Parameter
     - Effect on computation
   * - ``cf_boot``
     - Number of trees; more trees require more computation.
   * - ``cf_m_share_min``, ``cf_m_share_max``
     - Bounds for the share of candidate variables.
   * - ``cf_m_grid``
     - Number of candidate-count grid values, not variables per split.
   * - ``cf_alpha_reg_min``, ``cf_alpha_reg_max``
     - Split-balance bounds; discrete treatments use treatment-specific balance by default.
   * - ``cf_n_min_min``, ``cf_n_min_max``
     - Minimum-leaf-size search range.
   * - ``cf_tune_all``
     - Expands tuning grids; increases computation.
   * - ``cf_random_thresholds``
     - Limits initial split candidates. If no admissible split is found, the algorithm retries all available variables and supported thresholds.
   * - ``cf_chunks_maxsize``
     - Splits large training samples into chunks and averages the resulting forests.
   * - ``cf_subsample_power_eval``
     - Filling-sample exponent, default 0.90; construction uses 0.85.
   * - ``cf_subsample_factor_eval``
     - None/True resolves to factor 2; False uses all available filling observations.

See :doc:`../algorithm_reference/training` for filling and candidate-variable
rules. Binary features should be listed as ordered variables.

Minimization of RAM usage
-------------------------

Low-memory prediction is enabled by default. It processes forest weights in
chunks rather than retaining the full weight matrix. Consider chunking and
fewer workers before discarding prediction observations. QIATEs and
``p_iate_m_ate`` require ``_int_low_memory_predict=False``.

The prediction-row cap is 1,000,000 with low-memory prediction and
250,000 otherwise, unless changed through ``_int_max_obs_prediction``. A smaller
prediction sample can change the population over which effects are averaged;
it is not solely a performance setting. Training-sample reductions can also
reduce precision.

For BGATE and CBGATE estimation, **mcf** draws a reference sample from the
prediction data and reuses it at each value of the heterogeneity variable
being evaluated. At least 50 observations are used per evaluation value,
or all available observations if fewer than 50 remain. As a result, the
total number of reference rows can exceed the number implied by
``p_bgate_sample_share``. See :doc:`../algorithm_reference/bgates_cbgates`
for details.
``p_choice_based_sampling`` reweights aggregate prediction observations by
treatment. It does not draw a subsample to speed up computation.

Example
-------

This configuration fragment illustrates resource controls. See the API for
sample-dependent defaults and the remaining options.

.. code-block:: python

    from mcf.mcf_main import ModifiedCausalForest

    my_mcf = ModifiedCausalForest(
        var_y_name='outcome', var_d_name='treat',
        var_x_name_ord=['x_cont0', 'x_cont1', 'x_ord1'],
        cf_boot=500,
        gen_mp_parallel=2,
        _int_low_memory_predict=True
    )

GPU-related requirements are described in :doc:`experimental_features`.
