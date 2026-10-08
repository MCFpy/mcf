Local centering
===============

Method
--------

Local centering is a form of residualization and can improve the performance of forest estimators by regressing out the impact of the features on the outcome.
Let us define the conditionally centered outcome :math:`\tilde{Y}_i` as:

.. math::

   \tilde{Y}_i = Y_i - \hat{y}_{-i}(X_i)

where:

- :math:`Y_i` is the outcome for observation :math:`i`.
- :math:`\hat{y}_{-i}(X_i)` is an estimate of the conditional outcome expectation :math:`E[Y_i | X_i = x]`, given the realised :math:`x` of the feature vector :math:`X_i`, and computed without using the observation :math:`i`.

Implementation
--------------

Centered outcomes subtract fitted conditional means from observed outcomes.
``lc_estimator='RandomForest'`` is the default; other estimators and automatic
selection are available. ``lc_yes=True`` enables centering.

With ``lc_cs_cv=True``, construction-sample predictions are cross-fitted using
``lc_cs_cv_k`` folds. Filling and prediction observations use the fitted models.
With ``lc_cs_cv=False``, ``lc_cs_share`` reserves a separate sample for model
fitting that is excluded from subsequent forest estimation.

Uncentered potential outcomes are expressed on the original outcome scale
and can be used as policy scores. In the single-outcome example, their
column names are available in
``results['iate_names_dic'][0]['names_y_pot_uncenter']``.
When efficiency improvements are enabled, the corresponding local
centering models are used to restore the outcome levels.

See the :py:class:`API <mcf_main.ModifiedCausalForest>` for estimator choices
and resolved settings.


Example
~~~~~~~

.. code-block:: python

   from mcf.example_data import example_data
   from mcf.mcf_main import ModifiedCausalForest

   # Generate example data using the built-in function `example_data()`
   training_df, prediction_df, name_dict = example_data()


   my_mcf = ModifiedCausalForest(
       var_y_name="outcome",
       var_d_name="treat",
       var_x_name_ord=["x_cont0", "x_cont1", "x_ord1"],
       # Activates local centering
       lc_yes = True,
       # Data for local centering & common support adjustment by crossvalidation
       lc_cs_cv = True,
       # Number of folds in cross-validation
       lc_cs_cv_k = 5
   )

   my_mcf.train(training_df)
   results = my_mcf.predict(prediction_df)
