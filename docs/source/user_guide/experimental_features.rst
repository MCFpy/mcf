Experimental features
=====================

Note: All features in this section are experimental and thus not yet fully documented and tested. Please open an issue `here <https://github.com/MCFpy/mcf/issues>`__ if you encounter any problems or have any questions. 

Sensitivity checks
------------------

The method :py:meth:`~mcf_mini.ModifiedCausalForest.sensitivity` of the :class:`~mcf_mini.ModifiedCausalForest` class contains some simulation-based tools to check how well the Modified Causal Forest works in removing selection bias and how sensitive the results are with respect to potentially missing confounding covariates (i.e., those related to treatment and potential outcomes).

A paper by Armendariz-Pacheco, Frischknecht, Lechner, and Mareckova (2024) will discuss and investigate the different methods in detail. So far, please note that all methods are simulation based.

The sensitivity checks consist of the following steps:

1. Estimate all treatment probabilities.
2. Remove all observations from treatment states other than one (largest treatment or user-determined).
3. Use estimated probabilities to simulate treated observations, respecting the original treatment shares (pseudo-treatments).
4. Estimate the effects of pseudo-treatments. The true effects are known to be zero, so the deviation from 0 is used as a measure of result sensitivity.

Steps 3 and 4 may be repeated, and results averaged to reduce simulation noise.

Please consult the API for details on how to use the :py:meth:`~mcf_mini.ModifiedCausalForest.sensitivity` method.