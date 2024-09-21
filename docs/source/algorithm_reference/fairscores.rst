Fairness Extensions in Optimal Policy
===========================================

The :py:class:`~optpolicy_functions.OptimalPolicy` class in the **mcf** includes experimental features for fairness adjustments, accessible through the ``fairscores`` method. 
These features are designed to ensure that policy scores are fair with respect to certain protected variables. 
The fairness adjustments are based on the work by Bearth, Lechner, Mareckova, and Muny (2024).

This method can be configured using several parameters to control the type and extent of fairness adjustments. 

Usage
------
To use the fairness adjustments, configure the :py:class:`~optpolicy_functions.OptimalPolicy` class with the appropriate parameters and call the ``fairscores`` method on your data. This will return a DataFrame with adjusted policy scores that account for fairness considerations.

Note
------
These features are experimental and may require further testing and validation. For more details or additional parameters, please consult the :py:class:`API <mcf_functions.ModifiedCausalForest>` documentation.
