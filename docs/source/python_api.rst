Python API
==========


Overview of classes 
-------------------

..  
    If you add a '~' before a reference like `mymodule.MyClass.mymethod`, the
    link text will then only show 'mymethod' which is often desirable.
.. autosummary::
    ~mcf_main


Modified Causal Forest
----------------------

.. currentmodule:: mcf_main

.. autoclass:: ModifiedCausalForest


Methods
^^^^^^^

..
   HACK -- Also list the methods in the "commented out" paragraph below. The
   point here is that we don't want this to appear in the output, but the
   autosummary below will, even when commented out, generate the separate
   documentation page that can be cross-referenced.

   .. autosummary:: 
      :toctree:
      ModifiedCausalForest.train
      ModifiedCausalForest.predict
      ModifiedCausalForest.analyse
      ModifiedCausalForest.sensitivity

.. autosummary:: 
    ~ModifiedCausalForest.train
    ~ModifiedCausalForest.predict
    ~ModifiedCausalForest.analyse
    ~ModifiedCausalForest.sensitivity

