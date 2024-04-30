FAQ
==========================

.. contents::
   :local:
   :depth: 2

Installation
------------

- **How do I install the package?**

  You can install the package following the :ref:`Installation Guide <installation-guide>`.

  As a quick reference, you can install the package using pip:

  .. code-block:: bash

     pip install mcf

Usage
-----

- **Where can I find the results of the ModifiedCausalForest class?**

  The results are stored in a dictionary returned by the :py:meth:`~mcf_functions.ModifiedCausalForest.predict` method of the :py:class:`~mcf_functions.ModifiedCausalForest` class. This dictionary contains various estimated treatment 
  effects, their corresponding standard errors and related objects, all of which can be inspected in your variable explorer.

  Here is a brief demonstration on how to retrieve these results:

  .. code-block:: python

     # Train the Modified Causal Forest:
     my_mcf.train(df)
     # Assign the output of the predict method to a variable:
     results = my_mcf.predict(df)
     # The 'results' dictionary contains the estimated treatment effects, standard errors and others:
     print(results.keys())

  For more examples you can check out the :ref:`Getting Started <getting-started>` or the :doc:`user_guide`.

  Moreover, when using the :py:meth:`~mcf_functions.ModifiedCausalForest.train`, :py:meth:`~mcf_functions.ModifiedCausalForest.predict`, :py:meth:`~mcf_functions.ModifiedCausalForest.analyse` methods, and the :py:class:`~reporting.McfOptPolReport`: class, a folder is generated in your specified output path. If no output path is specified, all files will be saved where the Anaconda distribution is installed. You can learn more about working directories and output paths at `w3schools <https://www.w3schools.com/python/ref_os_chdir.asp>`_.

  The "out" folder contains a PDF with crucial information regarding the estimation of the :py:class:`~mcf_functions.ModifiedCausalForest`. For more comprehensive insights, we recommend reviewing the `Full example with all parameters used <https://github.com/MCFpy/mcf/blob/main/examples/all_parameters_mcf.py>`__.


- **Where can I find the results of the OptimalPolicy class?**

  The results are stored mainly in two diccionaries which you can access once you have used the :py:meth:`~optpolicy_functions.OptimalPolicy.evaluate` and :py:meth:`~optpolicy_functions.OptimalPolicy.allocate` and :py:meth:`~optpolicy_functions.OptimalPolicy.solve` methods. Additionally, you can access further results stored in your instance of the :py:class:`~optpolicy_functions.OptimalPolicy` class where you can access multiple dictionaries containing additional results. 

  For more examples you can check out the :ref:`Getting Started <getting-started>` or the :doc:`user_guide`.

  As with the :py:class:`~optpolicy_functions.OptimalPolicy` class, further results are also stored in the "out" folder either as PDF, txt or csv files. 

- **How can I determine which data points were excluded during common support checks and access the corresponding dataframe?**

  We recommend you check the :ref:`Common Support <common-support>` section. Additonally, you can check which data points were excluded in the in the common support section of the PDF file which is automatically generated.

- **How do I access the dataframe representing the final sample that passed common support criteria?**

  You can access the final sample that passed the common support criteria from the results dictionary returned by the :py:meth:`~mcf_functions.ModifiedCausalForest.predict` method of the :py:class:`~mcf_functions.ModifiedCausalForest` class. The dataframe is stored under the key `"iate_data_df"`.

  .. code-block:: python

     # Access the dataframe from the results dictionary
     final_sample_df = results["iate_data_df"]
     print(final_sample_df)

- **Do I include the heterogeneity variable in the covariates?**

  Yes, you must include the heterogeneity variable that you are interested in with the rest of your covariates.

- **What's the difference between ordered and unordered variables?**

  Ordered variables are numerical variables that have a natural order, such as age or income. Unordered variables, also known as categorical variables, are variables that don't have a natural order, such as gender or nationality.

Troubleshooting
---------------

- **I'm getting an error when I try to install the package. What should I do?**

  Make sure you have the latest version of pip installed. If the problem persists, please use the `issue tracker <https://github.com/MCFpy/mcf/issues>`__.

- **I'm getting an error with Ray. What should I do?**

  If you're getting an error with Ray, try resetting the kernel before every training. This can often solve issues related to Ray. If the problem persists, please use the `issue tracker <https://github.com/MCFpy/mcf/issues>`__.

- **The package installed successfully, but I'm getting an error when I try to import it. What should I do?**

  This could be due to a conflict with other packages or an `issue <https://github.com/MCFpy/mcf/issues>`__ with your Python environment. Try creating a new virtual environment and installing the package there. If the problem persists, please open an `issue <https://github.com/MCFpy/mcf/issues>`__ on the GitHub repository.

- **I'm getting unexpected results when I use the package. What should I do?**

  Make sure you're using the package as intended. Check the documentation and examples to see if you're using the functions and classes correctly. If you believe the results are incorrect, please open an `issue <https://github.com/MCFpy/mcf/issues>`__ on the GitHub repository.

- **The package is running slower than I expected. What can I do to improve performance?**

  Performance can depend on many factors, including the size of your data and your hardware. Check the documentation for tips on improving performance, specially the :ref:`Computational <computational-speed>` section.

- **I'm having trouble understanding how to use a certain feature of the package. Where can I find help?**

  The documentation is the best place to start. It provides a detailed explanation of all features and how to use them. If you're still having trouble, consider opening an `issue <https://github.com/MCFpy/mcf/issues>`__ on the GitHub repository.

