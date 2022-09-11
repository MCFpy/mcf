# Python API - Short Version


```python
modified_causal_forest(d_name=d_name, x_name_always_in_ord=x_name_always_in_ord,
                       x_name_always_in_unord=x_name_always_in_unord, x_name_ord=x_name_ord,
                       x_name_remain_ord=x_name_remain_ord, x_name_remain_unord=x_name_remain_unord,
                       x_name_unord=x_name_unord, y_name=y_name,
                       z_name_list=z_name_list, z_name_split_ord=z_name_split_ord,
                       z_name_split_unord=z_name_split_unord, 
                       atet_flag=atet_flag,
                       boot=boot, datpfad=datpfad, gatet_flag=gatet_flag,
                       indata=indata,
                       l_centering=l_centering, 
                       l_centering_replicatoin=l_centering_replication,
                       mp_parallel=mp_parallel,
                       outpfad=outpfad)
```


## Variable Names

**d**

* <a id="d_name">**d_name**</a> - list with **String**
	* Specifies name of treatment, which must be discrete.

**x**

* <a id="x_name_always_in_ord">**x_name_always_in_ord**</a> - list with **String**
	* Specifies names of ordered features, which are always included when deciding upon the next split.
* <a id="x_name_always_in_unord">**x_name_always_in_unord**</a> - list with **String**
	* Specifies names of unordered variables, which are always included when deciding upon next split.
* <a id="x_name_ord">**x_name_ord**</a> - list with **String**
	* Specifies names of ordered variables.
* <a id="x_name_remain_ord">**x_name_remain_ord**</a> - list with **String**
	* Specifies names of ordered variables to be excluded from preliminary feature selection.
* <a id="x_name_remain_unord">**x_name_remain_unord**</a> - list with **String**
	* Specifies names of unordered variables to be excluded from preliminary feature selection.
* <a id="x_name_unord">**x_name_unord**</a> - list with **String**
	* Specifies names of unordered variables.

**y**

* <a id="y_name">**y_name**</a> - list with **String**
	* Specifies outcome variables.


**z**

* <a id="z_name_list">**z_name_list**</a> - list with **String**
	* Specifies names of ordered variables with many values; these variables are recoded to define the split, they will be added to the list of confounders. Since they are broken up in categories for all their observed values, it does not matter whether they are coded as ordered or unordered.
* <a id="z_name_split_ord">**z_name_split_ord**</a> - list with **String**
	* Specifies names of ordered variables that are discrete and define a unique sample split for each value.
* <a id="z_name_split_unord">**z_name_split_unord**</a> - list with **String**
	* Specifies names of unordered variables that are discrete and define a unique sample split for each value.

## Parameters

**a**

* <a id="atet_flag">**atet_flag**</a> - **Boolean**
	* If  True, average effects for subpopulations are computed by treatments (if available); this works only if at least one $z$ variable is specified; the default is False.

**b**

* <a id="boot">**boot**</a> - positive **Integer**
	* Gives the number of trees in the forest to be estimated; the default is 1000.


**d**
* <a id="datpfad">**datpfad**</a>  - **String**
	* Specifies the directory, in which the data is saved for estimation and/or prediction.

**g**

* <a id="gatet_flag">**gatet_flag**</a> - **Boolean**
	* If True, GATE(T)s are computed  for subpopulations by treatments; the default is False.

**i**
* <a id="indata">**indata**</a> - **String**
	* Specifies the file name for the data, which is used for estimation; the file needs to be in *csv* format.

**l**

* <a id="l_centering">**l_centering**</a>  - **Boolean**
	* Determines whether local centering is used; the default value is True.
* <a id="l_centering_cv_k">**l_centering_cv_k**</a> - **Boolean**
	* Specifies number of folds used in cross-validation; only valid if *l_centering_new_sample* is Fals
* <a id="l_centering_replication">**l_centering_replication**</a> - **Boolean** 
	* Disables threading in the estimation of the regression forests for repliable reults; default is True.


**m**

* <a id="mp_parallel">**mp_parallel**</a> - **None** or **Float** larger **0**
	* Specifies the number of parallel processes; the default value is None; for a value of None the number of parallel processes is set to two less than the number of logical cores; for values between -0.5 and 1.5, the value is set to 1; for number greater than 1.5, the value is set to the integer part of the specified processes.


**o**

* <a id="outpfad">**outpfad**</a> - **String**
	* Creates a folder *out* in the application directory, in which the output files are written.
