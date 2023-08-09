# Python API


```python
modified_causal_forest(cf_boot=cf_boot,
                       lc_yes=lc_yes,
                       p_atet=p_atet,
                       p_gatet_flag=p_gatet_flag,
                       var_d_name=var_d_name,
                       var_x_name_always_in_ord=var_x_name_always_in_ord,
                       var_x_name_always_in_unord=x_name_always_in_unord,
                       var_x_name_ord=x_name_ord,
                       var_x_name_remain_ord=var_x_name_remain_ord,
                       var_x_name_remain_unord=var_x_name_remain_unord,
                       var_x_name_unord=var_x_name_unord,
                       var_y_name=var_y_name,
                       var_z_name_list=var_z_name_list,
                       var_z_name_ord=var_z_name_ord,
                       var_z_name_unord=var_z_name_ord,)
```


## Variable Names

- <a id="var_d_name"><strong>var_d_name</strong></a>
	* Name of treatment.

- <a id="var_x_name_always_in_ord"><strong>var_x_name_always_in_ord</strong></a>
  * Names of ordered variables, which should be always included when deciding on next split.

- <a id="var_x_name_always_in_unord"><strong>var_x_name_always_in_unord</strong></a>
  * Names of unordered variables, which should be always included when deciding on next split.

- <a id="var_x_name_ord"><strong>var_x_name_ord</strong></a>
  *	Names of ordered features.

- <a id="var_x_name_unord"><strong>var_x_name_unord</strong></a>
  * Names of unordered features.

- <a id="var_x_name_remain_ord"><strong>var_x_name_remain_ord</strong></a>
  *	Names of unordered features.

- <a id="var_x_name_remain_unord"><strong>var_x_name_remain_unord</strong></a>
  * Names of variables excluded from preliminary feature selection. Default is None.

- <a id="var_y_name"><strong>var_y_name</strong></a>
  * Name(s) of outcome variable(s).

- <a id="var_z_name_list"><strong>var_z_name_list</strong></a>
  * Names of variables for heterogeneity analysis.

- <a id="var_z_name_ord"><strong>var_z_name_ord</strong></a>
  * Names of ordered variables to define policy relevant heterogeneity.

- <a id="var_z_name_unord"><strong>var_z_name_unord</strong></a>
  * Names of unordered variables to define policy relevant heterogeneity.

## Parameters

- <a id="cf_boot"><strong>cf_boot</strong></a>
  * Number of bootstrap or subsampling replications. Default is 1000.

- <a id="lc_yes"><strong>lc_yes</strong></a>
  * If True, local centering is deployed. The default is True.

- <a id="p_atet"><strong>p_atet</strong></a>
  * If True, the average effects are estimated by treatment group. This works only if at least one heterogeneity variable is defined. The default is False.

- <a id="p_gatet"><strong>p_gatet</strong></a>
  * If set to True, GATEs and GATETs are computed. If no variables are specified for GATE estimation, **p_bgate** is set to False. The default is False.
