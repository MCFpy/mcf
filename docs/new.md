# Release Updates 

## Version 0.1.0

### Bug fixes
 - Bug fix for dealing with missings.
 - Bug fixes for problems computing treatment effects for treatment populations.
 - Bug fixes for the use of panel data and clustering.

### What's New
- [post_kmeans_no_of_groups](./core_6.md#post_kmeans_no_of_groups) can now be a list or tuple with multiple values for the number of clusters; the optimal value is chosen through silhouette analysis.
- Detection of numerical variables added; raises an exception for non-numerical inputs.
- All variables used are shown in initial treatment specific statistics to detect common support issues.
- Improved statistics for common support analysis.

### Experimental
- Optimal Policy Tool building policy trees included bases on estimated IATEs (allowing implicitly for constraints and programme costs).
