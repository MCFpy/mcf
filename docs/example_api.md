<a id="numpy_ex_cross_ref.ExampleError"></a>

## ExampleError Objects

```python
class ExampleError(Exception)
```

Exceptions are documented in the same way as classes.
The init method may be documented in either the class level
docstring, or as a docstring on the init method itself. $`\sqrt{3x-1}+(1+x)^2`$
Either form is acceptable, but the two should not be mixed. Choose one
convention to document the init method and be consistent with it.

### Note
----
-  Do not include the `self` parameter in the ``Parameters`` section.

### Parameters
----------      
<a id="var_bgate_name"><strong>var_bgate_name</strong></a> 
-  Human readable string describing the exception 

<a id="var_cluster_name"><strong>var_cluster_name</strong></a>
-  Numeric error code.
    -  describing the exception

### Attributes
----------      
<a id="msg"><strong>msg</strong></a>
  - Human $\sqrt{3x-1}+(1+x)^2$ readable string describing the exception
    - $n_d^{0.4}/6$.

<a id="code"><strong>code</strong></a>
- obj: `int`, optional
  - Numeric error code. $n^{0.3}$.
    - $round(60000 + \sqrt{n - 60000)}$, where $n$, this is a formatted docstring.

<a id="numpy_ex_cross_ref.ExampleClass"></a>

## ExampleClass Objects

```python
class ExampleClass(object)
```

The summary line for a class docstring should fit on one line.

 If the class has public attributes, they may be documented here
 in an ``Attributes`` $`\sqrt{3x-1}+(1+x)^2`$ section and follow the same formatting as a
 function's ``Args`` section. Alternatively, attributes may be documented
 inline with the attribute's declaration (see __init__ method below).
 $`\sqrt{3x-1}+(1+x)^2`$
 Properties created with the ``@property`` decorator should be documented
 in the property's getter method.

 ### Attributes
 ----------
 <a id="attr1"><strong>attr1</strong></a>
 - str
     - Description of `attr1`. $`\sqrt{3x-1}+(1+x)^2`$

 <a id="attr2"><strong>attr2</strong></a>
 - obj:`int`, optional
   - Description of `attr2`.

 <a id="p_gmate_sample_share"><strong>p_gmate_sample_share</strong></a>
 - **p_gmate_sample_share** : Float (or None), optiona

-  Implementation of (AM)BGATE estimation is very cpu intensive.
 Therefore, random samples are used to speed up the programme if
 there are number observations  / number of evaluation points > 10.
 None: If observation in prediction data $(n) < 1000: 1$
       If $n >= 1000: 1000 + (n-1000)**(3/4)$ / evaluation points.
 Default is None.

<a id="numpy_ex_cross_ref.ExampleClass.__init__"></a>

#### \_\_init\_\_

```python
def __init__(param1, param2, param3)
```

Example of docstring on the __init__ method.

The __init__ method may be documented in either the class level
docstring, or as a docstring on the __init__ method itself.

Either form is acceptable, but the two should not be mixed. Choose one
convention to document the __init__ method and be consistent with it.

### Note
----
Do not include the `self` parameter in the ``Parameters`` section.

### Parameters
----------
<a id="param1"><strong>param1</strong></a>
-  **param1** : str
   -  Description of `param1`.

<a id="param2"><strong>param2</strong></a>
-  **param2** : :obj:`list` of :obj:`str`
   -  Description of `param2`. Multiple lines are supported.

<a id="param3"><strong>param3</strong></a>
-  **param3** : :obj:`int`, optional
   -  Description of `param3`.

To reference the **msg** parameter in the same document like this: [Message parameter](#msg).
Similarly, to reference the **code** parameter like this: [Code parameter](#code).
