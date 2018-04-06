[TOC]

---

# Python Snippets 



## Using `pass`

pass is the “no-op” statement in Python. It can be used in blocks where no action is to be taken; it is only required because Python uses whitespace to delimit blocks:

```
if x < 0:
    print 'negative!'
elif x == 0:
    # TODO: put something
    pass
else:
	print 'positive!'
```

It’s common to use pass as a place-holder in code while working on a new piece of functionality:

```
def f(x, y, z):
    # TODO: implement this function!
    pass
```


## Exceptions handling

Handling Python errors or exceptions gracefully is an important part of building robust programs. In data analysis applications, many functions only work on certain kinds of input.

Example: Python’s `float` function is capable of casting a string to a floating point number, but fails with ValueError on improper inputs:

```
In [343]: float('1.2345')
Out[343]: 1.2345
In [344]: float('something')

ValueError                                Traceback (most recent call last)
<ipython-input-344-439904410854> in <module>()
----> 1 float('something')
ValueError: could not convert string to float: something
```



### Using `try` and `except`

Suppose we wanted a version of float that fails gracefully, returning the input argument.
We can do this by writing a function that encloses the call to float in a try/except block:

```Python
def attempt_float(x):
    try:
        return float(x)
    except:
        return x
```

The code in the except part of the block will only be executed if float(x) raises an exception:

```Python
In [346]: attempt_float('1.2345')
Out[346]: 1.2345

In [347]: attempt_float('something')
In [348]: 'something'

In [350]: attempt_float((1, 2))

TypeError                                 Traceback (most recent call last)
<ipython-input-350-9bdfd730cead> in <module>()
----> 1 attempt_float((1, 2))
TypeError: float() argument must be a string or a number
```

You can **catch multiple exception types** by writing a tuple of exception types:

```python
def attempt_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x
```



### Using `else` and  `finally`

```python
f = open(path, 'w')


try:
    write_to_file(f)
except:
    # executes only if the 'try' block fails
    print 'Failed'
else:
    # executes only if the 'try' block succeeds
    print 'Succeeded'
finally:
    # executes always
    f.close()
```

## Use a `dict` to replace values in a `Series` 

```python
df = DataFrame({
    'grade': list('ABC' * 4),
    'val': np.random.randn(12)}
)

lookup = {
    'A': 'Excellent',
    'B': 'Satisfactory',
    'C': 'Improve'
}

df['grade_desc'] = df['grade'].map(lookup)
```



## Export a `dict` as a `json`

```Python
with open('dest.json', 'w') as f:
	f.write(json.dumps(my-dictionary))
```



## Check `dir()` for objects in the environment

```python
[x for x in dir() if '<pattern>' in x]
```



## Flatten a list of lists

```python
# using Comprehensions
list_flat = [item for sublist in list_nested for item in sublist]

# using a function
from more_itertools import flatten
flatten(list_nested)
```



## Read UNIX timestamps (and convert them to IST)

- The choice of `unit=` below depends on the granularity of the UNIX timestamp. 
- Check [here](http://epochconverter.com) for details

```Python
df['pandas_ts'] = \
(pd
 .to_datetime(df['unix_ts'], unit='ms')
 .map(lambda x: x.tz_localize('UTC').tz_convert('Asia/Kolkata'))
)
```



## Cartesian Product of two or more lists.

```python
X = list('xyz')
A = list('abc')
P = ...
[(x, a) for x in X for a in A for p in ...]
```



## Useful `pd.Series()` methods


| method        | syntax                                              | action                                                       |
| ------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| `between`     | `s.between(left, right, inclusive=True)`            | Return boolean Series equivalent to left <= series <= right. NA values will be treated as False |
| `corr`        | `s.corr(other, method='pearson', min_periods=None)` | Compute correlation with other Series, excluding missing values. Method is one of `'kendall', 'spearman', 'pearson'` |
| `kurt`        | `s.kurt()`                                          | kurtosis over requested axis                                 |
| `skew`        | `s.skew()`                                          | skewness over requested axis                                 |
| `nlargest`    | `s.nlargest(10)`                                    | Return the largest n elements.                               |
| `nsmallest`   | `s.nsmallest(10)`                                   | Return the smallest n elements.                              |
| `quantile`    | `s.quantile([.25, .5, .75])`                        | Return value at the given quantile                           |
| `nunique`     | `s.nunique()`                                       | Return number of unique elements in the object.              |
| `sample`      | `s.sample(n or frac, replace, random_state)`        | Return n or frac% of items, with or without replacement using a seed for the random number generator |
| `where`       | `s.where(cond, else)`                               | Return an object of same shape as self, whose entries are from self where cond is True and from other where cond is False. |
| `interpolate` | `s.interpolate(method, axis)`                       | Interpolate values according to different methods (such as `'nearest', 'linear', 'quadratic', 'polynomial'`) |
| `idxmax`      |                                                     | Find the index of the largest value                          |
|               |                                                     |                                                              |



## Copy/Paste/Move Files & Folders

```python
import shutil

# copy a file
shutil.copy(src, dst) 

# copy a directory
shutil.copytree(src, dst)

# move file or directory
shutil.move(src, dst)
```



## Today's date to use in filenames

```Python
import datetime
str(datetime.datetime.now().date())
```



## To avoid unalignable indexes before `join/merge`

- Find a UNION of all the indexes in the objects that need to be merged.
- Reindex all the objects with this UNIONd index
- NaNs will be introduced whereever there are missing rows/columns based on that index
- Handle missings as required.




## Sort `DataFrame` by multiple columns

```python
df = df.sort(['col1','col2','col3'], ascending = [1,1,0])
```



## Display an image/Video in Jupyter

```python
from IPython.display import Image, display, SVG, YouTubeVideo

# To display an image
display(Image(url='http://history.nasa.gov/ap11ann/kippsphotos/5903.jpg'))

# To display an SVG
SVG(url='http://upload.wikimedia.org/wikipedia/en/a/a4/Flag_of_the_United_States.svg')

# To display a YouTube Video
YouTubeVideo('link-to-video')
```



## Replace multiple whitespaces with single

```Python
' '.join(mystring.split())
```



## Conditional Formatting on `DataFrame`

```python
# by columns
df.style.background_gradient(cmap='Greens')

# by rows
df.style.background_gradient(cmap='Greens', axis=1)
```


## Remove a file

````Python
import os
os.remove('path/to/file')
````



## JavaScript `PivotTables` in Jupyter

```Python
from pivottablejs import pivot_ui
pivot_ui(df)
```



## Creating Pipelines in `sklearn`

- Usual method of building a classification model using SVM involves
  - Standardization
  - Dimensionality Reduction
  - Model Fitting
  - Scoring

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = RandomizedPCA(n_components=10)

X_train_pca = pca.fit_transform(X_train_scaled)
svm = SVC(C=0.1, gamma=1e-3)
svm.fit(X_train_pca, y_train)

X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

y_pred = svm.predict(X_test_pca)
accuracy_score(y_test, y_pred)
```

- Using Pipelines, this can be reduced to ...

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
 StandardScaler(),
 RandomizedPCA(n_components=10),
 SVC(C=0.1, gamma=1e-3),
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy_score(y_test, y_pred)
```



## Grid Search on `Pipeline`

```python
import numpy as np
from sklearn.grid_search import RandomizedSearchCV

params = {
 'randomizedpca__n_components': [5, 10, 20],
 'svc__C': np.logspace(-3, 3, 7),
 'svc__gamma': np.logspace(-6, 0, 7),
}

search = RandomizedSearchCV(pipeline, params, n_iter=30, cv=5)
search.fit(X_train, y_train)

print search.best_params_
print search.grid_scores_
```



## Fixing Locale Errors

This happens because `bash` loads locale settings automatically and they might conflict with Pandas' defaults (which change on upgrading)

- Open up the Terminal

- Run `open ~/.bash_profile`

- On the TextEdit file that opens up, append

   ```bash
   export LC_ALL=en_US.UTF-8
   export LANG=en_US.UTF-8
   ```

- Save and quit.

- Run `source ~/.bash_profile`




## Extract all numbers from a string

```python
import re
str_ = '25 is greater than -4'

re.findall(r'\d+', str_)
```



## Month as Text from Number

```python
df.my_date.dt.strftime('%b')
```



## Scatterplot with a Regression Line

```python
sns.regplot(x="X", y = "Y", data=df)
```


## Objects and Memory Locations

```python
x = 'hello'
memloc_1 = id(x)
print memloc_1
print x

x += 'world'
memloc_2 = id(x)
print memloc_2
print x

import ctypes
print ctypes.cast(memloc_1, ctypes.py_object).value
print ctypes.cast(memloc_2, ctypes.py_object).value
```



## Convert `TimeDelta` to `int`

```python
# taking the difference of two pd.datetime objects gives a timeDelta object
x_td = x['ts1'] - x['ts2'] 			

x_td_int = x_td/np.timedelta64(1, 'D')
```



## See: `np.allclose()`



## Running t-tests

```Python
from scipy.stats import ttest_ind

# Find all booleans/cateogorical variables with 2 levels
catg_and_bools = [x for x in df.columns if df[x].value_counts().count() == 2 and 'target' not in x]

# Test the catg_and_bools against the target
def runTtestAgainstTarget(df=DataFrame(), col='', numr=''):
    """
    Splits a numeric variable into 2 parts based on the values of a categorical
    Performs a ttest on the 2 parts
    """
    a = df.loc[df[col] == df[col].unique()[0], numr]
    b = df.loc[df[col] == df[col].unique()[1], numr]

    tstat, pval = ttest_ind(a, b)

    return {'Variable': col, 'T-statistic': tstat, 'P-value': pval}
```

```python
DataFrame(map(lambda x: runTtestAgainstTarget(df=df, col=x, numr='target'), catg_and_bools))
```



## Running ANOVA

```python
from scipy.stats import f_oneway

# Find all categorical columns with >2 levels
catg_3 = [x for x in df.dtypes[df.dtypes == object].index.tolist() \
          if df[x].value_counts().count() > 2 and 'target' not in x]

def runANOVA(df=DataFrame(), catg='', numr=''):
    """
    Run ANOVA on a Categorical IV and Numeric DV
    """
    grpd = df.groupby(catg)

    dict_anova_input = {k:grpd.get_group(k).loc[:, numr] for k in grpd.groups.keys()}
    anova_res = f_oneway(*dict_anova_input.values())

    return {'Variable': catg, 'F-value': anova_res[0], 'P-value': anova_res[1]}
```

```python
DataFrame(map(lambda x: runANOVA(df=df, catg=x, numr='target'), catg_3))
```



## Using `where` to filter a series

The lambda function returns the value of the series where condition == True, and returns `NaN`s where False. We chain a `dropna()` so that the output contains only those values in the Series that satisfy the condition.

```python
pd.Series.where(lambda x: condition).dropna()
```



## `bisect_left` to find the insertion point in a sorted array

```python
from bisect import bisect_left
import numpy as np

x = np.random.randint(0, 50, 15)
# array([29, 10, 18, 27,  3, 32, 22, 32, 41, 34, 30, 30, 24, 24, 20])

x.sort()
# array([ 3, 10, 18, 20, 22, 24, 24, 27, 29, 30, 30, 32, 32, 34, 41])

# The number fifteen would be inserted between
x[bisect_left(x, 15)] # 18
# and
x[bisect_left(x, 15) - 1] # 10

# Can also be used for quickly searching a number in an array



```



## `ParseError` when importing large files into Pandas

- Cause 1: too many columns in a particular row
- Remedies
    - Extract a portion of the file (say 250 rows) and find the number of columns and specify `usecols=` with `read_csv`
    - Use `error_bad_lines=False` to _skip_ lines with too many fields

- Cause 2: an EOF character encountererd in a row
- Remedy
    - Use `quoting=csv.QUOTE_NONE` when calling `read_csv`




## Run bash commands on Python, receive output in a DataFrame

```Python
import subprocess as sbp
from io import StringIO
import pandas as pd

run_on_bash = \
lambda i: sbp.check_output("{}".format(i), shell=True).decode('utf-8').strip()

result_ = run_on_bash("""
seq 1 20 | awk 'BEGIN{OFS=","}{print rand(),rand(),rand()}'
""")

pd.read_csv(StringIO(result_), header=None)
```

## Using Python-env with Hydrogen on Atom

```bash
# create env, activate
conda create -n py3-env python=3.6 pandas scikit-learn seaborn 
source activate py3-env

# pass it to ipython
python -m ipykernel install --user --name py3-env
```

After you run these two lines and restart Atom, you will be prompted to choose which environment you use when evaluating code using Hydrogen.


## Add/Remove Kernels to IPython/Jupyter (also for use with Atom)

```bash
# see installed kernels
jupyter kernelspec list

# remove a kernel
jupyter kernelspec remove <kernel-to-drop>
```

## Installing packages from a running Notebook

```python
import sys

# Conda Packages
!conda install --yes --prefix {sys.prefix} numpy

# pip Packages
!{sys.executable} -m pip install numpy
```

Aside // Even when installing packages with pip inside an environment, it is better to write `python -m pip install <package>` than `pip install <package>`.

## `.translate()` in Python3

The translate method now requires a table mapping ordinals to be replaced to the replacements. This table could be a dictionary mapping Unicode ordinals (to be replaced) to Unicode ordinals, strings, or None (replacements).
The following code can be used to remove all punctuation:

```python
import string
remove_ = str.maketrans(dict.fromkeys(string.punctuation))
cleaned_str = dirty_str.translate(remove_)
```

## `ModuleNotFoundError` when running scripts that use local modules

Python tries to look for modules on the system path and raises this error when it doesn't find it. To solve this, update the path as:

```python
import os
import sys

if __name__ == '__main__':
	# Set project root to the path where the local module exists
	PROJECT_ROOT = os.getcwd()
	
	# Add it to the system path
	sys.path.append(PROJECT_ROOT)
	
	# Now you can import anything from the local module
	from foo.bar import baz
```


## Setting Formats for Charts

```python
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

plt.style.use('seaborn-white')
sns.set_context("talk")

rcParams['figure.figsize'] = 12, 5 
rcParams['font.family'] = 'sans-serif'

font_title = {
    'size': 18, 
    'weight': 600, 
    'name': 'Georgia'
}

font_axes = {
    'size': 16, 
    'weight': 500, 
    'name': 'Georgia'
}

font_text = {
    'size': 14, 
    'weight': 300, 
    'name': 'Verdana'
}

%pylab inline
```

## Making Publication Quality Charts

```python
# set up the canvas, draw the plot
fig, ax = plt.subplots(figsize=(10, 5))
df_.plot.barh(ax=ax)

TITLE_ = "Lorem Ipsum"
XLABEL_ = "Lorem Ipsum"
YLABEL_ = "Lorem Ipsum"
VLINE_AT = 0
VLINE_X_LOW = 
VLINE_X_HIGH = 
INSIGHT_X = 
INSIGHT_Y =


# Title, Labels
ax.set_title("{}\n".format(TITLE_), fontdict=font_title);
ax.set_xlabel("\n {}".format(XLABEL_), fontdict=font_axes)
ax.set_ylabel("{}\n".format(YLABEL_), fontdict=font_axes)

# Draw a vertical line, annotate it
ax.vlines(VLINE_AT, VLINE_X_LOW, VLINE_X_HIGH, color='green', linestyle='dashed', alpha=0.4)
ax.annotate(s="Avg. Survival %", xy=(VLINE_AT - 0.095 , XHIGH-2), color='darkgreen', fontsize=12)

# Add grid, remove spines
ax.grid(True, linestyle=":", alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(INSIGHT_X, INSIGHT_Y, 
        """Almost half the people on the ship were \nin the 20-29 age group.
        \nApparently, this group was the least fortunate
        """, 
        fontdict=font_text)

plt.savefig("reports/figures/{}.png".format("-".join(TITLE_.split(" "))), bbox_inches='tight', pad_inches=0.5)
```
