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
    'weight': 200, 
    'name': 'Georgia'
}

font_axes = {
    'size': 16, 
    'weight': 200, 
    'name': 'Verdana'
}

font_text = {
    'size': 14, 
    'weight': 200, 
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
# Can also use ax.set(title=, ...)
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
ax.legend(bbox_to_anchor=(1.05, 1))

ax.text(INSIGHT_X, INSIGHT_Y, 
        """Almost half the people on the ship were \nin the 20-29 age group.
        \nApparently, this group was the least fortunate
        """, 
        fontdict=font_text)

plt.savefig("reports/figures/{}.png".format("-".join(TITLE_.split(" "))), bbox_inches='tight', pad_inches=0.5)
```

## Directory Sizes in UNIX

```bash
du -h -d 1 [PATH]
```

- the `-h` switch returns human-readable file sizes (in MB)
- the `-d` switch limits the depth of directories to 1

## Check available (free) RAM in Unix

```bash
free -m
```

## Run Jupyter in the background

```bash
jupyter lab --allow-root --ip 0.0.0.0 --port 8008 &> /dev/null &

# get the token
jupyter notebook list

# stop a kernel (using the port number)
jupyter notebook stop --NbserverStopApp.port=8008
```

## Copy a file into a running docker container

```bash
# get the container id
docker ps -a

# copy file 
docker cp [SRC] container_id:[DEST]
```

## Format floats for print()

```python
print("Dataset has {:.2f}% duplicate records.".format(100*(df_.shape[0] - df_.drop_duplicates().shape[0])/df_.shape[0]))
```

## Parallelize for-loops

```python
import multiprocessing
from sklearn.externals.joblib import delayed, Parallel

CPUs = multiprocessing.cpu_count()

result = \
Parallel(n_jobs=CPUs, verbose=3)(
    delayed(my_function)
    (*args, **kwargs) for i in iterable
)

# example
# Parallel(n_jobs=2)(
# 	delayed(sqrt)
#	(i) for i in [4, 9, 16, 25, 36, 49]
# )
```

## Download a Youtube Playlist as MP3 files

```bash
# brew install ffmpeg
# brew link ffmepg
youtube-dl --extract-audio --audio-format mp3 -o "%(title)s.%(ext)s" [youtube-playlist-tag]
```

## Suppress Warnings

```python
import warnings
warnings.simplefilter('ignore', category=FutureWarning)
```

## Read SPSS files (.sav) into Python via R

```python
# Dependencies
!conda install -y -c r rpy2 r-foreign

sav_file = "path/to/sav"
csv_file = "path/to/csv"

from rpy2.robjects.packages import importr
pkg_foreign = importr('foreign')

r_df = pkg_foreign.read_spss(sav_file, to_data_frame=True, reencode=True)
r_df.to_csvfile(csv_file)
```

## Round integers to nearest 10 (useful with Binning) with `round(x, -1)`

```python
numeric_col = 'foo'
num_bins = 5
LL = 0
UL = round(df.loc[: 'numeric_col'].max(), -1)


BINS = np.linspace(LL, UL, num_bins)
LBLS = [f"{str(i).zfill(2)}-to-{str(j).zfill(2)}" for i, j in zip(BINS[:-1], BINS[1:])]

srs_binned = df.loc[:, numeric_col].pipe(pd.cut, bins=BINS, labels=LBLS)
srs_binned.value_counts(normalize=True)
```
## Chart Formatting v1.0

```python
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

from ipywidgets import interactive, Dropdown

sns.set_style("whitegrid")
sns.set_context("talk")

plt.rcParams['figure.figsize'] = 12, 5 
plt.rcParams['font.family'] = 'monospace'

%matplotlib inline

def format_plot(ax):
    """
    Modify the font dicts for the design language you're following
    
    Titles and axes labels 
    - should be set before calling this function
    - here they will be spaced away from the chart
    
    Text, annotations etc. should also be set before 
    See: ax.annotate, ax.text, ax.vlines, ax.hlines ...
    """
    font_title = {
        'size': 20, 
        'weight': 600, 
        'name': 'monospace'
    }

    font_axes = {
        'size': 14, 
        'weight': 'bold', 
        'name': 'monospace'
    }

    ax.grid(True, linestyle=":", alpha=0.6)
    sns.despine(ax=ax, left=True)

    if ax.get_legend():
        ax.legend(bbox_to_anchor=(1.1, 1))
    
    ax.set_title(f"\n\n{ax.get_title().title()}\n", fontdict=font_title)
    ax.set_xlabel(f"\n{ax.get_xlabel().title()} ➞", fontdict=font_axes)
    ax.set_ylabel(f"{ax.get_ylabel().title()} ➞\n", fontdict=font_axes)
```    

## Change number format for plot axes

```
# Get thousands separator 
ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
```

## Store pwd as hidden file, get it on clipboard

```bash
cat ~/.pwds/.gw | tr -d '\n' | pbcopy
```

## Make seaborn `annot` text larger

```python
sns.set(font_scale=1)
sns.heatmap(df_, annot=True)
```
## Limit output to a few rows when printing a DataFrame

```python
pd.set_option("display.max_rows", 20)
```

## Handy Aliases

- Put these in your `.bashrc` or `.zshrc`

```bash
alias cel='conda env list'

alias cact='function _f0(){conda activate $1;};_f0'
# usage: cact dataviz

alias cdeac='conda deactivate'

alias jlab='function _f1(){jupyter lab --no-browser --port=$1 &> /dev/null & ;};_f1'
# usage: jlab 8800

alias jnl='jupyter notebook list'

alias jstop='function _f2(){jupyter notebook stop --NbserverStopApp.port=$1;};_f2'
# usage jstop 8800
```

## Setting up ZSH + iTerm2

- Follow https://medium.freecodecamp.org/jazz-up-your-zsh-terminal-in-seven-steps-a-visual-guide-e81a8fd59a38
- Customize Agnoster https://github.com/agnoster/agnoster-zsh-theme
- Remove machine name from prompt https://stackoverflow.com/questions/28491458/zsh-agnoster-theme-showing-machine-name

## Remove redundant (and annoying) files

```bash
find . -name '*.pyc' -delete
find . -name '__pycache__' -delete
```

## Using Hydrogen with Atom to run Python code in-line like Jupyter

```
conda activate my-env
conda install ipykernel
python -m ipykernel install --user
```

## Using DataClasses

- code generator that removes boilerplate from Class definitions
- essentially still a flexible data and methods holder
- use the `@dataclass` decorator with params
- by default, it is 
	- mutable (and hence unhashable)
	- unordered
	- but, these defaults can be overridden using the `frozen` and `order` params

```python
# Example 1
from data class import dataclass, asdict, astuple, replace, field
@dataclass
class Color:
	hue: int
	saturation: float
	lightness: float = 0.5
	
c = Color(120, 4.2)
c
c.lightness
replace(c, hue=120)
asdict(c)

# Example 2
@dataclass(frozen=True, order=True)
class Color:
	hue: int
	saturation: float
	lightness: float = 0.5

list_colors = [Color(10, 2.0), Color(1, 90.5), Color(120, 45.2), Color(45, 77.8), Color(45, 77.8)]
sorted(list_colors)
set(list_colors) # will ignore dups
```

## Using Jinja2 Templates

- Templating language
- In a nutshell, fills placeholders with text
- Templates are text files with `{{a-variable}}`
- We fill or `render` the template by passing data to `variables`
- Basic Usage

```bash
mkdir templates
touch templates/namaste.txt
echo 'Namaste, {{name}}!' > namaste.txt
```

```python
from jinja2 import Environment, FileSystemLoader, Template
fsl = FileSystemLoader('./templates')
env = Environment(loader=fsl)
template_1 = env.get_template('namaste.txt')
template_1.render(name='Dushyant')

# or 
template_2 = Template("Hello {{name}}!")
template_2.render(name='Dushyant')
```
- Conditionals and Loops

```
# Conditionals
{% if x > 0 %}
	X is positive
{% else %}
	X is Negative
{% endif %}

# Looping
{% for color in colors %}
	{{ color }}
{% endfor %}	
```

- Templates can be combined
	- If you have `header.html, body.html`, you can call them in an `index.html` with `include`

```
<HTML>
	{% include 'header.html' %}
	<BODY>
	{% include 'body.html' %}
	</BODY>
</HTML>
```

## Create combinations with `itertools`

- useful to generate search grids or test-cases 

```python
from itertools import product, permutations, combinations

for t in product('ABC', 'DE', 'xyz'):
	print(t)
	
# Generate nPm
for t in permutations('HELLO', 2):
	print(t)

# Generate nCm
for t in combinations('HELLO', 2):
	print(t)
```


## Writing robust Python code

- Write docstrings with examples, check with `doctest`
- Detect common errors with `pyflakes`
- Add type checking with `mypy`
- Run unit tests with `pytest` and `hypothesis`

```bash
python3 -m pyflakes my-module.py
mypy my-module.py
pytest my-module.py
```

## Convert 1D numpy array to 2D

```python
x = np.linspace(start=0, stop=1, num=10**3)
X = x[:, np.newaxis]
print(x.ndim, X.ndim)
```

## Jupyter Widgets: Passing fixed arguments to function under `interact`

- An interactive viz. is created as `interact(func, par_1=widget_1, ...)`
- Here, func usually takes values from the widgets like Dropdowns or Sliders
- Sometimes, we want to pass values not taken from widgets (ex. a dataframe or a sqlite connection)
- Then, we use the `fixed` pseudo-widget

```python
from ipywidgets import Dropdown, Slider, interactive, fixed

def f(x, y, kwargs):
    '''
    '''
    result = ... kwargs['a'] ...
    return result
    
interactive(f,
	    x=Dropdown(...),
	    y=Slider(...),
	    kwargs=fixed({'a': a, 'b': b}))
```

## Using `scp` to transfer data between local machine and remote

```bash
scp -r path-to-source path-to-destination
```

- path to a remote directory will be built as USERNAME@REMOTE-MACHINE-IP:path/to/directory

## Exiting a program during execution

```python
import os
import sys

if not os.path.exists('config.ini'):
    print('Could not find config file.\nPlease ensure config.ini exists and try again.')
    sys.exit()
```

## Find files and perform actions

- Use the bash `find` command
- Syntax: `find DIRECTORY -type xxx -size xxx -iname "xxx" -exec xxx {} \;`

```bash
find ./src -type f -name "*.pyc" -exec rm -f {} \; 
```

## Start Jupyter Lab, direct output to file

- works for any command (syntax: `nohup <command> &`)
- output is redirected to a file `nohup.out`

```bash
nohup jupyter lab --no-browser --allow-root --ip "*" --port 1508 &
```

## Regex over pd.Series

- The following pattern will keep only alphabets and numbers
- Removes all other/special characters

```python
pd.Series(['abcd-1234-##$', 'wxyz-7890-@@&']).replace('[^a-z|0-9]', '', regex=True)
```

## Flatten column names in a Pandas MultiIndex

```python
df.columns = ['_'.join(col).strip() for col in df.columns.values]
```

## Multiprocessing

```python
import multiprocessing

def my_func():
    pass
    
pool = multiprocessing.Pool()
pool.map(my_func, list_of_inputs)
```

## Run a function over a Pandas GroupBy object in parallel using Multiprocessing

```python
from multiprocessing import Pool

def aggregate():
    '''
    Reduces a GroupBy group into a row
    '''
    pass
    
## list(df.groupby('KEY')) returns a list of tuples like (key, group)
## pool.map() --> returns a list 
    
with Pool() as pool:
    list_result = \
    pool.map(func=aggregate,
             iterable=[x[1] for x in list(df.groupby("KEY"))])

df_aggregated = pd.concat(list_result, axis=1).T
```

## Postgres + Pandas

```python
# Get image
docker pull postgres

# Start a container
# see: https://docs.docker.com/samples/library/postgres/
# options: POSTGRES_USER (default: postgres), POSTGRES_DB (default: postgres), PGDATA (to defined a directory for database files)
docker run --rm -d --name psql \
                   -e POSTGRES_PASSWORD=pg_docker_01 \
		   -p 5432:5432 \
		   -v $(PWD):/var/lib/postgresql/data \
		   postgres

# Requirements
pip install pandas==0.24.2
pip install psycopg2-binary

# Connect to the DB
from sqlalchemy import create_engine
engine = create_engine("postgresql://postgres:pg_docker_01@localhost:5432/postgres")

# Read data from text files, load into DB
df = pd.read_csv(...)
df.to_sql(
    'table_name', 
    con=engine, 
    if_exists='replace', 
    index=False, 
    method='multi', 
    chunksize=10**5
)

# PS: This should work for DataFrames under 10M rows.
# For larger dataframes, try loading incrementally
# See - https://stackoverflow.com/questions/23103962/how-to-write-dataframe-to-postgres-table (Pandas 0.24+ Solution)
```
