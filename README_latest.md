## Apply Linear Regression to each column of a DF and plot slope on line chart

```python
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

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
        ax.legend(loc='best', bbox_to_anchor=(0, 1.15), prop={'size': 12})
    
    ax.set_title(f"\n{ax.get_title()}\n", fontdict=font_title)
    ax.set_xlabel(f"\n{ax.get_xlabel()} ➞", fontdict=font_axes)
    ax.set_ylabel(f"{ax.get_ylabel()} ➞\n", fontdict=font_axes)
    
    # Format x- and y- axis ticks
    if isinstance(pd.Series(ax.get_yticks().tolist()).mean(), float) or isinstance(pd.Series(ax.get_yticks().tolist()).mean(), int):
        if pd.Series(ax.get_yticks().tolist()).count() > 1:
            ax.set_yticklabels([f"{i:.0%}" if i <= 1 else f"{int(i):,d}" for i in ax.get_yticks().tolist()])
        
    if isinstance(pd.Series(ax.get_xticks().tolist()).mean(), float) or isinstance(pd.Series(ax.get_xticks().tolist()).mean(), int):
        if pd.Series(ax.get_xticks().tolist()).count() > 1:
            ax.set_xticklabels([f"{i:.0%}" if i <= 1 else f"{int(i):,d}" for i in ax.get_xticks().tolist()])


def fit_linear_model(srs_y):
    """
    Apply LinearRegression to a single time-series
    Return beta, intercept, actual, fitted
    """
    df_linear = \
    pd.DataFrame({
        'y': srs_y.values.tolist(),
        'x': list(range(1, len(srs_y) + 1))
    }).fillna(0)

    lr = LinearRegression()
    lr.fit(df_linear[['x']], df_linear['y'])

    y_pred = \
    pd.Series(
        data=lr.predict(df_linear[['x']]),
        index=srs_y.index,
        name=f"{srs_y.name} trend"
    )

    result = {
        'beta': lr.coef_[0],
        'intercept': lr.intercept_,
        'y_pred': y_pred,
        'y_actual': srs_y
    }
    
    return result

def fit_linear_model_to_df(fr):
    """
    Apply LinearRegression to each Series in a DataFrame
    Return: Predicted as DataFrame; Beta, Intercepts as Series
    """
    dict_lm = fr.apply(lambda col: fit_linear_model(col.dropna()))
    
    fr_fitted = \
    (pd.Series(dict_lm)
     .map(lambda i: i.get('y_pred'))
     .pipe(lambda srs: pd.concat(srs.tolist(), axis=1)))
    
    srs_betas = pd.Series({k:dict_lm.get(k).get('beta') for k in dict_lm.keys()})
    srs_intercepts = pd.Series({k:dict_lm.get(k).get('intercept') for k in dict_lm.keys()})
    
    result = {
        'fitted': fr_fitted,
        'betas': srs_betas,
        'intercepts': srs_intercepts
    }
    
    return result

def plot_lm(fr, layout, figsize, ylim, ylabel, xloc, perc=None):
    """
    """
    dict_lm = fit_linear_model_to_df(fr=fr)
    
    axs = \
    (fr
     .pipe(lambda fr: fr.loc[:, fr.mean().sort_values().index])
     .plot(subplots=True, 
           linewidth=3,
           sharey=True, 
           sharex=True, 
           ylim=ylim,
           layout=layout,
           figsize=figsize))
            
    for ax in axs.flatten():
        ax.set_ylabel(ylabel)
        if bool(ax.get_legend_handles_labels()[1]):       
            label = ax.get_legend_handles_labels()[1][0]    
            dict_lm['fitted'].loc[:, f"{label} trend"].plot(ax=ax, linestyle='dashed', color='k', linewidth=2)
            
        if bool(perc):
            ax.set_yticklabels([f'{x:.0%}' for x in ax.get_yticks().tolist()])
            ax.annotate(text=f"m={dict_lm['betas'][label]:.2%}", xy=(xloc, 1.1 * dict_lm['fitted'].loc[:, f"{label} trend"].max()), fontsize=12)
        else:
            ax.set_yticklabels([f'{int(x):,d}' for x in ax.get_yticks().tolist()])
            ax.annotate(text=f"m={int(dict_lm['betas'][label]):,d}", xy=(xloc, 1.1 * dict_lm['fitted'].loc[:, f"{label} trend"].max()), fontsize=12)
        format_plot(ax)    
```
