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

## Use Conditional Formatting on a pandas.DataFrame

```python
# by columns
df.style.background_gradient(cmap='Greens')

# by rows
df.style.background_gradient(cmap='Greens', axis=1)
```

## JavaScript Pivot Tables

```python
from pivottablejs import pivot_ui
pivot_ui(df)
```
