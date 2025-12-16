import numpy as np
import matplotlib.pyplot as plt


def plot_sorted_bar(ax, series, n_classes='all', **bar_kwargs):
    """
    Bar plot of a count series that sort the columns in descending order.
    If n_classes is set as an integer it clamps the plot to the top n_classes
    most populated classes and show a final bar with the remaining counts.
    """
    series = _sort_and_clamp_series(series, n_classes)
    series.plot(kind='bar', ax=ax, **bar_kwargs)
    ax.set_xlabel('')
    return ax
        
    
def plot_count_rank(ax, counts, xlabel='rank', ylabel='count', loglog=True, **scatter_kwargs):
    """
    Ranked statistics of the elements in "frequencies". 
    """
    sort_i = np.argsort(counts)[::-1]
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.scatter(np.arange(len(counts))+1, counts[sort_i], **scatter_kwargs)
    return ax


def plot_count_hist_nobin(ax, values, xlabel='', ylabel='', loglog=True, **scatter_kwargs):
    """
    Plotting a histogram of counts of repeating elements in "values" without binning. 
    The x-axis are the integer values.
    """
    uni_c, count_c = np.unique(values, return_counts=True)
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if xlabel != '':
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel != '':
          ax.set_ylabel(ylabel, fontsize=12)
    ax.scatter(uni_c, count_c, **scatter_kwargs)
    return ax
    
    
### AUXILIARY FUNCTIONS

def _sort_and_clamp_series(series, n_classes='all'):
    series = series.sort_values(ascending=False)
    if type(n_classes) == int:
        counts_left = np.sum(series.iloc[n_classes:])
        n_class_left = len(series.iloc[n_classes:])
        series = series.iloc[:n_classes]
        series['other '+str(n_classes)+' classes'] = counts_left
    return series
