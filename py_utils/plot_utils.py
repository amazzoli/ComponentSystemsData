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


def _sort_and_clamp_series(series, n_classes='all'):
    series = series.sort_values(ascending=False)
    if type(n_classes) == int:
        counts_left = np.sum(series[n_classes:])
        n_class_left = len(series[n_classes:])
        series = series[:n_classes]
        series['other '+str(n_classes)+' classes'] = counts_left
    return series
