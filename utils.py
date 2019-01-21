from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_df(df, filename):
    df.to_pickle(filename)

def load_df(filename):
    return pd.read_pickle(filename)

def linear(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def exponential(x, a, b, c):
    return a * np.log(-b*x) + c

def plot_bestfit(func, x, y):
    popt, pcov = curve_fit(func, x, y)
    plt.plot(x, func(x, *popt), '-')
    plt.show()

def simple_correlation(df, xmetric, ymetric, agg=False):
    df = df[[xmetric, ymetric]]
    if agg:
        grouped = df.groupby(xmetric).agg(np.mean)
    else:
        grouped = df.groupby(xmetric)
    return grouped.index, grouped


def scatter_2d(x, y, xlabel="x", ylabel="y", bestfit=False, func=None):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not bestfit:
        plt.show()
    else:
        plot_bestfit(func, x, y.values.flatten())
