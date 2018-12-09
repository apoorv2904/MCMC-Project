
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from collections import defaultdict
from scipy.stats import norm
import time
sns.set_style('white')
sns.set_context('talk')

def set_seed( seed ):
    np.random.seed(seed)
      

def plot_schedule( b_vector, title=""):
    """Plots temperature and beta schedule over time."""
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(14,5))
    ax1.plot(np.arange(len(b_vector)), b_vector )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Beta Value")
    ax1.set_title( "Beta Schedule")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Temperature")
    ax2.set_title( "Temperature Schedule")
    ax2.plot(np.arange(len(b_vector)), 1.0/b_vector )
    plt.suptitle(title)
    
    
def plot_error(e_vectors, x_axis_vector=None, title="", legend=[],
               xlabel="Time"):
    """Plots error over time (or the values in x_axis_vector). Inputs are
    lists of vectors to allow comparisons."""
    if x_axis_vector is None:
        x_axis_vector = np.arange(len(e_vector))
    if not legend:
        legend = [" "] * len(h_vectors)
    plt.figure(figsize=(10,5))
    plt.hlines([0.1, 0.2, 0.3, 0.4], x_axis_vector[0], x_axis_vector[-1],
               linestyles="--", color="lightgray")
    for i, e_vector in enumerate(e_vectors):
        plt.plot(x_axis_vector, e_vector, label=legend[i])
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Error")
    plt.ylim(0, 1)

    
def plot_energy_error(h_vectors, e_vectors, title="", legend=[]):
    """Plots energy and error over time. Inputs are lists of vectors to allow
    comparisons."""
    show_legend = True
    if not legend:
        legend = [" "] * len(h_vectors)
        show_legend = False
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(20,4))
    ax1.set_title(title)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Energy")
    ax1.set_yscale("log")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Error")
    ax2.set_ylim(0, 1)
    for y in [0.1, 0.2, 0.3, 0.4]:
        ax2.axhline(y, ls="--", color="lightgray")
    for i, (h_vector, e_vector) in enumerate(zip(h_vectors, e_vectors)):
        ax1.plot(np.arange(len(h_vector)), h_vector, label=legend[i])
        ax2.plot(np.arange(len(e_vector)), e_vector, label=legend[i])
    if show_legend:
        ax1.legend()
        ax2.legend()
      
    
def plot_acceptance_trend( a_vector, s_vector, title=""):
    """Plots acceptance probabilities and binary acceptance variable over time."""
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(20,4))
    ax1.plot(np.arange(len(s_vector)), s_vector,'.')
    ax1.set_title(title)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Acceptance Result")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Acceptance Probability")
    for y in [0.2, 0.4, 0.6, 0.8]:
        ax2.axhline(y, ls="--", color="lightgray")
    ax2.plot(np.arange(len(a_vector)), a_vector, '.')
    
    
def plot_beta_schedule( b_vector ):
    plt.figure(figsize=(10,4))
    plt.plot(np.arange(len(b_vector)),b_vector)
    plt.xlabel("Time")
    plt.ylabel("Beta")

