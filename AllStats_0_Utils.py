import os, psutil, gc, glob, pdb, sys, functools
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import pandas as pd
import statsmodels as sm
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.nonparametric.kde import KDEUnivariate
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'image.cmap': 'gray'})
mpl.style.use('fivethirtyeight')
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 6
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['figure.subplot.bottom'] = 0.3
mpl.rcParams['axes.unicode_minus'] = False

p = psutil.Process(os.getpid())
# p.nice(psutil.HIGH_PRIORITY_CLASS)

vslice = np.vectorize(slice)

def dkw_conf_bands(ecdf_data, alpha = 0.05):
    
    alpha = alpha
    
    epsilon = np.sqrt(np.log(2./alpha) / (2 * (ecdf_data.n - 1)))
    
    lower = np.clip(ecdf_data.y - epsilon, 0, 1)
    
    upper = np.clip(ecdf_data.y + epsilon, 0, 1)
    
    return lower, upper

def distro_plots(data, ecdf_data, cdf_data = None):
    
    lower, upper = dkw_conf_bands(ecdf_data)
    
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))

    ax1.hist(data)
    ax1.set_title('histogram', size = 12)

    ax2.plot(ecdf_data.x, ecdf_data.y, label = 'ECDF',  c = 'r')
    ax2.set_title('CDF confidence bands', size = 12)
    
    if cdf_data is not None:
        ax2.plot(ecdf_data.x, cdf_data, label = 'CDF',  c = 'g')
        
    ax2.plot(ecdf_data.x, lower, label = 'L_n', color = 'b', alpha = 0.55)
    ax2.plot(ecdf_data.x, upper, label = 'U_n', color = 'b', alpha = 0.55)
    ax2.legend(loc = 0, fontsize = 12)
    
    for ax in [ax1, ax2]:
        ax.tick_params(axis = 'x', labelsize = 12)
        ax.tick_params(axis = 'y', labelsize = 12)

def exp_params(nsamples, min_nsamples):
    
    nsamples_scale = np.log10(nsamples).astype(np.int64)
    
    nsamples_step = max(50, int(10**(nsamples_scale//2)))
    
    exp_sample_sizes = np.unique(np.linspace(min_nsamples, nsamples, nsamples_step).astype(np.int64))
    
    exp_initial = np.zeros_like(exp_sample_sizes)
    
    exp_slices = vslice(exp_initial, exp_sample_sizes)

    return exp_sample_sizes, exp_slices

def corrcoeff(x,y):
    
    x_mc = x - x.mean()
    y_mc = y - y.mean()
    
    nx_std = np.sqrt((x_mc**2).sum())
    ny_std = np.sqrt((y_mc**2).sum())
    
    cov = (x_mc*y_mc).sum()
    
    corr = cov/(nx_std*ny_std)
    
    return corr

def boot_corrcoeff(x, y, sample_axis = 1):
    
    x_repmean = np.repeat(x.mean(axis = sample_axis).reshape(-1,1), x.shape[sample_axis], axis = sample_axis)
    y_repmean = np.repeat(y.mean(axis = sample_axis).reshape(-1,1), y.shape[sample_axis], axis = sample_axis)
    
    x_mc = x - x_repmean
    y_mc = y - y_repmean
    
    nx_std = np.sqrt((x_mc**2).sum(axis = sample_axis))
    ny_std = np.sqrt((y_mc**2).sum(axis = sample_axis))
    
    cov = np.diag(np.tensordot(x_mc, y_mc, axes = ([sample_axis], [sample_axis])))
    
    corr = cov/(nx_std*ny_std)
    
    return corr

def exp_mean(exp_data, slices):
    
    boot_mean = []
    
    for sl in slices:
        
        mean_sl = np.mean(exp_data[sl], axis = 0)
        
        boot_mean.append(mean_sl)
    
    boot_mean = np.asarray(boot_mean)
    
    return boot_mean

def exp_skew(exp_data, slices):
    
    boot_skew = []
    
    for sl in slices:
        
        skew_sl = sp.stats.skew(exp_data[sl], axis = 0)
        
        boot_skew.append(skew_sl)
    
    boot_skew = np.asarray(boot_skew)
    
    return boot_skew

def exp_perc_diff(exp_data, slices):
    
    boot_reps = []
    
    for sl in slices:
        
        perc_vec = np.percentile(exp_data[sl], [25., 75.], axis = 0)
        perc_diff = (perc_vec[-1] - perc_vec[0])/1.34
        
        boot_reps.append(perc_diff)
    
    boot_reps = np.asarray(boot_reps)
    
    return boot_reps

def exp_max(exp_data, slices): 
    
    boot_reps = []
    
    for sl in slices:
        
        max_sl = np.amax(exp_data[sl], axis = 0)
        
        boot_reps.append(max_sl)
    
    boot_reps = np.asarray(boot_reps)
    
    return boot_reps

def conf_interval_coverage(true_statistic, bounds):
    
    within_interval = (true_statistic > bounds[:, 0]) & (true_statistic < bounds[:, 1])
    
    coverage = within_interval.sum()/within_interval.shape[0]
    
    return coverage

def exp_coverages(true_statistic, bounds):
    
    boot_coverage = []
    
    for idx in np.arange(bounds.shape[0]):
        
        boot_coverage.append(conf_interval_coverage(true_statistic, bounds = bounds[idx]))

    boot_coverage = np.asarray(boot_coverage)
    
    return boot_coverage

def exp_intervals(exp_sample_sizes, bootstrap_estimate, plugin_estimate, true_statistic):
    
    bootstrap_stderr = bootstrap_estimate.std(axis = 1)
    bootstrap_25per, boostrap_975per = np.percentile(bootstrap_estimate, [2.5, 97.5], axis = 1)

    plugin_estimate = plugin_estimate.reshape(-1,1)

    z_5pc = np.sqrt(2)*sp.special.erfinv(1-0.05)
    normal_lbound = plugin_estimate - z_5pc*bootstrap_stderr
    normal_ubound = plugin_estimate + z_5pc*bootstrap_stderr
    normal_bounds = np.transpose(np.stack((normal_lbound, normal_ubound)), axes = [1,-1,0])
    normal_coverage = exp_coverages(true_statistic = true_statistic, bounds = normal_bounds)

    percentile_bounds = np.transpose(np.stack((bootstrap_25per, boostrap_975per)), axes = [1,-1,0])
    percentile_coverage = exp_coverages(true_statistic = true_statistic, bounds = percentile_bounds)

    pivot_lbound = 2*plugin_estimate - percentile_bounds[:,:,1]
    pivot_ubound = 2*plugin_estimate - percentile_bounds[:,:,0]
    pivot_bounds = np.transpose(np.stack((pivot_lbound, pivot_ubound)), axes = [1,-1,0])
    pivot_coverage = exp_coverages(true_statistic = true_statistic, bounds = pivot_bounds)

    bootstrap_intervals = [normal_bounds, percentile_bounds, pivot_bounds]
    bootstrap_intervals_names = ['normal_bounds', 'percentile_bounds', 'pivot_bounds']

    bootstrap_intervals = dict(zip(bootstrap_intervals_names, bootstrap_intervals))

    bootstrap_coverage = np.vstack([exp_sample_sizes, normal_coverage, percentile_coverage, pivot_coverage]).T

    bootstrap_coverage = pd.DataFrame(bootstrap_coverage, columns = ['no_samples', 'normal_coverage', 'percentile_coverage', 'pivot_coverage'])

    bootstrap_coverage['no_samples'] = bootstrap_coverage['no_samples'].astype(np.int64)
    bootstrap_coverage.set_index('no_samples', inplace = True)
    
    return bootstrap_intervals, bootstrap_coverage

def interval_plots(exp_sample_sizes, bootstrap_intervals, true_statistic, plugin_estimate, statistic_name):

    bootstrap_intervals_names = list(bootstrap_intervals.keys())
    
    fig, axes  = plt.subplots(1, len(bootstrap_intervals), figsize = (18,6), sharey = True)

    for idx, ax in enumerate(axes.flat):
        
        name = bootstrap_intervals_names[idx]
        bounds = bootstrap_intervals[name]
        
        ax.plot(exp_sample_sizes, np.median(bounds, axis = 1)[:,0], color = 'blue', label = 'interval bands')
        ax.plot(exp_sample_sizes, np.median(bounds, axis = 1)[:,1], color = 'blue')
        ax.plot(exp_sample_sizes, plugin_estimate, color = 'green', label = 'plugin %s' %(statistic_name))

        ax.axhline(true_statistic, color = 'red', label = 'true %s' %(statistic_name))
        
        ax.set_title(name, size = 12)
        ax.legend(loc = 1, fontsize = 12)
        ax.tick_params(axis='x', labelsize = 12)
        ax.tick_params(axis='y', labelsize = 12)

def coverage_plots(bootstrap_coverage):

    bootstrap_coverage_names = bootstrap_coverage.columns

    fig, axes = plt.subplots(1 ,bootstrap_coverage_names.size, figsize = (18,6), sharey = True)

    for idx, ax in enumerate(axes.flat):

        bootstrap_coverage[bootstrap_coverage_names[idx]].plot(ax = ax)
        ax.set_title(bootstrap_coverage_names[idx], size = 12)
        ax.tick_params(axis='x', labelsize = 12)
        ax.tick_params(axis='y', labelsize = 12)

def bootstrap_estimate_distro(exp_sample_sizes, bootstrap_estimate, true_statistic, plugin_estimate, statistic_name):

    lnsp_exp_idx = np.linspace(0, bootstrap_estimate.shape[0] - 1, 6).astype(np.int64)
    
    fig, axes  = plt.subplots(lnsp_exp_idx.size//3 ,lnsp_exp_idx.size//2, figsize = (18,12), sharey = True, sharex = True)
    
    for exp_idx, ax in enumerate(axes.flat):
        
        ax.hist(bootstrap_estimate[lnsp_exp_idx[exp_idx]].flatten(), density = True)
        ax.set_title('sample size = ' + str(exp_sample_sizes[lnsp_exp_idx[exp_idx]]), size = 12)
        
        ax.axvline(true_statistic, color = 'red', label = 'true %s' %(statistic_name))
        ax.axvline(plugin_estimate[exp_idx], color = 'green', label = 'plugin %s' %(statistic_name))
        ax.legend(loc = 0, fontsize = 12)
        ax.tick_params(axis='x', labelsize = 12)
        ax.tick_params(axis='y', labelsize = 12)

class Max_uniform(sp.stats.rv_continuous):
    def _pdf(self, x, theta, nsamples):
        self.theta = theta
        self.nasampes = nsamples
        return nsamples*x**(nsamples - 1)/theta**nsamples
    def _cdf(self, x, theta, nsamples):
        self.theta = theta
        self.nasampes = nsamples
        return x**nsamples/theta**nsamples