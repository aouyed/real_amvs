#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:29:41 2022

@author: aouyed
"""

import xarray as xr
import pandas as pd
import numpy as np 
import plotter 
import matplotlib.pyplot as plt

from scipy import stats


def big_histogram(ds, column_x, column_y, xedges, yedges, bins=100):
    #xedges = [np.inf, -np.inf]
    #yedges = [np.inf, -np.inf]
    
        
    #xedges[0] = np.minimum(ds[column_x].min().item(), xedges[0])
    #xedges[1] = np.maximum(ds[column_x].max().item(), xedges[1])
    
    #yedges[0] = np.minimum(ds[column_y].min().item(), yedges[0])
    #yedges[1] = np.maximum(ds[column_y].max().item(), yedges[1])
    
    xbins = np.linspace(xedges[0], xedges[1], bins+1)
    ybins = np.linspace(yedges[0], yedges[1], bins+1)
    heatmap = np.zeros((bins, bins), np.uint)
    
    df=ds[[column_x, column_y]].to_dataframe().reset_index().dropna()
        
    heatmap, _, _ = np.histogram2d(
                df[column_x].values, df[column_y].values, bins=[xbins, ybins])
      
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent



def binned_statistics(ds, column_x, column_y, bins=20):
    df=ds[[column_x, column_y]].to_dataframe().reset_index().dropna()
    binned_values=stats.binned_statistic(df[column_x].values,  df[column_y].values, 'mean', bins=bins)
    
    return binned_values

def hist_unit(label1, label2, ax, xedges, yedges):
    hist,edges=big_histogram(ds, label1, label2, xedges, yedges)
    ax.imshow(hist, vmin=0, vmax=10, extent=edges,aspect='auto',origin='lower', cmap='CMRmap_r')
    return ax    
    
def three_panel_hist(ds, label, yedges):
    fig, axes = plt.subplots(nrows=3, ncols=1)
    axlist = axes.flat
    axlist[0]=hist_unit('grad_q', label, axlist[0],[0,0.002],yedges)
    axlist[1]=hist_unit('speed', label, axlist[1],[0,40], yedges)
    axlist[2]=hist_unit('angle_q', label, axlist[2],[0,180], yedges)

        
    plt.tight_layout()
    #fig.subplots_adjust(hspace=0.15)

    plt.savefig('../data/processed/plots/test_'+ label+'.png', bbox_inches='tight', dpi=500)
    plt.show()
    plt.close()
    
    
def two_panel_bins(ds):
    binned_values=binned_statistics(ds,'humidity_overlap', 'speed_diff', 15)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axlist = axes.flat
    axlist[0].plot(binned_values[1][:-1], binned_values[0])
    axlist[1].plot(binned_values[1][:-1], binned_values[0])      

    plt.tight_layout()
    #fig.subplots_adjust(hspace=0.15)

    plt.savefig('../data/processed/plots/test_bins.png', bbox_inches='tight', dpi=500)
    plt.show()
    plt.close()

def binned_plot(ds):
    fig, ax = plt.subplots()
    binned_values=binned_statistics(ds,'humidity_overlap', 'error_square', 10)
    ax.plot(binned_values[1][:-1],np.sqrt( binned_values[0]))
    ax.set_xlabel('specific humidity [g/g]')    
    
    
    plt.savefig('../data/processed/plots/test_bins.png', bbox_inches='tight', dpi=500)
    plt.show()
    plt.close()
    
    
ds=xr.open_dataset('../data/processed/tvl1_coarse_january_5_t10_01_01_2020_am.nc')
#s=xr.open_dataset('../data/processed/ratio.nc')

ds=ds.sel(satellite='j1',day=ds['day'].values[0])
ds=ds.sel(plev=700, method='nearest')
ds=plotter.compute(ds)

ds['angle']=abs(ds['angle'])
ds['angle_q']=abs(ds['angle_q'])

ds['humidity_overlap']=1000*ds['humidity_overlap']
print(xr.corr(ds.angle_q, ds.error_mag))
binned_plot(ds)
#three_panel_hist(ds, 'angle',[0,180])
#three_panel_hist(ds, 'speed_diff',[-5,5])

#hist,edges=big_histogram(ds, 'angle_q', 'angle')
#hist,edges=big_histogram(ds, 'grad_q', 'angle')
#plt.imshow(hist, vmin=0, vmax=100, extent=[0,1,0,1],origin='upper')
#plt.show()
