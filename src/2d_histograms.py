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
def big_histogram(ds, column_x, column_y,  bins=100):
    xedges = [np.inf, -np.inf]
    yedges = [np.inf, -np.inf]
    
        
    xedges[0] = np.minimum(ds[column_x].min().item(), xedges[0])
    xedges[1] = np.maximum(ds[column_x].max().item(), xedges[1])
    
    yedges[0] = np.minimum(ds[column_y].min().item(), yedges[0])
    yedges[1] = np.maximum(ds[column_y].max().item(), yedges[1])
    
    xbins = np.linspace(xedges[0], xedges[1], bins+1)
    ybins = np.linspace(yedges[0], yedges[1], bins+1)
    heatmap = np.zeros((bins, bins), np.uint)
    
    df=ds[[column_x, column_y]].to_dataframe().reset_index().dropna()
        
    heatmap, _, _ = np.histogram2d(
                df[column_x].values, df[column_y].values, bins=[xbins, ybins])
      
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent



ds=xr.open_dataset('../data/processed/july.nc')
ds=ds.sel(satellite='j1',day=ds['day'].values[0])
ds=plotter.angle(ds)
ds['dqdx']=ds['humidity_overlap'].differentiate('longitude')
ds['dqdy']=ds['humidity_overlap'].differentiate('latitude')
ds['grad_q']=np.sqrt(ds.dqdx**2+ds.dqdy**2)
ds=plotter.angle_grad_q(ds)
ds=plotter.compute(ds)
#ds=ds.where(ds.error_mag<10)
ds['angle']=abs(ds['angle'])
ds['angle_q']=abs(ds['angle_q'])
ds=ds.where(ds.angle_q<20)
print(np.sqrt(ds['error_square'].mean()))
breakpoint()
#hist,edges=big_histogram(ds, 'angle_q', 'angle')
hist,edges=big_histogram(ds, 'angle_q', 'error_mag')
plt.imshow(hist, vmin=0, vmax=100, extent=[0,1,0,1],origin='upper')
plt.show()
