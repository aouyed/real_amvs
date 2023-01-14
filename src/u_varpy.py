#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:29:33 2023

@author: aouyed
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def map_plotter_cartopy(da,  filename,title, units_label=''):
    values=da
    fig=plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()
    im = ax.imshow(values, cmap='viridis', extent=[ds['longitude'].min(
        ), ds['longitude'].max(), ds['latitude'].min(), ds['latitude'].max()], vmin=0, vmax=15,origin='lower')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    ax.set_title(title)
    #ax.scatter(df.longitude, df.latitude)
    #plt.title(label)
    cbar_ax = fig.add_axes([0.1, 0.05, 0.78, 0.05])
    fig.colorbar(im, cax=cbar_ax,orientation="horizontal", pad=0.5, label=units_label)    
    plt.savefig('../data/processed/plots/'+filename+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()  
    
    
def ax_cartopy(da, ax,title, units_label=''):
    values=da
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()
    im = ax.imshow(values, cmap='viridis', extent=[ds['longitude'].min(
        ), ds['longitude'].max(), ds['latitude'].min(), ds['latitude'].max()], vmin=0, vmax=15,origin='lower')
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    ax.set_title(title)
    return ax


def multiple_plots(ds):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axlist = axes.flat




ds=xr.open_dataset('../data/interim/model_07_01_2020.nc')

print(ds['level'].values)

ds=ds.sel(level=slice(200, 900))
u_min=ds['u'].coarsen(longitude=4, latitude=4, time=2, level=2, boundary='trim').min()
u_max=ds['u'].coarsen(longitude=4, latitude=4, time=2, level=2, boundary='trim').max()

diff=u_max-u_min
print(diff['level'].values)
diff_h=diff.sel(level=850, time=diff['time'].values[1], method='nearest')
mean=diff_h.mean().item()
mean=round(mean,2)
std=diff_h.std().item()
std=round(std,2)
title='mean = ' + str(mean) + ' m/s,  std = ' + str(std)+' m/s' 
level=diff_h['level'].values
level=str(int(level.item()))
map_plotter_cartopy(diff_h,  'u_diff_'+level, title,  'm/s')

diff_h=diff.sel(level=200, time=diff['time'].values[1], method='nearest')
level=diff_h['level'].values
level=str(int(level.item()))
mean=diff_h.mean().item()
mean=round(mean,2)
std=diff_h.std().item()
std=round(std,2)
title='mean = ' + str(mean) + ' m/s,  std = ' + str(std)+' m/s' 
map_plotter_cartopy(diff_h,  'u_diff_'+level, title,  'm/s')

