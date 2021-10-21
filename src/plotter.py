#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:06:15 2021

@author: aouyed

"""
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
from datetime import timedelta
import numpy as np
import cv2
from dateutil.parser import parse
import first_stage_amv as fsa 
import cartopy.crs as ccrs




def map_plotter_cartopy(ds, title, label,vmin, vmax,  units_label=''):
    values=np.squeeze(ds[label].values)
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    im = ax.imshow(values, cmap='viridis', extent=[ds['longitude'].min(
        ), ds['longitude'].max(), ds['latitude'].min(), ds['latitude'].max()], vmin=vmin, vmax=vmax)
    plt.title(label)
    cbar_ax = fig.add_axes([0.1, 0.05, 0.78, 0.05])
    fig.colorbar(im, cax=cbar_ax,orientation="horizontal", pad=0.5, label=units_label)    
    plt.savefig('../data/processed/plots/'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()  

def map_plotter(ds, title, label, units_label='',color='viridis'):
    values=np.squeeze(ds[label].values)
    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap=color, extent=[ds['longitude'].min(
        ), ds['longitude'].max(), ds['latitude'].min(), ds['latitude'].max()])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(units_label)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig('../data/processed/plots/'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.title(label)
    plt.show()
    plt.close()    

def map_plotter_vmax(ds, title, label, vmin, vmax, units_label='',color='viridis'):
    values=np.squeeze(ds[label].values)
    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap=color, extent=[ds['longitude'].min(
        ), ds['longitude'].max(), ds['latitude'].min(), ds['latitude'].max()], vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(units_label)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig('../data/processed/plots/'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.title(label)
    plt.show()
    plt.close()    
    
   
def overlap(ds):
    ds_snpp=ds.loc[{'satellite':'snpp'}]
    ds_j1=ds.loc[{'satellite':'j1'}]
    condition1=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_j1['obs_time']))
    condition2=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_snpp['obs_time']))
    ds_snpp=ds_snpp[['specific_humidity_mean','obs_time']].where(condition1 & condition2)
    ds_j1=ds_j1[['specific_humidity_mean','obs_time']].where(condition1 & condition2)
    return ds_snpp, ds_j1

def main():
    #ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_july.nc') 
    #ds_map=ds.loc[{'day':datetime(2020,7,3),'time':'am','satellite':'snpp'}]
    #ds_map=ds_map.sel(plev=706, method='nearest')
    #map_plotter(ds_map, 'snpp', 'specific_humidity_mean')
    
    #ds_map=ds.loc[{'day':datetime(2020,7,3),'time':'am'}]
    #ds_map=ds_map.sel(plev=706, method='nearest')
    #ds_snpp, ds_j1=overlap(ds_map)
    #map_plotter(ds_snpp, 'snpp', 'specific_humidity_mean')
    
    time='pm'
    ds=xr.open_dataset('../data/processed/07_01_2020_'+time+'.nc')
    ds_map=ds.loc[{'time':time,'satellite':'snpp'}]
    ds_map=ds_map.sel(plev=706, method='nearest')
    map_plotter(ds_map, 'snpp', 'humidity_overlap')
    print(ds_map)
    
    
if __name__=="__main__":
    main()
    