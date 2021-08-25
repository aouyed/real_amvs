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
    
def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_july.nc') 
    #ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test2.nc')
    print(ds)
    #print(ds)
    ds_map=ds.loc[{'day':datetime(2020,7,3),'time':'am','satellite':'snpp'}]
    ds_map=ds_map.sel(plev=706, method='nearest')
    mind=ds_map['obs_time'].min(skipna=True).values
    print(mind)
    timedelta=mind+np.timedelta64(5, 'm')
    ds_map=ds_map.where((ds_map.obs_time>mind) & (ds_map.obs_time<timedelta))
    map_plotter(ds_map, 'snpp', 'specific_humidity_mean')
    
    ds_map=ds.loc[{'day':datetime(2020,7,3),'plev':706.6,'time':'pm','satellite':'snpp'}]
    mind=ds_map['obs_time'].min(skipna=True).values
    print(mind)
    timedelta=mind+np.timedelta64(10, 'm')
    ds_map=ds_map.where((ds_map.obs_time>mind) & (ds_map.obs_time<timedelta))
    map_plotter(ds_map, 'j1', 'specific_humidity_mean')
    
if __name__=="__main__":
    main()
    