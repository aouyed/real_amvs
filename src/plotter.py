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
import stats_calculators as sc




def map_plotter_cartopy(ds, title, label, units_label=''):
    values=np.squeeze(ds[label].values)
    fig=plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    im = ax.imshow(values, cmap='viridis', extent=[ds['longitude'].min(
        ), ds['longitude'].max(), ds['latitude'].min(), ds['latitude'].max()])
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    #plt.title(label)
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

def corr(ds, thresh):
    ds['speed']=np.sqrt(ds.u**2+ds.v**2)
    ds['speed_era5']=np.sqrt(ds.u_era5**2+ds.v_era5**2)
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=np.sqrt(ds['u_error']**2+ds['v_error']**2)
    ds['error_square']=ds['u_error']**2+ds['v_error']**2
    ds=ds.where(ds.error_mag<thresh)
    print(np.sqrt(sc.weighted_mean(ds['error_square'])))

    df=ds.to_dataframe()
    df=df.reset_index().dropna()
    print(df[['u','u_era5']].corr())

def main():
  
    time='am'
    ds=xr.open_dataset('../data/processed/07_01_2020_'+time+'.nc')
    ds_map=ds.loc[{'time':time,'satellite':'j1'}]
    ds_map=ds_map.sel(plev=190, method='nearest')
    #corr(ds_map, 10)
    start=ds_map['obs_time'].min(skipna=True).values
    end=start + np.timedelta64(285, 'm')
    ds_map=ds_map.where((ds_map.obs_time >= start) & (ds_map.obs_time <= end))

    map_plotter_cartopy(ds_map, 'humidity_overlap_map_test_'+time, 'humidity_overlap','[g/kg]')
    

    print(ds_map)
    
    
if __name__=="__main__":
    main()
    