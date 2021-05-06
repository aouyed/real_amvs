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

def map_plotter(ds, title, label, units_label, vmin, vmax):
    values = np.squeeze(ds[label].values)
    values[values == 0] = np.nan
    #values=values[:50,:]
    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap='viridis', extent=[ds['longitude'].min(
        ), ds['longitude'].max(), ds['latitude'].min(), ds['latitude'].max()])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(units_label)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig(title+'.png', dpi=300)
    plt.title(label)
    plt.show()
    plt.close()
    
    
def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test2.nc')
    print(ds)
    #print(ds)
    ds_map=ds.loc[{'day':datetime(2020,7,3),'plev':706.6,'time':'pm','satellite':'snpp'}]
    mind=ds_map['obs_time'].min(skipna=True).values
    #print(mind)
    timedelta=mind+np.timedelta64(1, 'h')
    ds_map=ds_map.where((ds_map.obs_time>mind) & (ds_map.obs_time<timedelta))
    map_plotter(ds_map, 'snpp_o', 'humidity_overlap', ' ', 0, 0.014)
    map_plotter(ds_map, 'snpp', 'specific_humidity_mean', ' ', 0, 0.014)
    
    ds_map=ds.loc[{'day':datetime(2020,7,3),'plev':706.6,'time':'pm','satellite':'j1'}]
    
    timedelta=mind+np.timedelta64(2, 'h')
    #ds_map=ds_map.where((ds_map.obs_time>mind) & (ds_map.obs_time<timedelta))
    map_plotter(ds_map, 'j1_o', 'humidity_overlap', ' ', 0, 0.014)
    map_plotter(ds_map, 'j1', 'specific_humidity_mean', ' ', 0, 0.014)
    
if __name__=="__main__":
    main()
    