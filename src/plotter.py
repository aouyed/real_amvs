#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:06:15 2021

@author: aouyed

"""
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import numpy as np
import cv2

def map_plotter(ds, title, label, units_label, vmin, vmax):
    values = np.squeeze(ds[label].values)
    values[values == 0] = np.nan
    #values=values[:50,:]
    fig, ax = plt.subplots()
    if vmin == vmax:
        #im = ax.imshow(values, cmap='viridis', extent=[ds['lon'].min(
        #), ds['lon'].max(), ds['lat'].min(), ds['lat'].max()])
        im = ax.imshow(values, cmap='viridis')
    else:
        im = ax.imshow(values, cmap='viridis', extent=[ds['lon'].min(
        ), ds['lon'].max(), ds['lat'].min(), ds['lat'].max()], vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(units_label)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig(title+'.png', dpi=300)
    plt.title(label)
    plt.show()
    plt.close()
    
    
def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_0.nc')
    print(ds['satellite'].values)
    #print(ds)
    ds_map=ds.loc[{'day':datetime(2020,7,1),'pressure':878.6,'satellite':'snpp','time':'am'}]
    map_plotter(ds_map, 'test', 'specific_humidity_mu', ' ', 0, 0)
    
if __name__=="__main__":
    main()
    