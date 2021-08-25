# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import first_stage_amv as fsa
from datetime import datetime 
import cartopy.crs as ccrs


def quiver_plot(ds, title, u, v):
    ds=ds.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    ax.set_title(title)
    Q = ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values))
    qk = ax.quiverkey(Q, 0.8, 0.9, 5, r'5 m/s', labelpos='E',
                      coordinates='figure')
    fig.tight_layout()
    plt.savefig('../data/processed/plots/quiver_'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
def quiver_plot_cartopy(ds, title, u, v):
    ds=ds.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    Q = ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values))
    qk = ax.quiverkey(Q, 0.8, 0.9, 5, r'5 m/s', labelpos='E',
                      coordinates='figure')
    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()  
    
    

def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test_3d_'+ fsa.ALG+'.nc')
    print(ds)
    ds['theta']=  np.arctan(ds['v']/ ds['u'])
    print(abs(ds['flowx']).mean())
    print(abs(ds['u']).mean())
    print(abs(ds['u_era5']).mean())
    print(abs(ds['flowy']).mean())
    print(abs(ds['v']).mean())
    print(abs(ds['dt_inv']).mean())
    print(abs(ds['theta']).mean())
    

    ds_map=ds.loc[{'day':datetime(2020,7,3),'time':'am','satellite':'j1'}]
    ds_map=ds_map.sel(plev='850',method='nearest')
    ds_map['u'].plot.hist(bins=100)
    plt.show()
    plt.close()
    ds_map['u_era5'].plot.hist(bins=100)
    plt.show()
    plt.close()
    ds_map['theta'].plot.hist(bins=100)
    plt.show()
    plt.close()
    #ds_map = ds_map.coarsen(longitude=3, boundary='trim').mean().coarsen(
                #latitude=3, boundary='trim').mean()
    
    mint=np.datetime64('2020-07-03T05:30')
    maxt=np.datetime64('2020-07-03T06:30')
    #ds_map=ds_map.where((ds_map.obs_time>mint) & (ds_map.obs_time<maxt))
    quiver_plot(ds_map, 'test_era5','u_era5','v_era5')
    quiver_plot(ds_map, 'test','u','v')
    
    
    ds_u=xr.open_dataset('../data/raw/reanalysis/u_700_2020_07_03_18:00:00_era5.nc')
    ds_v=xr.open_dataset('../data/raw/reanalysis/v_700_2020_07_03_18:00:00_era5.nc')
    ds=xr.merge([ds_u,ds_v])
    ds['theta']=  np.arctan(ds['v']/ ds['u'])
    print(abs(ds['u']).mean())
    print(abs(ds['v']).mean())
    print(abs(ds['theta']).mean())
   
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds=ds.reindex(longitude=np.sort(ds['longitude'].values))
    #ds=ds.rename({'u':'flowx','v': 'flowy'})
    #ds_map=ds.loc[{'day':datetime(2020,7,3),'plev':slice(706,707),'time':'am','satellite':'snpp'}]
    ds_map = ds.coarsen(longitude=50, boundary='trim').mean().coarsen(
                latitude=50, boundary='trim').mean()
    quiver_plot(ds_map, 'era5_18z','u','v')
    print(ds)

if __name__ == '__main__':
    main()

