# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime 

def quiver_plot(ds, title):
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    ax.set_title(title)
    Q = ax.quiver(X, Y, np.squeeze(
        ds['u'].values), np.squeeze(ds['v'].values))
    qk = ax.quiverkey(Q, 0.8, 0.9, 1, r'$0.1$ pixel', labelpos='E',
                      coordinates='figure')
    fig.tight_layout()
    plt.savefig('../data/processed/plots/quiver_'+title+'.png',
                bbox_inches='tight', dpi=300)

    plt.close()
    
    
ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test2.nc')
print(ds)
ds_map=ds.loc[{'day':datetime(2020,7,3),'plev':slice(706,707),'time':'am','satellite':'snpp'}]
ds_map = ds_map.coarsen(longitude=10, boundary='trim').mean().coarsen(
            latitude=10, boundary='trim').mean()
mint=np.datetime64('2020-07-03T05:30')
maxt=np.datetime64('2020-07-03T06:30')
#ds_map=ds_map.where((ds_map.obs_time>mint) & (ds_map.obs_time<maxt))
quiver_plot(ds_map, 'test')

ds_u=xr.open_dataset('../data/raw/reanalysis/u_700_2020_07_03_18:00:00_era5.nc')
ds_v=xr.open_dataset('../data/raw/reanalysis/v_700_2020_07_03_18:00:00_era5.nc')
ds=xr.merge([ds_u,ds_v])
ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
ds=ds.reindex(longitude=np.sort(ds['longitude'].values))
#ds=ds.rename({'u':'flowx','v': 'flowy'})
#ds_map=ds.loc[{'day':datetime(2020,7,3),'plev':slice(706,707),'time':'am','satellite':'snpp'}]
ds_map = ds.coarsen(longitude=50, boundary='trim').mean().coarsen(
            latitude=50, boundary='trim').mean()
quiver_plot(ds_map, 'era5_18z')
print(ds)
