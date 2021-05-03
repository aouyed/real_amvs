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
        ds['flowx'].values), np.squeeze(ds['flowy'].values))
    qk = ax.quiverkey(Q, 0.8, 0.9, 0.1, r'$0.1$ pixel', labelpos='E',
                      coordinates='figure')
    fig.tight_layout()
    plt.savefig('../data/processed/plots/quiver_'+title+'.png',
                bbox_inches='tight', dpi=300)

    plt.close()
    
    
ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test.nc')
ds_map=ds.loc[{'day':datetime(2020,1,1),'plev':slice(706,707),'time':'pm'}]
ds_map = ds_map.coarsen(longitude=10, boundary='trim').mean().coarsen(
            latitude=10, boundary='trim').mean()
quiver_plot(ds_map, 'test')

