# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

ds=xr.open_dataset('climcaps_20200101_am_j1_gridded_specific_humidity_1deg_noqc.nc')

fig, ax = plt.subplots()

ax = plt.axes(projection=ccrs.Miller())
lat=ds['latitude'].values
lon=ds['longitude'].values
ds_unit=ds['specific_humidity_mean'].loc[{'plev':41}]
var=np.squeeze(ds_unit.values)
ax.pcolormesh(lon, lat, var,cmap='RdBu')

ax.coastlines()

plt.show()
