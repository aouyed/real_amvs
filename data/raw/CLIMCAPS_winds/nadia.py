# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import cartopy
import matplotlib

print(cartopy.__version__)
ds=xr.open_dataset('climcaps_20200703_pm_snpp_gridded_specific_humidity_1deg_noqc.nc')
df=ds[['specific_humidity_mean','longitude','latitude']].to_dataframe()
df=df.reset_index()
df=df.set_index(['longitude','latitude','plev'])
ds=xr.Dataset.from_dataframe(df)
#df=df.reset_index(drop=True).set_index(['latitude','longitude'])
#print(df)
fig, ax = plt.subplots()
#ds['longitude']=ds.longitude%360

proj=ccrs.PlateCarree()
ax = plt.axes(projection=proj)
ax.set_global()
#gl = ax.gridlines(crs=proj, draw_labels=True,
 #                 linewidth=2, color='gray', alpha=0.5, linestyle='--')

lat=ds['latitude'].values
lon=ds['longitude'].values
ds_unit=ds['specific_humidity_mean'].loc[{'plev':41}]
print(lon)                            
var=np.squeeze(ds_unit.values)
lat,lon=np.meshgrid(lat,lon)
ax.pcolormesh(lon, lat, var,cmap='viridis',transform=proj)
ax.coastlines()
plt.savefig('test.png',dpi=300)
plt.show()

