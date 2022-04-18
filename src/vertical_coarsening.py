#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:58:19 2022

@author: aouyed
"""
import xarray as xr
import pandas as pd 
import numpy as np


def vertical_coarsening 
ds=xr.open_dataset('../data/processed/01_01_2020_am.nc')
ds=ds.sel(satellite='j1')
ds=ds[['u','v','u_era5','v_era5']].coarsen(plev=5, boundary='trim').median()

data={'plev':[],'rmsvd':[]}
for plev in ds['plev'].values:
    ds_unit=ds.sel(plev=plev)
    error_sq=(ds_unit.u-ds_unit.u_era5)**2+(ds_unit.v-ds_unit.v_era5)**2
    rmsvd= np.sqrt(error_sq.mean().item())
    data['plev'].append(plev)
    data['rmsvd'].append(rmsvd)
df=pd.DataFrame(data=data)
print(df)





