#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:19:49 2022

@author: aouyed
"""

import xarray as xr
import numpy as np


def compute(ds, thresh):
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=np.sqrt(ds.u_error**2+ds.v_error**2)
    return ds

ds_old=xr.open_dataset('../data/processed/01_02_2020_am.nc')
ds=xr.open_dataset('../data/processed/coarse_january_5_t10_01_01_2020_am.nc')



ds=compute(ds,10)
ds_old=compute(ds_old,10)
ds_old=ds_old.where(ds_old.error_mag<10)
print(ds['u_error'].mean())
print(ds_old['u_error'].mean())


