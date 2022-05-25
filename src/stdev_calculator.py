#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:59:52 2022

@author: aouyed
"""

import xarray as xr
import numpy as np
import pandas as pd



d={'pressure':[], 'sigma_u': [], 'sigma_v': [], 'rmsvd':[], 'sigma':[]}
fname='../data/processed/filtered_thick_plev_tlv1_01_01_2020_am.nc'
ds=xr.open_dataset(fname)
ds['u_error']=ds['u'] - ds['u_era5']
ds['v_error']=ds['v'] - ds['v_era5']
ds['error_squared']=ds['u_error']**2+ds['v_error']**2



for pressure in (850,700, 500, 400):
    ds_unit=ds.sel(plev=pressure, method='nearest')
    sigma_u=ds_unit['u_era5'].std().item()
    sigma_v=ds_unit['v_era5'].std().item()
    sigma=np.sqrt(sigma_u**2+sigma_v**2)
    rmsvd=np.sqrt(ds_unit['error_squared'].mean().item())
    d['pressure'].append(pressure)  
    d['sigma_u'].append(sigma_u)    
    d['sigma_v'].append(sigma_v)
    d['sigma'].append(sigma)
    d['rmsvd'].append(rmsvd)
df=pd.DataFrame(data=d)
print(df)



    