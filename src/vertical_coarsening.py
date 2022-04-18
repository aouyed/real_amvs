#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:58:19 2022

@author: aouyed
"""
import xarray as xr
import pandas as pd 
import numpy as np
import config 
import datetime 
from tqdm import tqdm 

start=config.MONTH
end=start + datetime.timedelta(days=6)

dates=pd.date_range(start=start, end=end, freq='d')

def vertical_coarse(ds):
    ds=ds.reindex(plev=list(reversed(ds.plev)))
    ds_c=ds[['u','v','u_era5','v_era5','humidity_overlap']].coarsen(plev=5, boundary='trim').median()
    ds_c['obs_time']=ds['obs_time']
    return ds_c

for date in tqdm(dates): 
    date_string=date.strftime('%m_%d_%Y')
    for orbit in ('am','pm'):
        ds=xr.open_dataset('../data/processed/'+date_string+'_'+orbit+'.nc')
        ds=vertical_coarse(ds)
        ds.to_netcdf('../data/processed/'+date_string+'_'+orbit+'_thick_plev.nc')






