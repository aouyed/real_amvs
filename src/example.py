#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:22:42 2023

@author: aouyed
"""

from amv_calculator import amv_calculator
import xarray as xr
import warnings
import numpy as np

warnings.simplefilter("ignore") 



ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_january.nc')
ds=ds.loc[{'day':ds['day'].values[0] ,'time':ds['time'].values[0]}]
ds_snpp=ds.loc[{'satellite':'snpp'}]
ds_j1=ds.loc[{'satellite':'j1'}]

ds_model=xr.open_dataset('../data/interim/model_01_01_2020.nc')
ds_model = ds_model.assign_coords(longitude=(((ds_model.longitude + 180) % 360) - 180))
ds_model=ds_model.reindex(longitude=np.sort(ds_model['longitude'].values))
  

ac=amv_calculator(ds_snpp, ds_j1, ds_model)
ds=ac.ds_amv
ds.to_netcdf('../data/processed/test.nc')
breakpoint()