#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:22:42 2023

@author: aouyed
"""

from amv_calculator import amv_calculator
import xarray as xr


ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_january.nc')
ds=ds.loc[{'day':ds['day'].values[0] ,'time':ds['time'].values[0]}]
ds_snpp=ds.loc[{'satellite':'snpp'}]
ds_j1=ds.loc[{'satellite':'j1'}]

ds_era5=xr.open_dataset('../data/interim/coarse_model_01_01_2020.nc')

ac=amv_calculator(ds_snpp, ds_j1, ds_era5)
breakpoint()