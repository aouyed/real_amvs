#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 21:06:08 2021

@author: aouyed
"""

import xarray as xr
import numpy as np



ds_model=xr.open_dataset('../data/raw/reanalysis/07_03_20.nc')
ds_model2=xr.open_dataset('../data/raw/reanalysis/07_03_20_coarse.nc')
ds_model=ds_model[['u','v']]
    #print(ds_model)
ds_model = ds_model.assign_coords(longitude=(((ds_model.longitude + 180) % 360) - 180))
ds_model=ds_model.reindex(longitude=np.sort(ds_model['longitude'].values))
ds_model = ds_model.coarsen(longitude=4,latitude=4, boundary='trim').mean()

ds_model.to_netcdf('../data/raw/reanalysis/07_03_20_coarse.nc')