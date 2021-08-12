#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 21:06:08 2021

@author: aouyed
"""

import xarray as xr
import numpy as np
import quiver as q



ds_model=xr.open_dataset('../data/raw/reanalysis/07_03_20.nc')
ds_model2=xr.open_dataset('../data/raw/reanalysis/07_03_20_coarse.nc')
ds_model=ds_model[['u','v']]
ds_model = ds_model.assign_coords(longitude=(((ds_model.longitude + 180) % 360) - 180))
ds_model=ds_model.reindex(longitude=np.sort(ds_model['longitude'].values))

model2=ds_model2.sel(level=ds_model2['level'].values[0], time=ds_model2['time'].values[0])
model=ds_model.sel(level=ds_model['level'].values[0], time=ds_model['time'].values[0])
model1=model.coarsen(longitude=40, boundary='trim').mean().coarsen(latitude=40, boundary='trim').mean()
model2=model.coarsen(longitude=40,latitude=40, boundary='trim').mean()

q.quiver_plot(model1, '','u','v')
q.quiver_plot(model2, '','u','v')
q.quiver_plot(model2-model1, '','u','v')

