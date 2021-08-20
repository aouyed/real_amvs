#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:17:31 2021

@author: aouyed
"""
import cv2
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import inpainter 
import quiver
import first_stage_amv as fsa
import cross_section as cs
import plotter



import xarray as xr

ds_model=  xr.open_dataset('../data/interim/model_07_01_2020.nc')
ds=ds_model.sel(level=850, method="nearest")
ds=ds.sel(time=ds['time'].values[0])
ds=ds.drop(['level','time'])
plotter.map_plotter(ds, 'd', 'd', units_label='',color='viridis')

ds['vort_era5_smooth']=  ds['vo'].rolling(
        latitude=50, longitude=50, center=True).mean()
ds['div_era5_smooth']=  ds['d'].rolling(
        latitude=50, longitude=50, center=True).mean()
plotter.map_plotter(ds, 'smooth', 'div_era5_smooth', units_label='',color='viridis')
plotter.map_plotter(ds, 'd2', 'd', units_label='',color='viridis')

print(ds)



