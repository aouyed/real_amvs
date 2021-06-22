#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:06:15 2021

@author: aouyed

"""
import matplotlib.pyplot as plt
import first_stage_amv as fsa
import xarray as xr
from datetime import datetime
from datetime import timedelta
import numpy as np
import cv2
from dateutil.parser import parse
from metpy.interpolate import cross_section


def contourf_plotter(cross,  label, vmin, vmax):
    fig, ax = plt.subplots()
    ax.contourf(cross['latitude'], cross['plev'], cross[label],
                         levels=np.linspace(vmin, vmax, 10), cmap='YlGnBu')
    plt.show()
    plt.close()


def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test_3d_'+fsa.ALG+'.nc')   
    ds=ds.loc[{'day':datetime(2020,7,3),'time':'am','satellite':'snpp'}].squeeze()
    lat,lon=np.meshgrid(ds['latitude'].values, ds['longitude'].values)
    #ds=ds.rename({'latitude':'y','longitude':'x'})
    #ds['lat']=(['longitude','latitude'],lat)
    #ds['lon']=(['longitude','latitude'],lon)
    #breakpoint()
    ds=ds.drop(['day','satellite','time','flowx','flowy','obs_time'])
    ds=ds.metpy.assign_crs(grid_mapping_name='latitude_longitude',
    earth_radius=6371229.0)
    
    data = ds.metpy.parse_cf().squeeze()
    print(data)
    #print(ds)
    start = (-44.5, 38.5)
    end = (-29.5, 4.5)
    print(data)
    cross = cross_section(data, start, end).set_coords(('latitude', 'longitude'))
    print(cross)
    contourf_plotter(cross,  'u_era5', -10,10)
    
if __name__=="__main__":
    main()
    