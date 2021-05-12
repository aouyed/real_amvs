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
import amv_calculators as calc



R = 6371000

drad = np.deg2rad(1)
dx = R*drad
scale_x = dx
dy = R*drad
scale_y = dy


LABEL='specific_humidity_mean'


    
def ds_unit_calc(ds, day,pressure, time):
    ds=ds.loc[{'day':day ,'plev':pressure,'time':time}]
    ds=ds.drop(['day','plev','time'])
    ds=calc.prepare_ds(ds)
    
    df=ds.to_dataframe()
    swathes=calc.swath_initializer(ds)
  
    for swath in swathes:
        ds_snpp=ds.loc[{'satellite':'snpp'}]
        ds_j1=ds.loc[{'satellite':'j1'}]
        start=swath[0]
        end=swath[1]
        print(start)
        ds_merged, ds_snpp, ds_j1=calc.prepare_patch(ds_snpp, ds_j1, start, end)
    
        if (ds_snpp['specific_humidity_mean'].values.shape[0]>0) & (
                ds_j1['specific_humidity_mean'].values.shape[0]>0):                 
            df=calc.amv_calculator(ds_merged, df)
        
    ds=xr.Dataset.from_dataframe(df)
   
    return ds   
        
def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc.nc')
   
    print(ds)
    ds_total=xr.Dataset()
    for day in [ds['day'].values[3]]:
        print(day)
        ds_unit=xr.Dataset()
        for pressure in [ds['plev'].values[41]]:
            ds_unit1=xr.Dataset()
            for time in ds['time'].values:        
                ds_unit0=ds_unit_calc(ds, day,pressure, time)
                ds_unit0 = ds_unit0.expand_dims('day').assign_coords(day=np.array([day]))
                ds_unit0 = ds_unit0.expand_dims('time').assign_coords(time=np.array([time]))
                ds_unit0 = ds_unit0.expand_dims('plev').assign_coords(plev=np.array([pressure]))
                
                if not ds_unit1:
                    ds_unit1=ds_unit0
                else:
                    ds_unit1=xr.concat([ds_unit1,ds_unit0],'time')
            if not ds_unit:
                ds_unit=ds_unit1
            else:
                ds_unit=xr.concat([ds_unit,ds_unit1], 'plev') 
        if not ds_total:
            ds_total=ds_unit
        else:
            ds_total=xr.concat([ds_total,ds_unit], 'day')
    print(ds_total)
        
        
    ds_total.to_netcdf('../data/processed/real_water_vapor_noqc_test2.nc')

if __name__ == '__main__':
    main()
