#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:17:31 2021

@author: aouyed
"""
import os 
import cv2
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import inpainter 
import amv_calculators as calc
import quiver
import plotter
import time
from dask.diagnostics import ProgressBar
import pandas as pd
import era5_downloader as ed
pd.options.mode.chained_assignment = None  # default='warn'

ALG='farneback'


R = 6371000

drad = np.deg2rad(1)
dx = R*drad
scale_x = dx
dy = R*drad
scale_y = dy
swath_hours=24


LABEL='specific_humidity_mean'

def model_closer(date,ds,time):
    date = pd.to_datetime(str(date)) 
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day  = date.strftime('%d')
    os.remove('../data/interim/model_'+month+'_'+day+'_'+year+'.nc')
    ds.to_netcdf('../data/processed/'+month+'_'+day+'_'+year+'_'+time+'.nc')
   

def model_loader(date, pressure):
    date = pd.to_datetime(str(date)) 
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day  = date.strftime('%d')
    ds_model=  xr.open_dataset('../data/interim/model_'+month+'_'+day+'_'+year+'.nc')
    ds_model=ds_model.sel(level=pressure, method='nearest')
    ds_model=ds_model.drop('level')
    ds_model['vort_era5_smooth']=  ds_model['vo'].rolling(
        latitude=5, longitude=5, center=True).mean()
    ds_model['div_era5_smooth']=  ds_model['d'].rolling(
        latitude=5, longitude=5, center=True).mean()
    ds_model = ds_model.assign_coords(longitude=(((ds_model.longitude + 180) % 360) - 180))
    ds_model=ds_model.reindex(longitude=np.sort(ds_model['longitude'].values))
    print(ds_model)
    return ds_model
    
    
def ds_unit_calc(ds, day,pressure, time):
    ds=ds.loc[{'day':day ,'plev':pressure,'time':time}]
    ds=ds.drop(['day','plev','time'])
    ds=calc.prepare_ds(ds)
    ds_model=model_loader(day,pressure)
    df=ds.to_dataframe()
    swathes=calc.swath_initializer(ds,5,swath_hours)
    print('swathes prepared')
    for swath in swathes:
        ds_snpp=ds.loc[{'satellite':'snpp'}]
        ds_j1=ds.loc[{'satellite':'j1'}]
        start=swath[0]
        end=swath[1]
        print(swath[0])
        ds_merged, ds_snpp, ds_j1, df_snpp=calc.prepare_patch(ds_snpp, ds_j1, ds_model, start, end)
   
        if (df_snpp.shape[0]>100):
            df, ds_snpp_p,ds_j1_p, ds_model_p=calc.amv_calculator(ds_merged, df)
                     
    ds=xr.Dataset.from_dataframe(df)
    ds_model.close()
   
    return ds   

    


def serial_loop(ds):
    ds_total=xr.Dataset()
    for day in ds['day'].values:
        print(day)
        ed.downloader(day)
        
        ds_unit=xr.Dataset()
        for pressure in ds['plev']:
            ds_unit1=xr.Dataset()
            print('pressure:')
            print(pressure.item())
            for time in ['pm']:  
                print('time')
                print(time)
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
        model_closer(day,ds_unit,time)


def dask_func(ds):
    return xr.apply_ufunc(
        serial_loop, ds,
        dask='parallelized',output_dtypes=[float])
        
def main():
    #ds=xr.open_dataset('../data/processed/real_water_vapor_noqc.nc', chunks={"plev": 20})
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_july.nc')
    serial_loop(ds)
    
   

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

