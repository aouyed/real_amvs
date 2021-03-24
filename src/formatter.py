#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 15:21:36 2021

@author: aouyed
"""
import xarray as xr
import numpy as np
from datetime import datetime
from datetime import timedelta

PATH='../data/raw/'

def ds_unit_calc(day,time,satellite):
    day_string=day.strftime("%Y%m%d")
    print(day_string)
    d0=datetime(1993, 1, 1)    ds=xr.open_dataset(PATH+ 'climcaps_'+ day_string+'_'+time+'_'+satellite+'_gridded_specific_humidity_1deg.nc')
    ds['Lat']=ds['Lat']+ds['lat'].min()
    ds['Lat']=ds['Lat'].astype(np.float32)
    ds['Lon']=ds['Lon']+ds['lon'].min()
    ds['Lon']=ds['Lon'].astype(np.float32)
    ds['plev']=np.around(np.sort(np.unique(ds['pressure'].values)),decimals=1)
    ds=ds.drop('lat')
    ds=ds.drop('lon')
    ds=ds.drop('pressure')
    dts=ds['obs_time'].values
    
    mask=np.isnan(dts)
    dts=np.nan_to_num(dts)
    helper = np.vectorize(lambda x: d0 + timedelta(seconds=x))
    dt_sec=helper(dts)
    dt_sec[mask]=np.nan
    
    
    ds['obs_time']=(['Lon','Lat'],dt_sec)
    
    ds=ds.rename({'Lat':'lat','Lon':'lon','plev':'pressure'})
    ds = ds.expand_dims('day').assign_coords(day=np.array([day]))
    ds = ds.expand_dims('time').assign_coords(time=np.array([time]))
    ds = ds.expand_dims('satellite').assign_coords(satellite=np.array([satellite]))
    
    return ds


def satellite_loop(satellites,day,time):
    ds_total=xr.Dataset()
    for satellite in satellites:  
        ds_unit=ds_unit_calc(day, time, satellite)
        if not ds_total:
            ds_total = ds_unit
        else:       
            ds_total = xr.concat([ds_total, ds_unit], 'satellite')
    return ds_total

def time_loop(times, satellites, day):
    ds_total=xr.Dataset()
    for time in times:
        ds_unit=satellite_loop(satellites,day,time)
        if not ds_total:
            ds_total = ds_unit
        else:       
            ds_total = xr.concat([ds_total, ds_unit], 'time')
    return ds_total
        
def main():
    times=['am','pm']
    satellites=['j1','snpp']
    days=[datetime(2020,1,1),datetime(2020,1,2),datetime(2020,1,3),
      datetime(2020,7,1),datetime(2020,7,2),datetime(2020,7,3)]
    ds_total=xr.Dataset() 
    for day in days:    
        print(day)
        ds_unit=time_loop(times, satellites, day)
        if not ds_total:
            ds_total = ds_unit
        else:       
            ds_total = xr.concat([ds_total, ds_unit], 'day')
    print(ds_total)
    ds_total.to_netcdf('../data/processed/real_water_vapor.nc')
 
    
if __name__=="__main__":
    main()
    
    
    