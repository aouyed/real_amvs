#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:17:31 2021

@author: aouyed
"""
import os 
import xarray as xr
import numpy as np
import amv_calculators as calc
import time
import pandas as pd
import era5_downloader as ed
from datetime import datetime 
import config as c
from tqdm import tqdm
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

month_string=c.MONTH.strftime("%B").lower()
QC=c.QC


ALG='farneback'


R = 6371000

drad = np.deg2rad(1)
dx = R*drad
scale_x = dx
dy = R*drad
scale_y = dy
swath_hours=24


LABEL='specific_humidity_mean'

def ds_closer(date,ds,time):
    date = pd.to_datetime(str(date)) 
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day  = date.strftime('%d')
    ds.to_netcdf('../data/processed/'+month+'_'+day+'_'+year+'_'+time+'.nc')

def prepare_patch(ds_snpp, ds_j1, start, end):
    ds_j1=ds_j1.where((ds_j1.obs_time >= start) & (ds_j1.obs_time <= end))
    start=start+np.timedelta64(50, 'm')
    end=end+np.timedelta64(50, 'm')
    ds_snpp=ds_snpp.where((ds_snpp.obs_time >= start) & (ds_snpp.obs_time <= end))
    

    condition1=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_j1['obs_time']))
    condition2=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_snpp['obs_time']))
    ds_j1_unit=ds_j1[['specific_humidity_mean','obs_time']].where(condition1 & condition2)
    ds_snpp_unit=ds_snpp[['specific_humidity_mean','obs_time']].where(condition1 & condition2)
    df_j1=ds_j1_unit.to_dataframe().dropna(subset=[LABEL]).set_index('satellite',append=True)
    df_snpp=ds_snpp_unit.to_dataframe().dropna(subset=[LABEL]).set_index('satellite',append=True)
       
    ds_j1=xr.Dataset.from_dataframe(df_j1)
    ds_snpp=xr.Dataset.from_dataframe(df_snpp)
    ds_merged=xr.merge([ds_j1,ds_snpp])
    
    
    return ds_merged, ds_snpp, ds_j1, df_snpp


def map_plotter(ds, label, units_label='',color='viridis'):
    values=np.squeeze(ds[label].values)
    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap=color, extent=[ds['longitude'].min(
        ), ds['longitude'].max(), ds['latitude'].min(), ds['latitude'].max()])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label(units_label)
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.show()
    plt.close()       
    
def ds_unit_calc(ds, day,pressure, time):
    ds=ds.loc[{'day':day ,'plev':pressure,'time':time}]
    ds=ds.drop(['day','plev','time'])
    ds=calc.prepare_ds(ds)
    df=ds.to_dataframe()
    swathes=calc.swath_initializer(ds,5,swath_hours)
    for swath in swathes:
        print(swath)
        ds_snpp=ds.loc[{'satellite':'snpp'}]
        ds_j1=ds.loc[{'satellite':'j1'}]
        start=swath[0]
        end=swath[1]
        ds_merged, ds_snpp, ds_j1, df_snpp=prepare_patch(ds_snpp, ds_j1,  start, end)
        df_j1=ds_j1.to_dataframe().reset_index().dropna().set_index(['latitude','longitude','satellite'])
        df_snpp=ds_snpp.to_dataframe().reset_index().dropna().set_index(['latitude','longitude','satellite'])
        print(df_snpp.shape[0])
        if (df_snpp.shape[0]>100):
            df_snpp=df_snpp.reorder_levels(['latitude','longitude','satellite']).sort_index()
            df_j1=df_j1.reorder_levels(['latitude','longitude','satellite']).sort_index()
            df=df_filler(df, df_snpp)
            df=df_filler(df, df_j1)        
            ds_unit=xr.Dataset.from_dataframe(df)
            ds_unit=ds_unit.sel(satellite='j1')
            ds_unit=ds_unit.squeeze()
            map_plotter(ds_unit, 'humidity_overlap')
    ds=xr.Dataset.from_dataframe(df)
   
    return ds   

def df_filler(df, df_sat):
    swathi=df_sat.index.values 
    df['humidity_overlap'].loc[df.index.isin(swathi)]=df_sat['specific_humidity_mean']
    return df


def serial_loop(ds):
    day=ds['day'].values[0]
    time=ds['time'].values[0]
    pressure=ds['plev'].values[41]
    ds_unit = ds_unit_calc(ds, day,pressure, time)
    ds_unit = ds_unit.expand_dims('day').assign_coords(day=np.array([day]))
    ds_unit = ds_unit.expand_dims('time').assign_coords(time=np.array([time]))
    ds_unit = ds_unit.expand_dims('plev').assign_coords(plev=np.array([pressure]))
                
                
    ds_closer(day,ds_total,time)


        
def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_'+ month_string +'.nc')
    serial_loop(ds)
    
   

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

