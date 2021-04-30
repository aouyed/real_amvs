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
import cProfile


R = 6371000
dt_inv=1/3600

label='specific_humidity_mean'



def calc(frame0, frame):
     nframe0 = cv2.normalize(src=frame0, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
     nframe = cv2.normalize(src=frame, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
     optical_flow = cv2.optflow.createOptFlow_DeepFlow()
     flowd = optical_flow.calc(nframe0, nframe, None)
     flowx=flowd[:,:,0]
     flowy=flowd[:,:,1]
     
     return flowx, flowy
 
def frame_retreiver(ds, satellite):
    frame= ds[label].loc[{'satellite':satellite}].values        
    frame[frame == 0] = np.nan
    return frame
    
def ds_unit_calc0(ds, day,pressure, time):
    ds_unit=ds.loc[{'day':day ,'plev':pressure,'time':time}]
    frame0=frame_retreiver(ds_unit,'j1')
    frame=frame_retreiver(ds_unit, 'snpp')
    flowx,flowy=calc(frame0, frame)
    ds_unit['flowx']=(['latitude','longitude'],flowx)
    ds_unit['flowy']=(['latitude','longitude'],flowy)
    ds_unit['u']=ds_unit['flowx']*ds_unit['g_factor_x']
    ds_unit['v']=ds_unit['flowy']*ds_unit['g_factor_y']
    return ds_unit

def ds_unit_calc(ds, day,pressure, time):
    ds_unit=ds.loc[{'day':day ,'plev':pressure,'time':time}]
    df=ds_unit.to_dataframe()
    df=df.dropna().set_index('obs_time', append=True)
    ds_df=xr.Dataset.from_dataframe(df)
    ds_df["flowx"] = xr.full_like(ds_df['specific_humidity_mean'].loc[
        {'satellite':'j1','obs_time':ds_df['obs_time'].values[0]}
        ].drop(['satellite','obs_time']), fill_value=0)
    ds_df["flowy"] = xr.full_like(ds_df['flowx'], fill_value=0)
    for obs_time in ds_df.loc[{'satellite':'j1'}]['obs_time'].values:
        print(obs_time)
        ds=ds_df.loc[{'obs_time':obs_time,'satellite':'j1'}]
        df=ds.to_dataframe()
        ds=xr.Dataset.from_dataframe(df.dropna())
        ds=ds_df.loc[{'obs_time':obs_time,'latitude':ds['latitude'].values,
                          'longitude': ds['longitude'].values}]
        print('formatting done')
        frame0=frame_retreiver(ds,'j1')
        frame=frame_retreiver(ds, 'snpp')
        print(np.shape(frame))
        if 0 not in np.shape(frame):
            flowx,flowy=calc(frame0, frame)
            #flowx=frame0
            #flowy=frame
        
            ds['flowx']=(['latitude','longitude'],flowx)
            ds['flowy']=(['latitude','longitude'],flowy)
            ds_df['flowx'].loc[{'latitude':ds['latitude'].values,
                              'longitude': ds['longitude'].values}]=ds['flowx']
            ds_df['flowy'].loc[{'latitude':ds['latitude'].values,
                              'longitude': ds['longitude'].values}]=ds['flowy']
    
        
    return ds_df[['flowx','flowy']]


def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_inpainted.nc')
    drad = np.deg2rad(1)
    dx = R*abs(np.cos(np.deg2rad(ds.latitude)))*drad
    scale_x = dx*dt_inv
    dy = R*drad
    scale_y = dy*dt_inv
    ds['g_factor_x']=  scale_x 
    ds['g_factor_y']=  scale_y 
    
    print(ds)
    ds_total=xr.Dataset()
    for day in ds['day'].values:
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
        
        
    ds_total.to_netcdf('../data/processed/real_water_vapor_noqc_test.nc')

if __name__ == '__main__':
    main()
