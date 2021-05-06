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
import cProfile



R = 6371000
dt_inv=1/3600

LABEL='specific_humidity_mean'



def calc(frame0, frame):
    if frame0.shape != frame.shape:
        frame=np.resize(frame, frame0.shape)
    
    
    nframe0 = cv2.normalize(src=frame0, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    nframe = cv2.normalize(src=frame, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    optical_flow = cv2.optflow.createOptFlow_DeepFlow()
    flowd = optical_flow.calc(nframe0, nframe, None)
    flowx=flowd[:,:,0]
    flowy=flowd[:,:,1]
     
    return flowx, flowy
 
def frame_retreiver(ds):
    frame= np.squeeze(ds[LABEL].values)    
    #frame[frame == 0] = np.nan
    frame=inpainter.drop_nan(frame)
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
    suma=0
    latlona=ds['latitude'].values.shape[0]*ds['longitude'].values.shape[0]
    print(latlona)
    ds=ds.loc[{'day':day ,'plev':pressure,'time':time}]
    ds=ds.drop(['day','plev','time'])
    ds['humidity_overlap'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['flowx'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['flowy'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)

    df=ds.to_dataframe()

    dhours=60   
    mind=ds['obs_time'].min(skipna=True).values
    hours=np.arange(dhours,dhours*24,dhours)
    swathes=[]

    swathes.append([mind, mind+np.timedelta64(dhours, 'm')])
    for hour in hours:
        swathes.append([mind+np.timedelta64(hour, 'm'),
                        mind+np.timedelta64(hour+dhours, 'm')])
        
    for swath in swathes:
        ds_snpp=ds.loc[{'satellite':'snpp'}]
        ds_j1=ds.loc[{'satellite':'j1'}]
        start=swath[0]
        end=swath[1]
        
        print('end1:')
        print(end)
        ds_snpp=ds_snpp.where((ds_snpp.obs_time > start) & (ds_snpp.obs_time<end))
        df_helper=ds_snpp.to_dataframe().dropna(subset=[LABEL])
        ds_helper=xr.Dataset.from_dataframe(df_helper)

        start=start+np.timedelta64(50, 'm')
        end=end+np.timedelta64(50, 'm')
        print('end2:')
        print(end)
        
        ds_j1=ds_j1.where((ds_j1.obs_time > start) & (ds_j1.obs_time<end))
      
        #help1=ds_j1.latitude.isin(ds_helper.latitude.values)
        #help2=ds_j1.longitude.isin(ds_helper.longitude.values)
                                          
        #ds_j1=ds_j1.where(help1 & help2)
        #ds_j1=ds_j1.sel(latitude=ds_snpp['latitude'].values, longitude=ds_snpp[])
        condition1=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_j1['obs_time']))
        condition2=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_snpp['obs_time']))
        ds_j1_unit=ds_j1['specific_humidity_mean'].where(condition1 & condition2)
        ds_snpp_unit=ds_snpp['specific_humidity_mean'].where(condition1 & condition2)
        df_j1=ds_j1_unit.to_dataframe().dropna(subset=[LABEL]).set_index('satellite',append=True)
        df_snpp=ds_snpp_unit.to_dataframe().dropna(subset=[LABEL]).set_index('satellite',append=True)
       
        ds_j1=xr.Dataset.from_dataframe(df_j1)
        ds_snpp=xr.Dataset.from_dataframe(df_snpp)
        ds_merged=xr.merge([ds_j1,ds_snpp])
        if (ds_snpp['specific_humidity_mean'].values.shape[0]>0) & (
                ds_j1['specific_humidity_mean'].values.shape[0]>0):
                    
            ds_snpp=ds_merged.loc[{'satellite':'snpp'}].expand_dims('satellite')
            ds_j1=ds_merged.loc[{'satellite':'j1'}].expand_dims('satellite')
            frame0=frame_retreiver(ds_snpp)
            frame=frame_retreiver(ds_j1)
            flowx,flowy=calc(frame0, frame)
            flowx = np.expand_dims(flowx, axis=2)
            flowy = np.expand_dims(flowy, axis=2)
            ds_snpp['flowx']=(['latitude','longitude','satellite'],flowx)
            ds_snpp['flowy']=(['latitude','longitude','satellite'],flowy)
            ds_j1['flowx']=(['latitude','longitude','satellite'],flowx)
            ds_j1['flowy']=(['latitude','longitude','satellite'],flowy) 
            
            df_j1=ds_j1.to_dataframe().dropna()
            df_snpp=ds_snpp.to_dataframe().dropna()
        
            swathi=df_snpp.index.values 
            df['humidity_overlap'].loc[df.index.isin(swathi)]=df_snpp['specific_humidity_mean']
            df['flowx'].loc[df.index.isin(swathi)]=df_snpp['flowx']
            df['flowy'].loc[df.index.isin(swathi)]=df_snpp['flowy']
    
            swathi=df_j1.index.values 
            #print(swathi)
            df['humidity_overlap'].loc[df.index.isin(swathi)]=df_j1['specific_humidity_mean']
            df['flowx'].loc[df.index.isin(swathi)]=df_j1['flowx']
            df['flowy'].loc[df.index.isin(swathi)]=df_j1['flowy']
        
        

    ds=xr.Dataset.from_dataframe(df)
 
    
    return ds   
        
    
        
      



def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc.nc')
    drad = np.deg2rad(1)
    dx = R*abs(np.cos(np.deg2rad(ds.latitude)))*drad
    scale_x = dx*dt_inv
    dy = R*drad
    scale_y = dy*dt_inv
    ds['g_factor_x']=  scale_x 
    ds['g_factor_y']=  scale_y 
    
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
