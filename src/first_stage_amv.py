#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:17:31 2021

@author: aouyed
"""
import cv2
import xarray as xr
import numpy as np
label='specific_humidity_mu'
def wind_calculator(ds):
    flow_x=np.squeeze(ds['flow_x'].values)
    flow_y=np.squeeze(ds['flow_y'].values)
    lon,lat=np.meshgrid(ds['lon'].values,ds['lat'].values)
    dthetax = GRID*flow_x
    dradsx = dthetax * np.pi / 180
    lat = lat*np.pi/180
    dx = R*abs(np.cos(lat))*dradsx
    u= dx/1800
    
    dthetay =GRID*flow_y
    dradsy = dthetay * np.pi / 180
    dy = R*dradsy
    v= dy/1800
    
    ds['u']=(('time','lat','lon'),np.expand_dims(u,axis=0))
    ds['v']=(('time','lat','lon'),np.expand_dims(v,axis=0))
    return ds


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
    
 
def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_inpainted.nc')
    ds=ds.loc[{'lat':slice(-50,50)}].copy()
    print(ds)
    for day in ds['day'].values:
        print(day)
        for pressure in ds['pressure'].values:
            print(pressure)
            for time in ds['time'].values:
                ds_unit=ds.loc[{'day':day ,'pressure':pressure,'time':time}]
                frame0=frame_retreiver(ds_unit,'snpp')
                frame=frame_retreiver(ds_unit, 'j1')
                flowx,flowy=calc(frame0, frame)
                breakpoint()
                ds_unit['flowx']=flowx
        
                #ds[label].loc[{'day':day ,'pressure':pressure,
                 #            'satellite':satellite,'time':time}]=frame
    #ds.to_netcdf('../data/processed/real_water_vapor_amvs.nc')

if __name__ == '__main__':
    main()
