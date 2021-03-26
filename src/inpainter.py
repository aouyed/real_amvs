#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:40:11 2021

@author: aouyed
"""
import cv2
import xarray as xr
import numpy as np

def drop_nan(frame):
    mask=np.isnan(frame)
    mask = np.uint8(mask)
    frame = np.nan_to_num(frame)
    frame = cv2.inpaint(frame, mask, inpaintRadius=10, flags=cv2.INPAINT_NS)
    return frame

def main():
    label='specific_humidity_mu'
    ds=xr.open_dataset('../data/processed/real_water_vapor.nc')
    print(ds)
    for day in ds['day'].values:
        print(day)
        for pressure in ds['pressure'].values:
            print(pressure)
            for satellite in ds['satellite'].values:
                for time in ds['time'].values:
                    frame= ds[label].loc[{'day':day ,'pressure':pressure,
                                          'satellite':satellite,'time':time}].values       
                    frame = np.squeeze(frame)
                    frame[frame == 0] = np.nan
                    frame=drop_nan(frame)
                    breakpoint()
                    ds[label].loc[{'day':day ,'pressure':pressure,
                             'satellite':satellite,'time':time}]=frame
    ds.to_netcdf('../data/processed/real_water_vapor_inpainted.nc')



if __name__=="__main__":
    main()
 