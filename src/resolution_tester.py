#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 12:30:39 2022

@author: aouyed
"""

import xarray as xr
import numpy as np
import amv_calculators as ac
import time
import quiver as q 
import pandas as pd 

THETA=0.0625
R=ac.R


dt=3600
dt_inv=1/dt

def prepare_ds(ds,factor):
    ds1=ds.sel(pressure=500, method='nearest')
    ds1=ds1.sel(lat=slice(-70,70))
    
    ds1=ds1.coarsen(lat=factor, lon=factor, boundary='trim').mean()
    return ds1
    

speed_biases={'resolution':[],'speed_bias':[],'rmsvd':[]}


for factor in (16,8,4,2, 1):
    res=factor*THETA
    drad = np.deg2rad(res)
    dx = R*drad
    scale_x = dx
    dy = R*drad
    scale_y = dy    
    ds=xr.open_dataset('../data/raw/nature_run/NRData_20060101_1700.nc')
    ds1=prepare_ds(ds,factor)
    frame0=np.squeeze(ds1['qv'].values)
    
    
    ds=xr.open_dataset('../data/raw/nature_run/NRData_20060101_1800.nc')
    ds2=prepare_ds(ds,factor)
    frame=np.squeeze(ds2['qv'].values)
    
    
    start = time.time()
    flowx, flowy=ac.calc(frame0,frame, 0.15)
    end = time.time()
    print(end - start)
    
    ds=(ds1+ds2)*0.5
    
    ds['flowx']=(['lat','lon'], flowx)
    ds['flowy']=(['lat','lon'], flowy)
    
    dx_conv=abs(np.cos(np.deg2rad(ds.lat)))
    ds['utrack']= scale_x*dx_conv*dt_inv*ds['flowx']
    ds['vtrack']= scale_y*dt_inv*ds['flowy']
    ds['speed']=np.sqrt(ds.u**2+ds.v**2)
    ds['speed_track']=np.sqrt(ds.utrack**2+ds.vtrack**2)
    ds['speed_bias']=ds.speed_track-ds.speed
    ds['u_error']=ds.utrack-ds.u
    ds['v_error']=ds.vtrack-ds.v
    ds['square_error']=ds.u_error**2+ds.v_error**2
    
    rmsvd=np.sqrt(ds['square_error'].mean().item())
    speed_bias=ds['speed_bias'].mean().item()
    print('speed bias: ' + str(speed_bias))
    speed_biases['resolution'].append(res)
    speed_biases['speed_bias'].append(speed_bias)
    speed_biases['rmsvd'].append(rmsvd)

    ds=ds.rename({'lat':'latitude','lon':'longitude'})
    q.quiver_plot_cartopy(ds,'res_test_'+str(factor),'utrack','vtrack')
    q.quiver_plot_cartopy(ds,'nr_test'+str(factor),'u','v')
    
df=pd.DataFrame(data=speed_biases)
print(df)
breakpoint()

