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

drad = np.deg2rad(1)
dx = R*drad
scale_x = dx
dy = R*drad
scale_y = dy


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
    frame=inpainter.drop_nan(frame)
    return frame
    
def prepare_ds(ds):
    ds['humidity_overlap'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['flowx'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['flowy'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['u'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['v'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    return ds

def swath_initializer(ds):   
    dhours=60   
    mind=ds['obs_time'].min(skipna=True).values
    hours=np.arange(dhours,dhours*24,dhours)
    swathes=[]

    swathes.append([mind, mind+np.timedelta64(dhours, 'm')])
    for hour in hours:
        swathes.append([mind+np.timedelta64(hour, 'm'),
                        mind+np.timedelta64(hour+dhours, 'm')])
    return swathes
    

def prepare_patch(ds_snpp, ds_j1, start, end):
    condition1=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_j1['obs_time']))
    condition2=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_snpp['obs_time']))
    ds_j1_unit=ds_j1[['specific_humidity_mean','obs_time']].where(condition1 & condition2)
    ds_snpp_unit=ds_snpp[['specific_humidity_mean','obs_time']].where(condition1 & condition2)
    df_j1=ds_j1_unit.to_dataframe().dropna(subset=[LABEL]).set_index('satellite',append=True)
    df_snpp=ds_snpp_unit.to_dataframe().dropna(subset=[LABEL]).set_index('satellite',append=True)
       
    ds_j1=xr.Dataset.from_dataframe(df_j1)
    ds_snpp=xr.Dataset.from_dataframe(df_snpp)
    ds_merged=xr.merge([ds_j1,ds_snpp])
    
    return ds_merged, ds_snpp, ds_j1

def flow_calculator(ds_snpp, ds_j1):
    frame0=frame_retreiver(ds_snpp)
    frame=frame_retreiver(ds_j1)
    flowx,flowy=calc(frame0, frame)
    flowx = np.expand_dims(flowx, axis=2)
    flowy = np.expand_dims(flowy, axis=2)
    ds_snpp['flowx']=(['latitude','longitude','satellite'],flowx)
    ds_snpp['flowy']=(['latitude','longitude','satellite'],flowy)
    ds_j1['flowx']=(['latitude','longitude','satellite'],flowx)
    ds_j1['flowy']=(['latitude','longitude','satellite'],flowy)
    return ds_snpp, ds_j1

def amv_calculator(ds_merged, df):
    ds_snpp=ds_merged.loc[{'satellite':'snpp'}].expand_dims('satellite')
    ds_j1=ds_merged.loc[{'satellite':'j1'}].expand_dims('satellite')
    ds_snpp, ds_j1=flow_calculator(ds_snpp, ds_j1)
            
    dt=ds_merged['obs_time'].loc[
        {'satellite':'j1'}]-ds_merged['obs_time'].loc[{'satellite':'snpp'}]
    dt_int=dt.values.astype('timedelta64[s]').astype(np.int32)
    dt_inv=1/dt_int
    dt_inv[dt_inv==np.inf]=np.nan
    dx_conv=abs(np.cos(np.deg2rad(ds_merged.latitude)))
    ds_merged['dt_inv']=(['latitude','longitude'],dt_inv)
    ds_snpp['u']= scale_x*dx_conv*ds_merged['dt_inv']*ds_snpp['flowx']
    ds_snpp['v']= scale_y*ds_merged['dt_inv']*ds_snpp['flowy']
            
    ds_j1['u']= scale_x*dx_conv*ds_merged['dt_inv']*ds_j1['flowx']
    ds_j1['v']= scale_y*ds_merged['dt_inv']*ds_j1['flowy']
    df_j1=ds_j1.to_dataframe().dropna()
    df_snpp=ds_snpp.to_dataframe().dropna()
        
    swathi=df_snpp.index.values 
    df['humidity_overlap'].loc[df.index.isin(swathi)]=df_snpp['specific_humidity_mean']
    df['flowx'].loc[df.index.isin(swathi)]=df_snpp['flowx']
    df['flowy'].loc[df.index.isin(swathi)]=df_snpp['flowy']
    df['u'].loc[df.index.isin(swathi)]=df_snpp['u']
    df['v'].loc[df.index.isin(swathi)]=df_snpp['v']
    swathi=df_j1.index.values 
            #print(swathi)
    df['humidity_overlap'].loc[df.index.isin(swathi)]=df_j1['specific_humidity_mean']
    df['flowx'].loc[df.index.isin(swathi)]=df_j1['flowx']
    df['flowy'].loc[df.index.isin(swathi)]=df_j1['flowy']
    df['u'].loc[df.index.isin(swathi)]=df_j1['u']
    df['v'].loc[df.index.isin(swathi)]=df_j1['v']
    return df
    

