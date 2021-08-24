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
import quiver
import first_stage_amv as fsa
import cross_section as cs



R = 6371000

drad = np.deg2rad(1)
dx = R*drad
scale_x = dx
dy = R*drad
scale_y = dy
np.seterr(divide='ignore')


LABEL='specific_humidity_mean'


def calc(frame0, frame):
    if frame0.shape != frame.shape:
        frame=np.resize(frame, frame0.shape)
    
    
    nframe0 = cv2.normalize(src=frame0, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    nframe = cv2.normalize(src=frame, dst=None,
                            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    #optical_flow = cv2.optflow.createOptFlow_DeepFlow()
    if fsa.ALG == 'tvl1':
        print('tvl1')
        print('tvl1')
        print('tvl1')
        print('tvl1')
        print('tvl1')
        print('tvl1')
        optical_flow=cv2.optflow.createOptFlow_DualTVL1()
        flowd = optical_flow.calc(nframe0, nframe, None)
    else:
        flowd=cv2.calcOpticalFlowFarneback(nframe0,nframe, None, 0.5, 3, 20, 3, 7, 1.2, 0)
    flowx=flowd[:,:,0]
    flowy=flowd[:,:,1]
     
    return flowx, flowy
 
def frame_retreiver(ds):
    frame= np.squeeze(ds[LABEL].values)  
    #frame=np.nan_to_num(frame)
    frame=inpainter.drop_nan(frame)
    return frame
    
def prepare_ds(ds):
    ds['humidity_overlap'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['flowx'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['flowy'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['u'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['v'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['u_era5'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['v_era5'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['div_era5'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['vort_era5'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['div'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['vort'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['vort_era5_smooth'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['div_era5_smooth'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['div_smooth'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['vort_smooth'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)
    ds['dt_inv'] = xr.full_like(ds['specific_humidity_mean'], fill_value=np.nan)

    return ds

def swath_initializer(ds, dmins, swath_hours):   
    number=(swath_hours*60)/dmins
    #mind=ds['obs_time'].min(skipna=True).values
    mind=np.nanmin(ds['obs_time'].values)
    times=np.arange(dmins,dmins*number,dmins)
    swathes=[]
    swathes.append([mind, mind+np.timedelta64(dmins, 'm')])
    for time in times:
        time=int(round(time))
        swathes.append([mind+np.timedelta64(time, 'm'),
                        mind+np.timedelta64(time+dmins, 'm')])
    return swathes
    

def prepare_patch(ds_snpp, ds_j1, ds_model, start, end):
    ds_snpp=ds_snpp.where((ds_snpp.obs_time > start) & (ds_snpp.obs_time<end))
    model_t=start+np.timedelta64(25, 'm')
    start=start+np.timedelta64(50, 'm')
    end=end+np.timedelta64(50, 'm')
    ds_j1=ds_j1.where((ds_j1.obs_time > start) & +(ds_j1.obs_time<end))
    
    
    
    condition1=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_j1['obs_time']))
    condition2=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_snpp['obs_time']))
    ds_j1_unit=ds_j1[['specific_humidity_mean','obs_time']].where(condition1 & condition2)
    ds_snpp_unit=ds_snpp[['specific_humidity_mean','obs_time']].where(condition1 & condition2)
    df_j1=ds_j1_unit.to_dataframe().dropna(subset=[LABEL]).set_index('satellite',append=True)
    df_snpp=ds_snpp_unit.to_dataframe().dropna(subset=[LABEL]).set_index('satellite',append=True)
       
    ds_j1=xr.Dataset.from_dataframe(df_j1)
    ds_snpp=xr.Dataset.from_dataframe(df_snpp)
    ds_merged=xr.merge([ds_j1,ds_snpp])
    ds_model=ds_model.sel(time=model_t, latitude=ds_merged['latitude'].values, longitude=ds_merged['longitude'].values, method='nearest' )
    ds_model['latitude']=ds_merged['latitude'].values
    ds_model['longitude']=ds_merged['longitude'].values
    ds_model=ds_model.drop('time')
    ds_merged=xr.merge([ds_merged, ds_model])
    
    return ds_merged, ds_snpp, ds_j1, df_snpp

def flow_calculator(ds_snpp, ds_j1, ds_merged):
    frame0=frame_retreiver(ds_snpp)
    frame=frame_retreiver(ds_j1)
    flowx,flowy=calc(frame0, frame)
    flowx = np.expand_dims(flowx, axis=2)
    flowy = np.expand_dims(flowy, axis=2)
    ds_snpp['flowx']=(['latitude','longitude','satellite'],flowx)
    ds_snpp['flowy']=(['latitude','longitude','satellite'],flowy)
    ds_j1['flowx']=(['latitude','longitude','satellite'],flowx)
    ds_j1['flowy']=(['latitude','longitude','satellite'],flowy)
    frame = np.expand_dims(frame, axis=2)
    frame0 = np.expand_dims(frame0, axis=2) 
    dt=ds_merged['obs_time'].loc[
        {'satellite':'j1'}]-ds_merged['obs_time'].loc[{'satellite':'snpp'}]
    dt_int=dt.values.astype('timedelta64[s]').astype(np.int32)
    dt_inv=1/dt_int
    dt_inv[dt_inv==np.inf]=np.nan
    dx_conv=abs(np.cos(np.deg2rad(ds_merged.latitude)))
    ds_merged['dt_inv']=(['latitude','longitude'],dt_inv)
    
    ds_snpp['dt_inv']= ds_merged['dt_inv'] 
    ds_snpp['u']= scale_x*dx_conv*ds_merged['dt_inv']*ds_snpp['flowx']
    ds_snpp['v']= scale_y*ds_merged['dt_inv']*ds_snpp['flowy']
    ds_snpp=cs.grad_calculator(ds_snpp,'')
 
    ds_j1['dt_inv']= ds_merged['dt_inv']        
    ds_j1['u']= scale_x*dx_conv*ds_merged['dt_inv']*ds_j1['flowx']
    ds_j1['v']= scale_y*ds_merged['dt_inv']*ds_j1['flowy']
    ds_j1=cs.grad_calculator(ds_j1,'')
    return ds_snpp, ds_j1

def df_filler(df, df_sat):
    swathi=df_sat.index.values 
    df['humidity_overlap'].loc[df.index.isin(swathi)]=df_sat['specific_humidity_mean']
    df['flowx'].loc[df.index.isin(swathi)]=df_sat['flowx']
    df['flowy'].loc[df.index.isin(swathi)]=df_sat['flowy']
    df['u'].loc[df.index.isin(swathi)]=df_sat['u']
    df['v'].loc[df.index.isin(swathi)]=df_sat['v']
    df['vort'].loc[df.index.isin(swathi)]=df_sat['vort']
    df['div'].loc[df.index.isin(swathi)]=df_sat['div']
    df['vort_smooth'].loc[df.index.isin(swathi)]=df_sat['vort_smooth']
    df['div_smooth'].loc[df.index.isin(swathi)]=df_sat['div_smooth']
    df['dt_inv'].loc[df.index.isin(swathi)]=df_sat['dt_inv']
    return df

def df_filler_model(df, df_sat, df_m):
    swathi=df_sat.index.values 
    df['u_era5'].loc[df.index.isin(swathi)]=df_m['u']
    df['v_era5'].loc[df.index.isin(swathi)]=df_m['v']
    df['vort_era5'].loc[df.index.isin(swathi)]=df_m['vo']
    df['div_era5'].loc[df.index.isin(swathi)]=df_m['d']
    df['vort_era5_smooth'].loc[df.index.isin(swathi)]=df_m['vort_era5_smooth']
    df['div_era5_smooth'].loc[df.index.isin(swathi)]=df_m['div_era5_smooth']


    return df
    

def amv_calculator(ds_merged, df):
    ds_snpp=ds_merged.loc[{'satellite':'snpp'}].expand_dims('satellite')
    ds_j1=ds_merged.loc[{'satellite':'j1'}].expand_dims('satellite')
    ds_snpp, ds_j1=flow_calculator(ds_snpp, ds_j1, ds_merged)
    #ds_merged=ds_merged.where(ds_merged['specific_humidity_mean'])
    #df_j1=ds_j1.to_dataframe().dropna()
    #df_snpp=ds_snpp.to_dataframe().dropna()
    df_j1=ds_j1.to_dataframe()
    df_snpp=ds_snpp.to_dataframe()
    df_model=ds_merged.loc[{'satellite':'snpp'}].drop('satellite').to_dataframe().dropna()   
    df=df_filler(df, df_snpp)
    df=df_filler(df, df_j1)
    df=df_filler_model(df, df_j1, df_model)
    df=df_filler_model(df, df_snpp, df_model)
    
    return df, ds_snpp, ds_j1, ds_merged


def grad_quants(ds,ulabel,vlabel, dx,dy):
    u = ds[ulabel].values
    v = ds[vlabel].values
    u = np.squeeze(u)
    v = np.squeeze(v)
    mask=np.isnan(u)
    u,v=vel_filter(u, v)
    u, v, div = div_calc(
        u, v, dx, dy)
    u, v, vort =vort_calc(
        u, v, dx, dy)
    div[mask]=np.nan
    vort[mask]=np.nan
    return div, vort, u, v


def grad_calculator(ds, tag):
    lat = ds.latitude.values
    lon = ds.longitude.values
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    div, vort, u, v = grad_quants(ds, 'u'+tag,'v'+tag,dx, dy)
    ds['div'] = (['latitude', 'longitude'], div)
    ds['vort'] = (['latitude', 'longitude'], vort)
    ds['vort_smooth']= ds['vort'].rolling(latitude=5, 
                                         longitude=5, center=True).mean()
    ds['div_smooth']= ds['div'].rolling(latitude=5, 
                                          longitude=5, center=True).mean()

    return ds
    
def vort_calc(u, v, dx, dy):
    vort = mpcalc.vorticity(
        u * units['m/s'], v * units['m/s'], dx=dx, dy=dy)
    vort = SCALE*vort.magnitude
    
    return u, v, vort


def div_calc(u, v, dx, dy):
    div = mpcalc.divergence(u * units['m/s'], v * units['m/s'], 
                            dx=dx, dy=dy)
    div = SCALE*div.magnitude
    return u, v, div



    

