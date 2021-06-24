#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:06:15 2021

@author: aouyed

"""
import matplotlib.pyplot as plt
import first_stage_amv as fsa
import xarray as xr
from datetime import datetime
from datetime import timedelta
import numpy as np
import cv2
from dateutil.parser import parse
from metpy.interpolate import cross_section
import metpy.calc as mpcalc
from metpy.units import units

def preprocess(ds, thresh):
    #ds=ds.loc[{'day':datetime(2020,7,3),'time':'pm','satellite':'j1'}]
    #ds=ds.drop(['satellite','time','day'])
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=np.sqrt(ds['u_error']**2+ds['v_error']**2)
    ds['speed']=np.sqrt(ds['u']**2+ds['v']**2)
    ds['speed_era5']=np.sqrt(ds['u_era5']**2+ds['v_era5']**2)
    ds['speed_diff']=ds['speed']-ds['speed_era5']
    ds=ds.where(np.sqrt(ds.error_mag)<thresh)
    return ds

def grad_quants(ds,dx,dy):
    u = ds[ulabel]''.values
    v = ds[vlabel].values
    u = np.squeeze(u)
    v = np.squeeze(v)
    u, v, div = tc.div_calc(
        u.copy(), v.copy(), dx.copy(), dy.copy(), gauss_param, True)
    u, v, vort = tc.vort_calc(
        u.copy(), v.copy(), dx.copy(), dy.copy(), gauss_param, True)
    return div, vort, u, v


def grad_calculator(ds):
    lat = ds.lat.values
    lon = ds.lon.values
    print('calculating deltas...')

    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    div, vort, u, v = grad_quants(
        ds, 'u', 'v', dx, dy)
    ds['divergence'] = (['lat', 'lon'], div)
    ds['vorticity'] = (['lat', 'lon'], vort)
    
def preprocess_loop(ds_total):
   
    pressures = ds_total['plev'].values
    for pressure in pressures:
       
        ds_unit =ds_total.sel(plev=pressure, method='nearest' )
        ds_unit = grad_calculator(ds.copy(), gauss_param, -1)
        ds_unit = ds_unit.assign_coords(pressure=np.array([pressure]))
        ds_total.loc[{'plev'=pressure}]=ds_unit
        



def div_calc(u, v, dx, dy):
    div = mpcalc.divergence(
        u * units['m/s'], v * units['m/s'], dx, dy, dim_order='yx')
    div = div.magnitude
    return u, v, div
def vort_calc(u, v, dx, dy, kernel, is_track):
    vort = mpcalc.vorticity(
        u * units['m/s'], v * units['m/s'], dx, dy, dim_order='yx')
    vort = vort.magnitude
    return u, v, vort


def contourf_plotter(cross,  label, vmin, vmax):
    fig, ax = plt.subplots()
    im=ax.contourf(cross['longitude'], cross['plev'], cross[label],
                         levels=np.linspace(vmin, vmax, 10), cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    plt.savefig('../data/processed/plots/'+label+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()

    plt.close()

def latlon_uniques(ds):
    lat,lon=np.meshgrid(ds['latitude'].values, ds['longitude'].values)
    #ds=ds.rename({'latitude':'y','longitude':'x'})
    ds['lat']=(['longitude','latitude'],lat)
    ds['lon']=(['longitude','latitude'],lon)
    condition1=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds['humidity_overlap']))
    ds_test=ds.where(condition1)
    var=ds_test['lat'].values
    uniques=np.unique(var[~np.isnan(var)])
    print('lat unique')
    print(uniques)
    var=ds_test['lon'].values
    uniques=np.unique(var[~np.isnan(var)])
    print('lon unique')
    print(uniques)
    var=ds_test['humidity_overlap'].values
    uniques=np.unique(var[~np.isnan(var)]).shape
    print('humidity_overlap')
    print(uniques)
    

def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test_3d_'+fsa.ALG+'.nc')   
    ds=ds.loc[{'day':datetime(2020,7,3),'time':'am','satellite':'snpp'}].squeeze()
    ds=preprocess(ds,100)
    ds=preprocess_loop(ds)

    lat,lon=np.meshgrid(ds['latitude'].values, ds['longitude'].values)
    latlon_uniques(ds.copy())
    ds=ds.drop(['day','satellite','time','flowx','flowy','obs_time'])
    
    ds=ds.metpy.assign_crs(grid_mapping_name='latitude_longitude',
    earth_radius=6371229.0)
    
    data = ds.metpy.parse_cf().squeeze()
    latlon_uniques(data.copy())

    #print(ds)
    #start = (-44.5, -29.5)
    #end = (38.5, 4.5)
    start = (5.5, -149.5)
    end = (5.5, 4.5)
    cross = cross_section(data, start, end).set_coords(('latitude', 'longitude'))
    #cross['u_diff']
    print(cross)
    contourf_plotter(cross,  'specific_humidity_mean', 0,0.001)
    contourf_plotter(cross,  'u_era5', -10,10)
    contourf_plotter(cross,  'u', -10,10)
    contourf_plotter(cross,  'error_mag', 0,10)
    contourf_plotter(cross,'humidity_overlap', 0,0.001)
    
if __name__=="__main__":
    main()
    