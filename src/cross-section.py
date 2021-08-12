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
import inpainter 
import plotter 

SCALE=1e4

def shear_calc(ds, tag=''):
    u_diff=ds['u'+tag].diff('plev')
    v_diff=ds['v'+tag].diff('plev')
    shear=np.sqrt(u_diff**2+v_diff**2)
    return shear
    
def preprocess(ds, thresh):
    #ds=ds.loc[{'day':datetime(2020,7,3),'time':'pm','satellite':'j1'}]
    #ds=ds.drop(['satellite','time','day'])
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['vort_error']=ds['vorticity']-ds['vorticity_era5']
    ds['div_error']=ds['divergence']-ds['divergence_era5']
    ds['error_mag']=np.sqrt(ds['u_error']**2+ds['v_error']**2)
    ds['speed']=np.sqrt(ds['u']**2+ds['v']**2)
    ds['speed_era5']=np.sqrt(ds['u_era5']**2+ds['v_era5']**2)
    ds['speed_diff']=ds['speed']-ds['speed_era5']
    ds['shear']=shear_calc(ds)
    ds['shear_era5']=shear_calc(ds,tag='_era5')
    ds['diff_vort']=ds['vorticity'].diff('plev')
    ds['diff_div']=ds['divergence'].diff('plev')
    ds['diff_vort_era5']=ds['vorticity_era5'].diff('plev')
    ds['diff_div_era5']=ds['divergence_era5'].diff('plev')
    ds['diff_div_error']=ds['diff_div']-ds['diff_div_era5']
    ds['diff_vort_error']=ds['diff_vort']-ds['diff_vort_era5']
    ds=ds.where(ds.error_mag<thresh)
    print('stats')
    print(abs(ds['diff_vort_error']).mean())
    print(abs(ds['diff_vort']).mean())
    #ds=ds.where(abs(ds.diff_vort_error)<0.02)
    print(abs(ds['diff_vort_error']).mean())
    print(abs(ds['diff_vort']).mean())
    print(abs(ds['diff_vort_era5']).mean())

    return ds

def vel_filter(u, v):
   
    u =inpainter.drop_nan(u)
    v = inpainter.drop_nan(v)
    #u=np.nan_to_num(u)
    #v=np.nan_to_num(v)

    return u, v


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
    ds['divergence'] = (['latitude', 'longitude'], div)
    ds['vorticity'] = (['latitude', 'longitude'], vort)
    return ds
    
def preprocess_loop(ds_total, tag=''):
    ds_total['vorticity'+tag]= xr.full_like(ds_total['specific_humidity_mean'], fill_value=np.nan)
    ds_total['divergence'+tag]= xr.full_like(ds_total['specific_humidity_mean'], fill_value=np.nan)
    pressures = ds_total['plev'].values
    for pressure in pressures:
        print(pressure)
        print(pressures.shape)
       
        ds_unit =ds_total.sel(plev=pressure, method='nearest' )
        ds_unit = grad_calculator(ds_unit, tag)
        ds_total['vorticity'+tag].loc[{'plev':pressure}]=ds_unit['vorticity']
        ds_total['divergence'+tag].loc[{'plev':pressure}]=ds_unit['divergence']

    return ds_total

def quiver_plot(ds, title, u, v):
    ds = ds.coarsen(index=2, boundary='trim').mean().coarsen(
                plev=2, boundary='trim').mean()
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(ds['longitude'].values, ds['plev'].values)
    ax.set_title(title)
    Q = ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values))
    qk = ax.quiverkey(Q, 0.8, 0.9, 5, r'5 m/s', labelpos='E',
                      coordinates='figure')
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.tight_layout()
    plt.savefig('../data/processed/plots/quiver_'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def div_calc(u, v, dx, dy):
    div = mpcalc.divergence(u * units['m/s'], v * units['m/s'], 
                            dx=dx, dy=dy)
    div = SCALE*div.magnitude
    return u, v, div

def vort_calc(u, v, dx, dy):
    vort = mpcalc.vorticity(
        u * units['m/s'], v * units['m/s'], dx=dx, dy=dy)
    vort = SCALE*vort.magnitude
    
    return u, v, vort


def contourf_plotter(cross,  label, vmin, vmax, color='viridis'):
    fig, ax = plt.subplots()
    im=ax.contourf(cross['longitude'], cross['plev'], cross[label],
                         levels=np.linspace(vmin, vmax, 10), cmap=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    ax.set_title(label)
    ax.set_ylim(ax.get_ylim()[::-1])
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
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test_3d_8'+fsa.ALG+'.nc')   
    ds=ds.loc[{'day':datetime(2020,7,3),'time':'am','satellite':'snpp'}].squeeze()
    ds=preprocess_loop(ds)
    ds=preprocess_loop(ds, tag='_era5')
    ds.to_netcdf('../data/interim/cross.nc')
    ds=xr.open_dataset('../data/interim/cross.nc')
    ds=preprocess(ds,10)
    
   

    ds['diff_vort_lowres']=ds['vorticity'].sel(plev=300, method='nearest')-ds['vorticity'].sel(plev=850, method='nearest')
    plotter.map_plotter(ds.sel(plev=850, method='nearest'),
                        'divergence_map', 'divergence', units_label='')
    ds['diff_vort_era5_lowres']=ds['vorticity_era5'].sel(plev=300, method='nearest')-ds['vorticity_era5'].sel(plev=850, method='nearest')
    plotter.map_plotter(ds.sel(plev=850, method='nearest'),
                        'humidity_overlap_map', 'humidity_overlap', units_label='')
    plotter.map_plotter_vmax(ds,'diff_vort_lowres', 'diff_vort_lowres',-0.1,0.1 ,color='RdBu')
    plotter.map_plotter_vmax(ds,'diff_vort_era5_lowres', 'diff_vort_era5_lowres',-0.5,0.5 ,color='RdBu')

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
    #start = (-44.5, -29.5)
    #end = (38.5, 4.5)
    start = (6.5, -149.5)
    end = (6.5, 4.5)
    #start = (6.5, 0)
    #end = (6.5, 150)
    cross = cross_section(data, start, end).set_coords(('latitude', 'longitude'))
    cross=cross.reindex(plev=list(reversed(cross.plev)))
    #cross['u_diff']
    print(cross)
    contourf_plotter(cross,  'specific_humidity_mean', 0,0.001)
    contourf_plotter(cross,  'u_era5', -10,10)
    contourf_plotter(cross,  'u', -10,10)
    contourf_plotter(cross,  'error_mag', 0,5)
    contourf_plotter(cross,'humidity_overlap', 0,0.001)
    contourf_plotter(cross,'divergence', -0.1,0.1, color='RdBu')
    contourf_plotter(cross,'vorticity', -0.1,0.1, color='RdBu')
    contourf_plotter(cross,'divergence_era5', -0.15,0.15, color='RdBu')
    contourf_plotter(cross,'vorticity_era5', -0.15,0.15, color='RdBu')
    contourf_plotter(cross,'vort_error', -0.15,0.15, color='RdBu')
    contourf_plotter(cross,'div_error', -0.15,0.15, color='RdBu')
    contourf_plotter(cross,'diff_vort', -0.05,0.05, color='RdBu')
    contourf_plotter(cross,'diff_div', -0.05,0.05, color='RdBu')
    contourf_plotter(cross,'diff_vort_era5', -0.05,0.05, color='RdBu')
    contourf_plotter(cross,'diff_div_era5', -0.05,0.05, color='RdBu')
    contourf_plotter(cross,'diff_vort_error', -0.02,0.02, color='RdBu')
    contourf_plotter(cross,'diff_div_error', -0.02,0.02, color='RdBu')
    contourf_plotter(cross,'shear', 0,6)
    contourf_plotter(cross,'shear_era5', 0,6)
    quiver_plot(cross, 'cross', 'u', 'v')
    quiver_plot(cross, 'cross_era5', 'u_era5', 'v_era5')

    
if __name__=="__main__":
    main()
    