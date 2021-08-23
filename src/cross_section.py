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
import quiver 


SCALE=1e4
PRESSURES=[850, 750, 650, 550]
GEODESICS={'equator':[(-47.5, -60), (45, -30)],
                'swath':[(6.5, -149.5),(6.5, 4.5)]}



def shear_calc(ds, tag=''):
    u_diff=ds['u'+tag].diff('plev')
    v_diff=ds['v'+tag].diff('plev')
    shear=np.sqrt(u_diff**2+v_diff**2)
    return shear
    
def preprocess(ds, thresh):
 
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=np.sqrt(ds['u_error']**2+ds['v_error']**2)
    ds['speed']=np.sqrt(ds['u']**2+ds['v']**2)
    ds['speed_era5']=np.sqrt(ds['u_era5']**2+ds['v_era5']**2)
    ds['speed_diff']=ds['speed']-ds['speed_era5']
    ds['shear']=shear_calc(ds)
    ds['shear_era5']=shear_calc(ds,tag='_era5')
    ds=ds.where(ds.error_mag<thresh)
    return ds

def vel_filter(u, v):
   
    u =inpainter.drop_nan(u)
    v = inpainter.drop_nan(v)
    mask=np.isnan(u)
    u=cv2.blur(u, (5,5))
    v=cv2.blur(v, (5,5))
    u[mask]=np.nan
    v[mask]=np.nan
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
    ds['div'] = (['latitude', 'longitude'], div)
    ds['vort'] = (['latitude', 'longitude'], vort)
    ds['vort_smooth']= ds['vort'].rolling(latitude=5, 
                                         longitude=5, center=True).mean()
    ds['div_smooth']= ds['div'].rolling(latitude=5, 
                                          longitude=5, center=True).mean()

    return ds
    

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


def contourf_plotter(cross,  label, geo, vmin, vmax, color='viridis'):
    fig, ax = plt.subplots()
    im=ax.contourf(cross['longitude'], cross['plev'], cross[label],
                         levels=np.linspace(vmin, vmax, 10), cmap=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    ax.set_title(label)
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.savefig('../data/processed/plots/'+label+'_'+geo+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()

    plt.close()

def latlon_uniques(ds):
    lat,lon=np.meshgrid(ds['latitude'].values, ds['longitude'].values)
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
    
    
def map_loop(ds):
    for pressure in PRESSURES:
        plotter.map_plotter(ds.sel(plev=pressure, method='nearest'),
                            'humidity_overlap_map_'+str(pressure), 'humidity_overlap', units_label='')
        plotter.map_plotter_vmax(ds.sel(plev=pressure, method='nearest'),
                            'shear_'+str(pressure), 'shear',0,6, units_label='')
        plotter.map_plotter_vmax(ds.sel(plev=pressure, method='nearest'),
                            'shear_era5_'+str(pressure), 'shear_era5',0,6, units_label='')
        quiver.quiver_plot(ds.sel(plev=pressure,method='nearest'), 'quivermap_'+str(pressure), 'u', 'v')
        quiver.quiver_plot(ds.sel(plev=pressure,method='nearest'), 'quivermap_era5_'+str(pressure), 'u_era5', 'v_era5')

   

def cross_sequence(ds):
    ds=ds.metpy.assign_crs(grid_mapping_name='latitude_longitude',earth_radius=6371229.0)
    data = ds.metpy.parse_cf().squeeze()
    latlon_uniques(data.copy())
    for geokey in GEODESICS:
        geodesic=GEODESICS[geokey]
        start=geodesic[0]
        end=geodesic[1]
        cross = cross_section(data, start, end).set_coords(('latitude', 'longitude'))
        cross=cross.reindex(plev=list(reversed(cross.plev)))
        contourf_plotter(cross,  'error_mag', geokey,  0,5)
        quiver_plot(cross, 'quiver_'+geokey, 'u', 'v')
        quiver_plot(cross, 'quiver_era5_'+geokey, 'u_era5', 'v_era5')

    

def main():
    ds=xr.open_dataset('../data/processed/07_01_2020.nc')   
    ds=ds.loc[{'day':datetime(2020,7,1),'time':'am','satellite':'snpp'}].squeeze()
    ds['vort_era5']=SCALE* ds['vort_era5']
    ds['div_era5']=SCALE* ds['div_era5']
    ds['vort_era5_smooth']=SCALE*ds['vort_era5_smooth']
    ds['div_era5_smooth']=SCALE*ds['div_era5_smooth']
    ds=preprocess(ds,10)
    ds=ds.drop(['day','satellite','time','flowx','flowy','obs_time'])
    map_loop(ds)
    cross_sequence(ds)
   
    
if __name__=="__main__":
    main()
    