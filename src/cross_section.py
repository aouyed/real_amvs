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
import cartopy.crs as ccrs



SCALE=1e4
PRESSURES=[850, 750, 650, 550]
GEODESICS={'swath':[(-47.5, -60), (45, -30),'latitude'],
                'equator':[(6.5, -149.5),(6.5, 4.5),'longitude']}



def shear_calc(ds, tag=''):
    u_diff=ds['u'+tag].diff('plev')
    v_diff=ds['v'+tag].diff('plev')
    p_diff=ds['plev'].diff('plev')
    shear=np.sqrt(u_diff**2+v_diff**2)/p_diff
    return shear


def cloud_filter(ds,date):
    
    ds_qc=xr.open_dataset('../data/processed/real_water_vapor_qc.nc')
    ds_snpp=ds_qc.loc[{'satellite':'snpp','time':'am'}]
    ds_j1=ds_qc.loc[{'satellite':'j1','time':'am'}]
    ds_j1=ds_j1.drop(['day','satellite','time'])
    ds_snpp=ds_snpp.drop(['day','satellite','time'])
    condition1=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_j1['obs_time']))
    condition2=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_snpp['obs_time']))
    ds=ds.where(condition1 & condition2)
    return ds

def preprocess(ds, thresh,date):
    ds=ds.drop(['day','satellite','time','flowx','flowy','obs_time'])
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=np.sqrt(ds['u_error']**2+ds['v_error']**2)
    ds['speed']=np.sqrt(ds['u']**2+ds['v']**2)
    ds['speed_era5']=np.sqrt(ds['u_era5']**2+ds['v_era5']**2)
    ds['speed_diff']=ds['speed']-ds['speed_era5']
    ds['shear']=shear_calc(ds)
    ds['shear_era5']=shear_calc(ds,tag='_era5')
    ds=cloud_filter(ds,date)
    #ds=ds.where(ds['error_mag']<thresh)
    return ds


def quiver_plot(ds, title, u, v,xlabel):
    ds = ds.coarsen(index=2, boundary='trim').mean().coarsen(
                plev=2, boundary='trim').mean()
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(ds[xlabel].values, ds['plev'].values)
    ax.set_title(title)
    Q = ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values))
    ax.quiverkey(Q, 0.8, 0.9, 5, r'5 m/s', labelpos='E',
                      coordinates='figure')
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.tight_layout()
    plt.savefig('../data/processed/plots/quiver_'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
def quiver_ax(ax,ds, title, u, v,xlabel):
    ds = ds.coarsen(index=2, boundary='trim').mean().coarsen(
                plev=2, boundary='trim').mean()
    X, Y = np.meshgrid(ds[xlabel].values, ds['plev'].values)
    ax.set_title(title)
    Q=ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values))
    ax.quiverkey(Q, 0.8, 0.9, 5, r'5 m/s', labelpos='E',
                      coordinates='figure')
    ax.set_ylim(ax.get_ylim()[::-1])
    return ax


def div_calc(u, v, dx, dy):
    div = mpcalc.divergence(u * units['m/s'], v * units['m/s'], 
                            dx=dx, dy=dy)
    div = SCALE*div.magnitude
    return u, v, div



def contourf_plotter(cross,  label, geo, vmin, vmax, xlabel, color='viridis'):
    fig, ax = plt.subplots()
    im=ax.contourf(cross[xlabel], cross['plev'], cross[label],
                         levels=np.linspace(vmin, vmax, 10), cmap=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    ax.set_title(label)
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.savefig('../data/processed/plots/'+label+'_'+geo+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()

    plt.close()
    
    
def contourf_ax(ax, cross,  label, vmin, vmax, xlabel, color='viridis'):
    im=ax.contourf(cross[xlabel], cross['plev'], cross[label],
                         levels=np.linspace(vmin, vmax, 10), cmap=color)
    ax.set_title(label)
    ax.set_ylim(ax.get_ylim()[::-1])
    return ax, im
    
    
    
def multiple_contourf(cross,  label, geo, vmin, vmax, xlabel, geodesic, color='viridis'):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axlist = axes.flat
    axlist[0], im =contourf_ax(axlist[0],cross,  label, vmin,vmax,xlabel)
    axlist[1], im =contourf_ax(axlist[1],cross, label+'_era5',  vmin,vmax,xlabel)
    cbar_ax = fig.add_axes([0.1, -0.05, 0.78, 0.05])
    ax_inset = fig.add_axes([0.125, 0.665, 0.25, 0.25], projection=ccrs.PlateCarree())
    start=geodesic[0]
    end=geodesic[1]
    
    ax_inset.plot(start[1],end[1],start[0],end[0])
    ax_inset.coastlines()

    #fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='[m/s]',pad=0.2)
    

    fig.colorbar(im, cax=cbar_ax,orientation="horizontal", pad=0.2)
    fig.tight_layout()

    #fig.subplots_adjust(hspace=0.7, wspace=0.16)
    plt.savefig('../data/processed/plots/'+label + '_'+geo +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def multiple_quiver(ds, title,xlabel):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axlist = axes.flat
    axlist[0]=quiver_ax(axlist[0],ds, title, 'u', 'v',xlabel)
    axlist[1]=quiver_ax(axlist[1],ds, title+'_era5','u_era5','v_era5',xlabel)
    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+title +
                '.png', bbox_inches='tight', dpi=300)
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
                            'shear_'+str(pressure), 'shear',0,1, units_label='')
        plotter.map_plotter_vmax(ds.sel(plev=pressure, method='nearest'),
                            'shear_era5_'+str(pressure), 'shear_era5',0,1, units_label='')
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
        xlabel=geodesic[2]
        cross = cross_section(data, start, end).set_coords(('latitude', 'longitude'))
        cross=cross.reindex(plev=list(reversed(cross.plev)))
        contourf_plotter(cross,  'humidity_overlap', geokey,  0,0.01,xlabel)
        contourf_plotter(cross,  'error_mag', geokey,  0,5,xlabel)
        multiple_contourf(cross,  'shear', geokey,  0,1,xlabel, geodesic)
        multiple_quiver(cross, 'quiver_'+ geokey,xlabel)
        

    

def main():
    ds=xr.open_dataset('../data/processed/07_01_2020.nc')   
    date=datetime(2020,7,1)
    ds=ds.loc[{'day':date,'time':'am','satellite':'snpp'}].squeeze()
    ds['vort_era5']=SCALE* ds['vort_era5']
    ds['div_era5']=SCALE* ds['div_era5']
    ds['vort_era5_smooth']=SCALE*ds['vort_era5_smooth']
    ds['div_era5_smooth']=SCALE*ds['div_era5_smooth']
    ds=preprocess(ds,10,date)
    map_loop(ds)
    cross_sequence(ds)
   
    
if __name__=="__main__":
    main()
    