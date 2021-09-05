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
scale_g=1000
PRESSURES=[850, 700, 500, 400]
GEODESICS={'swath':[(-47.5, -60), (45, -30),'latitude'],
                'equator':[(6.5, -149.5),(6.5, 4.5),'longitude']}



def shear_calc(ds, tag=''):
    u=ds['u'+tag].rolling(plev=3, latitude=3, longitude=3).mean()
    v=ds['v'+tag].rolling(plev=3, latitude=3, longitude=3).mean()
    u_diff=u.differentiate('plev')
    v_diff=v.differentiate('plev')
    p_diff=ds['plev'].differentiate('plev')
    ds['u_shear'+tag]=u_diff/p_diff
    ds['v_shear'+tag]=v_diff/p_diff
    return ds


def shear_two_levels(ds, tag=''):
    u_diff=ds['u'+tag].sel(plev=850, method='nearest')-ds['u'+tag].sel(plev=500, method='nearest')
    v_diff=ds['v'+tag].sel(plev=850, method='nearest')-ds['v'+tag].sel(plev=500, method='nearest')
    shear=np.sqrt(u_diff**2+v_diff**2)
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

def preprocess(ds, thresh):
    ds=ds.drop(['day','satellite','time','flowx','flowy','obs_time'])
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=np.sqrt(ds['u_error']**2+ds['v_error']**2)
    ds['squared_error']=ds['u_error']**2+ds['v_error']**2

    ds['speed']=np.sqrt(ds['u']**2+ds['v']**2)
    ds['speed_era5']=np.sqrt(ds['u_era5']**2+ds['v_era5']**2)
    ds['speed_diff']=ds['speed']-ds['speed_era5']
    ds=shear_calc(ds)
    ds=shear_calc(ds,tag='_era5')
    if thresh>0:
        ds=ds.where(ds['error_mag']<thresh)
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
    
def quiver_ax(ax,ds, title, u, v,xlabel, qkey=5, units='m/s'):
    ds = ds.coarsen(index=2, boundary='trim').mean().coarsen(
                plev=2, boundary='trim').mean()
    X, Y = np.meshgrid(ds[xlabel].values, ds['plev'].values)
    ax.set_title(title)
    Q=ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values))
    ax.quiverkey(Q, 0.8, 0.9, qkey, str(qkey)+' m/s', labelpos='E',
                      coordinates='figure')
    ax.set_ylim(ax.get_ylim()[::-1])
    #ax.set_ylim(1000,500)
    return ax


def div_calc(u, v, dx, dy):
    div = mpcalc.divergence(u * units['m/s'], v * units['m/s'], 
                            dx=dx, dy=dy)
    div = SCALE*div.magnitude
    return u, v, div



def contourf_plotter(cross,  label, geo, vmin, vmax, xlabel,geodesic,thresh,  units_label='', color='viridis'):
    fig, ax = plt.subplots()
    im=ax.contourf(cross[xlabel], cross['plev'], cross[label],
                         levels=np.linspace(vmin, vmax, 10), cmap=color)
    cbar_ax = fig.add_axes([0.1, -0.05, 0.78, 0.05])
    fig.colorbar(im, cax=cbar_ax,orientation="horizontal", pad=0.5, label=units_label)    
    ax_inset=inset_plot(geodesic, fig)

    ax.set_title(label)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel(geodesic[2])
    ax.set_ylabel('[hPa]')
    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+label+'_'+geo+'_'+str(thresh)+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()

    plt.close()
    
    
def contourf_ax(ax, cross,  label, vmin, vmax, xlabel, color='viridis'):
    im=ax.contourf(cross[xlabel], cross['plev'], cross[label],
                         levels=np.linspace(vmin, vmax, 10), cmap=color)
    ax.set_title(label)
    ax.set_ylim(ax.get_ylim()[::-1])
    return ax, im
    

def inset_plot(geodesic, fig):
    ax_inset = fig.add_axes([0.085, 0.9, 0.25, 0.25], projection=ccrs.PlateCarree())
    start=geodesic[0]
    end=geodesic[1]
    ax_inset.set_global()
    ax_inset.plot([start[1],end[1]],[start[0],end[0]],transform=ccrs.Geodetic(), linewidth=3)
    ax_inset.coastlines()
    return ax_inset
    
    
def multiple_contourf(cross,  label, geo, vmin, vmax, xlabel, geodesic, color='viridis', units_label=''):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axlist = axes.flat
    axlist[0], im =contourf_ax(axlist[0],cross,  label, vmin,vmax,xlabel)
    axlist[0].set_xlabel(geodesic[2])
    axlist[0].set_ylabel('[hPa]')
    axlist[1], im =contourf_ax(axlist[1],cross, label+'_era5',  vmin,vmax,xlabel)
    cbar_ax = fig.add_axes([0.1, -0.05, 0.78, 0.05])
    fig.colorbar(im, cax=cbar_ax,orientation="horizontal", pad=0.5,label=units_label) 

    ax_inset=inset_plot(geodesic, fig)
    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+label + '_'+geo +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def multiple_quiver(ds, title, geodesic, xlabel, tag='',qkey=5, units='m/s'):
    fig, axes = plt.subplots(nrows=2, ncols=1)
    axlist = axes.flat
    axlist[0]=quiver_ax(axlist[0],ds, title, 'u'+tag, 'v'+tag,xlabel, qkey, units)
    axlist[0].set_xlabel(geodesic[2])
    axlist[1]=quiver_ax(axlist[1],ds, title+'_era5','u'+tag+'_era5','v'+tag+'_era5',xlabel, qkey, units)
    ax_inset=inset_plot(geodesic, fig)
    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+title +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def multiple_quiver_map(ds, title, tag=''):
    fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={
                             'projection': ccrs.PlateCarree()})
    axlist = axes.flat
    
    for index, pressure in enumerate(PRESSURES):
        print(axlist[index])
        quiver.quiver_ax_cartopy( axlist[index],ds.sel(plev=pressure, method='nearest'), str(pressure)+' hPa', 'u'+tag, 'v'+tag)
   
    fig.tight_layout()
    print('saving ' + title)
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
        plotter.map_plotter_cartopy(ds.sel(plev=pressure, method='nearest'),
                            'humidity_overlap_map_'+str(pressure), 'humidity_overlap', 0,0.015, units_label='[g/g]')
        plotter.map_plotter_cartopy(ds.sel(plev=pressure, method='nearest'),
                            'shear_'+str(pressure), 'shear',0,1, units_label='[m/(s hPa)]')
        plotter.map_plotter_cartopy(ds.sel(plev=pressure, method='nearest'),
                            'shear_era5_'+str(pressure), 'shear_era5',0,1, units_label='[m/(s hPa)]')
        quiver.quiver_plot_cartopy(ds.sel(plev=pressure,method='nearest'), 'quivermap_'+str(pressure), 'u', 'v')
        quiver.quiver_plot_cartopy(ds.sel(plev=pressure,method='nearest'), 'quivermap_era5_'+str(pressure), 'u_era5', 'v_era5')

   

def cross_sequence(ds, thresh):
    ds=ds.metpy.assign_crs(grid_mapping_name='latitude_longitude',earth_radius=6371229.0)
    data = ds.metpy.parse_cf().squeeze()
    latlon_uniques(data.copy())
    tag='_t'+str(thresh)
    for geokey in GEODESICS:
        geodesic=GEODESICS[geokey]
        start=geodesic[0]
        end=geodesic[1]
        xlabel=geodesic[2]
        cross = cross_section(data, start, end).set_coords(('latitude', 'longitude'))
        cross=cross.reindex(plev=list(reversed(cross.plev)))
        contourf_plotter(cross,  'humidity_overlap', geokey,  0,10,xlabel, geodesic, thresh, units_label='[g/kg]')
        contourf_plotter(cross,  'error_mag', geokey,  0,5,xlabel, geodesic,thresh,  units_label='[m/s]')
        #multiple_contourf(cross,  'shear', geokey+tag,  0,0.6,xlabel, geodesic, units_label='[m/(s hPa)]')
        multiple_quiver(cross, 'quiver_'+ geokey+tag, geodesic, xlabel, qkey=5)
        multiple_quiver(cross, 'quiver_shear'+ geokey+tag, geodesic,xlabel,  tag='_shear', qkey=0.1,units='m/(s hPa)')

        

    

def main():
    thresh=10
    ds=xr.open_dataset('../data/processed/07_01_2020.nc')   
    date=datetime(2020,7,1)
    ds=ds.loc[{'day':date,'time':'am','satellite':'snpp'}].squeeze()
    ds['vort_era5']=SCALE* ds['vort_era5']
    ds['div_era5']=SCALE* ds['div_era5']
    ds['vort_era5_smooth']=SCALE*ds['vort_era5_smooth']
    ds['div_era5_smooth']=SCALE*ds['div_era5_smooth']
    ds['humidity_overlap']=scale_g*ds['humidity_overlap']
    ds=preprocess(ds,thresh)
    
    #map_loop(ds)
    multiple_quiver_map(ds, 'quiver_'+str(thresh))
    multiple_quiver_map(ds, 'quiver_'+str(thresh)+'_era5','_era5')

    cross_sequence(ds, thresh)
   
    
if __name__=="__main__":
    main()
    