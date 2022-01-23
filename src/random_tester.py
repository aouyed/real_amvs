#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:06:15 2021

@author: aouyed

"""
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from metpy.interpolate import cross_section
import metpy.calc as mpcalc
from metpy.units import units
import plotter 
import quiver 
import cartopy.crs as ccrs
from datetime import datetime 
import stats_pressurer as sp
import stats_calculators as sc
import config as c


SCALE=1e4
scale_g=1000
PRESSURES=c.PRESSURES
GEODESICS=c.GEODESICS

month_string=c.MONTH.strftime("%B").lower()

def preprocess(ds, thresh):
    ds=ds.drop(['day','satellite','time','flowx','flowy'])
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=np.sqrt(ds['u_error']**2+ds['v_error']**2)
    ds['squared_error']=ds['u_error']**2+ds['v_error']**2
    ds['speed']=np.sqrt(ds['u']**2+ds['v']**2)
    ds['speed_era5']=np.sqrt(ds['u_era5']**2+ds['v_era5']**2)
    ds['speed_diff']=ds['speed']-ds['speed_era5']
    condition=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds['humidity_overlap']))
    ds=ds.where(condition)
    if thresh>0:
        ds=ds.where(ds['error_mag']<thresh)
    return ds

def quiver_plot_cartopy(ds, title, u, v):
    ds=ds.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    fig=plt.figure()
    X, Y = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    Q = ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values))
    qk = ax.quiverkey(Q, 0.5, 0.5, 10, r'10 m/s', labelpos='E',
                      coordinates='figure')
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()     
    
    
def metrics(ds):
    values=ds['u'].values
    count=np.count_nonzero(~np.isnan(values))
    speed_mean=abs(ds['speed']).mean().item()
    rmsvd=np.sqrt(ds['squared_error'].mean()).item()
    ratio=rmsvd/speed_mean
    print('speed_mean: ' + str(speed_mean))
    print('rmsvd: ' + str(rmsvd))
    print('ratio: ' + str(ratio))
    print('count: ' + str(count))

    
def stats(ds,label):
        
    mu=ds[label].mean().item()
    std=ds[label].std().item()
   
    return mu,std   

def random_fields(ds):
    mu_u,std_u = stats(ds, 'u')  
    mu_v,std_v = stats(ds, 'v')
    
    u_rand= np.random.normal(mu_u, std_u, (180, 360))
    v_rand= np.random.normal(mu_v, std_v, (180, 360))
    ds['u']=(['latitude','longitude'],u_rand)
    ds['v']=(['latitude','longitude'],v_rand)
    return ds
    

def main():
    time='pm'
    dsdate=c.MONTH.strftime('%m_%d_%Y')
    thresh=4
    ds=xr.open_dataset('../data/processed/' + dsdate+'_'+time+'.nc')
    date=c.MONTH
    ds=ds.loc[{'day':date,'time':time,'satellite':'snpp'}].squeeze()
    ds['humidity_overlap']=scale_g*ds['humidity_overlap']
    ds=ds.sel(plev=700, method='nearest')
    ds_original=preprocess(ds.copy(),thresh)

    quiver_plot_cartopy(ds_original, 'amv_thresh_'+str(thresh), 'u', 'v')
    quiver_plot_cartopy(ds_original, 'era_thresh_ '+str(thresh), 'u_era5', 'v_era5')
    metrics(ds_original)
    
    ds_rand=random_fields(ds.copy())
    ds_rand=preprocess(ds_rand,thresh)
    quiver_plot_cartopy(ds_rand, 'noise_thresh_ '+str(thresh), 'u', 'v')
    metrics(ds_rand)
    
    
    #7183

   
    
if __name__=="__main__":
    main()
    