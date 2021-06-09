# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import first_stage_amv as fsa
import quiver as q
from datetime import datetime 
import qv_grad as qg
BINS=20
    


def filter_plotter(df0, values, title):
    fig, ax = plt.subplots()

    df = df0[df0.exp_filter == 'exp2']
    ax.plot(df['latlon'], df['rmse'], '-o', label='UA')

    df = df0[df0.exp_filter == 'ground_t']
    ax.plot(df['latlon'], df['rmse'], '-o',
            label='model error')

    df = df0[df0.exp_filter == 'df']
    ax.plot(df['latlon'], df['rmse'], '-o', label='UA First Stage')

    df = df0[df0.exp_filter == 'jpl']
    ax.plot(df['latlon'], df['rmse'], '-o', label='JPL')

    ax.legend(frameon=None)
    ax.set_ylim(0, ERROR_MAX)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSVD [m/s]")
    ax.set_title(title)
    directory = '../data/processed/density_plots'
    plt.savefig(values+'.png', bbox_inches='tight', dpi=300)
    plt.close()



def weighted_mean(da):
    '''Calculates regional cosine weighted means from dataset'''
    
    weights = np.cos(np.deg2rad(da.latitude))
    weights.name='weights'
    

    da_weighted=da.weighted(weights)
    w_mean=da_weighted.mean(('longitude','latitude'), skipna=True)
    return w_mean.item()
    


def preprocess(ds):
    ds=ds.loc[{'day':datetime(2020,7,3),'time':'pm','satellite':'j1'}]
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['mag_error']=ds['u_error']**2+ds['v_error']**2
    ds['speed']=np.sqrt(ds['u']**2+ds['v']**2)
    ds['speed_era5']=np.sqrt(ds['u_era5']**2+ds['v_era5']**2)
    ds['speed_diff']=ds['speed']-ds['speed_era5']
    ds=qg.calculator(ds)
    var=ds['u'].values
    print(np.count_nonzero(var[~np.isnan(var)]))
    ds=ds.where(np.sqrt(ds.mag_error)<10)
    return ds

def rmse_calc(ds):
    rmse_dict={'edges':[],'rmse':[]}
    edges=[[-30,30],[30,60],[60,90],[-60,-30],[-90,-60]]
    
    for edge in edges:
        ds_unit=ds.sel(latitude=slice(edge[1],edge[0]))
        rmse= np.sqrt(weighted_mean(ds_unit['mag_error']))
        rmse_dict['edges'].append(edge)
        rmse_dict['rmse'].append(rmse)
    df=pd.DataFrame(data=rmse_dict)
    print(df)
    return df
        
def hist2d(ds, title, label, xedges, yedges):
    print('2dhistogramming..')

    bins = BINS
    xbins = np.linspace(xedges[0], xedges[1], bins+1)
    ybins = np.linspace(yedges[0], yedges[1], bins+1)
    df = ds.to_dataframe().reset_index().dropna()
    df = df[label]
    df = df.astype(np.float32)
    
    img, x_edges, y_edges = np.histogram2d(df[label[0]].values, df[label[1]].values, bins=[
                               xbins, ybins])
    img=img.T
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, ax = plt.subplots()

    im = ax.imshow(img, origin='lower',
                   cmap='CMRmap_r', aspect='auto', extent=extent)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.tight_layout()
    plt.show()
    plt.savefig('../data/processed/plots/hist2d_'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
def scatter2d(ds, title, label, xedges, yedges):
    print('scattering...')
    fig, ax = plt.subplots()
    df=ds.to_dataframe().reset_index().dropna()
    df=df[label]
    print(df.shape)
    X=df[label[0]]
    Y=df[label[1]]
    ax.scatter(X,Y, marker = 'o', facecolors='none', edgecolors='r')
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    #plt.ylim(yedges)
    plt.show()
    plt.savefig(title+'_scatter2d.png', dpi=300)
    plt.close()
def main():
    ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test2_'+ fsa.ALG+'.nc')
    ds=preprocess(ds)
    print(ds['specific_humidity_mean'])
    df=rmse_calc(ds)
    hist2d(ds, 'speed', ['speed','speed_diff'], [0,10], [-10,10])
    scatter2d(ds, 'humidity', ['specific_humidity_mean','speed_diff'], [0,10], [-10,10])
    hist2d(ds, 'humidity', ['specific_humidity_mean','speed_diff'], [0,0.014], [-10,10])
    scatter2d(ds, 'humidity', ['specific_humidity_mean','q_era5'], [0,0.014], [0,0.014])
    scatter2d(ds, 'grad', ['grad_mag_qv','speed_diff'], [0,0.014], [0,0.014])

    ds= ds.coarsen(longitude=10, boundary='trim').mean().coarsen(
                latitude=10, boundary='trim').mean()
    
    q.quiver_plot(ds, 'test_era5','u_era5','v_era5')
    q.quiver_plot(ds, 'test','u','v')
if __name__ == '__main__':
    main()

