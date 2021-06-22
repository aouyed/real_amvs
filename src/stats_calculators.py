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
import stats

def line_plotter(df0, values, title):
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
    

def sorting_latlon(df0):
    df0.edges[df0.edges == '90°S,60°S'] = '(0) 90°S,60°S'
    df0.edges[df0.edges == '60°S,30°S'] = '(1) 60°S,30°S'
    df0.edges[df0.edges == '30°S,30°N'] = '(2) 30°S,30°N'
    df0.edges[df0.edges == '30°N,60°N'] = '(3) 30°N,60°N'
    df0.edges[df0.edges == '60°N,90°N'] = '(4) 60°N,90°N'
    df0.sort_values(by=['edges'], inplace=True)
    return df0



def rmse_calc(ds, thresh):
    rmse_dict={'edges':[],'rmse':[]}
    edges=[[-30,30],[30,60],[60,90],[-60,-30],[-90,-60]]
    
    for edge in edges:
        ds_unit=ds.sel(latitude=slice(edge[1],edge[0]))
        rmse= np.sqrt(weighted_mean(ds_unit['mag_error']))
        edge=coord_to_string(edge)
        rmse_dict['edges'].append(edge)
        rmse_dict['rmse'].append(rmse)
    df=pd.DataFrame(data=rmse_dict)
    df.to_csv('../data/interim/dataframes/t'+str(thresh)+'.csv')
    print(df)
    return df
        
def hist2d(ds, title, label, xedges, yedges):
    print('2dhistogramming..')

    bins = stats.BINS
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
    
    
def coord_to_string(coord):
    if coord[0] < 0:
        uplat = str(abs(coord[0])) + '°S'
    else:
        uplat = str(coord[0]) + '°N'

    if coord[1] < 0:
        lowlat = str(abs(coord[1])) + '°S'
    else:
        lowlat = str(coord[1]) + '°N'
    stringd = str(str(lowlat)+',' + str(uplat))
    return stringd

    

