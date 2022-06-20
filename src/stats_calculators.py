
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import stats
import glob
import cross_section as cs
from natsort import natsorted

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


def weighted_mean_cross(da):
    '''Calculates regional cosine weighted means from dataset'''
    
    weights = np.cos(np.deg2rad(da.latitude))
    weights.name='weights'
    

    da_weighted=da.weighted(weights)
    w_mean=da_weighted.mean(skipna=True)
    return w_mean.item()


def weighted_sum(da):
    '''Calculates regional cosine weighted means from dataset'''
    
    weights = np.cos(np.deg2rad(da.latitude))
    weights.name='weights'
    da_weighted=da.weighted(weights)
    w_sum=da_weighted.sum(('longitude','latitude'), skipna=True)
    condition=xr.ufuncs.logical_not(xr.ufuncs.isnan(da))
    weights=weights.where(condition)
    return w_sum.item()

def weights_sum(da):
    '''Calculates regional cosine weighted means from dataset'''
    
    weights = np.cos(np.deg2rad(da.latitude))
    weights.name='weights'
    condition=xr.ufuncs.logical_not(xr.ufuncs.isnan(da))
    weights=weights.where(condition)
    return weights.sum().item()
    
    

def sorting_latlon(df0):
    df0.edges[df0.edges == '-70,-30'] = '(1) 70°S,30°S'
    df0.edges[df0.edges == '-30,30'] = '(2) 30°S,30°N'
    df0.edges[df0.edges == '30,70'] = '(3) 30°N,70°N'
    df0.sort_values(by=['edges'], inplace=True)
    return df0



def rmse_calc(ds, thresh):
    ds=ds.sel(plev=850, method='nearest')
    rmse_dict={'edges':[],'rmse':[],'shear':[],'shear_era5':[]}
    edges=[[-30,30],[30,60],[60,90],[-60,-30],[-90,-60]]
    
    for edge in edges:
        ds_unit=ds.sel(latitude=slice(edge[1],edge[0]))
        rmse= np.sqrt(weighted_mean(ds_unit['error_mag']))
        shear=weighted_mean(ds_unit['shear_two_levels'])
        shear_era5=weighted_mean(ds_unit['shear_two_levels_era5'])
        edge=coord_to_string(edge)
        rmse_dict['edges'].append(str(edge[0])+','+str(edge[1]))
        rmse_dict['rmse'].append(rmse)
        rmse_dict['shear'].append(shear)
        rmse_dict['shear_era5'].append(shear_era5)
    df=pd.DataFrame(data=rmse_dict)
    df.to_csv('../data/interim/dataframes/t'+str(thresh)+'.csv')
    print(df)
    return df

def calc_week(thresh):
    for pressure in cs.PRESSURES:
        print(pressure)
        rmse_dict={'edges':[],'rmse':[],'shear':[],'shear_era5':[]}
        edges=[[-30,30],[30,70],[-70,-30]]
        file_names=natsorted(glob.glob('../data/processed/07*20*.nc'))
    
        for edge in edges:
            print(edge)
            sums={'error':0, 'shear':0, 'shear_era5':0,'denominator':0,'shear_d':0}
            for file_name in file_names:
                print(file_name)
                ds=xr.open_dataset(file_name)
                ds_unit=ds.sel(latitude=slice(edge[1],edge[0]))
                ds_unit=ds_unit.loc[{'satellite':'snpp'}]
                ds_unit=cs.preprocess(ds_unit, thresh)
                shear=cs.shear_two_levels(ds_unit)
                shear_era5=cs.shear_two_levels(ds_unit,'_era5')

                sums['shear']= sums['shear']+ weighted_sum(shear)
                sums['shear_era5']= sums['shear_era5']+ weighted_sum(shear_era5)
                sums['shear_d']= sums['shear_d']+ weights_sum(shear_era5)
                ds_unit=ds_unit.sel(plev=pressure, method='nearest')
              
                sums['error']= sums['error']+ weighted_sum(ds_unit['error_mag'])
                sums['denominator']= sums['denominator']+ weights_sum(ds_unit['error_mag'])
           
            rmse_dict['edges'].append(str(edge[0])+','+str(edge[1]))
            rmse_dict['rmse'].append(np.sqrt(sums['error']/sums['denominator']))
            rmse_dict['shear'].append(sums['shear']/sums['shear_d'])
            rmse_dict['shear_era5'].append(sums['shear_era5']/sums['shear_d'])
        df=pd.DataFrame(data=rmse_dict)
        df.to_csv('../data/interim/dataframes/t'+str(thresh)+'_'+str(pressure) + '.csv')
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
    if coord[1] < 0:
        uplat = str(abs(coord[1])) + '°S'
    else:
        uplat = str(coord[1]) + '°N'

    if coord[0] < 0:
        lowlat = str(abs(coord[0])) + '°S'
    else:
        lowlat = str(coord[0]) + '°N'
    stringd = str(str(lowlat)+',' + str(uplat))
    return stringd


    

