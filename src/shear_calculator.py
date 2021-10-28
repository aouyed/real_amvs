#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:04:41 2021

@author: aouyed
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime 
import stats_calculators as sc
import cross_section as cs
from matplotlib.pyplot import cm
import glob

THRESHOLDS=[4,10,100]


def preprocess(ds, thresh):
    ds=ds.drop(['day','satellite','time','flowx','flowy'])
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=np.sqrt(ds['u_error']**2+ds['v_error']**2)
    condition=xr.ufuncs.logical_not(xr.ufuncs.isnan(ds['humidity_overlap']))
    ds=ds.where(condition)
    if thresh>0:
        ds=ds.where(ds['error_mag']<thresh)
    return ds


def thresh_loop():
    for thresh in THRESHOLDS:
        df=calc_shear(thresh)
        print(df)
        
        
def edge_calculator():
    edge_l=np.arange(-70,70,10).tolist()
    edge_u=np.arange(-60,80,10).tolist()
    edges=list(zip(edge_l,edge_u))
    return edges



def calc_shear(thresh):
    shear_dict={'edges':[],'shear':[],'shear_era5':[]}
    edges=edge_calculator()
    file_names=glob.glob('../data/processed/07*2020*.nc')
    print(file_names)

    for edge in edges:
        print(edge)
        sums={'error':0, 'shear':0, 'shear_era5':0,'shear_denominator':0}
        for file_name in file_names:
            ds=xr.open_dataset(file_name)
            ds_unit=ds.sel(latitude=slice(edge[1],edge[0]))
            ds_unit=ds_unit.loc[{'satellite':'snpp'}]
            ds_unit=preprocess(ds_unit,thresh)
            shear=cs.shear_two_levels(ds_unit)
            shear_era5=cs.shear_two_levels(ds_unit,'_era5')

            sums['shear']= sums['shear']+ sc.weighted_sum(shear)
            sums['shear_era5']= sums['shear_era5']+ sc.weighted_sum(shear_era5)
            sums['shear_denominator']= sums['shear_denominator']+ sc.weights_sum(shear_era5)
          
           
        edge=sc.coord_to_string(edge)
        print(edge)
        shear_dict['edges'].append(edge)
        shear_dict['shear'].append(sums['shear']/sums['shear_denominator'])
        shear_dict['shear_era5'].append(sums['shear_era5']/sums['shear_denominator'])
    df=pd.DataFrame(data=shear_dict)
    df.to_csv('../data/processed/dataframes/t'+str(thresh)+'_shear.csv')
    print(df)
    return df


def line_plotter(label):
    fig, ax = plt.subplots()
    colors = cm.tab10(np.linspace(0, 1, len(THRESHOLDS)))
    for i, thresh in enumerate(THRESHOLDS):
        df=pd.read_csv('../data/processed/dataframes/t'+str(thresh)+'_shear.csv')
        ax.plot(df['edges'], df[label], '-o', label='δ = '+str(thresh)+' m/s', color=colors[i])
        ax.plot(df['edges'], df[label+'_era5'], '-o', linestyle='dashed', label='era5, δ = '+str(thresh)+' m/s', color=colors[i])
    
    
    ax.legend(frameon=None)
    #ax.set_ylim(0, 10)
    ax.set_xlabel("Region")
    ax.set_ylabel('Shear [m/s]')
    plt.xticks(rotation=45, ha="right")
    plt.savefig('../data/processed/plots/line_plots.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def main():
    #thresh_loop()
    line_plotter('shear')

   
if __name__ == '__main__':
    main()
