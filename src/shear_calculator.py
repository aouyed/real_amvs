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
from datetime import timedelta
import stats_calculators as sc
import cross_section as cs
from matplotlib.pyplot import cm
import glob
import config as c
import amv_calculators as ac
from tqdm import tqdm
THRESHOLDS=c.THRESHOLDS


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
    start_date=c.MONTH
    end_date=c.MONTH + timedelta(days=6)
    days=ac.daterange(start_date, end_date, 24)
    
    for edge in tqdm(edges):
        sums={'error':0, 'shear':0, 'shear_era5':0,'shear_denominator':0}
        for day in days:
            for time in ('am','pm'):
                file_name= day.strftime('../data/processed/%m_%d_%Y_')+time +'.nc'
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
        shear_dict['edges'].append(edge)
        shear_dict['shear'].append(sums['shear']/sums['shear_denominator'])
        shear_dict['shear_era5'].append(sums['shear_era5']/sums['shear_denominator'])
    df=pd.DataFrame(data=shear_dict)
    df.to_csv('../data/processed/dataframes/'+c.month_string+'_t'+str(thresh)+'_shear.csv')
    print(df)
    return df


def line_plotter(label,month_string, tag):
    fig, ax = plt.subplots()
    colors = cm.tab10(np.linspace(0, 1, len(THRESHOLDS)))
    for i, thresh in enumerate(THRESHOLDS):
        df=pd.read_csv('../data/processed/dataframes/'+month_string+'_t'+str(thresh)+'_shear.csv')
        ax.plot(df['edges'], df[label], '-o', label='δ = '+str(thresh)+' m/s', color=colors[i])
    df=pd.read_csv('../data/processed/dataframes/'+c.month_string+'_t100_shear.csv')
    ax.plot(df['edges'], df[label+'_era5'], '-o', linestyle='dashed', label='ERA 5', color ='red')
    
    ax.text(0.05,0.9,tag, transform=ax.transAxes)
    ax.legend(frameon=None)
    #ax.set_ylim(0, 10)
    ax.set_xlabel("Region")
    ax.set_ylabel('Shear [m/s]')
    plt.xticks(rotation=45, ha="right")
    plt.savefig('../data/processed/plots/'+month_string+'_line_plots.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def line_ax(ax,label, month_string, tag):
    colors = cm.tab10(np.linspace(0, 1, len(THRESHOLDS)))
    lats=np.arange(-65,70,10)
    lats=pd.Series(lats)
    for i, thresh in enumerate(THRESHOLDS):
        df=pd.read_csv('../data/processed/dataframes/'+month_string+'_t'+str(thresh)+'_shear.csv')
        df['edges']=lats
        ax.plot(df['edges'], df[label], '-o', label='δ = '+str(thresh)+' m/s', color=colors[i])
    df=pd.read_csv('../data/processed/dataframes/'+c.month_string+'_t100_shear.csv')
    df['edges']=lats
    ax.plot(df['edges'], df[label+'_era5'], '-o', linestyle='dashed', label='ERA 5', color ='red')
    ax.set_ylim(5,20)
    ax.text(0.05,0.9,tag, transform=ax.transAxes)
    return ax, df['edges']
    

def  sample_calc():
    shear_total=pd.DataFrame()
    for month_string in ('january','july'):
        shear_sample=pd.DataFrame()
        for i, thresh in enumerate(THRESHOLDS):
            df=pd.read_csv('../data/processed/dataframes/'+month_string+'_t'+str(thresh)+'_shear.csv')
            sample_unit=df[df.edges=='30°S,20°S'].copy()
            sample_unit['thresh']=thresh
            sample_unit['month']=month_string
            if shear_sample.empty:
                shear_sample=sample_unit
            else:
                shear_sample=shear_sample.append(sample_unit)
             
            
        df=pd.read_csv('../data/processed/dataframes/'+c.month_string+'_t100_shear.csv')
        sample_unit=df[df.edges=='30°S,20°S'].copy()
        shear_sample['shear_era5']=sample_unit['shear_era5'].values.item()    
        if shear_total.empty:
            shear_total=shear_sample
        else:
            shear_total=shear_total.append(shear_sample)
    
    shear_total['diff']=shear_total['shear']-shear_total['shear_era5']
    shear_total.to_csv('../data/processed/dataframes/shear_samples.csv')
    
    print(shear_total)
   
    
 
 
def multiple_lineplots(title):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    axlist = axes.flat
    axlist[0],_=line_ax(axlist[0],'shear','january', '(a)')
    axlist[1],labels=line_ax(axlist[1],'shear','july', '(b)')
    axlist[1].legend(frameon=None, loc='upper right')
    axlist[1].set_xlabel("Latitude")
    axlist[0].set_ylabel('Shear [m/s]')
    #plt.xticks(rotation=45, ha="right")
    axlist[1].set_xticks(np.arange(-70,80,10))
    fig.tight_layout()
 
    plt.savefig('../data/processed/plots/'+title +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
    
def main():
    sample_calc()
    multiple_lineplots('shear_double')
    #thresh_loop()
     #line_plotter('shear','january', '(a)')
     #line_plotter('shear','july', '(b)')


   
if __name__ == '__main__':
    main()
