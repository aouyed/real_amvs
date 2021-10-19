# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import stats_calculators as sc
import glob
import cross_section as cs
from natsort import natsorted
from matplotlib.pyplot import cm


THRESHOLDS=[5,10,100]


def calc_days(thresh):
    file_names=natsorted(glob.glob('../data/processed/07*20*.nc'))
    for file_name in file_names: 
        d={'pressure':[],'error_sum':[],'speed_sum':[],
           'denominator':[],'yield':[]}
        print(file_name)
        ds=xr.open_dataset(file_name)
        ds=ds.loc[{'satellite':'snpp'}]
        ds=cs.preprocess(ds, thresh)      
        for pressure in ds['plev'].values:
            ds_unit=ds.sel(plev=pressure, method='nearest')
            error_sum=sc.weighted_sum(ds_unit['squared_error'])
            speed_sum=sc.weighted_sum(ds_unit['speed'])
            data=ds_unit['squared_error'].values
            counts=np.count_nonzero(~np.isnan(data))

            denominator= sc.weights_sum(ds_unit['squared_error'])
            pressure=int(round(pressure))
            print(pressure)
            d['pressure'].append(pressure)
            d['error_sum'].append(error_sum)
            d['speed_sum'].append(speed_sum)
            d['denominator'].append(denominator)
            d['yield'].append(counts)
        df=pd.DataFrame(data=d)
        df.set_index('pressure', drop=True)
        df.to_csv('../data/interim/dataframes/t'+str(thresh)+'_'+file_name[18:28] + '.csv')
    
def calc_pressure(thresh):
    d={'pressure':[],'rmsvd':[],'speed':[], 'yield': []}
    file_names=natsorted(glob.glob('../data/interim/dataframes/t'+str(thresh)+'_07*20.csv'))
    print(file_names)
    df=pd.read_csv(file_names[0])
    pressures=df['pressure'].values
    df=df.set_index('pressure',drop=True)
    for pressure in pressures:
        error_sum=0
        speed_sum=0
        denominator=0
        yield_sum=0
        for file in file_names:
            df_unit=pd.read_csv(file)
            df_unit=df_unit.set_index('pressure',drop=True)
            error_sum=error_sum+df_unit.loc[pressure,'error_sum']
            speed_sum=speed_sum+df_unit.loc[pressure,'speed_sum']
            yield_sum=yield_sum+df_unit.loc[pressure,'yield']
            denominator=denominator + df_unit.loc[pressure,'denominator']
        d['pressure'].append(pressure)
        d['speed'].append(speed_sum/denominator)
        d['rmsvd'].append(np.sqrt(error_sum/denominator))
        d['yield'].append(yield_sum)

    df=pd.DataFrame(data=d)
    print(df)
    df.set_index('pressure', drop=True)
    df.to_csv('../data/processed/dataframes/rmsvd_t'+str(thresh)+'.csv')
    
        
def  plot_rmsvd(ax, df, width, thresh):
     ax.plot(df['pressure'], df['rmsvd'],  label='RMSVD, δ = '+str(thresh)+' m/s', linewidth=width, color='black')
     ax.plot(df['pressure'], df['speed'], label='speed, δ = '+str(thresh)+' m/s',linestyle='dashed', linewidth=width,color='black')
     ax.set_ylabel('[m/s]')

def  plot_yield(ax, df, width, thresh):
     ax.plot(df['pressure'], df['yield']/7,  label='δ = '+str(thresh)+' m/s', linewidth=width,color='black')
     ax.set_ylabel('Counts per day')

    
def line_plotter(func, ax):
    colors = cm.tab10(np.linspace(0, 1, len(THRESHOLDS)))
    widths=np.linspace(1,3, len(THRESHOLDS))
    for i, thresh in enumerate(reversed(THRESHOLDS)):
        #color=colors[i]
        width=widths[i]
        df=pd.read_csv('../data/processed/dataframes/rmsvd_t'+str(thresh)+'.csv')
        func(ax, df, width, thresh)
    ax.legend(frameon=False)
    ax.set_xlabel("Pressure [hPa]")
    ax.set_xscale('symlog')
    ax.set_xticklabels(np.arange(1000, 50, -300))
    ax.set_xlim(df['pressure'].max(), df['pressure'].min())
    ax.set_xticks(np.arange(1000, 50, -300))

    return ax
    
    
def multiple_lineplots(title, plot_rmsvd,plot_yield):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axlist = axes.flat
    axlist[0]=line_plotter(plot_rmsvd, axlist[0])
    axlist[1]=line_plotter(plot_yield, axlist[1])
    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+title +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def threshold_fun():
    for thresh in THRESHOLDS:
        calc_days(thresh)
        calc_pressure(thresh)
    
def main():
    #threshold_fun()
    multiple_lineplots('pressure_plots', plot_rmsvd,plot_yield)
    


if __name__ == '__main__':
    main()
    

