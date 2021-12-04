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
import config as c
from tqdm import tqdm
import amv_calculators as ac
from datetime import timedelta 

THRESHOLDS=c.THRESHOLDS


def calc_days(thresh, days):

    
    for day in tqdm(days):
        for time in ('am','pm'):
            d={'pressure':[],'error_sum':[],'speed_sum':[],
               'denominator':[],'yield':[]}
            file_name= day.strftime('../data/processed/%m_%d_%Y_')+time +'.nc'
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
                d['pressure'].append(pressure)
                d['error_sum'].append(error_sum)
                d['speed_sum'].append(speed_sum)
                d['denominator'].append(denominator)
                d['yield'].append(counts)
            df=pd.DataFrame(data=d)
            df.set_index('pressure', drop=True)
            df.to_csv('../data/interim/dataframes/t'+str(thresh)+'_'+day.strftime('%m_%d_%Y_') + time + '.csv')
    return df['pressure'].values
    
def calc_pressure(thresh, pressures, days):
    dsdate = c.MONTH.strftime('_m_*_%Y*.csv')
    d={'pressure':[],'rmsvd':[],'speed':[], 'yield': []}

    for pressure in tqdm(pressures):
        error_sum=0
        speed_sum=0
        denominator=0
        yield_sum=0
        for day in days:
            for time in ('am','pm'):
                file= '../data/interim/dataframes/t'+str(thresh)+'_'+day.strftime('%m_%d_%Y_') + time + '.csv'
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
    df.to_csv('../data/processed/dataframes/'+c.month_string+'_rmsvd_t'+str(thresh)+'.csv')
    
        
def  plot_rmsvd(ax, df, width, thresh):
     ax.plot(df['rmsvd'],df['pressure'], label='δ = '+str(thresh)+' m/s', linewidth=width)
     #ax.plot(df['pressure'], df['speed'], label='speed, δ = '+str(thresh)+' m/s',linestyle='dashed', linewidth=width,color='black')
     ax.set_xlabel('RMSVD between AMVs and ERA5 [m/s]')
     ax.set_yscale('symlog')
     ax.set_yticklabels(np.arange(1000, 50, -150))
     ax.set_ylim(df['pressure'].max(), df['pressure'].min())
     ax.set_yticks(np.arange(1000, 50, -150))
     ax.set_xticklabels(np.arange(0, 35, 5))
     ax.set_xticks(np.arange(0, 35, 5))


def  plot_yield(ax, df, width, thresh):
     ax.plot( df['yield']/(7*1000),df['pressure'],  label='δ = '+str(thresh)+' m/s', linewidth=width)
     ax.set_xlabel('1000 counts per day')
     ax.set_yscale('symlog')
     ax.set_yticklabels(np.arange(1000, 50, -150))
     ax.set_ylim(df['pressure'].max(), df['pressure'].min())
     ax.set_yticks(np.arange(1000, 50, -150))
     nticks=np.arange(0, 30, 5)
     ax.set_xticklabels(nticks)
     ax.set_xticks(nticks)


    
def line_plotter(func, ax, month):
    colors = cm.tab10(np.linspace(0, 1, len(THRESHOLDS)))
    widths=np.linspace(1,3, len(THRESHOLDS))
    for i, thresh in enumerate(reversed(THRESHOLDS)):
        #color=colors[i]
        width=widths[i]
        df=pd.read_csv('../data/processed/dataframes/'+month+'_rmsvd_t'+str(thresh)+'.csv')
        func(ax, df, width, thresh)
    ax.legend(frameon=False)
    ax.set_ylabel("Pressure [hPa]")


    return ax
    
    
def multiple_lineplots(tag, month, title, plot_rmsvd,plot_yield):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axlist = axes.flat
    axlist[0]=line_plotter(plot_rmsvd, axlist[0], month)
    axlist[1]=line_plotter(plot_yield, axlist[1], month)
    #axlist[0].text(-5,150,tag[0],c='red')
    #axlist[1].text(-5,150,tag[1],c='red')
    axlist[0].text(0.5,0.5,tag[0], transform=axlist[0].transAxes)
    axlist[1].text(0.5,0.5,tag[1], transform=axlist[1].transAxes)

    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+title +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def threshold_fun():
    start_date=c.MONTH
    end_date=c.MONTH + timedelta(days=6)
    days=ac.daterange(start_date, end_date, 24)
    
    for thresh in THRESHOLDS:
        pressures=calc_days(thresh, days)
        calc_pressure(thresh, pressures, days)
    
def main():
    #threshold_fun()
    multiple_lineplots(['(a)','(b)'],'january','january_pressure_plots', plot_rmsvd,plot_yield)
    multiple_lineplots(['(c)','(d)'],'july','july_pressure_plots', plot_rmsvd,plot_yield)



if __name__ == '__main__':
    main()
    

