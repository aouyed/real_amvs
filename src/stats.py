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
import stats_calculators as sc
import cross_section as cs
import qv_grad as qg
BINS=30
THRESHOLD=10    

    



def thresh_loop():
     for thresh in [5]:
        ds=xr.open_dataset('../data/processed/07_01_2020.nc')  
        date=datetime(2020,7,1)
        ds=ds.loc[{'day':date,'time':'am','satellite':'snpp'}].squeeze()
        #df=sc.rmse_calc(ds, thresh)
        df=sc.calc_week(thresh)
        df=sc.sorting_latlon(df)
        thresh=str(thresh)
       

def rmse_plotter(label):
    fig, ax = plt.subplots()

    for pressure in cs.PRESSURES:
        df=pd.read_csv('../data/interim/dataframes/t5_' + str(pressure)+'.csv')
        df=sc.sorting_latlon(df)
        ax.plot(df['edges'], df[label], '-o', label=str(pressure)+' hPa')  
    ax.legend(frameon=None)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSVD [m/s]")
    ax.legend(frameon=None)
    plt.show()
    plt.savefig('../data/processed/plots/rmse_plots.png', bbox_inches='tight', dpi=300)
    plt.close()

def line_plotter(label):
    df1=pd.read_csv('../data/interim/dataframes/t0_850.csv')
    df2=pd.read_csv('../data/interim/dataframes/t5_850.csv')
    df3=pd.read_csv('../data/interim/dataframes/t10_850.csv')

    df1=sc.sorting_latlon(df1)
    df2=sc.sorting_latlon(df2)
    df3=sc.sorting_latlon(df3)

    fig, ax = plt.subplots()

 
    ax.plot(df1['edges'], df1[label], '-o', label='no quality control')
    ax.plot(df2['edges'], df2[label], '-o', label='thresh= 5 m/s')
    ax.plot(df3['edges'], df3[label], '-o', label='thresh= 10 m/s')
    ax.plot(df1['edges'], df1[label+'_era5'], '-o', linestyle='dashed', label='era5, no quality control')
    ax.plot(df2['edges'], df2[label+'_era5'], '-o', linestyle='dashed', label='era5, thresh= 5 m/s')
    ax.plot(df3['edges'], df3[label+'_era5'], '-o', linestyle='dashed', label='era5, thresh= 10 m/s')


    ax.legend(frameon=None)
    #ax.set_ylim(0, 10)
    ax.set_xlabel("Region")
    ax.set_ylabel('Shear [m/s]')
    plt.show()
    plt.savefig('../data/processed/plots/line_plots.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    
def main():
    #thresh_loop()
    rmse_plotter('rmse')
    line_plotter('shear')

   
if __name__ == '__main__':
    main()

