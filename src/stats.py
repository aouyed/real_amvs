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
import stats_pressurer as sp
from matplotlib.pyplot import cm

BINS=30
THRESHOLD=10    

    



def thresh_loop():
     for thresh in sp.THRESHOLDS:
        ds=xr.open_dataset('../data/processed/07_01_2020_am.nc')  
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
    fig, ax = plt.subplots()
    colors = cm.tab10(np.linspace(0, 1, len(sp.THRESHOLDS)))
    for i, thresh in enumerate(sp.THRESHOLDS):
        df=pd.read_csv('../data/interim/dataframes/t'+str(thresh)+'_850.csv')
        df=sc.sorting_latlon(df)
        ax.plot(df['edges'], df[label], '-o', label='δ = '+str(thresh)+' m/s', color=colors[i])
        ax.plot(df['edges'], df[label+'_era5'], '-o', linestyle='dashed', label='era5, δ = '+str(thresh)+' m/s', color=colors[i])
    
    
    ax.legend(frameon=None)
    #ax.set_ylim(0, 10)
    ax.set_xlabel("Region")
    ax.set_ylabel('Shear [m/s]')
    plt.savefig('../data/processed/plots/line_plots.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def main():
   # thresh_loop()
    #rmse_plotter('rmse')
    line_plotter('shear')

   
if __name__ == '__main__':
    main()

