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
     for thresh in (10,0):
        ds=xr.open_dataset('../data/processed/07_01_2020.nc')  
        date=datetime(2020,7,1)
        ds=ds.loc[{'day':date,'time':'am','satellite':'snpp'}].squeeze()
        #df=sc.rmse_calc(ds, thresh)
        df=sc.calc_week(thresh)
        df=sc.sorting_latlon(df)
        thresh=str(thresh)
       
def line_plotter(label):
    df1=pd.read_csv('../data/interim/dataframes/t0.csv')
    df2=pd.read_csv('../data/interim/dataframes/t10.csv')
    df1=sc.sorting_latlon(df1)
    df2=sc.sorting_latlon(df2)

    fig, ax = plt.subplots()


    ax.plot(df1['edges'], df1[label], '-o', label='error_thresh= 5 m/s')
    ax.plot(df2['edges'], df2[label], '-o', label='error_thresh= 10 m/s')


    ax.legend(frameon=None)
    #ax.set_ylim(0, 10)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSVD [m/s]")
    plt.show()
    plt.savefig('../data/processed/plots/line_plots.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    
def main():
    thresh_loop()
    #line_plotter('rmse')
    #line_plotter('shear')
    #line_plotter('shear_era5')

   
if __name__ == '__main__':
    main()

