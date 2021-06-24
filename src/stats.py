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
import qv_grad as qg
BINS=30
THRESHOLD=10    

    


def preprocess(ds, thresh):
    #ds=ds.loc[{'day':datetime(2020,7,3),'time':'pm','satellite':'j1'}]
    #ds=ds.drop(['satellite','time','day'])
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=ds['u_error']**2+ds['v_error']**2
    ds['speed']=np.sqrt(ds['u']**2+ds['v']**2)
    ds['speed_era5']=np.sqrt(ds['u_era5']**2+ds['v_era5']**2)
    ds['speed_diff']=ds['speed']-ds['speed_era5']
    ds=qg.calculator(ds)
    ds=qg.angle(ds)
    ds=ds.where(np.sqrt(ds.mag_error)<thresh)
    var=ds['u'].values
    print(np.count_nonzero(var[~np.isnan(var)]))
    return ds

def thresh_loop():
     for thresh in (5,10):
        ds=xr.open_dataset('../data/processed/real_water_vapor_noqc_test2_'+ fsa.ALG+'.nc')
        ds=preprocess(ds, thresh)
        print(ds['specific_humidity_mean'])
        df=sc.rmse_calc(ds, thresh)
        df=sc.sorting_latlon(df)
        thresh=str(thresh)
        sc.hist2d(ds, 'speed_t'+thresh, ['speed','speed_diff'], [0,10], [-10,10])
        sc.scatter2d(ds, 'humidity_t'+thresh, ['specific_humidity_mean','speed_diff'], [0,10], [-10,10])
        sc.hist2d(ds, 'humidity_t'+ thresh, ['specific_humidity_mean','speed_diff'], [0,0.014], [-10,10])
        sc.scatter2d(ds, 'humidity_t'+thresh, ['specific_humidity_mean','q_era5'], [0,0.014], [0,0.014])
        sc.hist2d(ds, 'angle_t'+thresh, ['angle','speed_diff'], [-180,180], [-10,10])
        sc.hist2d(ds, 'grad_t'+thresh, ['grad_mag_qv','speed_diff'], [0,0.04], [-10,10])
    
        ds= ds.coarsen(longitude=10, boundary='trim').mean().coarsen(
                latitude=10, boundary='trim').mean()
    
        q.quiver_plot(ds, 'test_era5_t'+thresh,'u_era5','v_era5')
        q.quiver_plot(ds, 'test_t'+thresh,'u','v')
    
def line_plotter():
    df1=pd.read_csv('../data/interim/dataframes/t5.csv')
    df2=pd.read_csv('../data/interim/dataframes/t10.csv')
    
    fig, ax = plt.subplots()


    ax.plot(df1['edges'], df1['rmse'], '-o', label='error_thresh= 5 m/s')
    ax.plot(df2['edges'], df2['rmse'], '-o', label='error_thresh= 10 m/s')


    ax.legend(frameon=None)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Region")
    ax.set_ylabel("RMSVD [m/s]")
    plt.show()
    plt.savefig('../data/processed/plots/line_plots.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    
def main():
    thresh_loop()
    line_plotter()
   
if __name__ == '__main__':
    main()

