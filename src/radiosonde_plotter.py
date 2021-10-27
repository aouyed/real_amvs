#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:15:56 2021

@author: aouyed
"""
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np


THRESHOLDS=[10,4]

def scatter(x,y):
    fig, ax=plt.subplots()
    ax.scatter(x,y)
    ax.set_xlabel('Radiosonde [m/s]')
    ax.set_ylabel('ERA 5 [m/s]')
    ax.set_title('Magnitude of Vector Difference')
    ax.set_xlim(0,20)
    ax.set_ylim(0,20)
    plt.show()
    plt.close()
    

def pressure_df(df):
    plevs=df['plev'].unique()
    d={'plev':[],'rmsvd':[],'rmsvd_era5':[]}
    for plev in plevs:
        
        df_unit=df[df.plev==plev]
        rmsvd=np.sqrt(df_unit['error_square'].mean())
        rmsvd_era5=np.sqrt(df_unit['error_square_era5'].mean())
        d['plev'].append(plev)
        d['rmsvd'].append(rmsvd)
        d['rmsvd_era5'].append(rmsvd_era5)
        
    df_pressure=pd.DataFrame(data=d)
    df_pressure=df_pressure.sort_values(by=['plev'])
    return df_pressure 
        
def pressure_plot(df,rmsvd_label, title):
    fig, ax=plt.subplots() 
    df_pressure_era= pressure_df(df)
    for thresh in THRESHOLDS:
        df_unit=df[df.error_era5<thresh]
        df_pressure= pressure_df(df_unit)
        ax.plot(df_pressure[rmsvd_label], df_pressure.plev, label='δ = '+str(thresh)+' m/s')
    ax.plot(df_pressure_era['rmsvd_era5'], df_pressure_era.plev, label='ERA 5')
    ax.axvline(4.05,linestyle='dashed',label='Aeolus (Mie)')
    ax.axvline(5.93,linestyle='dotted',label='GEO AMVs' )
    ax.legend(frameon=False)
    ax.set_xlabel('RMSVD [m/s]')
    ax.set_ylabel('Pressure [hPa]')
    ax.set_xlim(0,10)
    ax.set_yscale('symlog')
    ax.set_yticklabels(np.arange(900, 50, -100))
    ax.set_ylim(df['plev'].max(), df['plev'].min())
    ax.set_yticks(np.arange(900, 50, -100))
    ax.set_title(title)
    plt.show()
    plt.close()
           
       
def preprocess(df):
    df=df[df.UWND>-1000]
    df['speed']=np.sqrt(df.u**2+df.v**2)
    df['speed_era5']=np.sqrt(df.u_coarse_era5**2+df.v_coarse_era5**2)
    df['speed_rao']=np.sqrt(df.UWND**2+df.VWND**2)

    print(df.columns)
    df['error_square']=df.error_mag**2
    df['error_square_era5']=(df.UWND-df.u_coarse_era5)**2+(df.VWND-df.v_coarse_era5)**2
    return df


def main():
    df=pd.read_pickle('../data/processed/dataframes/collocated_radiosondes_july.pkl')
    df=preprocess(df)
    delta=timedelta(hours=1.5)

    df=df[df.deltat<delta]    
    print(df.shape)
    breakpoint()
    

    pressure_plot(df, 'rmsvd','Difference between 3D AMVs and radiosondes')
    #pressure_plot(df, 'rmsvd_era5', 'Difference between ERA5 and radiosondes')

    scatter(df['error_mag'],df['error_era5'])
    scatter(df['speed'],df['speed_rao'])
    scatter(df['speed_era5'],df['speed_rao'])
    scatter(df['u'],df['UWND'])
    scatter(df['u_era5'],df['UWND'])


    #scatter(df['UWND'],df['u'])
    #scatter(df['plev'],df['error_mag'])
   
    
    
    
    


if __name__=='__main__':
    main()