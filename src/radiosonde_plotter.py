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
        df_unit=df[df.error_mag<thresh]
        print(df_unit.shape)
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
    plt.savefig('../data/processed/plots/radiosonde_comparison.png', dpi=300)
    plt.show()
    plt.close()
           
       
def preprocess(df):
    df=df[df.u_wind>-1000]
    udiff=df.u_wind-df.u
    vdiff=df.v_wind-df.v
    df['error_mag']=np.sqrt((df.u-df.u_era5)**2+(df.v-df.v_era5)**2)
    df['error_square']=udiff**2+vdiff**2
    df['error_square_era5']=(df.u_wind-df.u_era5)**2+(df.v_wind-df.v_era5)**2
    return df


def main():
    df=pd.read_pickle('../data/processed/dataframes/winds_rs.pkl')
    df=preprocess(df)
    #delta=timedelta(hours=1.5)

    #df=df[df.deltat<delta]    
    print(df.shape)
    

    pressure_plot(df, 'rmsvd','Difference between 3D AMVs and radiosondes')
    
    
    
    
    


if __name__=='__main__':
    main()