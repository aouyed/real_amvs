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


THRESHOLDS=[100,10,5]

def scatter(x,y):
    fig, ax=plt.subplots()
    ax.scatter(x,y)
    ax.set_xlabel('radiosonde u')
    ax.set_ylabel('AMV u')
    #ax.set_xlim(0,30)
    #ax.set_ylim(0,30)
    plt.show()
    plt.close()
    

def pressure_df(df):
    plevs=df['plev'].unique()
    d={'plev':[],'rmsvd':[]}
    for plev in plevs:
        
        df_unit=df[df.plev==plev]
        rmsvd=np.sqrt(df_unit['error_square'].mean())
        d['plev'].append(plev)
        d['rmsvd'].append(rmsvd)
        
    df_pressure=pd.DataFrame(data=d)
    df_pressure=df_pressure.sort_values(by=['plev'])
    print(df_pressure)
    return df_pressure 
        
def pressure_plot(df):
    fig, ax=plt.subplots() 
    for thresh in THRESHOLDS:
        df_unit=df[df.error_era5<thresh]
        print(df_unit.shape)
        df_pressure= pressure_df(df_unit)
        ax.plot(df_pressure.plev, df_pressure.rmsvd, label=str(thresh))
    plt.show()
    plt.close()
           
       




def main():
    df=pd.read_pickle('../data/processed/dataframes/collocated_radiosondes_july.pkl')
    df=df[df.UWND>-1000]
    delta=timedelta(hours=3)
    df=df[df.deltat<delta]
    df['error_square']=df.error_mag**2
    pressure_plot(df)
    scatter(df['error_mag'],df['error_era5'])
    scatter(df['UWND'],df['u'])
    scatter(df['plev'],df['error_mag'])
   
    
    
    
    


if __name__=='__main__':
    main()