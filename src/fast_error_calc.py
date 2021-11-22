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
import config as c


THRESHOLDS=[10,4]

       
def preprocess(df):
    df=df[df.u_wind>-1000]
    udiff=df.u_wind-df.u
    vdiff=df.v_wind-df.v
    df['error_mag']=np.sqrt((df.u-df.u_era5)**2+(df.v-df.v_era5)**2)
    df['error_square']=udiff**2+vdiff**2
    df['error_square_era5']=(df.u_wind-df.u_era5)**2+(df.v_wind-df.v_era5)**2
    return df


def main():
    df_jan=pd.read_pickle('../data/processed/dataframes/january_winds_rs_model.pkl')
    df_july=pd.read_pickle('../data/processed/dataframes/july_winds_rs_model.pkl')

    df_jan=preprocess(df_jan)
    df_jan=df_jan.drop_duplicates()

    df_jan['speed_era5']=np.sqrt(df_jan.u_era5**2+df_jan.v_era5**2)
    df_jan['speed_amv']=np.sqrt(df_jan.u**2+df_jan.v**2)

    df_jan['rel_error']=np.sqrt(df_jan['error_square'])/df_jan.speed
    df_jan_300=df_jan.loc[df_jan.pressure==300]
    df_jan_850=df_jan.loc[df_jan.pressure==850]
    df_jan_300=df_jan_300.loc[df_jan_300.error_mag<20]
    df_jan_850=df_jan_850.loc[df_jan_850.error_mag<4]
    
    print('rmsvd')
    print(np.sqrt(df_jan_300['error_square'].mean()))
    df_jan_850[df_jan_850.rel_error==np.inf]=np.nan
    print(np.sqrt(df_jan_850['error_square'].mean()))
    
    print('relative_error')
    print(df_jan_300['rel_error'].mean())
    df_jan_850[df_jan_850.rel_error==np.inf]=np.nan
    print(df_jan_850['rel_error'].mean(skipna=True))

    
    
    
    
    


if __name__=='__main__':
    main()