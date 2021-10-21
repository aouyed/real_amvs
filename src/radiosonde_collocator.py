#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:04:26 2021

@author: aouyed
"""

import glob
from datetime import datetime 
import pandas as pd
import xarray as xr
import cross_section as cs
from datetime import timedelta
import numpy as np


def daterange(start_date, end_date, dhour):
    date_list = []
    delta = timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list




def preprocess(ds):
    ds=ds.drop(['day','satellite','time','flowx','flowy'])
    ds['u_error']=ds['u']-ds['u_era5']
    ds['v_error']=ds['v']-ds['v_era5']
    ds['error_mag']=np.sqrt(ds['u_error']**2+ds['v_error']**2)
    return ds



def df_calculator(df, df_s, lat, lon, sample):
     df['LAT']=lat
     df['LON']=lon
     df['lat']=sample['latitude'].values.item()
     df['lon']=sample['longitude'].values.item()
     df['time']=df_s['obs_time']
     df['deltat']=abs(df['TIME']-df['time'])
     df['plev']=df_s['plev']
     df['u']=df_s['u']
     df['v']=df_s['v']
     df['error_era5']=df_s['error_mag']
     df=df.dropna()
     return df
    
def collocator(file, ds,df_total):
    df=pd.read_csv(file, skiprows=6,delimiter='\s+')
    stem=file[24:]
    date=datetime.strptime(stem[:10],"%Y%m%d%H")
    df['TIME']=date
    with open (file) as myfile:
        top=myfile.readlines()[0:5]
    lat=float(top[4][7:15])
    lon=float(top[4][25:34])
    sample=ds.sel(latitude=lat, longitude=lon, day=date,plev=df['PRES'].values,  method='nearest')
    sample=sample.sel(satellite='snpp')
    sample=preprocess(sample)
    
    df=df.reset_index()
    df_s=sample.to_dataframe().reset_index()
    if not df_s['u'].isnull().all():
        df=df_calculator(df, df_s, lat, lon, sample)
        if df_total.empty:
            df_total=df
        else:
            df_total=df_total.append(df)
                
    return df_total
    
    
    
def main():
    days= daterange(datetime(2020,7,1), datetime(2020,7,7), 24)
    for day in days:
        print(day)
        df_total=pd.DataFrame()
        filedate=day.strftime('%Y%m%d')
        dsdate=day.strftime('%m_%d_%Y_')
        files=glob.glob('../data/raw/radiosondes/' + filedate+'*.dat')
        for time in ('am','pm'):
            ds=xr.open_dataset('../data/processed/'+ dsdate+time+'.nc') 
            for file in files:
                df_total= collocator(file, ds, df_total)      
    df_total['u_error']=df_total.UWND - df_total.u
    df_total['v_error']=df_total.VWND - df_total.v
    df_total['error_mag']=np.sqrt(df_total.v_error**2+df_total.u_error**2)
    df_total.to_pickle('../data/processed/dataframes/collocated_radiosondes_july.pkl')
 
        
        
        
    
    


if __name__=='__main__':
    main()