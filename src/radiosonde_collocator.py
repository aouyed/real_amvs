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

def main():
    files=glob.glob('../data/raw/radiosondes/20200702*.dat')
    print(files)
    
    time='am'
    ds=xr.open_dataset('../data/processed/07_02_2020_'+time+'.nc')  
    df_total=pd.DataFrame()
    for file in files:
        print(file)
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
        df=df.reset_index()
        df_s=sample.to_dataframe().reset_index()
        if not df_s['u'].isnull().all():
            df['LAT']=lat
            df['LON']=lon
            df['lat']=sample['latitude'].values.item()
            df['lon']=sample['longitude'].values.item()
            df['time']=df_s['obs_time']
            df['plev']=df_s['plev']
            df['u']=df_s['u']
            df['v']=df_s['v']
            print(df)
            if df_total.empty:
                df_total=df
            else:
                df_total=df_total.append(df)
    print(df_total[['TIME','time']])
        
        
        
    
    


if __name__=='__main__':
    main()