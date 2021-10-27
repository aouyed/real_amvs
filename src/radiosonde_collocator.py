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
    ds = ds.drop(['day', 'satellite', 'time', 'flowx', 'flowy'])
    ds['u_error'] = ds['u']-ds['u_era5']
    ds['v_error'] = ds['v']-ds['v_era5']
    ds['error_mag'] = np.sqrt(ds['u_error']**2+ds['v_error']**2)
    return ds


def df_calculator(df, df_s, df_m, lat, lon, sample):
    df['LAT'] = lat
    df['LON'] = lon
    df['lat'] = sample['latitude'].values.item()
    df['lon'] = sample['longitude'].values.item()
    df['time'] = df_s['obs_time']
    df['deltat'] = abs(df['TIME']-df['time'])
    df['plev'] = df_s['plev']
    df['u'] = df_s['u']
    df['v'] = df_s['v']
    df['u_era5'] = df_s['u_era5']
    df['v_era5'] = df_s['v_era5']
    df['error_era5'] = df_s['error_mag']
    df['time_coarse_era5']=df_m['time']
    df['u_coarse_era5']=df_m['u']
    df['v_coarse_era5']=df_m['v']
    df['pres_era5']=df_m['level']
    df['lat_era']=df_m['latitude'].values[0]
    df['lon_era']=df_m['longitude'].values[0]

    df = df.dropna()
    return df


def collocated_pressure_list(df, plevs):
    df_total = pd.DataFrame()
    for plev in plevs:
        df_unit = df.iloc[(df['PRES']-plev).abs().argsort()[:1]]
        df_unit = df_unit.reset_index(drop=True)
        df_unit['plev'] = plev
        if df_total.empty:
            df_total = df_unit
        else:
            df_total = df_total.append(df_unit)
    return df_total


def collocator(file, ds, ds_model, df_total):
    df = pd.read_csv(file, skiprows=6, delimiter='\s+')
    df = collocated_pressure_list(df, ds['plev'].values)
    stem = file[24:]
    date = datetime.strptime(stem[:10], "%Y%m%d%H")
    df['TIME'] = date
    with open(file) as myfile:
        top = myfile.readlines()[0:5]
    lat = float(top[4][7:15])
    lon = float(top[4][25:34])
    sample = ds.sel(latitude=lat, longitude=lon, day=date,
                    plev=df['plev'].values,  method='nearest')
    sample = sample.sel(satellite='snpp')
    sample = preprocess(sample)
    ds_coarse_era5= ds_model.sel(latitude=lat, longitude=lon, time=sample['obs_time'].values[0],
                    level=df['plev'].values,  method='nearest')
    

    df = df.reset_index()
    df_s = sample.to_dataframe().reset_index()
    df_coarse_era5=ds_coarse_era5.to_dataframe().reset_index()
    if not df_s['u'].isnull().all():
        df = df_calculator(df, df_s, df_coarse_era5, lat, lon, sample)
        if df_total.empty:
            df_total = df
        else:
            df_total = df_total.append(df)

    return df_total


def main():
    days = daterange(datetime(2020, 7, 1), datetime(2020, 7, 7), 24)
    for day in days:
        print(day)
        df_total = pd.DataFrame()
        filedate = day.strftime('%Y%m%d')
        dsdate = day.strftime('%m_%d_%Y')
        files = glob.glob('../data/raw/radiosondes/' + filedate+'*.dat')
        for time in ('am', 'pm'):
            ds = xr.open_dataset('../data/processed/' + dsdate+'_'+time+'.nc')
            ds_model=xr.open_dataset('../data/interim/coarse_model_' + dsdate+'.nc')
            ds_model = ds_model.assign_coords(longitude=(((ds_model.longitude + 180) % 360) - 180))
            ds_model=ds_model.reindex(longitude=np.sort(ds_model['longitude'].values))

            for file in files:
                df_total = collocator(file, ds,ds_model,  df_total)
    df_total['u_error'] = df_total.UWND - df_total.u
    df_total['v_error'] = df_total.VWND - df_total.v
    df_total['error_mag'] = np.sqrt(df_total.v_error**2+df_total.u_error**2)
    df_total.to_pickle(
        '../data/processed/dataframes/collocated_radiosondes_july.pkl')


if __name__ == '__main__':
    main()
