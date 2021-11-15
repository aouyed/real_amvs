
from datetime import datetime
import pandas as pd
import xarray as xr
from datetime import timedelta
import numpy as np
import igra
import time
from tqdm import tqdm
import os.path
import config as c
import amv_calculators as ac
from siphon.simplewebservice.igra2 import IGRAUpperAir

PATH='../data/interim/rs_dataframes/'

HOURS=1.5 


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


def space_time_collocator(days, deltat):
    df=pd.DataFrame()
    for day in days:
        print(day)
        for time in ('am','pm'):
            dsdate = day.strftime('%m_%d_%Y')
            ds = xr.open_dataset('../data/processed/' + dsdate+'_'+time+'.nc')
            ds-ds.sel(satellite='snpp')
            df_unit=ds[['obs_time','u']].to_dataframe()
            df_unit=df_unit.reset_index()
            df_unit=df_unit[['latitude','longitude','obs_time','u']]
            first_rao=datetime(day.year, day.month, day.day, 0)
            second_rao=datetime(day.year, day.month, day.day, 12)
            condition1=df_unit['obs_time'].between(first_rao-deltat,first_rao+deltat)
            condition2=df_unit['obs_time'].between(second_rao-deltat,second_rao+deltat)
            df_unit=df_unit.loc[condition1 | condition2]
            df_unit=df_unit.dropna()
            df_unit=df_unit[['latitude','longitude','obs_time']]
            df_unit=df_unit.drop_duplicates()
                
            if df.empty:
                df=df_unit
            else:
                df=df.append(df_unit)
    df=df.reset_index(drop=True)
    df=df.drop_duplicates()
    return df
        
def collocate_igra(stations, lat, lon):

    
    condition1=stations['lat'].between(lat-1,lat+1)
    condition2=stations['lon'].between(lon-1,lon+1)
    condition3=condition1 & condition2
    condition4=stations['end'] >=2020
    stations=stations.loc[condition3 & condition4]
    return stations
    
    
def collocated_igra_ids(df):
     df=df.reset_index(drop=True)
     start_time = time.time()
     stations = igra.download.stationlist('/tmp')

     station_dict={'lat':[],'lon':[],'lon_rs':[],'lat_rs':[],'stationid':[],'obs_time':[]}
     for latlon in tqdm(df.values):
         lat,lon,obs_time = latlon
         df_unit=collocate_igra(stations, lat, lon)
         if not df_unit.empty:
             ids=df_unit.index.values.tolist()
             station_dict['lat'].append(lat)
             station_dict['lon'].append(lon)
             station_dict['lat_rs'].append(df_unit['lat'].values[0])
             station_dict['lon_rs'].append(df_unit['lon'].values[0])
             station_dict['stationid'].append(ids[0])
             station_dict['obs_time'].append(obs_time)
             
     output_df=pd.DataFrame(data=station_dict)
     print("--- %s seconds ---" % (time.time() - start_time))    
     print(output_df)
     return output_df
 
    




def igra_downloader(df,days, month_string):
    station_list=np.unique(df['stationid'].values)
    for station in tqdm(station_list):
        fname=PATH+month_string+'_' +station+'.pkl'
        if not os.path.isfile(fname):
            df_total=pd.DataFrame()
    
            for day in days:
    
                first_rao=datetime(day.year, day.month, day.day, 0)
                second_rao=datetime(day.year, day.month, day.day, 12)
                for date in (first_rao, second_rao):    
                    try:
                        df_unit, header = IGRAUpperAir.request_data(date, station)
                        df_unit['date']=date
                        df_unit['stationid']=station
                        if df_total.empty:
                            df_total=df_unit
                        else:
                            df_total=df_total.append(df_unit)
                    except Exception as e:
                        pass
            if not df_total.empty:
                df_total.to_pickle('../data/interim/rs_dataframes/' + month_string+'_' +station +'.pkl')
    
    
                
            
    
         
   
     
    


def main():
    deltat=timedelta(hours=HOURS)
    start_date=c.MONTH
    end_date=c.MONTH + timedelta(days=6)
    days=ac.daterange(start_date, end_date, 24)
    month_string=c.month_string
    df=space_time_collocator(days, deltat)
    df=df.reset_index(drop=True)
    df=df.drop_duplicates()
    df=collocated_igra_ids(df)
    df.to_pickle('../data/interim/dataframes/'+month_string+'_igra_id.pkl')
    df=pd.read_pickle('../data/interim/dataframes/'+month_string+'_igra_id.pkl')
    print(df)
    igra_downloader(df,days, month_string)



if __name__ == '__main__':
    main()
