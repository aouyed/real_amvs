
import glob
from datetime import datetime
import pandas as pd
import xarray as xr
import cross_section as cs
from datetime import timedelta
import numpy as np
import igra
import time
from siphon.simplewebservice.igra2 import IGRAUpperAir
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
        time='am'
        dsdate = day.strftime('%m_%d_%Y')
        ds = xr.open_dataset('../data/processed/' + dsdate+'_'+time+'.nc')
        df_unit=ds['obs_time'].to_dataframe()
        df_unit=df_unit.reset_index()
        df_unit=df_unit[['latitude','longitude','obs_time']]
        first_rao=datetime(day.year, day.month, day.day, 0)
        second_rao=datetime(day.year, day.month, day.day, 12)
        condition1=df_unit['obs_time'].between(first_rao-deltat,first_rao+deltat)
        condition2=df_unit['obs_time'].between(second_rao-deltat,second_rao+deltat)
        df_unit=df_unit.loc[condition1 | condition2]
        df_unit=df_unit.drop_duplicates(ignore_index=True)
        
        if df.empty:
            df=df_unit
        else:
            df=df.append(df_unit)
    return df 
        
def collocate_igra(stations, latlon):
    lat=latlon[0]
    lon=latlon[1]
    
    condition1=stations['lat'].between(lat-1,lat+1)
    condition2=stations['lon'].between(lon-1,lon+1)
    condition3=condition1 & condition2
    condition4=stations['end'] >=2020
    stations=stations.loc[condition3 & condition4]
    return stations
    
    
def collocated_igra_ids(df):
     start_time = time.time()
     stations = igra.download.stationlist('/tmp')
     lat=df['latitude'].values
     lon=df['longitude'].values
     times=df['obs_time'].values
     latlons=list(zip(lat,lon, times))
     station_dict={'lat':[],'lon':[],'stationid':[],'obs_time':[]}
     for latlon in latlons:
         df_unit=collocate_igra(stations, latlon)
         if not df_unit.empty:
             ids=df_unit.index.values.tolist()
             station_dict['lat'].append(latlon[0])
             station_dict['lon'].append(latlon[1])
             station_dict['stationid'].append(ids[0])
             station_dict['obs_time'].append(latlon[2])
     output_df=pd.DataFrame(data=station_dict)
     print("--- %s seconds ---" % (time.time() - start_time))    
     print(output_df)
     return output_df
            

def collocated_winds(df):
    
    for station in df[['stationid','obs_time']].values:
        date=station[1]
        stationid=station[0]
        if date.hour> (24-HOURS):
            date=datetime(date.year,date.month, date.day+1,0)
        elif date.hour<= HOURS:
            date=datetime(date.year,date.month, date.day,0)
        else:
            date=datetime(date.year,date.month, date.day,12)
            
            
        print(station)
        print(date)
        start_time = time.time()

        df, header = IGRAUpperAir.request_data(date, stationid)
        print("--- %s seconds ---" % (time.time() - start_time))    


        breakpoint()
        



         


def igra_downloader(df,days):
    df_total=pd.DataFrame()
    station_list=np.unique(df['stationid'].values)
    for station in station_list:
        print(station)
        for day in days:

            print(day)
            first_rao=datetime(day.year, day.month, day.day, 0)
            second_rao=datetime(day.year, day.month, day.day, 12)
            for date in (first_rao, second_rao):
                start_time = time.time()

                try:
                    df_unit, header = IGRAUpperAir.request_data(date, station)
                    df_unit['date']=date
                    df_unit['stationid']=station
                    if df_total.empty:
                        df_total=df_unit
                    else:
                        df_total=df_total.append(df_unit)
                except ValueError:
                    print('exception')
                print("--- %s seconds ---" % (time.time() - start_time))    

    return df_total


            
            
    
         
   
     
    


def main():
    deltat=timedelta(hours=HOURS)
    days = daterange(datetime(2020, 7, 1), datetime(2020, 7, 7), 24)
    #df=space_time_collocator(days, deltat)
    #df=collocated_igra_ids(df)
    #df.to_pickle('../data/interim/dataframes/igra_id.pkl')
    df=pd.read_pickle('../data/interim/dataframes/igra_id.pkl')
    print(df)
    df_rao=igra_downloader(df,days)
    df_rao=pd.read_pickle('../data/interim/dataframes/igra_rao.pkl')



if __name__ == '__main__':
    main()
