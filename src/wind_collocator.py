
import glob
from datetime import datetime
import pandas as pd
import xarray as xr
from datetime import timedelta
import numpy as np
import igra
import time
import os.path
from tqdm import tqdm
HOURS=1.5 
PATH='../data/interim/rs_dataframes/'

def daterange(start_date, end_date, dhour):
    date_list = []
    delta = timedelta(hours=dhour)
    while start_date <= end_date:
        date_list.append(start_date)
        start_date += delta
    return date_list

def collocated_pressure_list(df, plevs):
    df_total = pd.DataFrame()
    for plev in plevs:
        df_unit = df.iloc[(df['pressure']-plev).abs().argsort()[:1]]
        df_unit = df_unit.reset_index(drop=True)
        df_unit['plev'] = plev
        if df_total.empty:
            df_total = df_unit
        else:
            df_total = df_total.append(df_unit)
    
    return df_total.reset_index(drop=True)
            

def df_calculator(df, df_s, df_m, lat, lon, lat_rs,lon_rs):
    df['lat'] = lat
    df['lon'] = lon
    df['lat_rs'] = lat_rs
    df['lon_rs'] = lon_rs
    df['date_amv'] = df_s['obs_time']
    df['deltat'] = abs(df['date']-df['date_amv'])
    df['plev'] = df_s['plev']
    df['u'] = df_s['u']
    df['v'] = df_s['v']
    df['u_era5'] = df_s['u_era5']
    df['v_era5'] = df_s['v_era5']
    df['time_coarse_era5']=df_m['time']
    df['u_coarse_era5']=df_m['u']
    df['v_coarse_era5']=df_m['v']
    df['pres_era5']=df_m['level']
    
    
    return df
   
    
def collocated_winds(df):
    df_total=pd.DataFrame()
    for parameters in tqdm(df.values):
        lat,lon,lon_rs, lat_rs, station,obs_time=parameters
        fname=PATH+station+'.pkl'
        if os.path.isfile(fname):
            df_rs=pd.read_pickle(fname)
            df_rs=df_rs.reset_index()
            if obs_time.hour >= (24-HOURS):
                date=datetime(obs_time.year,obs_time.month, obs_time.day+1,0)
            elif obs_time.hour<= HOURS:
                date=datetime(obs_time.year,obs_time.month, obs_time.day,0)
            else:
                date=datetime(obs_time.year,obs_time.month, obs_time.day,12)
         
            df_rs=df_rs.loc[df_rs['date']==date]
            if not df_rs.empty:
                
                ds_path=obs_time.strftime('../data/processed/%m_%d_%Y_am.nc')
                dsdate = obs_time.strftime('%m_%d_%Y')
                ds=xr.open_dataset(ds_path)
                ds=ds.sel(satellite='snpp')
                ds=ds.squeeze()
                
                ds_model=xr.open_dataset('../data/interim/coarse_model_' + dsdate+'.nc')
                ds_model = ds_model.assign_coords(longitude=(((ds_model.longitude + 180) % 360) - 180))
                ds_model=ds_model.reindex(longitude=np.sort(ds_model['longitude'].values))
               
                df_rs=collocated_pressure_list(df_rs, ds['plev'].values)
                ds=ds.sel(latitude=lat,longitude=lon, plev=df_rs['plev'].values, method='nearest')   
                ds_model= ds_model.sel(latitude=lat, longitude=lon, time=obs_time,
                    level=df_rs['plev'].values,  method='nearest')
               
                df_s=ds.to_dataframe().reset_index()
                df_m=ds_model.to_dataframe().reset_index()

                df_rs=df_calculator(df_rs, df_s, df_m, lat, lon, lat_rs, lon_rs)
                if df_total.empty:
                    df_total=df_rs
                else:
                    df_total=df_total.append(df_rs)
    df_total.to_pickle('../data/processed/dataframes/winds_rs_model.pkl')
    

                
                

        
        
        



    


def main():
    deltat=timedelta(hours=HOURS)
    days = daterange(datetime(2020, 7, 1), datetime(2020, 7, 7), 24)
    #df=space_time_collocator(days, deltat)
    #df=collocated_igra_ids(df)
    #df.to_pickle('../data/interim/dataframes/igra_id.pkl')
    df=pd.read_pickle('../data/interim/dataframes/igra_id.pkl')
    df=df.reset_index(drop=True)
    print(df)
    collocated_winds(df)



if __name__ == '__main__':
    main()
