import xarray as xr
import numpy as np
import cross_section as cs
import matplotlib.pyplot as plt
import pandas as pd
import config 
import datetime 
from tqdm import tqdm 

factor=0.1
min_period=100
slices=(slice(-30,-90), slice(30,-30), slice(90,30))
start=config.MONTH
end=start + datetime.timedelta(days=6)

dates=pd.date_range(start=start, end=end, freq='d')




for date in tqdm(dates):
    for orbit in ('am','pm'):
        date_string=date.strftime('%m_%d_%Y')
    
        ds_total=xr.open_dataset('../data/processed/'+date_string+'_'+orbit+'_thick_plev.nc')
        ds_total['u_nobias']=xr.full_like(ds_total['u'], np.nan, dtype=np.double)
        
        for plev in ds_total['plev'].values:
            for slicel in slices:
                ds=ds_total.sel(latitude=slicel)
                ds=cs.preprocess(ds, 0)
                ds=ds.sel(plev=plev, method='nearest')
                error=ds['u_error'].median().item()
                ds_total['u_nobias'].loc[{'latitude':slicel}]=ds['u']-error
        ds_total['u']=ds_total['u_nobias']
        ds_total=ds_total.drop('u_nobias')
        ds_total.to_netcdf('../data/processed/'+date_string+'_'+orbit+'_thick_plev_nobias.nc')
