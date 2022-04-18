import xarray as xr
import numpy as np

factor=0.1
min_period=100

ds=xr.open_dataset('../data/processed/01_01_2020_am.nc')
ds=ds.sel(satellite='j1')
ds=ds.sel(plev=ds['plev'].values[-10:])
ds['u_mean']=ds['u'].rolling({'latitude':11,'plev':3,'longitude':11}, min_periods=min_period).mean(skipna=True)
ds['u_std']=ds['u'].rolling({'latitude':11,'plev':3,'longitude':11}, min_periods=min_period).std(skipna=True)
ds['v_mean']=ds['v'].rolling({'latitude':11,'plev':3,'longitude':11}, min_periods=min_period).mean(skipna=True)
ds['v_std']=ds['v'].rolling({'latitude':11,'plev':3,'longitude':11}, min_periods=min_period).std(skipna=True)
ds['speed']=np.sqrt(ds.u**2+ds.v**2)
ds['speed_mean']=ds['speed'].rolling({'latitude':11,'plev':3,'longitude':11}, min_periods=min_period).mean(skipna=True)
ds['speed_std']=ds['speed'].rolling({'latitude':11,'plev':3,'longitude':11}, min_periods=min_period).std(skipna=True)

low_u=ds['u_mean']-factor*ds['u_std']
high_u=low_u=ds['u_mean']+factor*ds['u_std']
low_v=ds['v_mean']-factor*ds['v_std']
high_v=low_u=ds['v_mean']+factor*ds['v_std']
low_speed=ds['speed_mean']-factor*ds['speed_std']
high_speed=ds['speed_mean']+factor*ds['speed_std']

condition1=(ds.u>low_u) & (ds.u<high_u)
condition2=(ds.v>low_v) & (ds.v<high_v)
condition3=(ds.speed>low_speed) & (ds.speed<high_speed)
filtered_ds=ds.where(condition3)
error_sq=(ds.u-ds.u_era5)**2+(ds.v-ds.v_era5)**2
print(np.sqrt(error_sq.mean().item()))
error_sq_f=(filtered_ds.u-filtered_ds.u_era5)**2+(filtered_ds.v-filtered_ds.v_era5)**2
print(np.sqrt(error_sq_f.mean().item()))


