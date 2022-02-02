import xarray as xr
import numpy as np
import config as c


def ds_loader(filename):
    ds_total=xr.Dataset()

    ds=xr.open_dataset('../data/raw/'+filename)
    ds['air_pres_h2o']=ds['air_pres_h2o']/100

    for orbit_pass in ds['orbit_pass'].values:
        ds_orbit=xr.Dataset()ds_tot
        for pressure in c.PRESSURES:
            ds_unit=ds.sel(orbit_pass=orbit_pass)
            ds_unit=ds_unit.sel(air_pres_h2o=pressure, method='nearest')
            ds_unit=ds_unit.squeeze()
            lon, lat = np.meshgrid(ds_unit['lon'].values,
                                   ds_unit['lat'].values, indexing='xy')
            ds_unit['lat_coor']= (['lat','lon'], lat)
            ds_unit['lon_coor']= (['lat','lon'], lon)
            start=ds_unit['obs_time_tai93']
            dt_sec=240*ds_unit['lon_coor']
            dt=dt_sec.values.astype('timedelta64[s]')
            obs_time=start.values + dt
            ds_unit['obs_time']=(['lat','lon'], obs_time)
            ds_unit=ds_unit[['spec_hum','rel_hum']]
            if not ds_orbit:
                ds_orbit=ds_unit
            else:
                ds_orbit=xr.concat([ds_orbit,ds_unit],'air_pres_h2o')
        if not ds_orbit:
            ds_total=ds_orbit
        else:
            ds_total=xr.concat([ds_total,ds_orbit],'orbit_pass')
    print(ds_total)
    return ds_total

def main():
    ds_j1=ds_loader('j1.nc')
    print(ds_j1)


if __name__ == '__main__':
    main()

