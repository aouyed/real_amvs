#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:16:36 2021

@author: aouyed
"""

import xarray as xr
import numpy as np
import xesmf as xe
from datetime import datetime

pressures = [300, 800]

dates = (datetime(2009, 11, 4, 16, 45, 0, 0), datetime(
    2009, 11, 4, 17, 0, 0, 0), datetime(2009, 11, 4, 17, 15, 0, 0))


def date_loop(pressure_str):
    dr_total = xr.Dataset()
    for date in dates:
        print(date)
        year = str(date.year)
        month = str(date.month)
        day = '04'
        hour = date.hour
        min = date.minute
        if min == 0:
            min = '00'
        else:
            min = str(min)
        hour = str(hour)
        ds = xr.open_dataset('MCS_pressure_levels/MCS_Data4tracking_'+year+'-'+month +
                             '-'+day+'_'+hour+'_'+min+'_'+pressure_str+'hPa.nc')

        latmax = ds['lat'].max().item()
        latmin = ds['lat'].min().item()
        lonmax = ds['lon'].max().item()
        lonmin = ds['lon'].min().item()

        new_lat = np.arange(latmin, latmax, 0.0267)
        new_lon = np.arange(lonmin, lonmax, 0.0267)

        ds_out = xr.Dataset(
            {'lat': (['lat'], new_lat), 'lon': ('lon', new_lon), })
        regridder = xe.Regridder(ds, ds_out, 'bilinear', reuse_weights=True)
        dr_out = regridder(ds[['qv_nr', 'qv_cld', 'qv_clr', 'ua_nr', 'va_nr']])
        dr_out = dr_out.expand_dims('time')
        dr_out = dr_out.assign_coords(time=np.array([date]))
        dr_out = dr_out.rename({'ua_nr': 'u', 'va_nr': 'v'})
        if not dr_total:
            dr_total = dr_out
        else:
            dr_total = xr.concat([dr_total, dr_out], 'time')
    dr_total = dr_total.astype(np.float32)
    return dr_total


def pressure_loop():
    dr_total_p = xr.Dataset()
    for pressure in pressures:
        dr_unit_p = date_loop(str(pressure))
        dr_unit_p = dr_unit_p.expand_dims('pressure')
        dr_unit_p = dr_unit_p.assign_coords(pressure=np.array([pressure]))
        if not dr_total_p:
            dr_total_p = dr_unit_p
        else:
            dr_total_p = xr.concat([dr_total_p, dr_unit_p], 'pressure')
    return dr_total_p


# dr_total.to_netcdf('MCS_total.nc')

def main():
    dr_total = pressure_loop()
    dr_unit = dr_total[['qv_nr', 'u', 'v']]
    dr_unit = dr_unit.rename({'qv_nr': 'qv'})
    dr_unit = dr_unit.where(dr_unit != 0)
    dr_unit.to_netcdf('qv_nr.nc')

    dr_unit = dr_total[['qv_cld', 'u', 'v']]
    dr_unit = dr_unit.rename({'qv_cld': 'qv'})
    dr_unit = dr_unit.where(dr_unit != 0)
    dr_unit.to_netcdf('qv_cld.nc')

    dr_unit = dr_total[['qv_clr', 'u', 'v']]
    dr_unit = dr_unit.rename({'qv_clr': 'qv'})
    dr_unit = dr_unit.where(dr_unit != 0)
    dr_unit.to_netcdf('qv_clr.nc')

    print('Done!')


if __name__ == '__main__':
    main()