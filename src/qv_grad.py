import xarray as xr
import pdb
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime
import numpy as np
PATH = '../data/processed/experiments/'


def calculator(ds):

    
    lat = ds.latitude.values
    lon = ds.longitude.values
    print('calculating deltas...')
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)

    qv = np.squeeze(ds['q_era5'].values)
    print('calculating gradient ...')
    grad = mpcalc.gradient(qv, deltas=(dy, dx))
    grad = np.array([grad[0].magnitude, grad[1].magnitude])
    grady = grad[0]
    gradx = grad[1]
    grad_mag = np.sqrt(gradx**2+grady**2)
    print('building data arrays...')
    ds['grad_y_qv'] = (['latitude', 'longitude'], grady)
    ds['grad_x_qv'] = (['latitude', 'longitude'], gradx)
    ds['grad_mag_qv'] = (['latitude', 'longitude'], 1e6*grad_mag)
    
    return ds

def angle(ds):
    """Calculates angle between moisture and wind velocity."""
    dot = ds['grad_x_qv']*ds['u_era5']+ds['grad_y_qv']*ds['v_era5']
    mags = np.sqrt(ds['grad_x_qv']**2+ds['grad_y_qv']**2) * \
    np.sqrt(ds['u_era5']**2+ds['v_era5']**2)
    c = (dot/mags)
    ds['angle'] = np.arccos(c)
    ds['angle'] = ds.angle/np.pi*180
    ds['neg_function'] = ds['grad_x_qv'] * \
        ds['v_era5'] - ds['grad_y_qv']*ds['u_era5']
    df=ds.to_dataframe()
    df.loc[df.neg_function < 0, 'angle'] = -df.loc[df.neg_function < 0, 'angle']
    df = df.drop(columns=['neg_function'])
    ds=xr.Dataset.from_dataframe(df)
    return ds

