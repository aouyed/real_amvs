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

    qv = np.squeeze(ds['specific_humidity_mean'].values)
    print('calculating gradient ...')
    grad = mpcalc.gradient(qv, deltas=(dy, dx))
    grad = np.array([grad[0].magnitude, grad[1].magnitude])
    grady = grad[0]
    gradx = grad[1]
    grad_mag = np.sqrt(gradx**2+grady**2)
    print('building data arrays...')
    ds['grad_y_qv'] = (['latitude', 'longitude'], grady)
    ds['grad_x_qv'] = (['latitude', 'longitude'], gradx)
    ds['grad_mag_qv'] = (['latitude', 'longitude'], grad_mag)
    
    return ds


