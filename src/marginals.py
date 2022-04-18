import xarray as xr
import numpy as np
import cross_section as cs
import matplotlib.pyplot as plt

factor=0.1
min_period=100

ds=xr.open_dataset('../data/processed/01_01_2020_am.nc')
ds=cs.preprocess(ds, 0)
ds['u_error'].plot.hist(label='u')
ds['v_error'].plot.hist(label='v',alpha=0.5)
plt.legend()
plt.savefig('../data/processed/plots/error bias.png', 
            bbox_inches='tight', dpi=300)