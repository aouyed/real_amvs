#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 15:26:14 2021

@author: aouyed
"""

import xarray as xr
from datetime import datetime 
import numpy as np


def main():
    day=datetime(2020,7,1)
    ds=xr.open_dataset('../data/raw/aeolus/july/AE_OPER_ALD_U_N_2B_20200701T003647_20200701T020459_0002.nc')
    print(ds)  
    dsdate = day.strftime('%m_%d_%Y')  
    ds_model=xr.open_dataset('../data/interim/coarse_model_' + dsdate+'.nc')
    ds_model = ds_model.assign_coords(longitude=(((ds_model.longitude + 180) % 360) - 180))
    ds_model=ds_model.reindex(longitude=np.sort(ds_model['longitude'].values))
    breakpoint()
    
    
if __name__=="__main__":
    main()
    