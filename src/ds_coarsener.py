#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 19:31:50 2021

@author: aouyed
"""

import xarray as xr
import time






def main():
   ds=xr.open_dataset('../data/interim/model_07_01_2020.nc')
   ds=ds[['u','v']].coarsen(longitude=4, latitude=4, boundary='trim').mean()
   print(ds)

    
   

if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))