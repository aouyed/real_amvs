#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:04:26 2021

@author: aouyed
"""

import glob
from datetime import datetime 
import pandas as pd
import xarray as xr

def main():
    files=glob.glob('../data/raw/radiosondes/202007*.dat')
    time='am'
    ds=xr.open_dataset('../data/processed/07_01_2020_'+time+'.nc')  
    for file in files:
        df=pd.read_csv(file, skiprows=6,delimiter='\s+')
        stem=file[24:]
        date=datetime.strptime(stem[:10],"%Y%m%d%H")
        df['TIME']=date
        with open (file) as myfile:
            top=myfile.readlines()[0:5]
        lat=float(top[4][7:15])
        lon=float(top[4][25:34])
        
        print(file)
        breakpoint()
    
    


if __name__=='__main__':
    main()