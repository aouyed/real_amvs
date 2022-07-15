#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:46:41 2022

@author: aouyed
"""

import radiosonde_plotter as rp 
import pandas as pd
from parameters import parameters 


def loader(param):
    df=pd.read_pickle('../data/processed/dataframes/'+param.month_string+'_winds_rs_model_'+param.tag+'.pkl')
    df=rp.preprocess(df)
    lat_range=df['lat'].between(-25,30)
    lon_range=df['lon'].between(-7,27)
    #df = df[lat_range & lon_range]
    #df=df[['lat','lon','plev','date_amv','date']].dropna().drop_duplicates()
    return df
    
def main():
    param=parameters()
    param.set_Lambda(0.15)
    df1=loader(param)
    param.set_Lambda(0.45)
    df2=loader(param)
    

    breakpoint()
    
    



if __name__=='__main__':
    main()