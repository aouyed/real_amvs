# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import first_stage_amv as fsa
from datetime import datetime 
import cartopy.crs as ccrs
import matplotlib.ticker as mticker



def quiver_plot(ds, title, u, v):
    ds=ds.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    ax.set_title(title)
    Q = ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values))
    qk = ax.quiverkey(Q, 0.8, 0.9, 5, r'5 m/s', labelpos='E',
                      coordinates='figure')
    fig.tight_layout()
    plt.savefig('../data/processed/plots/quiver_'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
def quiver_plot_cartopy(ds, title, u, v):
    ds=ds.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    fig=plt.figure()
    X, Y = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    Q = ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values))
    qk = ax.quiverkey(Q, 0.5, 0.5, 10, r'10 m/s', labelpos='E',
                      coordinates='figure')
    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()  
    
def quiver_ax_cartopy(ax,ds, title, u, v):
    ds=ds.coarsen(latitude=5, longitude=5, boundary='trim').mean()
    X, Y = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
   
    Q = ax.quiver(X, Y, np.squeeze(
        ds[u].values), np.squeeze(ds[v].values),scale=250)
    qk=ax.quiverkey(Q, 0.5, 0.5, 5, r'5 m/s', labelpos='E',
                      coordinates='figure')
    ax.set_title(title, fontsize=8)
    ax.coastlines()
    gl=ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])

    return ax,gl
    
    

def main():
    print('main')

if __name__ == '__main__':
    main()

