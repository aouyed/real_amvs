#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 15:15:56 2021

@author: aouyed
"""
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import config as c
import cartopy.crs as ccrs



THRESHOLDS=[10,4]

def scatter(x,y):
    fig, ax=plt.subplots()
    ax.scatter(x,y)
    ax.set_xlabel('Radiosonde [m/s]')
    ax.set_ylabel('ERA 5 [m/s]')
    ax.set_title('Magnitude of Vector Difference')
    ax.set_xlim(0,20)
    ax.set_ylim(0,20)
    plt.show()
    plt.close()
    

def pressure_df(df):
    plevs=df['plev'].unique()
    d={'plev':[],'rmsvd':[],'rmsvd_era5':[]}
    for plev in plevs:
        
        df_unit=df[df.plev==plev]
        rmsvd=np.sqrt(df_unit['error_square'].mean())
        rmsvd_era5=np.sqrt(df_unit['error_square_era5'].mean())
        d['plev'].append(plev)
        d['rmsvd'].append(rmsvd)
        d['rmsvd_era5'].append(rmsvd_era5)
        
    df_pressure=pd.DataFrame(data=d)
    df_pressure=df_pressure.sort_values(by=['plev'])
    return df_pressure 
        
def pressure_plot(df,rmsvd_label, title):
    fig, ax=plt.subplots() 
    df_pressure_era= pressure_df(df)
    for thresh in THRESHOLDS:
        df_unit=df[df.error_mag<thresh]
        print(df_unit.shape)
        df_pressure= pressure_df(df_unit)
        ax.plot(df_pressure[rmsvd_label], df_pressure.plev, label='δ = '+str(thresh)+' m/s')
    ax.plot(df_pressure_era['rmsvd_era5'], df_pressure_era.plev, label='ERA 5')
    ax.axvline(4.05,linestyle='dashed',label='Aeolus (Mie)')
    ax.axvline(5.93,linestyle='dotted',label='GEO AMVs' )
    ax.legend(frameon=False)
    ax.set_xlabel('RMSVD [m/s]')
    ax.set_ylabel('Pressure [hPa]')
    ax.set_xlim(0,10)
    ax.set_yscale('symlog')
    ax.set_yticklabels(np.arange(900, 50, -100))
    ax.set_ylim(df['plev'].max(), df['plev'].min())
    ax.set_yticks(np.arange(900, 50, -100))
    ax.set_title(title)
    plt.savefig('../data/processed/plots/'+c.month_string+'_radiosonde_comparison.png', dpi=300)
    plt.show()
    plt.close()
           
def sample_stat_calc(df_pressure, df_pressure_era, thresh):
    sample_unit=df_pressure[df_pressure.plev.between(852.5,853)]
    sample_era5=df_pressure_era[df_pressure_era.plev.between(852.5,853)]
    sample_unit['rmsvd_era5']=  sample_era5['rmsvd_era5'].values.item()
    
    sample_unit2=df_pressure[df_pressure.plev.between(300,301)]
    sample_era5=df_pressure_era[df_pressure_era.plev.between(300,300.1)]
    sample_unit2['rmsvd_era5']=  sample_era5['rmsvd_era5'].values.item()
        
    sample_unit=sample_unit.append(sample_unit2)
    sample_unit['thresh']=thresh
    return sample_unit
    
def pressure_ax(ax, df,rmsvd_label):
    sample_stats=pd.DataFrame()
    df_pressure_era= pressure_df(df)
    for thresh in THRESHOLDS:
        df_unit=df[df.error_mag<thresh]
        df_pressure= pressure_df(df_unit)
        ax.plot(df_pressure[rmsvd_label], df_pressure.plev, label='δ = '+str(thresh)+' m/s')
        sample_unit=sample_stat_calc(df_pressure,  df_pressure_era, thresh)
        if sample_stats.empty:
            sample_stats=sample_unit
        else:
            sample_stats=sample_stats.append(sample_unit)
    ax.plot(df_pressure_era['rmsvd_era5'], df_pressure_era.plev, label='ERA 5')
    ax.axvspan(5.93, 8.97, alpha=0.25, color='grey')    
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlabel('RMSVD [m/s]')
    ax.set_ylabel('Pressure [hPa]')
    ax.set_xlim(2,10)
    ax.set_yscale('symlog')
    ax.set_yticklabels(np.arange(900, 50, -125))
    ax.set_ylim(df['plev'].max(), df['plev'].min())
    ax.set_yticks(np.arange(900, 50, -125))
    return ax, sample_stats
    

def scatter_plot_cartopy(ax, title, x, y):
    gl=ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.xlabels_top = False
    gl.ylabels_right=False
    ax.coastlines()
    ax.scatter(x,y,s=20)
    return ax



def location_loader():
    df1=pd.read_pickle('../data/processed/dataframes/january_winds_rs_model.pkl')
    df2=pd.read_pickle('../data/processed/dataframes/july_winds_rs_model.pkl')
    df1.reset_index(drop=True)
    df2.reset_index(drop=True)
    df=df1.append(df2).reset_index(drop=True)
    df=preprocess(df)
    df=df.loc[df.error_mag<4]
    df=df[['lat_rs','lon_rs','stationid']].drop_duplicates(ignore_index=True)
    print(df.shape)
    breakpoint()
    return(df)

    
def multiple_pressure_map(df_jan, df_july, fname):
    fig=plt.figure()
    ax1= plt.subplot(2,2,1)
    ax2= plt.subplot(2,2,2)
    ax3=plt.subplot(2,1,2,projection=ccrs.PlateCarree())

    axlist = [ax1,ax2,ax3]
    axlist[0], sample_stats_jan =pressure_ax(axlist[0], df_jan, 'rmsvd')
    axlist[1], sample_stats_july =pressure_ax(axlist[1], df_july, 'rmsvd')
    df=location_loader()
    axlist[2]=scatter_plot_cartopy(axlist[2],'rs_coords',df['lon_rs'],df['lat_rs'])

    axlist[0].text(2.5,775,'(a)')
    axlist[1].text(2.5,775,'(b)')
    axlist[2].text(-170,-60,'(c)')


    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+fname +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    sample_stats_jan['month']='january'
    sample_stats_july['month']='july'
    sample_stats=sample_stats_jan.append(sample_stats_july)
    sample_stats['rmsvd_diff']=sample_stats['rmsvd']-sample_stats['rmsvd_era5']
    print(sample_stats)
    sample_stats.to_csv('../data/processed/dataframes/sample_rao_stats.csv')
    
       
def preprocess(df):
    df=df[df.u_wind>-1000]
    udiff=df.u_wind-df.u
    vdiff=df.v_wind-df.v
    df['error_mag']=np.sqrt((df.u-df.u_era5)**2+(df.v-df.v_era5)**2)
    df['error_square']=udiff**2+vdiff**2
    df['error_square_era5']=(df.u_wind-df.u_era5)**2+(df.v_wind-df.v_era5)**2
    return df


def main():
    df_jan=pd.read_pickle('../data/processed/dataframes/january_winds_rs_model.pkl')
    df_july=pd.read_pickle('../data/processed/dataframes/july_winds_rs_model.pkl')

    df_jan=preprocess(df_jan)
    df_jan=df_jan.drop_duplicates()
    
    df_july=preprocess(df_july)
    df_july=df_july.drop_duplicates()
    
    multiple_pressure_map(df_jan,df_july,  'rmsvd_map')
    
    
    
    
    


if __name__=='__main__':
    main()