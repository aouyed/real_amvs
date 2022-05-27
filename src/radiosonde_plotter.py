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
from parameters import parameters 
from datetime import datetime 

THRESHOLDS=[10, 4]
TAGS=[['filtered_thick_plev_tlv1',10],['filtered_4_thick_plev_tlv1',4]]
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
    d={'plev':[],'rmsvd':[],'rmsvd_era5':[],'u_error':[], 'v_error':[],
       'u_error_era5':[], 'v_error_era5':[], 'angle':[], 
       'angle_era5':[],'angle_bias':[],'angle_bias_era5':[],
       'speed_bias':[],'speed_bias_era5':[]}
    for plev in plevs:
        
        df_unit=df[df.plev==plev]
        rmsvd=np.sqrt(df_unit['error_square'].mean())
        u_error=df_unit['u_error'].mean()
        u_error_era5=df_unit['u_error_era5'].mean()

        v_error=df_unit['v_error'].mean()
        v_error_era5=df_unit['v_error_era5'].mean()

        angle=df_unit['angle'].mean()
        angle_era5=df_unit['angle_era5'].mean()
        angle_bias=df_unit['signed_angle'].mean()
        angle_bias_era5=df_unit['signed_angle_era5'].mean()
        speed_bias=df_unit['speed_error'].mean()
        speed_bias_era5=df_unit['speed_error_era5'].mean()


        rmsvd_era5=np.sqrt(df_unit['error_square_era5'].mean())
        d['plev'].append(plev)
        d['rmsvd'].append(rmsvd)
        d['rmsvd_era5'].append(rmsvd_era5)
        d['u_error'].append(u_error)
        d['u_error_era5'].append(u_error_era5)

        d['v_error'].append(v_error)
        d['v_error_era5'].append(v_error_era5)


        d['angle'].append(angle)
        d['angle_era5'].append(angle_era5)
        
        d['speed_bias'].append(speed_bias)
        d['speed_bias_era5'].append(speed_bias_era5)
        
        d['angle_bias'].append(angle_bias)
        d['angle_bias_era5'].append(angle_bias_era5)



        
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
    
def pressure_ax(ax,  param,rmsvd_label,xlabel, xlim):
    sample_stats=pd.DataFrame()
    month_string=param.month_string
    param.set_thresh(100)

    df=pd.read_pickle('../data/processed/dataframes/'+month_string+'_winds_rs_model_'+ param.tag +'.pkl')
    df=preprocess(df)
    df=df.drop_duplicates()
    df_pressure_era= pressure_df(df)
    print('era_5')
    print(df['signed_angle_era5'].mean())
    print(df['speed_error_era5'].mean())
    
    print(df['signed_angle'].mean())
    print(df['speed_error'].mean())


    
    for thresh in THRESHOLDS:
        param.set_thresh(thresh)
        df_unit=pd.read_pickle('../data/processed/dataframes/'+month_string+'_winds_rs_model_'+ param.tag +'.pkl')
        df_unit=preprocess(df_unit)
        print(thresh)
        print(df_unit['signed_angle'].mean())
        print(df_unit['speed_error'].mean())

        df_pressure= pressure_df(df_unit)
        ax.plot(df_pressure[rmsvd_label], df_pressure.plev, label='δ = '+str(thresh)+' m/s')
        #sample_unit=sample_stat_calc(df_pressure,  df_pressure_era, thresh)
        #if sample_stats.empty:
         #   sample_stats=sample_unit
        #else:
         #   sample_stats=sample_stats.append(sample_unit)
    ax.plot(df_pressure_era[rmsvd_label+'_era5'], df_pressure_era.plev, label='ERA 5')
    #ax.axvspan(5.93, 8.97, alpha=0.25, color='grey')    
    #ax.legend(frameon=False, loc='upper left')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Pressure [hPa]')
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_yscale('symlog')
    ax.set_yticklabels(np.arange(900, 50, -125))
    ax.set_ylim(df['plev'].max(), df['plev'].min())
    ax.set_yticks(np.arange(900, 50, -125))
    return ax
    

def scatter_plot_cartopy(ax, title, x, y):
    gl=ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    gl.xlabels_top = False
    gl.ylabels_right=False
    ax.coastlines()
    ax.scatter(x,y,s=20)
    return ax



def location_loader(param):
    param.set_month(datetime(2020,1,1))
    df1=pd.read_pickle('../data/processed/dataframes/'+param.month_string+'_winds_rs_model_'+param.tag+'.pkl')
    param.set_month(datetime(2020,7,1))
    df2=pd.read_pickle('../data/processed/dataframes/'+param.month_string+'_winds_rs_model_'+param.tag+'.pkl')
    #df2=df1
    df1.reset_index(drop=True)
    df2.reset_index(drop=True)
    df=df1.append(df2).reset_index(drop=True)
    df=preprocess(df)
    #df=df.loc[df.error_mag<4]
    df=df[['lat_rs','lon_rs','stationid']].drop_duplicates(ignore_index=True)
    print(df.shape)
    return(df)

    
def multiple_pressure_map(fname, label, xlabel, param):
    fig=plt.figure()
    ax1= plt.subplot(2,2,1)
    ax2= plt.subplot(2,2,2)
    ax3=plt.subplot(2,1,2,projection=ccrs.PlateCarree())

    axlist = [ax1,ax2,ax3]
    param.set_month(datetime(2020,1,1))
    axlist[0]=pressure_ax(axlist[0],param, label,xlabel)
    param.set_month(datetime(2020,7,1))

    axlist[1]=pressure_ax(axlist[1], param, label, xlabel)
    df=location_loader(param)
    axlist[2]=scatter_plot_cartopy(axlist[2],'rs_coords',df['lon_rs'],df['lat_rs'])

    axlist[0].text(4,275,'(a)')
    axlist[1].text(4.,275,'(b)')
    axlist[2].text(-170,-40,'(c)')


    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+param.tag+fname +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    
def four_panel_plot(fname, param, var1='rmsvd', var2='angle', 
                     xlabel1='RMSVD [m/s]', 
                     xlabel2='Angle [deg]',xlim1=(0,12), xlim2=(10,60)):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    axlist = axes.flat
    
    two_radiosonde_panels(axlist[:2],var1,xlabel1,xlim1,param)
    two_radiosonde_panels(axlist[2:], var2,xlabel2,xlim2,param)
    axlist[0].legend(frameon=False)
    
    axlist[0].text(0.05,0.05,'(a)',transform=axlist[0].transAxes)
    axlist[1].text(0.05,0.05,'(b)',transform=axlist[1].transAxes)
    axlist[2].text(0.05,0.05,'(c)',transform=axlist[2].transAxes)
    axlist[3].text(0.05,0.05,'(d)',transform=axlist[3].transAxes)


   


    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+param.tag+fname +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    

def two_radiosonde_panels(axlist, label, xlabel, xlim, param):
    param.set_month(datetime(2020,1,1))
    axlist[0]=pressure_ax(axlist[0],param, label,xlabel, xlim)
    param.set_month(datetime(2020,7,1))

    axlist[1]=pressure_ax(axlist[1], param, label, xlabel, xlim)
    #df=location_loader(param)
    #axlist[2]=scatter_plot_cartopy(axlist[2],'rs_coords',df['lon_rs'],df['lat_rs'])

def location_plot(fname, param):
    fig=plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    param.set_thresh(10)
    df=location_loader(param)
    ax=scatter_plot_cartopy(ax,'rs_coords',df['lon_rs'],df['lat_rs'])

    fig.tight_layout()
    plt.savefig('../data/processed/plots/location_'+param.tag+fname +
                '.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    



def angle(df, ulabel, vlabel, angle_label):
    dot = df[ulabel]*df['u_wind']+df[vlabel]*df['v_wind']
    mags = np.sqrt(df[ulabel]**2+df[vlabel]**2) * \
        np.sqrt(df['u_wind']**2+df['v_wind']**2)
    c = (dot/mags)
    df[angle_label] = np.arccos(c)
    df[angle_label] = df[angle_label]/np.pi*180
    df['neg_function'] = df[ulabel] * \
        df['v_wind'] - df[vlabel]*df['u_wind']
    neg_function=df['neg_function'].values
    neg_function[neg_function< 0]=-1
    neg_function[neg_function > 0]=1
     
    
    #ds['neg_function']=np.positive( ds['neg_function'])
    df[angle_label]=neg_function*df[angle_label]
    return df

       
def preprocess(df):
    df=df[df.u_wind>-1000]
    udiff=df.u_wind-df.u
    vdiff=df.v_wind-df.v
    df['u_error']=-udiff
    df['v_error']=-vdiff
    df['speed']=np.sqrt(df.u**2+df.v**2)
    df['speed_era5']=np.sqrt(df.u_era5**2+df.v_era5**2)
    df['speed_wind']=np.sqrt(df.u_wind**2+df.v_wind**2)
    df['speed_error']=df.speed-df.speed_wind
    df['speed_error_era5']=df.speed_era5-df.speed_wind
    df['u_error_era5']=df.u_wind-df.u_era5
    df['v_error_era5']=df.v_wind-df.v_era5

    df['error_mag']=np.sqrt((df.u-df.u_era5)**2+(df.v-df.v_era5)**2)
    df['error_square']=udiff**2+vdiff**2
    df['error_square_era5']=(df.u_wind-df.u_era5)**2+(df.v_wind-df.v_era5)**2
    df=angle(df, 'u','v','signed_angle')
    df=angle(df, 'u_era5','v_era5','signed_angle_era5')
    df['angle']=abs(df['signed_angle'])
    df['angle_era5']=abs(df['signed_angle_era5'])
   # df=df[df.speed>3]


    return df


def main(param):
    four_panel_plot('test_rmsvd_angle', param)
    four_panel_plot('bias', param, var1='speed_bias', var2='angle_bias', 
                     xlabel1='Bias [m/s]', 
                     xlabel2='Angle bias [deg]', xlim1=(-5,5), xlim2=(-20,20))
    four_panel_plot('component bias', param, var1='u_error', var2='v_error', 
                     xlabel1='u bias [m/s]', 
                     xlabel2='v bias', xlim1=(-5,5), xlim2=(-5,5))
    location_plot('location', param)
    
    #df_jan=pd.read_pickle('../data/processed/dataframes/january_winds_rs_model'+c.TAG+'.pkl')
    #df_jan=pd.read_pickle('../data/processed/dataframes/july_winds_rs_model'+c.TAG+'.pkl')

    #df_july=pd.read_pickle('../data/processed/dataframes/july_winds_rs_model'+c.TAG+'.pkl')
    #df_jan=pd.read_pickle('../data/processed/dataframes/january_winds_rs_model.pkl')
    #df_july=pd.read_pickle('../data/processed/dataframes/july_winds_rs_model.pkl')
   
    #df_jan=preprocess(df_jan)
    #df_jan=df_jan.drop_duplicates()
    #df_july=preprocess(df_july)
    #df_july=df_july.drop_duplicates()
    
    #multiple_pressure_map(df_jan,df_july,  'rmsvd')
    #multiple_pressure_map('angle_'+c.TAG, 'angle','angle [deg]',param)
    #multiple_pressure_map('rmsvd_'+c.TAG, 'rmsvd','RMSVD [m/s]',param)

    
    
    
    


if __name__=='__main__':
    param=parameters()
    param.set_plev_coarse(5)
    main(param)