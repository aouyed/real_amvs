
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import config as c
import cartopy.crs as ccrs



THRESHOLDS=[10]

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
       'u_error_era5':[], 'angle':[], 'angle_era5':[]}
    for plev in plevs:
        
        df_unit=df[df.plev==plev]
        rmsvd=np.sqrt(df_unit['error_square'].mean())
        u_error=df_unit['u_error'].mean()
        u_error_era5=df_unit['u_error_era5'].mean()

        v_error=df_unit['v_error'].mean()
        angle=df_unit['angle'].mean()
        angle_era5=df_unit['angle_era5'].mean()


        rmsvd_era5=np.sqrt(df_unit['error_square_era5'].mean())
        d['plev'].append(plev)
        d['rmsvd'].append(rmsvd)
        d['rmsvd_era5'].append(rmsvd_era5)
        d['u_error'].append(u_error)
        d['u_error_era5'].append(u_error_era5)

        d['v_error'].append(v_error)
        d['angle'].append(angle)
        d['angle_era5'].append(angle_era5)


        
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
        #sample_unit=sample_stat_calc(df_pressure,  df_pressure_era, thresh)
        #if sample_stats.empty:
         #   sample_stats=sample_unit
        #else:
         #   sample_stats=sample_stats.append(sample_unit)
    ax.plot(df_pressure_era['angle_era5'], df_pressure.plev, label='ERA 5')
    #ax.axvspan(5.93, 8.97, alpha=0.25, color='grey')    
    ax.legend(frameon=False, loc='upper left')
    ax.set_xlabel('angle [deg]')
    ax.set_ylabel('Pressure [hPa]')
    #ax.set_xlim(-10,5)
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



def location_loader():
    #df1=pd.read_pickle('../data/processed/dataframes/january_winds_rs_modelfull_nn_tlv1.pkl')
    df2=pd.read_pickle('../data/processed/dataframes/july_winds_rs_modelfull_nn_tlv1.pkl') 
    df1=df2
    df1.reset_index(drop=True)
    df2.reset_index(drop=True)
    df=df1.append(df2).reset_index(drop=True)
    df=preprocess(df)
    df=df.loc[df.error_mag<4]
    df=df[['lat_rs','lon_rs','stationid']].drop_duplicates(ignore_index=True)
    print(df.shape)
    return(df)

   
def multiple_pressure_map(df_jan, df_july, fname):
    fig=plt.figure()
    ax1= plt.subplot(2,2,1)
    ax2= plt.subplot(2,2,2)
    ax3=plt.subplot(2,1,2,projection=ccrs.PlateCarree())

    axlist = [ax1,ax2,ax3]
    axlist[0]=pressure_ax(axlist[0], df_jan, 'angle')
    axlist[1]=pressure_ax(axlist[1], df_july, 'angle')
    df=location_loader()
    axlist[2]=scatter_plot_cartopy(axlist[2],'rs_coords',df['lon_rs'],df['lat_rs'])

    axlist[0].text(4,275,'(a)')
    axlist[1].text(4.,275,'(b)')
    axlist[2].text(-170,-40,'(c)')


    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+fname +
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
    df['u_error_era5']=df.u-df.u_era5
    df['v_error_era5']=df.v-df.v_era5

    df['v_error']=-vdiff
    df['error_mag']=np.sqrt((df.u-df.u_era5)**2+(df.v-df.v_era5)**2)
    df['error_square']=udiff**2+vdiff**2
    df['error_square_era5']=(df.u_wind-df.u_era5)**2+(df.v_wind-df.v_era5)**2
    df=angle(df, 'u','v','angle')
    df=angle(df, 'u_era5','v_era5','angle_era5')

    df['angle']=abs(df['angle'])
    df['angle_era5']=abs(df['angle_era5'])

    #df=df[df.lat.between(-30,30)]
    return df



def main():
    #df_jan=pd.read_pickle('../data/processed/dataframes/january_winds_rs_modelfull_nn_tlv1.pkl')
    df_july=pd.read_pickle('../data/processed/dataframes/july_winds_rs_modelfull_nn_tlv1.pkl')
    df_jan=df_july 
    df_jan=preprocess(df_jan)
    df_jan=df_jan.drop_duplicates()
    df_july=preprocess(df_july)
    df_july=df_july.drop_duplicates()
    
    multiple_pressure_map(df_jan,df_july,  'angle_july_og')
    
    
    
    
    


if __name__=='__main__':
    main()