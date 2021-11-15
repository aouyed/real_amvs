
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np




def scatter_plot_cartopy(title, x, y):
    fig=plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

    ax.coastlines()
    ax.scatter(x,y,s=20)
    fig.tight_layout()
    plt.savefig('../data/processed/plots/'+title+'.png',
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()  
    
def preprocess(df):
    df=df[df.u_wind>-1000]
    udiff=df.u_wind-df.u
    vdiff=df.v_wind-df.v
    df['error_mag']=np.sqrt((df.u-df.u_era5)**2+(df.v-df.v_era5)**2)
    df['error_square']=udiff**2+vdiff**2
    df['error_square_era5']=(df.u_wind-df.u_era5)**2+(df.v_wind-df.v_era5)**2
    return df    


def main():

    df1=pd.read_pickle('../data/processed/dataframes/january_winds_rs_model.pkl')
    df2=pd.read_pickle('../data/processed/dataframes/july_winds_rs_model.pkl')
    df1.reset_index(drop=True)
    df2.reset_index(drop=True)
    df=df1.append(df2).reset_index(drop=True)
    df=preprocess(df)
    df=df.loc[df.error_mag<10]
    df=df[['lat_rs','lon_rs','stationid']].drop_duplicates(ignore_index=True)
    print(df.shape)

    scatter_plot_cartopy('rs_coords',df['lon_rs'],df['lat_rs'])

if __name__ == '__main__':
    main()
