import pandas as pd
import numpy as np
import geopandas as gpd 
from shapely.geometry import Point, LineString, Polygon
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import os

"""
Identify flight legs of the octagon flights. 
This data is not needed for Bayesian inference, only plotting.
"""

directory = "data/bayesian_inference"
suffix = "_00.csv"

for filename in os.listdir(directory):
    if filename.endswith(suffix):
        fileloc = f"{directory}/{filename}"
        file_string = filename.replace(suffix, "") # remove suffix from the string

        # open dataframe
        dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
        df = pd.read_csv(fileloc, index_col='datetime', dtype=dtype_options)
        df.index = pd.to_datetime(df.index, format='ISO8601')

        # check that type of flight is octagon
        if df['NOTE.typeOfFlight'][0] == 'octagon':
            print(f'found octagon flight: {file_string}')

            df['PROC.OSD.rollingMeanHeight'] = df['OSD.height [m]'].rolling(window=10, center=True).mean()

            df['PROC.leg'] = np.nan
            df.loc[df['PROC.OSD.rollingMeanHeight'] > 9, 'PROC.leg'] = 33  # outer leg, top layer
            df.loc[(df['PROC.OSD.rollingMeanHeight'] > 6.4) & (df['PROC.OSD.rollingMeanHeight'] < 6.6), 'PROC.leg'] = 23  # outer leg, mid layer
            df.loc[(df['PROC.OSD.rollingMeanHeight'] > 3.89) & (df['PROC.OSD.rollingMeanHeight'] < 4.1), 'PROC.leg'] = 13
            df.loc[(df['PROC.OSD.rollingMeanHeight'] > 7.9) & (df['PROC.OSD.rollingMeanHeight'] < 8.2), 'PROC.leg'] = 32
            df.loc[(df['PROC.OSD.rollingMeanHeight'] > 5.4) & (df['PROC.OSD.rollingMeanHeight'] < 5.6), 'PROC.leg'] = 22
            df.loc[(df['PROC.OSD.rollingMeanHeight'] > 3.7) & (df['PROC.OSD.rollingMeanHeight'] < 3.88), 'PROC.leg'] = 12  # mid leg, bottom layer
            df.loc[(df['PROC.OSD.rollingMeanHeight'] > 5.9) & (df['PROC.OSD.rollingMeanHeight'] < 6.1), 'PROC.leg'] = 31
            df.loc[(df['PROC.OSD.rollingMeanHeight'] > 4.4) & (df['PROC.OSD.rollingMeanHeight'] < 4.6), 'PROC.leg'] = 21
            df.loc[(df['PROC.OSD.rollingMeanHeight'] > 3.35) & (df['PROC.OSD.rollingMeanHeight'] < 3.65), 'PROC.leg'] = 11  # inner leg, bottom layer

            df['PROC.leg.raw'] = df['PROC.leg'].copy()
            df['PROC.leg'] = df['PROC.leg'].rolling(window=2, center=True).mean()  # don't pick window=5 or window=3
            allowed_values = [11, 12, 13, 21, 22, 23, 31, 32, 33]
            df.loc[~df['PROC.leg'].isin(allowed_values), 'PROC.leg'] = np.nan

            df.drop(columns=['PROC.OSD.rollingMeanHeight'], inplace=True)

            # plot wind data with horizontal wind speed from the eddy covariance tower

            df.dropna(subset=['AERIS.CH4 [ppm]', 'PROC.x', 'PROC.y', 'PROC.h_drone_base'], inplace=True)

            vmin = 0
            vmax = 10
            s = 3
            cmap = 'Reds'

            norm = plt.Normalize(vmin=0, vmax=10)
            cmap = plt.cm.cool

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,9))

            df1 = df[df['PROC.leg'] < 15]
            ax[0].scatter(df1['PROC.x'].values, df1['PROC.y'].values, c=df1['AERIS.CH4 [ppm]'], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
            im1= ax[0].quiver(   df1['PROC.x'].values, 
                            df1['PROC.y'].values,
                            np.sin(np.deg2rad(df1['PROC.EC.WindDirection'].values+180)),
                            np.cos(np.deg2rad(df1['PROC.EC.WindDirection'].values+180)),
                            df1['AERIS.CH4 [ppm]'].values,
                            angles='xy',
                            cmap='Reds',
                            norm=norm,
                            scale=50/df1['PROC.EC.HorWindSpeed'].values,
                        )

            df2 = df[(df['PROC.leg'] > 20) & (df['PROC.leg'] < 25)]
            ax[1].scatter(df2['PROC.x'].values, df2['PROC.y'].values, c=df2['AERIS.CH4 [ppm]'], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
            ax[1].quiver(   df2['PROC.x'].values, 
                            df2['PROC.y'].values,
                            np.sin(np.deg2rad(df2['PROC.EC.WindDirection'].values+180)),
                            np.cos(np.deg2rad(df2['PROC.EC.WindDirection'].values+180)),
                            df2['AERIS.CH4 [ppm]'],
                            angles='xy',
                            cmap='Reds',
                            norm=norm,
                            scale=50/df2['PROC.EC.HorWindSpeed'].values,
                        )

            df3 = df[(df['PROC.leg'] > 30) & (df['PROC.leg'] < 35)]
            ax[2].scatter(df3['PROC.x'].values, df3['PROC.y'].values, c=df3['AERIS.CH4 [ppm]'], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
            ax[2].quiver(   df3['PROC.x'].values, 
                            df3['PROC.y'].values,
                            np.sin(np.deg2rad(df3['PROC.EC.WindDirection'].values+180)),
                            np.cos(np.deg2rad(df3['PROC.EC.WindDirection'].values+180)),
                            df3['AERIS.CH4 [ppm]'],
                            angles='xy',
                            cmap='Reds',
                            norm=norm,
                            scale=50/df3['PROC.EC.HorWindSpeed'].values,
                        )

            mean_winddirection = np.nanmean(df['PROC.EC.WindDirection'].values)
            mean_windspeed = np.nanmean(df['PROC.EC.HorWindSpeed'].values)
            for j in range(3):
                ax[j].set_xlim([-48,28])
                ax[j].set_ylim([-48,28])
                ax[j].set_aspect('equal')
                ax[j].scatter(0,0, s=20, c='r')
                ax[j].quiver(0, 0, np.sin(np.deg2rad(mean_winddirection+180)), np.cos(np.deg2rad(mean_winddirection+180)),
                    angles='xy', scale=50/mean_windspeed)

            ax[0].set_title('bottom layer')
            ax[1].set_title('mid layer')
            ax[2].set_title('top layer')

            cbar = plt.colorbar(im1, ax=ax[2])
            plt.tight_layout()
            plt.savefig(f"{directory}/01_Figures/{file_string}_winddata_EC.png")
            plt.close()

            # Another figure with wind speed estimated by the drone, NOTE wind direction from EC tower

            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,9))

            df1 = df[df['PROC.leg'] < 15]
            ax[0].scatter(df1['PROC.x'].values, df1['PROC.y'].values, c=df1['AERIS.CH4 [ppm]'], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
            im1= ax[0].quiver(   df1['PROC.x'].values, 
                            df1['PROC.y'].values,
                            np.sin(np.deg2rad(df1['PROC.EC.WindDirection'].values+180)),
                            np.cos(np.deg2rad(df1['PROC.EC.WindDirection'].values+180)),
                            df1['AERIS.CH4 [ppm]'].values,
                            angles='xy',
                            cmap='Reds',
                            norm=norm,
                            scale=50/df1['PROC.WEATHER.windSpeedCorrOrigin'].values,
                        )

            df2 = df[(df['PROC.leg'] > 20) & (df['PROC.leg'] < 25)]
            ax[1].scatter(df2['PROC.x'].values, df2['PROC.y'].values, c=df2['AERIS.CH4 [ppm]'], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
            ax[1].quiver(   df2['PROC.x'].values, 
                            df2['PROC.y'].values,
                            np.sin(np.deg2rad(df2['PROC.EC.WindDirection'].values+180)),
                            np.cos(np.deg2rad(df2['PROC.EC.WindDirection'].values+180)),
                            df2['AERIS.CH4 [ppm]'],
                            angles='xy',
                            cmap='Reds',
                            norm=norm,
                            scale=50/df2['PROC.WEATHER.windSpeedCorrOrigin'].values,
                        )

            df3 = df[(df['PROC.leg'] > 30) & (df['PROC.leg'] < 35)]
            ax[2].scatter(df3['PROC.x'].values, df3['PROC.y'].values, c=df3['AERIS.CH4 [ppm]'], cmap=cmap, s=s, vmin=vmin, vmax=vmax)
            ax[2].quiver(   df3['PROC.x'].values, 
                            df3['PROC.y'].values,
                            np.sin(np.deg2rad(df3['PROC.EC.WindDirection'].values+180)),
                            np.cos(np.deg2rad(df3['PROC.EC.WindDirection'].values+180)),
                            df3['AERIS.CH4 [ppm]'],
                            angles='xy',
                            cmap='Reds',
                            norm=norm,
                            scale=50/df3['PROC.WEATHER.windSpeedCorrOrigin'].values,
                        )

            mean_winddirection = np.nanmean(df['PROC.EC.WindDirection'].values)
            mean_windspeed = np.nanmean(df['PROC.WEATHER.windSpeedCorrOrigin'].values)
            for j in range(3):
                ax[j].set_xlim([-48,28])
                ax[j].set_ylim([-48,28])
                ax[j].set_aspect('equal')
                ax[j].scatter(0,0, s=20, c='r')
                ax[j].quiver(0, 0, np.sin(np.deg2rad(mean_winddirection+180)), np.cos(np.deg2rad(mean_winddirection+180)),
                    angles='xy', scale=50/mean_windspeed)

            ax[0].set_title('bottom layer')
            ax[1].set_title('mid layer')
            ax[2].set_title('top layer')

            cbar = plt.colorbar(im1, ax=ax[2])
            plt.tight_layout()
            plt.savefig(f"{directory}/01_Figures/{file_string}_winddata_drone.png")
            plt.close()

        df.to_csv(f"{directory}/{file_string}_01.csv", index=True, decimal='.', sep=',')