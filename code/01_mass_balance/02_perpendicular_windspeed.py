import pandas as pd
import numpy as np
import geopandas as gpd 
from shapely.geometry import Point, LineString, Polygon
from shapely import wkt
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import os

"""
Compute the windspeed perpendicular to the orientation of the walls
"""

WD = {      'N': 0,
            'NE': 45,
            'E': 90,
            'SE': 135,
            'S': 180,
            'SW': 225,
            'W':  270,
            'NW': 315,
}

directory = "data/mass_balance"
suffix = "_01.csv"

for filename in os.listdir(directory):
    if filename.endswith(suffix):
        fileloc = f"{directory}/{filename}"
        file_string = filename.replace(suffix, "") # remove suffix from the string
        print(file_string)

        # open dataframe
        dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
        df = pd.read_csv(fileloc, index_col='datetime', dtype=dtype_options)
        df.index = pd.to_datetime(df.index, format='ISO8601')

        for index, row in df.iterrows():
            if (row["BOX.wall"]) and (row["PROC.EC.WindDirection"]) and (row["PROC.WEATHER.windSpeedCorrOrigin"]) \
                and not (np.isnan(row["PROC.WEATHER.windSpeedCorrOrigin"])):
                
                wind_direction = row["PROC.EC.WindDirection"]
                wind_speed = row["PROC.WEATHER.windSpeedCorrOrigin"]
                wall = row["BOX.wall"]
                # print(f"wind_direction = {wind_direction} and wind_speed = {wind_speed} and wall = {wall}")

                wind_direction_radians = np.radians((wind_direction+180) % 360)
                wind_speed_0 = wind_speed * np.cos(wind_direction_radians)
                wind_speed_90 = wind_speed * np.sin(wind_direction_radians)
                # print(f"wind speed 0 = {wind_speed_0} and wind speed 90 = {wind_speed_90}")

                theta = np.radians(45)
                wind_speed_45 = np.cos(theta) * wind_speed_0 + np.sin(theta) * wind_speed_90
                wind_speed_135 = - np.sin(theta) * wind_speed_0 + np.cos(theta) * wind_speed_90
                # print(f"wind speed 45 = {wind_speed_45} and wind speed 135 = {wind_speed_135}")

                perp_wind_speed = None
                if wall == 'NE_wall':
                    perp_wind_speed = wind_speed_45
                elif wall == 'SW_wall':
                    perp_wind_speed = - wind_speed_45
                elif wall == 'SE_wall':
                    perp_wind_speed = wind_speed_135
                elif wall == 'NW_wall':
                    perp_wind_speed = - wind_speed_135

                if perp_wind_speed:
                    df.loc[index, 'BOX.perpWind'] = perp_wind_speed

        df['geometry'] = df['geometry'].apply(wkt.loads) 
        gdf = gpd.GeoDataFrame(df)

        df.to_csv(f"data/mass_balance/{file_string}_02.csv", index=True, decimal='.', sep=',')
