import pandas as pd
import numpy as np
import geopandas as gpd 
from shapely.geometry import Point, LineString, Polygon
from shapely import wkt
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import os

"""
Determine to which side of the box each of the observation points belongs, 
NE, SE, SW, NW, or top.
"""

# open gdf_boma
fileloc = "data/gdf_boma.csv"
df_boma = pd.read_csv(fileloc)
df_boma['geometry'] = df_boma['geometry'].apply(wkt.loads)
gdf_boma = gpd.GeoDataFrame(df_boma, geometry='geometry', crs=32737)
gdf_boma.set_index('name', inplace=True)
print(gdf_boma)

# make LinString for each wall and compute length of the wall
d = {   'name': ['NE_wall', 'SE_wall', 'SW_wall', 'NW_wall'],
        'geometry' : [  LineString([ gdf_boma.loc['N'].geometry, gdf_boma.loc['E'].geometry ]),
                        LineString([ gdf_boma.loc['E'].geometry, gdf_boma.loc['S'].geometry ]),
                        LineString([ gdf_boma.loc['S'].geometry, gdf_boma.loc['W'].geometry ]),
                        LineString([ gdf_boma.loc['W'].geometry, gdf_boma.loc['N'].geometry ])
        ] 
    }
gdf_walls = gpd.GeoDataFrame(d, geometry='geometry', crs=32737)
gdf_walls['length_meters'] = gdf_walls['geometry'].length

length_from_N = [0, 0, 0, 0, 0]
l = 0
for i, row in gdf_walls.iterrows():
    l += row['length_meters']
    length_from_N[i+1] = l
gdf_walls['length_from_N'] = length_from_N[:4]  # distance from 'starting point north'
print(gdf_walls)

# create polygon of the box
polygon = Polygon([ gdf_boma.loc['S'].geometry, 
                    gdf_boma.loc['E'].geometry, 
                    gdf_boma.loc['N'].geometry, 
                    gdf_boma.loc['W'].geometry ])

directory = "data/mass_balance"
suffix = "_00.csv"

for filename in os.listdir(directory):
    if filename.endswith(suffix):
        fileloc = f"{directory}/{filename}"
        file_string = filename.replace(suffix, "") # remove suffix from the string

        # open dataframe
        dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
        df = pd.read_csv(fileloc, index_col='datetime', dtype=dtype_options)
        df.index = pd.to_datetime(df.index, format='ISO8601')

        df['geometry'] = df['geometry'].apply(wkt.loads)  # from combine_dataframes step 08
        gdf = gpd.GeoDataFrame(df)
        # print(gdf['geometry'].head())

        for gdf_index, gdf_row in gdf.iterrows():

            point = gdf_row.geometry
            # print(point)

            if gdf_row['OSD.vpsHeight [m]'] > 2:
                min_distance = 0.7 # add points to wall that are within 70 cm of the wall
            else: # we should be a bit more tolerant for the bottom leg of the box, the manual flight.
                min_distance = 3.5 # add points to wall that are within 3.5 m of the wall
            closest_wall_index = None
            closest_wall_id = None
            closest_wall_geometry = None

            for index, row in gdf_walls.iterrows():
                distance = point.distance(row['geometry'])
                # print(distance)
                if distance < min_distance:
                    min_distance = distance
                    closest_wall_index = index
                    closest_wall_id = row["name"]
                    closest_wall_geometry = row.geometry

            # point was not close enough to any of the linestrings, if it is within the polygon and above > 9.5m, it is assigned 'top'.
            if (closest_wall_id == None) and (gdf_row['OSD.height [m]'] > 9.5) and point.within(polygon): 
                closest_wall_index = 4
                closest_wall_id = 'top'
                closest_wall_geometry = None

            gdf.loc[gdf_index, 'BOX.wall.index'] = closest_wall_index
            gdf.loc[gdf_index, 'BOX.wall'] = closest_wall_id
            gdf.loc[gdf_index, 'BOX.wall_geometry'] = closest_wall_geometry

            if closest_wall_geometry: # 'glue' the observations to a wall
                point_glued = closest_wall_geometry.interpolate( closest_wall_geometry.project( point ) )
                gdf.loc[gdf_index, 'BOX.closest_point_geometry'] = point_glued
                # e.g.  if NE_wall, distance from N
                #       if SE_wall, distance from E
                #       if SW_wall, distance from S
                #       if NW_wall, distance from W
                distance_from_corner = point_glued.distance(Point(closest_wall_geometry.coords[0]))  
                gdf.loc[gdf_index, 'BOX.distance_from_corner'] = distance_from_corner
                gdf.loc[gdf_index, 'BOX.distance_from_N'] = distance_from_corner + gdf_walls.loc[gdf_walls['name']==closest_wall_id,'length_from_N'].values[0]

        # plot how the points are assigned to the different walls
        fig, ax = plt.subplots()
        gdf.plot(ax=ax, c=gdf['BOX.wall.index'])
        gdf_walls.plot(ax=ax, color='black')
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(gdf['BOX.wall.index'])
        cbar = plt.colorbar(sm, ax=ax)
        plt.title(file_string)
        plt.savefig(f"{directory}/01_Figures/{file_string}.png")
        plt.close()

        # make the plot only for the first leg, manual flight
        fig, ax = plt.subplots()
        gdf_low = gdf[gdf['OSD.vpsHeight [m]'] < 2]
        gdf_low.plot(ax=ax, c=gdf_low['BOX.wall.index'])
        gdf_walls.plot(ax=ax, color='black')
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(gdf['BOX.wall.index'])
        cbar = plt.colorbar(sm, ax=ax)
        plt.title(file_string)
        plt.savefig(f"{directory}/01_Figures/{file_string}_low.png")
        plt.close()

        gdf.to_csv(f"data/mass_balance/{file_string}_01.csv", index=True, decimal='.', sep=',')

gdf_walls.to_csv("data/mass_balance/gdf_walls.csv", index=True, decimal='.', sep=',')