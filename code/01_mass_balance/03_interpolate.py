import pandas as pd
import numpy as np
import geopandas as gpd 
from shapely.geometry import Point, LineString, Polygon
from shapely import wkt
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import os
from scipy.interpolate import LinearNDInterpolator, griddata, RectBivariateSpline, NearestNDInterpolator

"""
Interpolate data, compute fluxes and emission rate
"""

fileloc = "data/mass_balance/gdf_walls.csv"
df_walls = pd.read_csv(fileloc, index_col=0)
df_walls['geometry'] = df_walls['geometry'].apply(wkt.loads)
gdf_walls = gpd.GeoDataFrame(df_walls, geometry='geometry', crs=32737)
# print(gdf_walls)

df_results = pd.DataFrame(columns=['BOX.total', 'BOX.per_animal'])

def do_interpolation(df, wall, parameter):
    """ linear interpolation between observations to create fine grid """

    # select wall
    df_wall = df[df['BOX.wall'] == wall]

    # keep the rows that have observations
    df_wall = df_wall.dropna(subset=[parameter])
    df_wall = df_wall.dropna(subset=['BOX.distance_from_corner'])
    df_wall = df_wall.dropna(subset=['PROC.h_drone_base'])

    x = df_wall['BOX.distance_from_corner'].values
    y = df_wall['PROC.h_drone_base'].values
    z = df_wall[parameter].values

    length_wall = gdf_walls[gdf_walls['name'] == wall]['length_meters'].values[0]  # approx 27m

    # add zero wind speeds at ground surface.
    if parameter == 'BOX.perpWind':
        n_wall = int(np.round(length_wall,0))
        x = np.hstack((x, np.linspace(start=0.1, stop=length_wall-0.1, num=n_wall, endpoint=True)))
        y = np.hstack((y, np.zeros(n_wall)))
        z = np.hstack((z, np.zeros(n_wall)))

    X_1D = np.linspace(start=0.1, stop=length_wall-0.1, num=27*5, endpoint=True)
    Y_1D = np.linspace(0.1, 9.9, 50)
    X,Y = np.meshgrid(X_1D, Y_1D)
    interp = LinearNDInterpolator(points=list(zip(x, y)), values=z)
    Z = interp(X, Y)

    return X, Y, Z, x, y, z

def do_nearest_interpolation(X, Y, Z, x, y, z):
    """ nearest value interpolator for empty cells """

    nearest_interp = NearestNDInterpolator(list(zip(x, y)), z)
    nearest_Z = nearest_interp(X, Y)
    # Replace NaN values in Z with corresponding values from nearest_Z
    Z[np.isnan(Z)] = nearest_Z[np.isnan(Z)]
    return Z

def do_mean(wall, X, Y, Z):
    """ compute mean of fine grid values to be used on a coarser 1mx1m grid """

    length_wall = gdf_walls[gdf_walls['name'] == wall]['length_meters'].values[0]

    x_edges = np.linspace(start=0, stop=length_wall, num=int(np.round(length_wall+1,0)), endpoint=True)
    y_edges = np.linspace(start=0, stop=10, num=11, endpoint=True)
    # print(f"x_edges = \n{x_edges}")

    x_indices = np.digitize(X, x_edges) - 1
    y_indices = np.digitize(Y, y_edges) - 1
    # print(f"x_indices = \n{x_indices}")

    z_new = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan)
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            mask = (x_indices == i) & (y_indices == j)
            if np.any(mask):
                z_new[j, i] = np.mean(Z[mask])

    x_mid = (x_edges[:-1] + x_edges[1:])/2
    y_mid = (y_edges[:-1] + y_edges[1:])/2
    x_new, y_new = np.meshgrid(x_mid, y_mid)
    # print(f"x_mid = {x_mid}")
    # print(f"z_new = {z_new}")

    return x_new, y_new, z_new

directory = "data/mass_balance"
suffix = "_02.csv"

count = 0
for filename in os.listdir(directory):
    if filename.endswith(suffix):
        count += 1

        fileloc = f"{directory}/{filename}"
        file_string = filename.replace(suffix, "") # remove suffix from the string
        print(file_string)

        # open dataframe
        dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
        df = pd.read_csv(fileloc, index_col='datetime', dtype=dtype_options)
        df.index = pd.to_datetime(df.index, format='ISO8601')

        # interpolate methane concentraton observations
        X_NE, Y_NE, Z_NE, x_NE, y_NE, z_NE = do_interpolation(df, wall='NE_wall', parameter='AERIS.CH4 [ppm]')
        X_SE, Y_SE, Z_SE, x_SE, y_SE, z_SE = do_interpolation(df, wall='SE_wall', parameter='AERIS.CH4 [ppm]')
        X_SW, Y_SW, Z_SW, x_SW, y_SW, z_SW = do_interpolation(df, wall='SW_wall', parameter='AERIS.CH4 [ppm]')
        X_NW, Y_NW, Z_NW, x_NW, y_NW, z_NW = do_interpolation(df, wall='NW_wall', parameter='AERIS.CH4 [ppm]')
        concatenated_Y = np.concatenate((Y_NE, Y_SE, Y_SW, Y_NW), axis=1)
        concatenated_Z = np.concatenate((Z_NE, Z_SE, Z_SW, Z_NW), axis=1)

        start_SE = X_NE[-1,-1] + 0.1
        start_SW = start_SE + X_SW[-1,-1] + 0.1
        start_NW = start_SW + X_NW[-1,-1] + 0.1
        concatenated_z_methane = np.concatenate((z_NE, z_SE, z_SW, z_NW))
        concatenated_y_methane = np.concatenate((y_NE, y_SE, y_SW, y_NW))
        concatenated_x_methane = np.concatenate((x_NE, start_SE+x_SE, start_SW+x_SW, start_NW+x_NW))

        # try computing mean per coarse grid cell
        X_NE, Y_NE, Z_NE = do_mean(wall='NE_wall', X=X_NE, Y=Y_NE, Z=Z_NE)
        X_SE, Y_SE, Z_SE = do_mean(wall='SE_wall', X=X_SE, Y=Y_SE, Z=Z_SE)
        X_SW, Y_SW, Z_SW = do_mean(wall='SW_wall', X=X_SW, Y=Y_SW, Z=Z_SW)
        X_NW, Y_NW, Z_NW = do_mean(wall='NW_wall', X=X_NW, Y=Y_NW, Z=Z_NW)

        # fill in empty grid cells with nearest neighbour observation
        Z_NE = do_nearest_interpolation(X_NE, Y_NE, Z_NE, x_NE, y_NE, z_NE)
        Z_SE = do_nearest_interpolation(X_SE, Y_SE, Z_SE, x_SE, y_SE, z_SE)
        Z_SW = do_nearest_interpolation(X_SW, Y_SW, Z_SW, x_SW, y_SW, z_SW)
        Z_NW = do_nearest_interpolation(X_NW, Y_NW, Z_NW, x_NW, y_NW, z_NW)

        concatenated_X = np.concatenate((X_NE, start_SE+X_SE, start_SW+X_SW, start_NW+X_NW), axis=1)
        concatenated_Y = np.concatenate((Y_NE, Y_SE, Y_SW, Y_NW), axis=1)
        concatenated_Z = np.concatenate((Z_NE, Z_SE, Z_SW, Z_NW), axis=1)

        # interpolate perpendicular wind speed
        X_NE, Y_NE, Z_NE, x_NE, y_NE, z_NE = do_interpolation(df, wall='NE_wall', parameter='BOX.perpWind')
        X_SE, Y_SE, Z_SE, x_SE, y_SE, z_SE = do_interpolation(df, wall='SE_wall', parameter='BOX.perpWind')
        X_SW, Y_SW, Z_SW, x_SW, y_SW, z_SW = do_interpolation(df, wall='SW_wall', parameter='BOX.perpWind')
        X_NW, Y_NW, Z_NW, x_NW, y_NW, z_NW = do_interpolation(df, wall='NW_wall', parameter='BOX.perpWind')
        concatenated_Z_wind = np.concatenate((Z_NE, Z_SE, Z_SW, Z_NW), axis=1)
        concatenated_y_wind = np.concatenate((y_NE, y_SE, y_SW, y_NW))
        concatenated_x_wind = np.concatenate((x_NE, start_SE+x_SE, start_SW+x_SW, start_NW+x_NW))
        
        # try computing mean per coarse grid cell
        X_NE, Y_NE, Z_NE = do_mean(wall='NE_wall', X=X_NE, Y=Y_NE, Z=Z_NE)
        X_SE, Y_SE, Z_SE = do_mean(wall='SE_wall', X=X_SE, Y=Y_SE, Z=Z_SE)
        X_SW, Y_SW, Z_SW = do_mean(wall='SW_wall', X=X_SW, Y=Y_SW, Z=Z_SW)
        X_NW, Y_NW, Z_NW = do_mean(wall='NW_wall', X=X_NW, Y=Y_NW, Z=Z_NW)
        
        # fill in empty grid cells with nearest neighbour observation
        Z_NE = do_nearest_interpolation(X_NE, Y_NE, Z_NE, x_NE, y_NE, z_NE)
        Z_SE = do_nearest_interpolation(X_SE, Y_SE, Z_SE, x_SE, y_SE, z_SE)
        Z_SW = do_nearest_interpolation(X_SW, Y_SW, Z_SW, x_SW, y_SW, z_SW)
        Z_NW = do_nearest_interpolation(X_NW, Y_NW, Z_NW, x_NW, y_NW, z_NW)
        concatenated_Z_wind = np.concatenate((Z_NE, Z_SE, Z_SW, Z_NW), axis=1)

        # estimate emission rate per animal, assuming each cell is 1mx1m.
        # convert ppm to mg/m3
        temperature = np.mean(df['EC.T [deg C]']) + 273.15  # Kelvin
        pressure = np.mean(df['EC.P [kPa]']) * 1_000  # Pa
        R = 8.314  # (m3*Pa)/(K*mol)
        molar_mass = 16.04  # g/mol
        molar_volume = R * temperature / pressure  # m3/mol
        conversion_factor = molar_volume / molar_mass  # m3/g
        conc = concatenated_Z / conversion_factor / 1e3  # mg/m3

        emission_rate = np.nansum(conc*concatenated_Z_wind)  # mg/m3 * m/s * m2 = mg/s
        amount_of_animals = df['NOTE.amount_int'].values[0]
        emission_rate_per_animal = np.round(emission_rate / amount_of_animals,2)

        # add result to dataframe
        df_results.loc[file_string] = [emission_rate, emission_rate_per_animal]  # mg/s

        # make figure
        fig, ax = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(26, 8), )

        p0 = ax[0].pcolormesh(concatenated_X, concatenated_Y, concatenated_Z, shading='auto', cmap='Reds')
        cbar = fig.colorbar(p0, ax=ax[0], fraction=0.02, pad=0.02)
        cbar.set_label(label='CH$_4$\n[ppm]', size=24)
        cbar.ax.tick_params(labelsize=24)
        ax[0].scatter(concatenated_x_methane, concatenated_y_methane, c='k', s=2, label="input point")
        ax[0].set_aspect('equal')
        ax[0].set_ylim([0,10])
        ax[0].set_xticks([])
        ax[0].set_yticks([0, 5, 10], ['0', '5', '10'], fontsize=24)

        vmax = np.max(np.abs(concatenated_Z_wind))
        vmin = -vmax
        p1 = ax[1].pcolormesh(concatenated_X, concatenated_Y, concatenated_Z_wind, shading='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(p1, ax=ax[1], fraction=0.02, pad=0.02)
        cbar.set_label(label='Wind$_{\perp}$\n[m$\,$s$^{-1}$]', size=24)
        cbar.ax.tick_params(labelsize=24)
        ax[1].scatter(concatenated_x_wind, concatenated_y_wind, c='k', s=2, label="input point")
        ax[1].set_aspect('equal')
        ax[1].set_ylim([0,10])
        ax[1].set_xticks([])
        ax[1].set_yticks([0, 5, 10], ['0', '5', '10'], fontsize=24)
   
        vmax = np.max(conc*concatenated_Z_wind*3600/1000)  # from mg/s to g/h 
        vmin = -vmax
        p2 = ax[2].pcolormesh(concatenated_X, concatenated_Y, conc*concatenated_Z_wind*3600/1000, shading='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(p2, ax=ax[2], fraction=0.02, pad=0.02)
        cbar.set_label(label='Flux\n[g$\,$h$^{-1}\,$m$^{-2}$]', size=24)
        cbar.ax.tick_params(labelsize=24)
        ax[2].set_aspect('equal')
        ax[2].set_ylim([0,10])
        ax[2].set_yticks([0, 5, 10], ['0', '5', '10'], fontsize=24)
        ax[2].set_xticks([start_SE/2, (start_SW+start_SE)/2, (start_SW+start_NW)/2, (start_NW+concatenated_X[-1,-1]+0.5)/2], ['NE', 'SE', 'SW', 'NW'], fontsize=28)

        fig.supylabel('Distance from ground [m]', fontsize=24, x=0.009)
        fig.align_labels()
        fig.tight_layout()
        plt.savefig(f"{directory}/03_Figures/{file_string}.png", bbox_inches='tight', dpi=300) 
        plt.close()

df_results.to_csv("data/mass_balance/mass_balance_results.csv", index=True, decimal='.', sep=',')