import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta, timezone
from scipy.stats import circstd, circmean

"""
Compute the mean wind speed and mean diffusivity of the plume for each drone flight
based on a plume height of h_max meters. 
"""

# open dataframe
directory = "data/bayesian_inference"
suffix = "_02.csv"

# we use a fixed vertical plume extent of h_max [m] for all drone flights
h_max = 8  # meter

data = []
indices = []

for filename in os.listdir(directory):
    if filename.endswith(suffix):
        fileloc = f"{directory}/{filename}"
        file_string = filename.replace(suffix, "") # remove suffix from the string

        dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
        df = pd.read_csv(fileloc, index_col='datetime', dtype=dtype_options)
        df.index = pd.to_datetime(df.index, format='ISO8601')

        kappa = 0.41  # von karmann constant
        z0 = 0.05  # estimated roughness length [m]
        d = 0.1  # estimated displacement height [m]

        # check that type of flight is octagon or plough
        if (df['NOTE.typeOfFlight'][0] == 'octagon') or (df['NOTE.typeOfFlight'][0] == 'plough'):
            print(f'found octagon or plough flight: {file_string}')

            L_mean = df['EDDYPRO.L'].mean()
            u_star_mean = df['EDDYPRO.u*'].mean()
            #print(f"L_mean = {L_mean}")
            #print(f"u_star_mean = {u_star_mean}")

            # diffusivity parameterization from EC
            z = np.linspace(0.101, h_max, 100)
            K_profile = kappa * (z-d) * df['EDDYPRO.u*'].mean() / df['PROC.EDDYPRO.PHI_H'].mean()  
            D_obs = np.mean(K_profile)
            #print(f"D_obs = {D_obs}")

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
            ax.hexbin(df['PROC.EDDYPRO.K'], df['PROC.h_drone_base'], extent=(0,20,0,20))
            ax.plot(K_profile, z, c='k')
            ax.set_ylabel('Height above ground [m]')
            ax.set_xlabel('Diffusivity (parameterization) [m2/s]')
            ax.set_ylim([-1,12])
            ax.set_xlim([0,20])
            plt.tight_layout()
            plt.savefig(f"data/bayesian_inference/03_Figures/{file_string}_D.png")
            plt.close()

            # wind speed parameterization from EC
            U_profile = df['EDDYPRO.u*'].mean() / kappa * ( np.log( (z-d) / z0 ) + df['PROC.EDDYPRO.PHI_M2'].mean() )
            U_obs = np.mean(U_profile)
            #print(f"U_obs = {U_obs}")

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
            ax.hexbin(df['PROC.EDDYPRO.U'], df['PROC.h_drone_base'], extent=(0,20,0,20))
            ax.plot(U_profile, z, c='k')
            ax.set_ylabel('Height above ground [m]')
            ax.set_xlabel('Wind speed (parameterization) [m/s]')
            ax.set_ylim([-1,12])
            ax.set_xlim([0,20])
            plt.tight_layout()
            plt.savefig(f"data/bayesian_inference/03_Figures/{file_string}_V_EC.png")
            plt.close()

            # drone wind speed
            h = range(h_max)
            V_profile = [0] + [df[(df['PROC.h_drone_base'] > i) & (df['PROC.h_drone_base'] < i+1)]['PROC.WEATHER.windSpeedCorrOrigin'].mean() for i in h]
            V_profile_h = [0] + [i+0.5 for i in h]
            V_obs = np.nanmean(V_profile)
            #print(f"V_obs = {V_obs}")

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
            ax.hexbin(df['PROC.WEATHER.windSpeedCorrOrigin'], df['PROC.h_drone_base'], extent=(0,20,0,20))
            ax.plot(V_profile, V_profile_h, c='k')
            ax.scatter(V_profile, V_profile_h, c='k')
            ax.set_ylabel('Height above ground [m]')
            ax.set_xlabel('Wind speed (linear correction through origin) [m/s]')
            ax.set_ylim([-1,12])
            ax.set_xlim([0,20])
            plt.tight_layout()
            plt.savefig(f"data/bayesian_inference/03_Figures/{file_string}_V_drone.png")
            plt.close()

            # wind direction from EC
            WD_rad = np.deg2rad(df['PROC.EC.WindDirection'])
            sin_sum = np.sum(np.sin(WD_rad))
            cos_sum = np.sum(np.cos(WD_rad))
            mean_theta = np.arctan2(sin_sum, cos_sum)
            mean_direction = np.rad2deg(mean_theta)
            WD_obs = mean_direction % 360
            #print(f"WD_obs from EC = {WD_obs}")

            # wind direction from drone
            wind_dic = {'N': 0,
                        'NE': 45,
                        'E': 90,
                        'SE': 135,
                        'S': 180,
                        'SW': 225,
                        'W': 270,
                        'NW': 315}
            df['WEATHER.windDirection [deg]'] = df['WEATHER.windDirection'].map(wind_dic)
            WD_rad = np.deg2rad(df['WEATHER.windDirection [deg]'])
            sin_sum = np.sum(np.sin(WD_rad))
            cos_sum = np.sum(np.cos(WD_rad))
            mean_theta = np.arctan2(sin_sum, cos_sum)
            mean_direction = np.rad2deg(mean_theta)
            WD_obs_drone = mean_direction % 360   # np.rad2deg(circmean(WD_rad)) works as well
            #print(f"WD_obs from drone = {WD_obs_drone}")
            
            # write to file
            data.append({'U_obs': U_obs, 'D_obs': D_obs, 'WD_obs': WD_obs, 'U_drone_obs': V_obs, 'WD_drone_obs': WD_obs_drone})
            indices.append(file_string)

df = pd.DataFrame(data)
df.index = indices

df.to_csv(f"{directory}/03_mean_obs.csv", index=True, decimal='.', sep=',')