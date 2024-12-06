import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import os

"""
Add diffusivity and wind speed based on eddy-covariance data
Use MOST parameterizations found in 
"An Introduction to Boundary Layer Meteorology", Stull, 1989. 
"""

# open dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df = pd.read_csv("data/10_df.csv", index_col='datetime', dtype=dtype_options)
df.index = pd.to_datetime(df.index, format='ISO8601')

d = 0.1  # displacement height [m]
kappa = 0.41  # von karmann constant [-]
z0 = 0.05  # roughness length [m]

### Diffusivity and windspeed at the drone location ###

# Check L < 0, except for 29/02/2024 15:30 - 16:00, but we did not do any measurements at that time
# print(df.index[df['EDDYPRO.L']>0].tolist())

# add businger-dyer relationship for dimensionless wind shear in the surface layer at measurement location, p. 383 (Stull).
df.loc[df['EDDYPRO.L'] < 0, 'PROC.EDDYPRO.PHI_H'] = 0.74 * (1 - (9 * (df['PROC.h_drone_base']-d) / df['EDDYPRO.L'])) ** (-1/2)
# add eddy viscosity for diabatic surface layer at measurement location, p. 209 (Stull)
df['PROC.EDDYPRO.K'] = kappa * (df['PROC.h_drone_base']-d) * df['EDDYPRO.u*'] / df['PROC.EDDYPRO.PHI_H']

# add wind speed parameterization at measurement location, p. 385 (Stull).
df.loc[df['EDDYPRO.L'] < 0, 'PROC.EDDYPRO.PHI_M'] = (1 - (15 * (df['PROC.h_drone_base']-d) / df['EDDYPRO.L'])) ** (-1/4)
df['PROC.EDDYPRO.PHI_M2'] = -2 * np.log( 1/2 * (1 +  1 /  df['PROC.EDDYPRO.PHI_M'] ) ) \
          - np.log( 1/2 * (1 + (1 / (df['PROC.EDDYPRO.PHI_M']**2) )) ) \
        + 2 * np.arctan( 1 / df['PROC.EDDYPRO.PHI_M'] ) - (np.pi/2)
df['PROC.EDDYPRO.U'] = df['EDDYPRO.u*'] / kappa * ( np.log( (df['PROC.h_drone_base'] - d) / z0 ) + df['PROC.EDDYPRO.PHI_M2'] )

### Windspeed at the height of the EC tower ###

h_EC = 5  # meter
df['PROC.EDDYPRO.U_EC'] = df['EDDYPRO.u*'] / kappa * ( np.log( (h_EC - d) / z0 ) + df['PROC.EDDYPRO.PHI_M2'] )

### Make plot for supplement ###

half_hourly_mean = df['PROC.EC.WindSpeed'].resample('30T').mean() # resample to get mean wind speed per half hour
df['mean_wind_speed_half_hour'] = df.index.to_series().apply(lambda x: half_hourly_mean.loc[x.floor('30T')])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
hb = ax.hexbin(df['PROC.EDDYPRO.U_EC'], df['mean_wind_speed_half_hour'], gridsize=20, extent=(0,10,0,10), bins='log', cmap='Reds')
cb = fig.colorbar(hb, ax=ax, label='Counts') 
ax.plot(np.arange(15), np.arange(15), c='k', linestyle='--')
ax.set_aspect('equal')
ax.set_ylabel('Mean wind speed from EC [m$\,$s$^{-1}$]')
ax.set_xlabel('Mean wind speed from MOST [m$\,$s$^{-1}$]')
ax.set_ylim([0,10])
ax.set_xlim([0,10])
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
plt.savefig(f"windspeed_MOST_supplement_article.png", dpi=300)
plt.close()

# save to df
df.to_csv("data/11_df.csv", index=True, decimal='.', sep=',')