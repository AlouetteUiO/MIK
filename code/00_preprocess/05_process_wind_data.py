import os
import pandas as pd
from datetime import date, time, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
from datetime import datetime, timedelta, timezone

"""
Correct the windspeed data from the drone
"""

# open dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df = pd.read_csv("data/04_df.csv", index_col='datetime', dtype=dtype_options)
df.index = pd.to_datetime(df.index, format='ISO8601')

# select hover flights based on field notes (field when the drone was hovering next to the eddy-covariance tower)
df_hover = df[df['NOTE.typeOfFlight']=='hover']
print(f"total hover time = {len(df_hover)/60} min")

# check that there are no NaN values 
if not (df_hover['WEATHER.windSpeed [m/s]'].isnull().any()) and not (df_hover['PROC.EC.WindSpeed'].isnull().any()):

    # linear fit through origin
    def linear_through_origin(x, beta):
        return beta * x
    beta, _ = curve_fit(linear_through_origin, df_hover['WEATHER.windSpeed [m/s]'], df_hover['PROC.EC.WindSpeed'])
    print(f"slope (linear fit through origin) = {beta}")

    # drone windspeed correction
    df['PROC.WEATHER.windSpeedCorrOrigin'] = df['WEATHER.windSpeed [m/s]'] * beta

    # windspeed figure for supplement of article
    fig, ax = plt.subplots(figsize=(8,6))
    hb = ax.hexbin(df_hover['WEATHER.windSpeed [m/s]'], df_hover['PROC.EC.WindSpeed'], gridsize=20, cmap='Reds', extent=(0,14,0,14), bins='log')
    cb = fig.colorbar(hb, ax=ax, label='Counts')
    ax.set_aspect('equal')
    ax.set_xlabel('Wind speed from drone [m$\,$s$^{-1}$]')
    ax.set_ylabel('Wind speed from EC [m$\,$s$^{-1}$]')
    ax.set_xlim([0,14])
    ax.set_ylim([0,14])
    x = np.linspace(0, 15, 400)
    y_lin_ori = linear_through_origin(x, beta)
    ax.plot(x, y_lin_ori, label = r'Linear fit: y=1.52x', linestyle='-', color='black')
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.legend()
    plt.savefig("windspeed_supplement_article.png", dpi=300)

    # Winddirection figure for supplement of article
    dict_winddirection = {  "N": 0,
                            "NE": 45,
                            "E": 90,
                            "SE": 135,
                            "S": 180,
                            "SW": 225,
                            "W": 270,
                            "NW": 315}
    winddir_mapping = df_hover['WEATHER.windDirection'].map(dict_winddirection).values
    df_hover = df_hover.copy()  # to avoid SettingWithCopyWarning
    df_hover['WEATHER.windDirection.Value'] = winddir_mapping

    fig, ax = plt.subplots(figsize=(8,6))
    hb = ax.hexbin(df_hover['WEATHER.windDirection.Value'], df_hover['PROC.EC.WindDirection'], gridsize=23, cmap='Reds', extent=(-15,350,-15,350), bins='log')
    cb = fig.colorbar(hb, ax=ax, label='Counts')
    ax.plot(np.arange(-45,360), np.arange(-45,360), c='k', linestyle='--')
    ax.set_aspect('equal')
    ax.set_xlabel('Wind direction from drone') 
    xticks = np.arange(0, 360, 45)
    xlabels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    ax.set_xticks(xticks, labels=xlabels)
    ax.set_ylabel('Wind direction from EC [$^{\circ}$]')
    ax.set_xlim([-45,360])
    ax.set_ylim([-20,370])
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig("winddirection_supplement_article.png", dpi=300)

    # save dataframe
    df.to_csv("data/05_df.csv", index=True, decimal='.', sep=',')