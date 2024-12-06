import os
import pandas as pd
from datetime import date, time, datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime, timedelta, timezone

"""
Convert the CH4 concentration from ppm to mg/m3
"""

# open dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df = pd.read_csv("data/05_df.csv", index_col='datetime', dtype=dtype_options)
df.index = pd.to_datetime(df.index, format='ISO8601')

def ppm_to_conc(ppm, temp, pressure):
    """
    Converts gas concentration in ppm to mg/m3 

    See
    https://www.lenntech.com/calculators/ppm/converter-parts-per-million.htm#Documentation
    """

    temp = temp + 273.15  # convert from deg C to Kelvin
    pressure = pressure / 101.325  # convert from kPa to atmospheres [atm]

    # ideal gas law: P V = n R T
    R = 0.0821  # [(L atm) / (mol K)]
    M = 16.04 # molar mass of methane in [g / mol]      

    molar_volume = R * temp / pressure  # molar volume [L / mol]
    # standard molar volume of ideal gas (1 bar and 0 deg C) = 22.71108 L/mol

    conversion_factor = molar_volume / M  # [L / g]
    conc = ppm / conversion_factor  # ppm is parts per million (/1,000,000) and [L / g] = [dm3 / g] = [m3 / mg] / 1,000,000

    return conc

df['PROC.CH4 [mg/m3]'] = ppm_to_conc(df['AERIS.CH4 [ppm]'], df['EC.T [deg C]'], df['EC.P [kPa]'])

# save dataframe
df.to_csv("data/06_df.csv", index=True, decimal='.', sep=',')