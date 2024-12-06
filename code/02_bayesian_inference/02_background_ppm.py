import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import os

"""
Estimate the background CH4 concentraton per flight. 
We use the median of the CH4 concentration observations below 1.8 ppm.
"""

directory = "data/bayesian_inference"
suffix = "_01.csv"

for filename in os.listdir(directory):
    if filename.endswith(suffix):
        fileloc = f"{directory}/{filename}"
        file_string = filename.replace(suffix, "") # remove suffix from the string

        # open dataframe
        dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
        df = pd.read_csv(fileloc, index_col='datetime', dtype=dtype_options)
        df.index = pd.to_datetime(df.index, format='ISO8601')

        # check that type of flight is octagon or plough
        if (df['NOTE.typeOfFlight'][0] == 'octagon') or (df['NOTE.typeOfFlight'][0] == 'plough'):
            print(f'found octagon or plough flight: {file_string}')

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
            bins = np.linspace(start=1.6, stop=12, num=104, endpoint=True)
            ax.hist(df['AERIS.CH4 [ppm]'].values, bins)
            ax.set_xlabel('methane concentration [ppm]')
            ax.set_ylabel('number of observations')
            ax.set_title(f'{file_string}')
            plt.tight_layout()
            plt.savefig(f"{directory}/02_Figures/{file_string}_observations.png")
            plt.close()

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,9))
            bins = np.linspace(start=1.65, stop=2, num=40, endpoint=True)
            ax.hist(df['AERIS.CH4 [ppm]'].values, bins)
            ax.set_xlabel('methane concentration [ppm]')
            ax.set_ylabel('number of observations')
            ax.set_title(f'{file_string}')
            mean = np.round(np.mean(df[df['AERIS.CH4 [ppm]']<=1.8]['AERIS.CH4 [ppm]']), 2)
            median = np.round(np.median(df[df['AERIS.CH4 [ppm]']<=1.8]['AERIS.CH4 [ppm]']), 2)
            std = np.round(np.std(df[df['AERIS.CH4 [ppm]']<=1.8]['AERIS.CH4 [ppm]']), 2)
            percentile = np.round(np.percentile(df[df['AERIS.CH4 [ppm]']<=1.8]['AERIS.CH4 [ppm]'], q=95), 2)
            ax.axvline(x=mean, c='red', label=f'mean = {mean} for obs <= 1.8\nstd = {std}') 
            ax.axvline(x=median, c='black', label=f'median = {median} for obs <= 1.8\n95 percentile = {percentile}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{directory}/02_Figures/{file_string}_background_ppm.png")
            plt.close()

            # save data in leading dataframe
            df['PROC.AERIS.background_ppm'] = median
            df.to_csv(f"{directory}/{file_string}_02.csv", index=True, decimal='.', sep=',')
