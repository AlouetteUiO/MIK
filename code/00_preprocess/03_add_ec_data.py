import os
import pandas as pd
from datetime import date, time, datetime
import numpy as np
import zipfile

"""
Add Eddy-covariance data.
"""

# create empty dataframe
df_total = pd.DataFrame()

# collect drone data
directory = "./ECtower_Kapiti"  # change to location of eddy-covariance data, available from zenodo
suffix = "_LI-7500DS.data"

columns_to_read = ['Seconds',
                'Nanoseconds',
                'U (m/s)', 
                'V (m/s)', 
                'W (m/s)',
                'CH4 (umol/mol)',
                'Temperature (C)',
                'Pressure (kPa)']

for ghgfile in os.listdir(directory):
    print(f"processing {ghgfile}")
    zip_file_path = f"{directory}/{ghgfile}"

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()

        for filename in file_list:
            if filename.endswith(suffix):

                target_filename = filename
                zip_ref.extract(target_filename)

    target_file_path = os.path.join(os.getcwd(), target_filename)

    df = pd.read_csv(target_file_path, skiprows=7, header=0, sep='\t', usecols=columns_to_read)
    os.remove(target_file_path)

    df['datetime'] = (pd.to_datetime(df['Seconds'], unit="s") + pd.to_timedelta(df['Nanoseconds'], unit="ns")).dt.tz_localize('UTC')
    df = df.set_index('datetime')
    df = df.drop( columns=['Seconds', 'Nanoseconds'] )

    # replace all -9999 with NaN, found for wind speed u,v,w on 07/03 around 07:55.
    df.replace(-9999, np.NaN, inplace=True)

    # rename columns
    df.rename(columns={ 'U (m/s)': 'EC.U [m/s]',
                        'V (m/s)': 'EC.V [m/s]',
                        'W (m/s)': 'EC.W [m/s]',
                        'CH4 (umol/mol)': 'EC.CH4 [ppm]',
                        'Temperature (C)': 'EC.T [deg C]',
                        'Pressure (kPa)': 'EC.P [kPa]'}, inplace=True)

    # yaw correction
    theta = np.radians(120)
    df['U_temp'] = np.cos(theta) * df['EC.U [m/s]'] + np.sin(theta) * df['EC.V [m/s]'] 
    df['PROC.EC.Vcorr'] = - np.sin(theta) * df['EC.U [m/s]'] + np.cos(theta) * df['EC.V [m/s]']
    # heading correction
    phi = np.radians(1)
    df['PROC.EC.Ucorr'] = np.cos(phi) * df['U_temp'] + np.sin(phi) * df['EC.W [m/s]']  
    df['PROC.EC.Wcorr'] = - np.sin(phi) * df['U_temp'] + np.cos(theta) * df['EC.W [m/s]']
    # compute wind speed and wind direction relative to north
    df['PROC.EC.WindSpeed'] = np.sqrt( df['PROC.EC.Ucorr']**2 + df['PROC.EC.Vcorr']**2 + df['PROC.EC.Wcorr']**2 )
    df['PROC.EC.HorWindSpeed'] = np.sqrt( df['PROC.EC.Ucorr']**2 + df['PROC.EC.Vcorr']**2)
    df['PROC.EC.WindDirection'] = (180 - np.degrees( np.arctan2(df['PROC.EC.Vcorr'], df['PROC.EC.Ucorr']) )) % 360

    df = df.drop(columns = ['U_temp'])

    # resample the data
    df = df.resample('S').mean()

    # concat
    df_total = pd.concat([df_total, df], ignore_index=False)

# add dataframe to leading dataframe
dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
df_old = pd.read_csv("02_df.csv", index_col='datetime', dtype=dtype_options)
df_old.index = pd.to_datetime(df_old.index, format='ISO8601')
merged_df = pd.merge(df_old, df_total, left_index=True, right_index=True, how='inner')

# save dataframe
merged_df.to_csv("03_df.csv", index=True, decimal='.', sep=',')
