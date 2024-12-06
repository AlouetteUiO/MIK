import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta, timezone
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones

from core.particle_filter import smoothing, resampling, mcmc_move
from core.observation_model import get_mu

seed = 1234

def get_mean_WD(data, weights=None):
    WD_rad = np.deg2rad(data)
    if weights is None:
        sin_sum = np.sum(np.sin(WD_rad))
        cos_sum = np.sum(np.cos(WD_rad))
    else:
        sin_sum = np.sum(weights * np.sin(WD_rad))
        cos_sum = np.sum(weights * np.cos(WD_rad))
    mean_theta = np.arctan2(sin_sum, cos_sum)
    mean_direction = np.rad2deg(mean_theta)
    return mean_direction % 360  # can also be done with scipy circmean

# open dataframe
directory = "data/bayesian_inference"
suffix = "_02.csv"

# read mean obs of wind speed, wind direction, etc.
df_obs = pd.read_csv("data/bayesian_inference/03_mean_obs.csv", index_col=0)
print(df_obs)

# read shapefile with source polygon
gdf_source = gpd.read_file("./source_polygons.shp")  # change to file location of shapefiles with outline of herd inside boma, available from zenodo
gdf_source = gdf_source.set_index('file_str')
gdf_source.drop(columns=['id'], inplace=True)
print(gdf_source)

# randomly select x_sources and y_sources from the shapefile
n_source = 100
gdf_source['sampled_geometry'] = gdf_source.sample_points(size=n_source, seed=seed)

# boma location, needed later to get x_obs and y_obs from sampled points
x_0 = 292271.466
y_0 = 9821576.904

results_dict = {}

for filename in os.listdir(directory):
    if filename.endswith(suffix):
        fileloc = f"{directory}/{filename}"
        file_string = filename.replace(suffix, "") # remove suffix from the string

        dtype_options = {'OSD.flycState': 'str', 'OSD.flightAction': 'str', 'WEATHER.windDirection': 'str', 'NOTE.animals':'string', 'NOTE.amount':'string'}
        df = pd.read_csv(fileloc, index_col='datetime', dtype=dtype_options)
        df.index = pd.to_datetime(df.index, format='ISO8601')

        # check that type of flight is octagon or plough
        if (df['NOTE.typeOfFlight'][0] == 'octagon') or (df['NOTE.typeOfFlight'][0] == 'plough'):
            print(f'found octagon or plough flight: {file_string}')

            # include this if you only want to process the plough flights of 29/02 and 01/03 that we include in the article
            if (file_string[5] == '2') and (file_string != '20240229_131600_pregnant_does'):
                continue
            if (file_string[5] == '3') and (int(file_string[7]) > 1):
                continue

            id_ = file_string
            # for some cases where we do not have shapefiles and we use the shapefile of the morning
            if file_string == '20240302_140200_heifers':
                id_ = '20240302_063000_heifers'
            if file_string == '20240306_125000_heifers':
                id_ = '20240306_070300_heifers'
            if file_string == '20240302_130500_commerical_boran_cows':
                id_ = '20240302_074500_commerical_boran_cows'
            if file_string == '20240303_120800_slick_herd':
                id_ = '20240303_070500_slick_herd'

            x_source = []
            y_source = []
            for point in gdf_source.loc[id_]['sampled_geometry'].geoms:
                x_source.append(point.x - x_0)
                y_source.append(point.y - y_0)
            x_source = np.array(x_source)
            y_source = np.array(y_source)
            z_source = df['PROC.z_source_mean'].values[0] 

            if not os.path.exists(f"{directory}/04_bayes_obs_c/{file_string}"):
                os.makedirs(f"{directory}/04_bayes_obs_c/{file_string}")

            # drop rows that contain NaN values
            df.dropna(subset=['AERIS.CH4 [ppm]', 'PROC.x', 'PROC.y', 'PROC.z_drone', 'PROC.z_surface', 'PROC.WEATHER.windSpeedCorr', 'PROC.WEATHER.windSpeedCorrOrigin', 'PROC.EC.WindDirection', 'PROC.EC.HorWindSpeed'], inplace=True)

            # get observations and their locations
            obs = df['AERIS.CH4 [ppm]'].values[:, np.newaxis]
            x_obs = df['PROC.x'].values[:, np.newaxis]
            y_obs = df['PROC.y'].values[:, np.newaxis]
            z_obs = df['PROC.z_drone_base'].values[:, np.newaxis]
            z_surface = df['PROC.z_surface'].values[:, np.newaxis]
            n_obs = obs.shape[0]
            print(f"n_obs = {n_obs}")
            
            # set global values
            TAU = 286_977_600  # lifetime of CH4 9.1 years
            TEMPERATURE = np.mean(df['EC.T [deg C]']) + 273.15  # temperature in Kelvin
            PRESSURE = np.mean(df['EC.P [kPa]']) * 1_000  # pressure in Pa
            background_ppm = df['PROC.AERIS.background_ppm'].values[0]
            print(f"tau = {TAU}")
            print(f"mean temperature = {np.mean(df['EC.T [deg C]'])} degC or {TEMPERATURE} K")
            print(f"mean pressure = {PRESSURE} Pa")
            print(f"mean background ppm = {background_ppm}")

            # make prior
            n_vars = 4
            n_particles = 25_000
            flux_min = 0.1
            flux_max = 10
            nr_animals = df['NOTE.amount_int'].values[0]
            diffusivity_min = 0.3 
            diffusivity_max = 3
            windspeed_max = 6 
            winddirection_min = -45  # adjust this to 30 for the plough/V-shaped flights
            winddirection_max = 135
            # to deal with high wind speed in steers, observation model gives NaN for high wind speed and low diffusivity
            if file_string == '20240305_115500_steers':
                diffusivity_min = 0.5

            bins_flux = np.linspace(start=0, stop=flux_max, num=50, endpoint=True)
            bins_windspeed = np.linspace(start=0, stop=windspeed_max, num=50, endpoint=True)
            bins_winddirection = np.linspace(start=winddirection_min, stop=winddirection_max, num=50, endpoint=True)  
            bins_diffusivity = np.linspace(start=0, stop=diffusivity_max, num=50, endpoint=True)

            bins_dict = {   0: bins_flux,
                            1: bins_windspeed,
                            2: bins_winddirection,
                            3: bins_diffusivity }

            prior = np.zeros(shape=(n_vars, n_particles))
            rng = np.random.default_rng(seed)
            prior[0,:] = rng.uniform(low=flux_min*nr_animals/n_source, high=flux_max*nr_animals/n_source, size=n_particles)  # flux [mg/s] per animal
            prior[1,:] = rng.uniform(low=0, high=windspeed_max, size=n_particles)  # wind speed [m/s]
            prior[2,:] = rng.uniform(low=winddirection_min, high=winddirection_max, size=n_particles)  # wind direction [deg]
            prior[3,:] = rng.uniform(low=diffusivity_min, high=diffusivity_max, size=n_particles)  # diffusivity [m2/s]
            #print(f"prior = \n{prior}")

            bounds = np.array([ [0, flux_max*nr_animals/n_source],
                                [0, windspeed_max],
                                [winddirection_min, winddirection_max],
                                [diffusivity_min, diffusivity_max],
                              ]
            )

            ### perform Bayesian inference ###

            abs_sigma_conc = 1.0

            # We are going to divide the observations in batches
            n_mcmc_steps = 5
            batch_size = 200

            # We want to make sure that the use all the observations that are higher than the background concentration
            indices_high = np.where(obs>background_ppm)[0]
            indices_low = np.where(obs<background_ppm)[0]
            np.random.shuffle(indices_low)
            indices = np.concatenate((indices_high, indices_low))
            n_batches = int(np.floor(n_obs/batch_size))
            indices = indices[:batch_size*n_batches]
            rng = np.random.default_rng(seed)
            selected_index = rng.choice(indices, size=(n_batches, batch_size), replace=False)
            #print(f"n_batches = {n_batches}")
            #print(f"selected_index = \n{selected_index}")

            variables = ['flux', 'wind speed', 'wind direction', 'diffusivity']
            colors = np.linspace(0.25, 1, num=n_batches+1) 

            ensemble = prior.copy()

            fig, ax = plt.subplots(ncols=2, nrows=2, figsize= (15,10))
            j = 0
            for k in range(n_vars):
                ax_0, ax_1 = np.divmod(k, 2)
                if (ax_0 == 0) and (ax_1 == 0):
                    ax[0,0].hist((ensemble[0,:]/nr_animals*n_source), bins=bins_dict[0], color=plt.cm.Oranges(colors[j+1]), density=True)
                else:
                    ax[ax_0, ax_1].hist(ensemble[k,:], bins=bins_dict[k], color=plt.cm.Oranges(colors[j+1]), density=True)
                ax[ax_0, ax_1].set_title(variables[k], fontsize=20)
            ax[0,0].hist((ensemble[0,:]/nr_animals*n_source), bins=bins_dict[0], color=plt.cm.Oranges(colors[j+1]), density=True)
            # plot the observation of wind speed, wind direction and concentration (in background_ppm plot)
            ax[0,1].hist(df['PROC.WEATHER.windSpeedCorrOrigin'].values, bins=bins_dict[1], color=plt.cm.Greens(colors[0]), density=True)

            for j in range(n_batches):

                selected_index_batch = selected_index[j]

                obs_batch = obs[selected_index_batch]
                x_obs_batch = x_obs[selected_index_batch]
                y_obs_batch = y_obs[selected_index_batch]
                z_obs_batch = z_obs[selected_index_batch]
                z_surface_batch = z_surface[selected_index_batch]
                windspeed_mean = df_obs.loc[file_string, 'U_obs']
                winddirection_mean = df_obs.loc[file_string, 'WD_obs']
                diffusivity_mean = df_obs.loc[file_string, 'D_obs']
                windspeed_obs_batch = [windspeed_mean] * obs_batch.shape[0]
                winddirection_obs_batch = [winddirection_mean] * obs_batch.shape[0]
                diffusivity_obs_batch = [diffusivity_mean] * obs_batch.shape[0]

                obs_wind = [    obs_batch,
                                windspeed_obs_batch,
                                winddirection_obs_batch,
                                diffusivity_obs_batch ]

                R_windspeed = 0.3
                R_winddirection = 10
                R_diffusivity = 0.15
                abs_sigma = np.concatenate((    [abs_sigma_conc]*obs_batch.shape[0],
                                                [R_windspeed*np.sqrt(n_obs)]*obs_batch.shape[0],
                                                [R_winddirection*np.sqrt(n_obs)]*obs_batch.shape[0],
                                                [R_diffusivity*np.sqrt(n_obs)]*obs_batch.shape[0] ), axis=None)[:,np.newaxis]

                weights, weights_exact, log_evidence = smoothing(ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, obs_wind, \
                    x_obs_batch, y_obs_batch, z_obs_batch, z_surface_batch, x_source, y_source, z_source, abs_sigma=abs_sigma, rel_sigma=0)

                Neff = 1/(np.sum(weights**2))  # number of effective particles

                flux_weighted_mean = np.round( np.average( ensemble[0,:]/nr_animals*n_source, weights = weights_exact), 2)
                windspeed_weighted_mean = np.round( np.average(ensemble[1,:], weights = weights_exact), 2)
                winddirection_weighted_mean = np.round( get_mean_WD(ensemble[2,:], weights=weights_exact), 2)
                diffusivity_weighted_mean = np.round( np.average(ensemble[3,:], weights = weights_exact), 2)
                flux_max_weight = np.round(ensemble[0, np.argmax(weights)]/nr_animals*n_source,2)

                # resample
                resampled_ensemble = resampling(ensemble, weights)
                ensemble = resampled_ensemble.copy()

                # mean_flux and std_flux of resampled ensemble
                mean_flux = np.round(np.average(ensemble[0,:]/nr_animals*n_source),2)
                std_flux = np.round(np.std(ensemble[0,:]/nr_animals*n_source),2)
                mean_V = np.round(np.average(ensemble[1,:]),2)
                std_V = np.round(np.std(ensemble[1,:]),2)
                mean_WD = np.round(np.average(ensemble[2,:]),2)
                std_WD = np.round(np.std(ensemble[2,:]),2)
                mean_D = np.round(np.average(ensemble[3,:]),2)
                std_D = np.round(np.std(ensemble[3,:]),2)

                ensemble_to_save = ensemble.copy()
                ensemble_to_save[0,:] = ensemble_to_save[0,:]/nr_animals*n_source
                results_dict[file_string] = ensemble_to_save

                # move
                bw = [silvermans_rule(ensemble[[i],:].T) for i in range(n_vars)]
                ensemble = resampled_ensemble.copy()
                for step in range(n_mcmc_steps):
                    print(f"mcmc step {step}")
                    ensemble = mcmc_move(ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, obs_wind, x_obs_batch, y_obs_batch, z_obs_batch, z_surface_batch, x_source, y_source, z_source, abs_sigma=abs_sigma, rel_sigma=0, std=bw, bounds=bounds, x=x_source, y=y_source)
                # print(f"posterior flux = \n{posterior[0]}")
                
                # plot resampled ensemble
                for k in range(n_vars):
                    ax_0, ax_1 = np.divmod(k, 2)
                    if (ax_0 == 0) and (ax_1 == 0):
                        ax[0,0].hist((resampled_ensemble[0,:]/nr_animals*n_source), bins=bins_dict[0], color=plt.cm.Blues(colors[j+1]), density=True)
                    else:
                        ax[ax_0, ax_1].hist(resampled_ensemble[k,:], bins=bins_dict[k], color=plt.cm.Blues(colors[j+1]), density=True)
                ax[0,0].hist((resampled_ensemble[0,:]/nr_animals*n_source), bins=bins_dict[0], color=plt.cm.Blues(colors[j+1]), density=True)
            
            ax[0,1].axvline(x=df_obs.loc[file_string, 'U_drone_obs'], c='violet')
            ax[0,0].axvline(x=flux_weighted_mean, c='green', linestyle='--')
            ax[0,1].axvline(x=windspeed_weighted_mean, c='green', linestyle='--')
            ax[1,0].axvline(x=winddirection_weighted_mean, c='green', linestyle='--')
            ax[1,1].axvline(x=diffusivity_weighted_mean, c='green', linestyle='--')
            fig.suptitle(f"abs_sigma = {abs_sigma[0][0], np.round(R_windspeed,2), np.round(R_winddirection,2), np.round(R_diffusivity,2)} | Neff = {np.round(Neff, 1)}/{n_particles} | \
                flux = {mean_flux} +- {std_flux} mg/s | V = {mean_V} +- {std_V} | WD = {mean_WD} +- {std_WD} | D = {mean_D} +- {std_D}")
            plt.tight_layout()
            plt.savefig(f"{directory}/04_bayes_obs_c/{file_string}/case_c.png")
            plt.close()
            print("saved figure")

# save results dictionary
np.save(f"{directory}/04_bayes_obs_c/results_dict.npy", results_dict)