import numpy as np
from scipy.special import kn
from itertools import product

def get_mu( flux, wind_speed, wind_direction, diffusivity, background_ppm,    # unknowns
            TAU, TEMPERATURE, PRESSURE,                                       # knowns
            x_obs, y_obs, z_obs, z_surface, x_source, y_source, z_source      # observation location and source location  
        ):

    n_particles = flux.shape[1]
    n_obs = x_obs.shape[0]
    #print(f"n_particles = {n_particles}")
    #print(f"n_obs = {n_obs}")

    flux = np.broadcast_to(flux, (n_obs, n_particles))
    wind_speed = np.broadcast_to(wind_speed, (n_obs, n_particles))
    wind_direction = np.broadcast_to(wind_direction, (n_obs, n_particles))
    diffusivity = np.broadcast_to(diffusivity, (n_obs, n_particles))

    lambda_var = np.sqrt(diffusivity * TAU / (1 + ((wind_speed**2 * TAU)/(4 * diffusivity))))
    assert lambda_var.shape == (n_obs, n_particles)

    mu_conc = np.zeros(shape=(n_obs,n_particles))
    mu_conc_mirror = np.zeros_like(mu_conc)

    for i in range(len(x_source)):

        dx = x_obs - x_source[i]
        dy = y_obs - y_source[i]

        dx = np.broadcast_to(dx, (n_obs, n_particles))
        dy = np.broadcast_to(dy, (n_obs, n_particles))
        assert dx.shape == dy.shape == (n_obs, n_particles)

        dz = z_obs-z_source
        z_source_mirror = z_surface - (z_source - z_surface)
        dz_mirror = z_obs - z_source_mirror
        dz = np.broadcast_to(dz, (n_obs, n_particles))
        dz_mirror = np.broadcast_to(dz_mirror, (n_obs, n_particles))
        assert dz.shape == (n_obs, n_particles)

        d = np.linalg.norm([dx, dy, dz], axis=0, ord=2)
        #print(f"d = \n{d}")
        d_mirror = np.linalg.norm([dx, dy, dz_mirror], axis=0, ord=2)  
        #print(f"d_mirror = \n{d_mirror}")
        assert d.shape == d_mirror.shape == (n_obs, n_particles)

        wind_direction_rad = np.deg2rad(wind_direction)
        mu_conc += (
                    (flux / (4 * np.pi * diffusivity * d))
                    * np.exp( - d / lambda_var )
                    * np.exp( (- dx * wind_speed * np.sin(wind_direction_rad)) / (2 * diffusivity) )
                    * np.exp( (- dy * wind_speed * np.cos(wind_direction_rad)) / (2 * diffusivity) )
                )
        #print(f"mu_conc = \n{mu_conc}")

        mu_conc_mirror += (
                    (flux / (4 * np.pi * diffusivity * d_mirror))
                    * np.exp( - d_mirror / lambda_var )
                    * np.exp( (- dx * wind_speed * np.sin(wind_direction_rad)) / (2 * diffusivity) )
                    * np.exp( (- dy * wind_speed * np.cos(wind_direction_rad)) / (2 * diffusivity) )
                )
        #print(f"mu_conc_mirror = \n{mu_conc_mirror}")
    
    mu_conc += mu_conc_mirror  # mg/m3
    assert mu_conc.shape == (n_obs, n_particles)

    mu_ppm = convert_to_ppm(mu_conc, TEMPERATURE, PRESSURE)
    assert mu_ppm.shape == (n_obs, n_particles)
    #print(f"mu_ppm = \n{mu_ppm}")

    mu_total_ppm = mu_ppm + background_ppm
    assert mu_total_ppm.shape == (n_obs, n_particles)

    return mu_total_ppm

def convert_to_ppm(mu_conc, TEMPERATURE, PRESSURE):
    """ mg/m3 to ppm """
    R = 8.314  # universal gas constant [(m3*Pa)/(K*mol)]
    molar_mass = 16.04  # [g/mol]
    molar_volume = R * TEMPERATURE / PRESSURE  # molar volume [m3/mol]
    conversion_factor = molar_volume / molar_mass  # [m3/g]
    conc_ppm = mu_conc * conversion_factor * 1e3
    #print(f"conc_ppm (no background) = \n{conc_ppm}")
    return conc_ppm