import numpy as np
from scipy.linalg import sqrtm
from scipy.special import logsumexp
from scipy import interpolate
from KDEpy import FFTKDE
from KDEpy.bw_selection import silvermans_rule, improved_sheather_jones
import itertools

from core.observation_model import get_mu


def get_prediction(ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, x_obs, y_obs, z_obs, z_surface, x_source, y_source, z_source):
    """ get concentration observation prediction """

    flux = ensemble[0,:][np.newaxis,:]
    wind_speed = ensemble[1,:][np.newaxis,:]
    wind_direction = ensemble[2,:][np.newaxis,:]
    diffusivity = ensemble[3,:][np.newaxis,:]

    mu_conc = get_mu(   flux,
                        wind_speed,
                        wind_direction,
                        diffusivity,
                        background_ppm,
                        TAU,
                        TEMPERATURE,
                        PRESSURE,
                        x_obs,
                        y_obs,
                        z_obs,
                        z_surface,
                        x_source,
                        y_source,
                        z_source)

    prediction = mu_conc # we get the rest of the prediction somewhere else
    return prediction

def get_likelihood(ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, obs, x_obs, y_obs, z_obs, z_surface, x_source, y_source, z_source, abs_sigma):
    """ compute the likelihood """

    n_particles = ensemble.shape[1]
    #print(f'n_particles = {n_particles}')

    conc_obs = obs[0]
    windspeed_obs = obs[1]
    winddirection_obs = obs[2]
    diffusivity_obs = obs[3]

    n_conc = len(conc_obs)
    n_windspeed = len(windspeed_obs)
    n_winddirection = len(winddirection_obs)
    n_diffusivity = len(diffusivity_obs)

    prediction = np.zeros( (n_conc+n_windspeed+n_winddirection+n_diffusivity, n_particles) )

    prediction_mu_conc = get_prediction(ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, x_obs, y_obs, z_obs, z_surface, x_source, y_source, z_source)  # prediction of the mean concentration

    prediction[                             : n_conc,                            :] = prediction_mu_conc
    prediction[ n_conc                      : n_conc+n_windspeed,                :] = ensemble[1,:][np.newaxis,:]
    prediction[ n_conc+n_windspeed          : n_conc+n_windspeed+n_winddirection,:] = ensemble[2,:][np.newaxis,:]
    prediction[ n_conc+n_windspeed+n_winddirection:                             ,:] = ensemble[3,:][np.newaxis,:]

    # reshape obs
    obs = np.concatenate(( obs[0], obs[1], obs[2], obs[3] ), axis=None)[:,np.newaxis]
    n_obs = obs.shape[0]

    residual = obs - prediction 
    residual[2,:] = abs(residual[2,:] + 180) % 360 - 180 
    assert residual.shape == (n_obs, n_particles)

    std = abs_sigma + 0 * np.array(obs)
    assert std.shape == (n_obs, 1)
    R = std**2
    assert R.shape == (n_obs, 1)
    #print(f"R = \n{R}")

    log_likelihood = -0.5 * ( (1/(R.T)) @ (residual**2) ) 
    # print(log_likelihood.shape)
    assert log_likelihood.shape == (1, n_particles)                                                         

    # compute normalized weights
    log_z = logsumexp(log_likelihood)
    log_weights = log_likelihood - log_z
    weights = np.exp(log_weights)
    weights = np.squeeze(weights)

    # compute variable proportional to the evidence, this will be a measure for the fit of the hyperparameters.
    # note that weights_exact and log_z_exact are not stictly necessary for the particle filter algorithm to work,
    # they can be omitted, and statistics can be computed from the resampled ensemble.
    log_likelihood_exact = log_likelihood - ((1/2) * np.sum(np.log(R))) - (n_obs/2 * np.log(2*np.pi))
    assert log_likelihood_exact.shape == (1, n_particles) 
    #print(f"log_likelihood_exact = \n{log_likelihood_exact}")

    log_z_exact = logsumexp(log_likelihood_exact) + np.log(1/n_particles)
    #print(f"log_z_exact = {log_z_exact}")
    z_exact = np.exp(log_z_exact)
    #print(f"z_exact = {z_exact}")

    log_weights_exact = log_likelihood_exact - log_z_exact
    weights_exact = np.exp(log_weights_exact)
    weigths_exact = np.squeeze(weights_exact)
    #print(f"weigths = \n{weights}")

    return weights, weigths_exact, log_z_exact


def smoothing(ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, obs, x_obs, y_obs, z_obs, z_surface, x_source, y_source, z_source, abs_sigma):
    """ update weights for mini-batch of data """

    weights, weights_exact, log_evidence = get_likelihood(ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, obs, x_obs, y_obs, z_obs, z_surface, x_source, y_source, z_source, abs_sigma)

    return weights, weights_exact, log_evidence

def resampling(ensemble, weights):
    """ resample the ensemble based on the weights of the ensemble members / particles """

    n_particles = np.shape(ensemble)[1]

    rng = np.random.default_rng(12)
    selection_index = rng.choice(a=n_particles, size=n_particles, p=weights)  # replace = True by default
    resampled_ensemble = ensemble[:, selection_index]

    return resampled_ensemble

def get_proposal_ensemble(ensemble, std):
    """ random walk proposal """

    noise = np.zeros_like(ensemble)
    n_vars, n_particles = np.shape(ensemble)
    rng = np.random.default_rng()
    for i in range(n_vars):
        noise[i,:] = rng.normal(loc=0, scale=std[i], size=n_particles)
    ensemble = ensemble + noise
    return ensemble

def get_in_bounds(ensemble, bounds, x, y):
    """ make sure that the ensemble stays inside the bounds set by the prior """

    if ensemble.ndim == 1:
        ensemble = ensemble[np.newaxis, :]
        bounds = bounds[np.newaxis, :]
    assert ensemble.ndim == bounds.ndim == 2

    #(f"ensemble = \n{ensemble}")
    n_vars, n_particles = ensemble.shape

    min_bounds = np.repeat(bounds[:, 0].reshape(-1, 1), n_particles, axis=1)
    max_bounds = np.repeat(bounds[:, 1].reshape(-1, 1), n_particles, axis=1)
    #print(f"min_bounds = \n{min_bounds}")
    #print(f"max_bounds = \n{max_bounds}")

    in_min_bounds = ensemble >= min_bounds
    in_max_bounds = ensemble <= max_bounds
    # assert in_min_bounds.shape == in_max_bounds.shape == ensemble.shape

    in_bounds = (in_min_bounds).all(axis=0) & (in_max_bounds).all(axis=0)
    # assert len(in_bounds) == n_particles
    # print(f"in_bounds = {in_bounds}")

    return in_bounds, in_min_bounds[0], in_max_bounds[0]

def get_reflected_ensemble(ensemble, bounds, x, y):
    """ use reflected boundaries for the ensemble """

    for i in range(ensemble.shape[0]):
        _, in_min_bounds, in_max_bounds = get_in_bounds(ensemble[i:i+1,:], bounds[i:i+1, :], x, y)
        ensemble[i, ~in_min_bounds] = bounds[i,0] + np.abs(ensemble[i, ~in_min_bounds] - bounds[i,0])
        ensemble[i, ~in_max_bounds] = bounds[i,1] - np.abs(ensemble[i, ~in_max_bounds] - bounds[i,1])

    return ensemble

def mcmc_move(resampled_ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, obs, x_obs, y_obs, z_obs, z_surface, x_source, y_source, z_source, abs_sigma, std, bounds, x, y):
    """ perform one mcmc step """

    n_vars, n_particles = np.shape(resampled_ensemble)

    current_likelihood, _, _ = get_likelihood(resampled_ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, obs, x_obs, y_obs, z_obs, z_surface, x_source, y_source, z_source, abs_sigma)

    # jitter
    proposal_ensemble = get_proposal_ensemble(resampled_ensemble, std)
    # reflective boundaries
    proposal_ensemble = get_reflected_ensemble(proposal_ensemble, bounds, x, y)
    # get proposal likelihood
    proposal_likelihood, _, _ = get_likelihood(proposal_ensemble, background_ppm, TAU, TEMPERATURE, PRESSURE, obs, x_obs, y_obs, z_obs, z_surface, x_source, y_source, z_source, abs_sigma)

    bounds_proposal, _, _ = get_in_bounds(proposal_ensemble, bounds, x, y)
    bounds_current, _, _ = get_in_bounds(resampled_ensemble, bounds, x, y)  
    acceptance_ratio = (proposal_likelihood * bounds_proposal) / \
        (current_likelihood * bounds_current)  
    acceptance_probability = np.minimum(1, acceptance_ratio)

    rng = np.random.default_rng()
    random_values = rng.random(n_particles)
    selection_index = random_values < acceptance_probability
    #print(f"selection_index = {selection_index}")

    new_ensemble = resampled_ensemble.copy()
    new_ensemble[:, selection_index] = proposal_ensemble[:, selection_index]
    assert new_ensemble.shape == (n_vars, n_particles)

    return new_ensemble
