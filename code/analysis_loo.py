import gpflow

import numpy as np
import os
import pandas as pd
import re

import cmip6_eb_funcs as my

np.random.seed(2022)

def rbf_maker():
    return gpflow.kernels.SquaredExponential()

# cols for each dataset
y_all = pd.DataFrame(np.full(3012, np.nan)) # hacky way to get long enough
theta_hat_all = pd.DataFrame(np.full(3012, np.nan))

for f in os.listdir('output/fits_preds_simplified/rbf_mcds_all_all_all/'):
    contains = re.search('train_single_', f)
    if contains is not None:
        #print(f)
        extant_output = pd.read_csv('output/fits_preds_simplified/rbf_mcds_all_all_all/'+f)
        y_all[f] = extant_output['obs']
        theta_hat_all[f] = extant_output['mean']

y_all.drop(labels=0, axis=1, inplace=True)
theta_hat_all.drop(labels=0, axis=1, inplace=True)

# forget about observed, do loo on models
y_all_sim = y_all.drop(labels='train_single_observed.csv', axis=1, inplace=False)
theta_hat_all_sim = theta_hat_all.drop(labels='train_single_observed.csv', axis=1, inplace=False)

cmip6s = pd.read_csv('../data/all_cmip6_simplified.csv')
x = cmip6s[cmip6s.name == 'ukesm1-0-ll']['x'].values


mses_model = []
mses_single_gp = []
datasets = []

for loo in y_all_sim.columns:
    y_all_loo = y_all_sim.copy()
    theta_hat_all_loo = theta_hat_all_sim.copy()

    obs_loo = y_all_loo.pop(loo) # the sim we'll treat as the observed ts
    y_all_loo.insert(0, 'observed', obs_loo[0:2061])

    gp_loo = theta_hat_all_loo.pop(loo)
    theta_hat_all_loo.insert(0, 'observed', gp_loo)

    if not all(theta_hat_all_loo.columns == y_all_loo.columns):
        raise Exception('columns do not match', theta_hat_all_loo.columns, 'vs', y_all_loo.columns)
        # this is important so the cov matrix lines up

    y_means = y_all_loo.mean(axis=1) # mean across rows

    fit = my.fit_ml_single(np.expand_dims(y_means,1), np.expand_dims(x,1), rbf_maker)
    m = fit['model']
    mu_hat, mu_var = m.predict_f(np.expand_dims(x,1))
    mu_hat = mu_hat.numpy()
    mu_var = mu_var.numpy()

    re_hat_all = theta_hat_all_loo.copy()

    for c in re_hat_all.columns:
        re_hat_all[c] = np.squeeze(mu_hat) - y_all_loo[c]

    re_hat_all_np = re_hat_all.values


    Sigma_hat = re_hat_all.cov().values
    sigma_0_hat = Sigma_hat[0,0]
    Sigma_0_hat = np.expand_dims(Sigma_hat[1:,0],1) # column vector
    Sigma_mod_hat = Sigma_hat[1:,1:]
    Sigma_mod_hat_inv = np.linalg.inv(Sigma_mod_hat)

    obs_pred = mu_hat[2061:3012] - np.matmul(np.matmul(Sigma_0_hat.T, Sigma_mod_hat_inv), re_hat_all_np[2061:3012,1:].T).T
    obs_var = sigma_0_hat - np.matmul(np.matmul(Sigma_0_hat.T, Sigma_mod_hat_inv), Sigma_0_hat)

    mse_our_model = np.mean((obs_loo[2061:3012] - np.squeeze(obs_pred))**2)
    mse_single_gp = np.mean((obs_loo[2061:3012] - theta_hat_all_loo['observed'][2061:3012])**2)

    mses_model.append(mse_our_model)
    mses_single_gp.append(mse_single_gp) # trained AND fit on ALL data

    #print('MSE. model:', mse_our_model, '. single gp:', mse_single_gp)

    output = pd.DataFrame([x,
                         y_means,
                         np.squeeze(mu_hat),
                         np.squeeze(mu_var),
                         obs_loo,
                         np.append(np.repeat(np.nan, len(x)-obs_pred.shape[0]), obs_pred),
                         np.repeat(np.squeeze(obs_var), y_means.shape[0]),
                         theta_hat_all_loo['observed']]).T
    output.columns = ['x',
                      'y_mean',
                      'mu_hat',
                      'mu_hat_var',
                      'observed',
                      'obs_pred',
                      'obs_var_scalar',
                      'single_gp_theta'] # remember to only use 2061:3012 if calc mse by hand later
    fn = loo[13:]
    datasets.append(fn)
    output.to_csv('output/gpre_vis/loo_mse/'+fn, index=False)



loo_output = pd.DataFrame([datasets, mses_model, mses_single_gp]).T
loo_output.columns = ['dataset_test', 'mse_model', 'mse_single_gp']

loo_output.to_csv('output/gpre_vis/loo_output_mse.csv', index=False)
