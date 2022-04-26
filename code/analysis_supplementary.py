# For completeness, this document contains the necessary code to reproduce our primary analysis with a Matern kernel.
# We did not find it improved results.

import gpflow

import numpy as np
import os
import pandas as pd
import re

import cmip6_eb_funcs as my

np.random.seed(2022)

cmip6s = pd.read_csv('../data/all_cmip6_simplified.csv')
xall_dsmc, yall_dsmc, ds_names = my.make_subset_simplified(cmip6s, 0, 1.1, 'nsa_ds', True)

vars = []
for i in range(len(xall_dsmc)):
    a = 1 if ds_names[i] == 'observed' else 0.1
    if ds_names[i] != 'observed':
        vars.append(np.var(yall_dsmc[i][0:2061]))
    else:
        print('observed var:', np.var(yall_dsmc[i]))
print('CMIP6 Avg Var 1850-2020:', np.mean(vars))




########################
# EB Kernel Param Ests #
########################

def matern_new_maker():
    return gpflow.kernels.Matern52()

matern_names = ['kern_var', 'kern_ls']

def matern_extractor(model):
    return [np.array(model.kernel.variance), np.array(model.kernel.lengthscales)]

matern_results = my.fit_ml(yall_dsmc, xall_dsmc, ds_names, matern_new_maker, matern_names, matern_extractor, 'output/kern_ests_simplified/eb_dsmc_matern52.csv')

##############
# GP Fitting #
##############

def matern_maker(params):
    kern = gpflow.kernels.SquaredExponential(variance = params.kern_var, lengthscales = params.kern_ls)
    return kern

matern_params = pd.read_csv('output/kern_ests_simplified/eb_dsmc_matern52.csv')
comp1matern = my.compare_loo_gp(matern_params, matern_maker, xall_dsmc, yall_dsmc, xall_dsmc, yall_dsmc, 'output/fits_preds_simplified/matern_mcds_all_all_all/')


##########################################
# Model Fit & Prediction for Observed TS #
##########################################

# cols for each dataset
y_all = pd.DataFrame(np.full(3012, np.nan)) # hacky way to get long enough
theta_hat_all = pd.DataFrame(np.full(3012, np.nan))
errs_all = pd.DataFrame(np.full(3012, np.nan))

for f in os.listdir('output/fits_preds_simplified/matern_mcds_all_all_all/'):
    contains = re.search('train_single_', f)
    if contains is not None:
        #print(f)
        extant_output = pd.read_csv('output/fits_preds_simplified/matern_mcds_all_all_all/'+f)
        y_all[f] = extant_output['obs']
        theta_hat_all[f] = extant_output['mean']
        errs_all[f] = extant_output['obs'] - extant_output['mean']

y_all.drop(labels=0, axis=1, inplace=True)
theta_hat_all.drop(labels=0, axis=1, inplace=True)
errs_all.drop(labels=0, axis=1,inplace=True)

obs = y_all.pop('train_single_observed.csv')
y_all.insert(0, 'observed', obs)
obs = theta_hat_all.pop('train_single_observed.csv')
theta_hat_all.insert(0, 'observed', obs)

y_means = y_all.mean(axis=1)

x = cmip6s[cmip6s.name == 'ukesm1-0-ll']['x'].values # same for every dataset

fit = my.fit_ml_single(np.expand_dims(y_means,1), np.expand_dims(x,1), matern_new_maker)
m = fit['model']
mu_hat, mu_var = m.predict_f(np.expand_dims(x,1))
mu_hat = mu_hat.numpy()
mu_var = mu_var.numpy()

re_hat_all = theta_hat_all.copy()

for c in re_hat_all.columns:
    re_hat_all[c] = np.squeeze(mu_hat) - y_all[c]

re_hat_all_np = re_hat_all.values

Sigma_hat = re_hat_all.cov().values

sigma_0_hat = Sigma_hat[0,0]
Sigma_0_hat = np.expand_dims(Sigma_hat[1:,0],1) # column vector
Sigma_mod_hat = Sigma_hat[1:,1:]
Sigma_mod_hat_inv = np.linalg.inv(Sigma_mod_hat)


obs_pred = mu_hat - np.matmul(np.matmul(Sigma_0_hat.T, Sigma_mod_hat_inv), re_hat_all_np[:,1:].T).T
obs_var = sigma_0_hat - np.matmul(np.matmul(Sigma_0_hat.T, Sigma_mod_hat_inv), Sigma_0_hat)

observed_ts = y_all['observed'][~np.isnan(y_all['observed'])]

gpre_vis = pd.DataFrame([x,
                         y_means,
                         np.squeeze(mu_hat),
                         np.squeeze(mu_var),
                         observed_ts,
                         np.squeeze(obs_pred),
                         np.repeat(np.squeeze(obs_var), observed_ts.shape[0])]).T
gpre_vis.columns = ['x', 'y_mean', 'mu_hat', 'mu_hat_var', 'observed', 'obs_pred', 'obs_var_scalar']

gpre_vis.to_csv('output/gpre_vis/preds_obs_long_matern.csv', index=False)



