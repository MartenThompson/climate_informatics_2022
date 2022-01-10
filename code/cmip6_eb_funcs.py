import numpy as np
import pandas as pd
import gpflow
from gpflow.utilities import print_summary


def make_subset_simplified(cmip6, start, end, column_name, mean_center=False):
    Xmake_all = []
    Ymake_all = []
    dataset_names = cmip6.name.unique()

    for n in dataset_names:
        p = cmip6[cmip6.name == n]
        X = np.expand_dims(p.x,1)
        
        if mean_center:
            # mean center based on 1850-2020 data
            temp_mean_centered = p[column_name] - np.mean(p[column_name][0:2061]) 
            Y = np.expand_dims(temp_mean_centered,1)
        else:
            Y = np.expand_dims(p[column_name],1)
        
        keep_tf = np.logical_and(X[:,0] >= start, X[:,0] < end)
        Xmake_all.append(X[keep_tf,:])
        Ymake_all.append(Y[keep_tf,:])
        
    return Xmake_all, Ymake_all, dataset_names

def make_subset(cmip6, start, end, column_name):
    Xmake_all = []
    Ymake_all = []
    dataset_names = cmip6.name.unique()

    for n in dataset_names:
        p = cmip6[cmip6.name == n]
        X = np.expand_dims(p.time_hrs_since01,1)
        # globally mean centered, but maybe not for w/e subset we have here. This bias might accidentally help gp?
        temp_mean_centered = p[column_name] - np.mean(p[column_name]) 
        Y = np.expand_dims(temp_mean_centered,1)
        
        keep_tf = np.logical_and(X[:,0] >= start, X[:,0] < end)
        Xmake_all.append(X[keep_tf,:])
        Ymake_all.append(Y[keep_tf,:])
        
    return Xmake_all, Ymake_all, dataset_names


def fit_ml_single(Y,X,kern_maker):
    opt = gpflow.optimizers.Scipy()
    max_iter = 1000
    
    kern = kern_maker()
    m = gpflow.models.GPR(data=(X, Y), kernel=kern, mean_function=None)
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=max_iter))
    
    return {'model':m,
           'converged':opt_logs['success']}

def fit_ml(Y_all, X_all, dataset_names, kern_maker, param_colnames, param_extractor, filename):
    eb_results = pd.DataFrame([], columns=['dataset','convergence','lik_var'] + param_colnames)
    opt = gpflow.optimizers.Scipy()
    max_iter = 1000
    
    if len(Y_all) != len(dataset_names):
        print('Size mismatch. Y:', len(Y_all), '. Names:', len(dataset_names))
        return 0
    
    for i in range(len(Y_all)):
        kern = kern_maker()
        X = X_all[i]
        Y = Y_all[i]
        m = gpflow.models.GPR(data=(X, Y), kernel=kern, mean_function=None)
        opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=max_iter))
        #print_summary(m)
        results = {'dataset': dataset_names[i],
                  'convergence': opt_logs['success'],
                  'lik_var': np.array(m.likelihood.variance)}
        
        param_values = param_extractor(m)
        for j in range(len(param_values)):
            results[param_colnames[j]] = param_values[j]
        
        eb_results = eb_results.append(results, ignore_index=True)

    eb_results.to_csv(filename, index=False)
    return eb_results


def compare_loo_gp(param_results, kernel_maker, Xtr_all, Ytr_all, Xte_all, Yte_all, pred_dir=None):
    M = param_results.shape[0]
    mse_group = np.zeros((M))
    mse_single = np.zeros((M))
    single_set = []
    
    # b/c we dropped observed
    if M != len(Xtr_all):
        print('Size mismatch: M', M, ', data', len(Xtr_all))
        return 0
    
    for m in range(M):
        dataset = param_results.dataset[m]

        X_tr = Xtr_all[m]
        Y_tr = Ytr_all[m]
        X_te = Xte_all[m]
        Y_te = Yte_all[m]
       
        group_ests = param_results.drop(m).mean(numeric_only=True)

        kern_group = kernel_maker(group_ests)
        kern_single = kernel_maker(param_results.loc[m])
        
        m_group = gpflow.models.GPR(data=(X_tr, Y_tr), kernel=kern_group, mean_function=None)
        m_group.likelihood.variance = np.double(group_ests.lik_var)
        
        mod = gpflow.models.GPR(data=(X_tr, Y_tr), kernel=kern_single, mean_function=None)
        mod.likelihood.variance = np.double(param_results.lik_var[m])

        pred_m_group, pred_var_group = m_group.predict_f(X_te)
        pred_m, pred_var = mod.predict_f(X_te)

        mse_group[m] = np.mean((Y_te[:,0] - pred_m_group[:,0])**2)
        mse_single[m] = np.mean((Y_te[:,0]- pred_m[:,0])**2)
        single_set.append(dataset)
        
        if pred_dir is not None:
            fn = pred_dir + 'test_group_' + param_results.dataset[m] + '.csv'
            d = np.array([pred_m_group[:,0], pred_var_group[:,0], Y_te[:,0]]).T
            dat = pd.DataFrame(d, columns=['mean', 'var', 'obs'])
            dat.to_csv(fn, index=False)
            
            fn = pred_dir + 'test_single_' + param_results.dataset[m] + '.csv'
            d = np.array([pred_m[:,0], pred_var[:,0], Y_te[:,0]]).T
            dat = pd.DataFrame(d, columns=['mean', 'var', 'obs'])
            dat.to_csv(fn, index=False)          
            
            train_m_group, train_var_group = m_group.predict_f(X_tr)
            train_m, train_var = mod.predict_f(X_tr)
            
            fn = pred_dir + 'train_group_' + param_results.dataset[m] + '.csv'
            d = np.array([train_m_group[:,0], train_var_group[:,0], Y_tr[:,0]]).T
            dat = pd.DataFrame(d, columns=['mean', 'var', 'obs'])
            dat.to_csv(fn, index=False)
            
            fn = pred_dir + 'train_single_' + param_results.dataset[m] + '.csv'
            d = np.array([train_m[:,0], train_var[:,0], Y_tr[:,0]]).T
            dat = pd.DataFrame(d, columns=['mean', 'var', 'obs'])
            dat.to_csv(fn, index=False)
        
    results = pd.DataFrame(np.array([mse_group, mse_single]).T, columns=['mse_group', 'mse_single'])
    results['single_set'] = single_set

    return results




# pred_dir   define to save all preds/obs
def compare_group_single_gp(group_param_results, single_param_results, make_kern, Xtr_all, Ytr_all, Xte_all, Yte_all, pred_dir=None):
    M = group_param_results.shape[0]
    mse_group = np.zeros((M))
    mse_single = np.zeros((M))
    single_set = []
    
    # b/c we dropped observed
    if M != len(Xtr_all):
        print('Size mismatch: M', M, ', data', len(Xtr_all))
        return 0
    if M != single_param_results.shape[0]:
        print('Size mismatch: M group:', M, ', M single:', single_param_results.shape[0])
        return 0
    
    for m in range(M):
        dataset = group_param_results.dataset[m]

        X_tr = Xtr_all[m]
        Y_tr = Ytr_all[m]
        X_te = Xte_all[m]
        Y_te = Yte_all[m]

        group_ests = group_param_results.drop(m).mean(numeric_only=True)
        
        kern_group = make_kern(group_ests)
        kern_single = make_kern(single_param_results.loc[m])
        
        m_group = gpflow.models.GPR(data=(X_tr, Y_tr), kernel=kern_group, mean_function=None)
        m_group.likelihood.variance = np.double(group_ests.lik_var)
        
        mod = gpflow.models.GPR(data=(X_tr, Y_tr), kernel=kern_single, mean_function=None)
        mod.likelihood.variance = np.double(single_param_results.lik_var[m])
        
        pred_m_group, pred_var_group = m_group.predict_f(X_te)
        pred_m, pred_var = mod.predict_f(X_te)

        mse_group[m] = np.mean((Y_te[:,0] - pred_m_group[:,0])**2)
        mse_single[m] = np.mean((Y_te[:,0]- pred_m[:,0])**2)
        single_set.append(dataset)
        
        if pred_dir is not None:
            fn = pred_dir + 'test_group_' + group_param_results.dataset[m] + '.csv'
            d = np.array([pred_m_group[:,0], pred_var_group[:,0], Y_te[:,0]]).T
            dat = pd.DataFrame(d, columns=['mean', 'var', 'obs'])
            dat.to_csv(fn, index=False)
            
            fn = pred_dir + 'test_single_' + group_param_results.dataset[m] + '.csv'
            d = np.array([pred_m[:,0], pred_var[:,0], Y_te[:,0]]).T
            dat = pd.DataFrame(d, columns=['mean', 'var', 'obs'])
            dat.to_csv(fn, index=False)          
            
            train_m_group, train_var_group = m_group.predict_f(X_tr)
            train_m, train_var = mod.predict_f(X_tr)
            
            fn = pred_dir + 'train_group_' + group_param_results.dataset[m] + '.csv'
            d = np.array([train_m_group[:,0], train_var_group[:,0], Y_tr[:,0]]).T
            dat = pd.DataFrame(d, columns=['mean', 'var', 'obs'])
            dat.to_csv(fn, index=False)
            
            fn = pred_dir + 'train_single_' + group_param_results.dataset[m] + '.csv'
            d = np.array([train_m[:,0], train_var[:,0], Y_tr[:,0]]).T
            dat = pd.DataFrame(d, columns=['mean', 'var', 'obs'])
            dat.to_csv(fn, index=False)
        
    results = pd.DataFrame(np.array([mse_group, mse_single]).T, columns=['mse_group', 'mse_single'])
    results['single_set'] = single_set

    return results
    








#X_all = []
#Y_all = []
#dataset_names = []
#for n in cmip6.name.unique():
#    p = cmip6[cmip6.name == n]
#    X = np.expand_dims(p.time_hrs_since01,1)
#    temp_mean_centered = p.nsa_dsdt - np.mean(p.nsa_dsdt)
#    Y = np.expand_dims(temp_mean_centered,1)
#    X_all.append(X)
#    Y_all.append(Y)
#    dataset_names.append(n)

# used to change [2.2] into 2.2 
#import re

#for i in range(garbo.shape[0]):
#    rbf_var_num = float(re.search('[0-9]*\.[0-9]*', garbo.rbf_var[i]).group(0))
#    rbf_ls_num = float(re.search('[0-9]*\.[0-9]*' , garbo.rbf_ls[i]).group(0))
#    p2_var_num = float(re.search('[0-9]*\.[0-9]*', garbo.p2_var[i]).group(0))
#    p2_off_num = float(re.search('[0-9]*\.[0-9]*' , garbo.p2_offset[i]).group(0))
#    #print(num)
#    garbo.rbf_var[i] = rbf_var_num
#    garbo.rbf_ls[i] = rbf_ls_num
#    garbo.p2_var[i] = p2_var_num
#    garbo.p2_offset[i] = p2_off_num

# garbo.head(2)
# garbo.to_csv('eb_kerns_rbfp2.csv', index=False)