import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from package.util import log_info


def high_diff_corr_pca(train_df, test_df, n_features=5):
    log_info('high_diff_corr_pca')
    log_info('n_features=%d'%n_features)

    tmp = [c for c in train_df.columns if c not in ['id','target']]
    cols = []
    for s in tmp:
        if not 'made' in s:
            cols.append(s)
    train_test_df = pd.concat([train_df[cols], test_df[cols]])
    train_df_ = train_df[cols]

    x = train_df[cols].columns.values
    y = train_df[cols].columns.values
    z = np.abs(train_df_[train_df['target']==0].corr().values - train_df_[train_df['target']==1].corr().values)
    z[np.isnan(z)] = 0

    for i in range(n_features):
        ind = np.argwhere(z == np.max(z))[0]
        col_x = x[ind[0]]
        col_y = y[ind[1]]
        corr = z[ind[0], ind[1]]

        log_info('%d,\t(%s,\t%s)\n%f'%(i+1, col_x, col_y, corr))

        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        pca = PCA(n_components=1, random_state=41)

        X_train_test = np.vstack((train_test_df[col_x], train_test_df[col_y])).T
        X_train = np.vstack((train_df[col_x], train_df[col_y])).T
        X_test = np.vstack((test_df[col_x], test_df[col_y])).T

        X_train_test = scaler.fit_transform(X_train_test)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        pca.fit(X_train_test)
        X_train = pca.transform(X_train)[:, 0]
        X_test = pca.transform(X_test)[:, 0]

        train_df['made_high_diff_corr_pca_' + col_x + '_' + col_y] = X_train
        test_df['made_high_diff_corr_pca_' + col_x + '_' + col_y] = X_test

        z[ind[0], ind[1]] = 0
        z[ind[1], ind[0]] = 0

    return train_df, test_df


def high_2_4_diff_corr_pca(train_df, test_df, n_features=10):
    log_info('high_2_4_diff_corr_pca')
    log_info('n_features=%d'%n_features)

    tmp = [c for c in train_df.columns if c not in ['id','target']]
    cols = []
    for s in tmp:
        if not 'made' in s:
            cols.append(s)
    train_test_df = pd.concat([train_df[cols], test_df[cols]])
    train_df_ = train_df[cols]
    test_df_ = test_df[cols]

    x = train_df[cols].columns.values
    y = train_df[cols].columns.values
    z = np.abs(train_df_[train_df['target']==0].corr().values - train_df_[train_df['target']==1].corr().values)
    z[np.isnan(z)] = 0

    for i in range(50):
        ind = np.argwhere(z == np.max(z))[0]
        col_x = x[ind[0]]
        col_y = y[ind[1]]
        corr = z[ind[0], ind[1]]

        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        pca = PCA(n_components=1, random_state=41)

        X_train_test = np.vstack((train_test_df[col_x], train_test_df[col_y])).T
        X_train = np.vstack((train_df[col_x], train_df[col_y])).T
        X_test = np.vstack((test_df[col_x], test_df[col_y])).T

        X_train_test = scaler.fit_transform(X_train_test)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        pca.fit(X_train_test)
        X_train = pca.transform(X_train)[:, 0]
        X_test = pca.transform(X_test)[:, 0]

        train_df_[col_x + '_' + col_y] = X_train
        test_df_[col_x + '_' + col_y] = X_test

        z[ind[0], ind[1]] = 0
        z[ind[1], ind[0]] = 0

    train_test_df = pd.concat([train_df_, test_df_])

    x = train_df_.columns.values
    y = train_df_.columns.values
    z = np.abs(train_df_[train_df['target']==0].corr().values - train_df_[train_df['target']==1].corr().values)
    z[np.isnan(z)] = 0

    for i in range(n_features):
        ind = np.argwhere(z == np.max(z))[0]
        col_x = x[ind[0]]
        col_y = y[ind[1]]
        corr = z[ind[0], ind[1]]

        log_info('%d,\t(%s,\t%s)\n%f'%(i+1, col_x, col_y, corr))

        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        pca = PCA(n_components=1, random_state=41)

        X_train_test = np.vstack((train_test_df[col_x], train_test_df[col_y])).T
        X_train = np.vstack((train_df_[col_x], train_df_[col_y])).T
        X_test = np.vstack((test_df_[col_x], test_df_[col_y])).T

        X_train_test = scaler.fit_transform(X_train_test)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        pca.fit(X_train_test)
        X_train = pca.transform(X_train)[:, 0]
        X_test = pca.transform(X_test)[:, 0]

        train_df['made_high_diff_corr_pca_' + col_x + '_' + col_y] = X_train
        test_df['made_high_diff_corr_pca_' + col_x + '_' + col_y] = X_test

        z[ind[0], ind[1]] = 0
        z[ind[1], ind[0]] = 0

    return train_df, test_df


def high_corr_pca(train_df, test_df, n_features=5):
    log_info('high_corr_pca')
    log_info('n_features=%d'%n_features)

    tmp = [c for c in train_df.columns if c not in ['id','target']]
    cols = []
    for s in tmp:
        if not 'made' in s and not 'cat' in s:
            cols.append(s)
    train_test_df = pd.concat([train_df[cols], test_df[cols]])

    x = train_df[cols].columns.values
    y = train_df[cols].columns.values
    z = np.abs(train_test_df.corr().values)
    z[np.isnan(z)] = 0

    for i in range(len(z)):
        z[i, i] = 0

    for i in range(n_features):
        ind = np.argwhere(z == np.max(z))[0]
        col_x = x[ind[0]]
        col_y = y[ind[1]]
        corr = z[ind[0], ind[1]]

        log_info('%d,\t(%s,\t%s)\n%f'%(i+1, col_x, col_y, corr))

        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        pca = PCA(n_components=1, random_state=41)

        X_train_test = np.vstack((train_test_df[col_x], train_test_df[col_y])).T
        X_train = np.vstack((train_df[col_x], train_df[col_y])).T
        X_test = np.vstack((test_df[col_x], test_df[col_y])).T

        X_train_test = scaler.fit_transform(X_train_test)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        pca.fit(X_train_test)
        X_train = pca.transform(X_train)[:, 0]
        X_test = pca.transform(X_test)[:, 0]

        train_df['made_high_corr_pca_' + col_x + '_' + col_y] = X_train
        test_df['made_high_corr_pca_' + col_x + '_' + col_y] = X_test

        z[ind[0], ind[1]] = 0
        z[ind[1], ind[0]] = 0

    return train_df, test_df


def all_feature_pca(train_df, test_df, n_components=5):
    log_info('all_feature_pca')
    log_info('n_components=%d'%n_components)

    tmp = [c for c in train_df.columns if c not in ['id','target']]
    cols = []
    for s in tmp:
        if not 'made' in s:
            cols.append(s)
    train_test_df = pd.concat([train_df[cols], test_df[cols]])

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    pca = PCA(n_components=n_components, random_state=41)

    X_train_test = train_test_df.as_matrix()
    X_train = train_df[cols].as_matrix()
    X_test = test_df[cols].as_matrix()

    X_train_test = scaler.fit_transform(X_train_test)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    pca.fit(X_train_test)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    for i in range(n_components):
        train_df['made_all_feature_pca_%d'%i] = X_train[:, i]
        test_df['made_all_feature_pca_%d'%i] = X_test[:, i]

    return train_df, test_df


def ps_ind_06_09_bin_count_encoding(train_df, test_df):
    log_info('ps_ind_06_09_bin_count_encoding')

    tmp = [c for c in train_df.columns if c not in ['id','target']]
    cols = []
    for s in tmp:
        if not 'made' in s:
            cols.append(s)
    train_test_df = pd.concat([train_df[cols], test_df[cols]])

    X_train = train_df['ps_ind_06_bin'] * 6
    X_test = test_df['ps_ind_06_bin'] * 6
    X_train_test = train_test_df['ps_ind_06_bin'] * 6
    for i in range(7, 10):
        X_train += train_df['ps_ind_06_bin'] * i
        X_test += test_df['ps_ind_06_bin'] * i
        X_train_test += train_test_df['ps_ind_06_bin'] * i

    cats = X_train_test.unique()
    for cat in cats:
        X_train = X_train.replace(cat, np.sum(X_train_test == cat) / len(X_train_test))
        X_test = X_test.replace(cat, np.sum(X_train_test == cat) / len(X_train_test))

    train_df['made_ps_ind_06_09_bin_count'] = X_train
    test_df['made_ps_ind_06_09_bin_count'] = X_test

    return train_df, test_df


def kinetic(row):
    probs=np.unique(row,return_counts=True)[1]/len(row)
    kinetic=np.sum(probs**2)
    return kinetic


def kinetic_feature(train_df, test_df):
    '''
    Kinetic And Transforms 0.482 UP the board
    [https://www.kaggle.com/alexandrudaia/kinetic-and-transforms-0-482-up-the-board]
    '''
    log_info('kinetic_feature')

    if not os.path.exists(os.path.join('processed', 'kinetic_train.npz')):
        first_kin_names = [col for col in train_df.columns if '_ind_' in col]
        subset_ind = train_df[first_kin_names]
        kinetic_1 = []
        for row in tqdm(range(subset_ind.shape[0])):
            row = subset_ind.iloc[row]
            k = kinetic(row)
            kinetic_1.append(k)
        second_kin_names = [col for col in train_df.columns if '_car_' in col and col.endswith('cat')]
        subset_ind = train_df[second_kin_names]
        kinetic_2 = []
        for row in tqdm(range(subset_ind.shape[0])):
            row = subset_ind.iloc[row]
            k = kinetic(row)
            kinetic_2.append(k)
        third_kin_names = [col for col in train_df.columns if '_calc_' in col and not col.endswith('bin')]
        subset_ind = train_df[second_kin_names]
        kinetic_3 = []
        for row in tqdm(range(subset_ind.shape[0])):
            row = subset_ind.iloc[row]
            k = kinetic(row)
            kinetic_3.append(k)
        fd_kin_names = [col for col in train_df.columns if '_calc_' in col and col.endswith('bin')]
        subset_ind = train_df[fd_kin_names]
        kinetic_4 = []
        for row in tqdm(range(subset_ind.shape[0])):
            row = subset_ind.iloc[row]
            k = kinetic(row)
            kinetic_4.append(k)
        train_df['made_kinetic_1'] = np.array(kinetic_1)
        train_df['made_kinetic_2'] = np.array(kinetic_2)
        train_df['made_kinetic_3'] = np.array(kinetic_3)
        train_df['made_kinetic_4'] = np.array(kinetic_4)

        np.savez(os.path.join('processed', 'kinetic_train.npz'),
            kinetic_1=np.array(kinetic_1),
            kinetic_2=np.array(kinetic_2),
            kinetic_3=np.array(kinetic_3),
            kinetic_4=np.array(kinetic_4))

    else:
        kinetic_train = np.load(os.path.join('processed', 'kinetic_train.npz'))
        train_df['made_kinetic_1'] = kinetic_train['kinetic_1']
        train_df['made_kinetic_2'] = kinetic_train['kinetic_2']
        train_df['made_kinetic_3'] = kinetic_train['kinetic_3']
        train_df['made_kinetic_4'] = kinetic_train['kinetic_4']

    if not os.path.exists(os.path.join('processed', 'kinetic_test.npz')):
        first_kin_names = [col for  col in test_df.columns  if '_ind_' in col]
        subset_ind = test_df[first_kin_names]
        kinetic_1 = []
        for row in tqdm(range(subset_ind.shape[0])):
            row = subset_ind.iloc[row]
            k = kinetic(row)
            kinetic_1.append(k)
        second_kin_names = [col for col in test_df.columns if '_car_' in col and col.endswith('cat')]
        subset_ind = test_df[second_kin_names]
        kinetic_2 = []
        for row in tqdm(range(subset_ind.shape[0])):
            row = subset_ind.iloc[row]
            k = kinetic(row)
            kinetic_2.append(k)
        third_kin_names = [col for col in test_df.columns if '_calc_' in col and not col.endswith('bin')]
        subset_ind = test_df[second_kin_names]
        kinetic_3 = []
        for row in tqdm(range(subset_ind.shape[0])):
            row = subset_ind.iloc[row]
            k = kinetic(row)
            kinetic_3.append(k)
        fd_kin_names = [col for col in test_df.columns if '_calc_' in col and col.endswith('bin')]
        subset_ind = test_df[fd_kin_names]
        kinetic_4 = []
        for row in tqdm(range(subset_ind.shape[0])):
            row = subset_ind.iloc[row]
            k = kinetic(row)
            kinetic_4.append(k)
        test_df['made_kinetic_1']=np.array(kinetic_1)
        test_df['made_kinetic_2']=np.array(kinetic_2)
        test_df['made_kinetic_3']=np.array(kinetic_3)
        test_df['made_kinetic_4']=np.array(kinetic_4)

        np.savez(os.path.join('processed', 'kinetic_test.npz'),
            kinetic_1=np.array(kinetic_1),
            kinetic_2=np.array(kinetic_2),
            kinetic_3=np.array(kinetic_3),
            kinetic_4=np.array(kinetic_4))

    else:
        kinetic_test = np.load(os.path.join('processed', 'kinetic_test.npz'))
        test_df['made_kinetic_1'] = kinetic_test['kinetic_1']
        test_df['made_kinetic_2'] = kinetic_test['kinetic_2']
        test_df['made_kinetic_3'] = kinetic_test['kinetic_3']
        test_df['made_kinetic_4'] = kinetic_test['kinetic_4']

    return train_df, test_df


def combine_continuous_features(train_df, test_df):
    log_info('combine_continuous_features')

    tmp = train_df.select_dtypes(include=['float64']).columns
    col_float = []
    for col in tmp:
        if not 'made' in col:
            col_float.append(col)

    for i in range(len(col_float)):
        for j in range(i+1, len(col_float)):
            train_df['made_plus_'+col_float[i]+'_'+col_float[j]] = train_df[col_float[i]] + train_df[col_float[j]]
            train_df['made_times_'+col_float[i]+'_'+col_float[j]] = train_df[col_float[i]] * train_df[col_float[j]]
            test_df['made_plus_'+col_float[i]+'_'+col_float[j]] = test_df[col_float[i]] + test_df[col_float[j]]
            test_df['made_times_'+col_float[i]+'_'+col_float[j]] = test_df[col_float[i]] * test_df[col_float[j]]

    return train_df, test_df


def sum_of_na(train_df, test_df):
    log_info('sum_of_na')

    tmp = [c for c in train_df.columns if c not in ['id','target']]
    cols = []
    for s in tmp:
        if not 'made' in s:
            cols.append(s)

    train_df['made_sum_of_na'] = np.sum((train_df[cols]==-1).values, axis=1)
    test_df['made_sum_of_na'] = np.sum((test_df[cols]==-1).values, axis=1)

    return train_df, test_df


def ps_car_13_x_ps_reg_03(train_df, test_df):
    log_info('ps_car_13_x_ps_reg_03')

    train_df['made_ps_car_13_x_ps_reg_03'] = train_df['ps_car_13'] * train_df['ps_reg_03']
    test_df['made_ps_car_13_x_ps_reg_03'] = test_df['ps_car_13'] * test_df['ps_reg_03']

    return train_df, test_df


def pascal_recon_ps_reg_03(train_df, test_df):
    '''
    Reconstruction of 'ps_reg_03'
    [https://www.kaggle.com/pnagel/reconstruction-of-ps-reg-03]
    '''
    log_info('pascal_recon_ps_reg_03')

    I = np.round((40*train_df['ps_reg_03'])**2)
    I = I.astype(int)
    M = (I - 1) // 31
    F = I - 31 * M
    train_df['ps_reg_03_M'] = M
    train_df['ps_reg_03_F'] = F

    I = np.round((40*test_df['ps_reg_03'])**2)
    I = I.astype(int)
    M = (I - 1) // 31
    F = I - 31 * M
    test_df['ps_reg_03_M'] = M
    test_df['ps_reg_03_F'] = F

    train_df.loc[train_df['ps_reg_03'] == -1, ('ps_reg_03_M', 'ps_reg_03_F')] = -1
    test_df.loc[test_df['ps_reg_03'] == -1, ('ps_reg_03_M', 'ps_reg_03_F')] = -1

    return train_df, test_df


def higher_than_median(train_df, test_df):
    log_info('higher_than_median')

    tmp = [c for c in train_df.columns if c not in ['id','target']]
    cols = []
    for s in tmp:
        if not 'made' in s:
            cols.append(s)

    train_test_df = pd.concat([train_df[cols], test_df[cols]])

    medians = train_test_df.median(axis=0)
    for col in cols:
        if not 'bin' in col:
            train_df['made_higher_than_median_'+col] = (train_df[col] > medians[col]).astype('int')
            test_df['made_higher_than_median_'+col] = (test_df[col] > medians[col]).astype('int')

    return train_df, test_df


def higher_than_mean(train_df, test_df):
    log_info('higher_than_mean')

    tmp = [c for c in train_df.columns if c not in ['id','target']]
    cols = []
    for s in tmp:
        if not 'made' in s:
            cols.append(s)

    train_test_df = pd.concat([train_df[cols], test_df[cols]])

    means = train_test_df.mean(axis=0)
    for col in cols:
        if not 'bin' in col:
            train_df['made_higher_than_mean_'+col] = (train_df[col] > means[col]).astype('int')
            test_df['made_higher_than_mean_'+col] = (test_df[col] > means[col]).astype('int')

    return train_df, test_df


def all_one_hot(train_df, test_df):
    log_info('higher_than_median')

    tmp = [c for c in train_df.columns if c not in ['id','target']]
    cols = []
    for s in tmp:
        if not 'made' in s:
            cols.append(s)

    train_test_df = pd.concat([train_df[cols], test_df[cols]])

    unique_vals = {col: list(train_test_df[col].unique()) for col in cols}
    for col in cols:
        if len(unique_vals[col]) > 2 and len(unique_vals[col]) < 7:
            for val in unique_vals[col]:
                train_df['made_all_one_hot_'+col+'_'+str(val)] = (train_df[col].values == val).astype('int')
                test_df['made_all_one_hot_'+col+'_'+str(val)] = (test_df[col].values == val).astype('int')

    return train_df, test_df
