import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

from package.preprocess import fillna, dummy_encoding, count_encoding, target_encoding, drop_calc, drop_cat
from package.loss_and_metric import gini_lgb, gini_norm
from package.util import Logger
from package.feature import high_diff_corr_pca, high_corr_pca, high_2_4_diff_corr_pca, kinetic_feature
from package.feature import kinetic_feature, combine_continuous_features
from package.feature import sum_of_na, ps_car_13_x_ps_reg_03, pascal_recon_ps_reg_03
from package.feature import higher_than_median, higher_than_mean, all_one_hot


def train_predict(train_df, test_df, params, model_name=None):
    if model_name == None:
        #model_name = 'l1_lgb_%s'%datetime.now().strftime('%m%d%H%M')
        model_name = 'l1_lgb'
    log = Logger(os.path.join('log', '%s.log'%model_name))

    cols = [c for c in train_df.columns if c not in ['id','target']]

    log.info('Features:')
    for col in cols:
        log.info('- %s'%col)
    log.info('\n')

    log.info('Parameters:')
    param_items = params.items()
    for param_item in param_items:
        log.info('- %s: %s' %(param_item[0], str(param_item[1])))
    log.info('\n')

    X = train_df[cols].values
    y = train_df['target'].values
    X_test = test_df[cols].values

    prob_train = np.zeros(len(X))
    prob_test = np.zeros(len(X_test))

    kfold = 5
    scores = []
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=41)
    for i, (train_ind, valid_ind) in enumerate(skf.split(X, y)):
        X_train, X_valid = X[train_ind], X[valid_ind]
        y_train, y_valid = y[train_ind], y[valid_ind]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        model = lgb.train(params,
            lgb_train,
            num_boost_round=10000,
            valid_sets=lgb_valid,
            feval=gini_lgb,
            early_stopping_rounds=100,
            verbose_eval=50)

        prob = model.predict(X_valid, num_iteration=model.best_iteration)
        prob_train[valid_ind] = prob
        score = gini_norm(prob, y_valid)
        scores.append(score)
        log.info('- Fold %d/%d score: %f' %(i + 1, kfold, score))

        prob = model.predict(X_test, num_iteration=model.best_iteration)
        prob_test += prob / kfold

    mean_score = np.mean(scores)
    log.info('- Mean score: %f' %mean_score)

    prob_train_df = pd.DataFrame({'id': train_df['id'], 'target': prob_train})
    prob_train_df.to_csv(os.path.join('local_cv', '%s.csv.gz'%model_name),
        index=False, compression='gzip')
    prob_test_df = pd.DataFrame({'id': test_df['id'], 'target': prob_test})
    prob_test_df.to_csv(os.path.join('submission', '%s.csv.gz'%model_name),
        index=False, compression='gzip')

    return mean_score


def main():
    # Read input data
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')

    # This prints out (rows, columns) in each dataframe
    print('original train shape: %s'%str(train_df.shape))
    print('original test shape: %s'%str(test_df.shape))


    train_df, test_df = fillna(train_df, test_df)
    train_df, test_df = count_encoding(train_df, test_df, replace=False)
    train_df, test_df = target_encoding(train_df, test_df, replace=False)
    train_df, test_df = dummy_encoding(train_df, test_df)
    train_df, test_df = high_diff_corr_pca(train_df, test_df, n_features=5)
    train_df, test_df = sum_of_na(train_df, test_df)
    train_df, test_df = drop_calc(train_df, test_df)


    # This prints out (rows, columns) in each dataframe
    print('Train shape: %s'%str(train_df.shape))
    print('Test shape: %s'%str(test_df.shape))

    params = {
        'application': 'binary',
        'num_threads': 2,
        'boosting': 'gbdt',
        'max_bin': 16,
        'learning_rate': 0.025,
        'num_leaves': 52,
        'feature_fraction': 0.45,
        'bagging_fraction': 0.75,
        'bagging_freq': 16,
        'min_data_in_leaf': 740,
        'min_child_weight': 2.0}

    train_predict(train_df, test_df, params)


if __name__ == '__main__':
    main()
