import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

from package.preprocess import fillna, dummy_encoding, count_encoding, target_encoding, drop_calc, drop_cat
from package.loss_and_metric import gini_lgb, gini_norm
from package.util import Logger
from package.feature import high_diff_corr_pca, high_corr_pca, high_2_4_diff_corr_pca, kinetic_feature
from package.feature import kinetic_feature, combine_continuous_features
from package.feature import sum_of_na, ps_car_13_x_ps_reg_03, pascal_recon_ps_reg_03


def pd2libffm(train_df, test_df):
    test_df.insert(1,'target',0)
    x = pd.concat([train_df, test_df])
    x = x.reset_index(drop=True)

    cols = [c for c in train_df.columns if c not in ['id','target']]

    for col in cols:
        if len(x[col].unique()) > 150:
            nbins = 50
        else:
            nbins = len(x[col].unique())
        x[col] = pd.cut(x[col], nbins, labels=False)

    train_df = x.loc[:train_df.shape[0]].copy()
    test_df = x.loc[train_df.shape[0]:].copy()

    train_df.drop('id', inplace=True, axis=1)
    test_df.drop('id', inplace=True, axis=1)

    cat_cols = train_df.columns[1:]
    num_cols = []

    currentcode = len(num_cols)
    catdict = {}
    catcodes = {}
    for x in num_cols:
        catdict[x] = 0
    for x in cat_cols:
        catdict[x] = 1

    noofrows = train_df.shape[0]
    noofcolumns = len(cols)
    with open('processed/alltrainffm.txt', 'w') as text_file:
        for n, r in enumerate(tqdm(range(noofrows))):
            datastring = ''
            datarow = train_df.iloc[r].to_dict()
            datastring += str(int(datarow['target']))

            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):
                    datastring = datastring + ' '+str(i)+':'+ str(i)+':'+ str(datarow[x])
                else:
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode
                    elif(datarow[x] not in catcodes[x]):
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode

                    code = catcodes[x][datarow[x]]
                    datastring = datastring + ' '+str(i)+':'+ str(int(code))+':1'
            datastring += '\n'
            text_file.write(datastring)

    noofrows = test_df.shape[0]
    noofcolumns = len(cols)
    with open('processed/alltestffm.txt', 'w') as text_file:
        for n, r in enumerate(tqdm(range(noofrows))):
            datastring = ''
            datarow = test_df.iloc[r].to_dict()
            datastring += str(int(datarow['target']))

            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):
                    datastring = datastring + ' '+str(i)+':'+ str(i)+':'+ str(datarow[x])
                else:
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode
                    elif(datarow[x] not in catcodes[x]):
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode

                    code = catcodes[x][datarow[x]]
                    datastring = datastring + ' '+str(i)+':'+ str(int(code))+':1'
            datastring += '\n'
            text_file.write(datastring)


def train_predict(train_df, test_df, params, model_name=None):
    if model_name == None:
        model_name = 'l1_ffm_%s'%datetime.now().strftime('%m%d%H%M')
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

    pd2libffm(train_df, test_df)

    X = pd.read_csv('processed/alltrainffm.txt', sep=' ', header=0)
    y = X[X.columns[0]]

    prob_train = np.zeros(len(X))
    prob_test = np.zeros(len(test_df))

    kfold = 5
    scores = []
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=41)
    for i, (train_ind, valid_ind) in enumerate(skf.split(X, y)):
        X_train, X_valid = X.iloc[train_ind], X.iloc[valid_ind]
        X_train = X_train.sample(frac=1).reset_index(drop=True)
        y_valid = y[valid_ind]

        X_train.to_csv('processed/libffm_train.txt', sep=' ', header=0, index=False)
        X_valid.to_csv('processed/libffm_valid.txt', sep=' ', header=0, index=False)

        os.system('./libffm_gini/ffm-train -l %f -k %d -t %d -b %d -r %f -s %d -S %d -p processed/libffm_valid.txt --auto-stop processed/libffm_train.txt processed/ffm.model' \
            %(params['lambda'], params['factor'], params['iteration'], params['patience'], params['eta'], params['nr_threads'], params['seed']))
        os.system('./libffm_gini/ffm-predict processed/libffm_valid.txt processed/ffm.model processed/prob_valid.txt')
        os.system('./libffm_gini/ffm-predict processed/alltestffm.txt processed/ffm.model processed/prob_test.txt')

        prob = pd.read_csv('processed/prob_valid.txt', header=None)
        prob_train[valid_ind] = prob[prob.columns[0]].values
        score = gini_norm(prob, y_valid)
        scores.append(score)
        log.info('- Fold %d/%d score: %f' %(i + 1, kfold, score))

        prob = pd.read_csv('processed/prob_test.txt', header=None)
        prob = prob[prob.columns[0]].values
        print(len(prob))
        print(len(test_df['id']))
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
    train_df, test_df = high_diff_corr_pca(train_df, test_df, n_features=5)
    train_df, test_df = sum_of_na(train_df, test_df)
    train_df, test_df = drop_calc(train_df, test_df)

    # This prints out (rows, columns) in each dataframe
    print('Train shape: %s'%str(train_df.shape))
    print('Test shape: %s'%str(test_df.shape))

    params = {
        'lambda': 1e-5,
        'factor': 6,
        'iteration': 100,
        'patience': 10,
        'eta': 0.2,
        'nr_threads': 1,
        'seed': 41}

    train_predict(train_df, test_df, params)


if __name__ == '__main__':
    main()
