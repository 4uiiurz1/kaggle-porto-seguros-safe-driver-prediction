import sys
import os
from datetime import datetime
import random
import math

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Merge, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate, Concatenate
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras import regularizers

from package.preprocess import fillna, dummy_encoding, count_encoding, target_encoding, drop_calc, drop_cat
from package.loss_and_metric import gini_lgb, gini_norm, roc_auc_callback
from package.util import Logger
from package.feature import high_diff_corr_pca, high_corr_pca, high_2_4_diff_corr_pca
from package.feature import kinetic_feature, combine_continuous_features
from package.feature import sum_of_na, ps_car_13_x_ps_reg_03, pascal_recon_ps_reg_03
from package.feature import higher_than_median, higher_than_mean, all_one_hot


def build_model(params, train_df, test_df, emb_cols, num_cols):
    models = []
    for col in emb_cols:
        unique_vals = np.unique(pd.concat([train_df[col], test_df[col]]).values)
        model = Sequential()
        if len(unique_vals) > 100:
            model.add(Embedding(len(unique_vals), round(math.sqrt(len(unique_vals))), input_length=1))
        else:
            model.add(Embedding(len(unique_vals), (len(unique_vals)+1)//2, input_length=1))
        model.add(Flatten())
        models.append(model)

    model_num = Sequential()
    model_num.add(Lambda(lambda x: x + 0, input_shape=(len(num_cols),)))
    models.append(model_num)

    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(params['l1_out'],
        ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(params['l1_drop']))
    model.add(Dense(params['l2_out'],
        ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(params['l2_drop']))
    model.add(Dense(1, activation='sigmoid'))

    return model


def generator(X, y, batch_size):
    while 1:
        rp = np.random.permutation(len(y))
        X_tmp = X.copy()
        X = []
        for arr in X_tmp:
            X.append(arr[rp])
        y = y[rp]

        for step in range(len(y) // batch_size):
            X_batch = []
            for arr in X:
                X_batch.append(arr[step * batch_size:(step + 1) * batch_size])
            y_batch = y[step * batch_size:(step + 1) * batch_size]

            yield X_batch, y_batch


def convert(df, test_df, train_ind, valid_ind, emb_cols, num_cols):
    train_df = df.iloc[train_ind, :]
    valid_df = df.iloc[valid_ind, :]

    X_train = []
    X_valid = []
    X_test = []
    for col in emb_cols:
        raw_vals = np.unique(pd.concat([df[col], test_df[col]]))
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i
        X_train.append(train_df[col].map(val_map).values)
        X_valid.append(valid_df[col].map(val_map).values)
        X_test.append(test_df[col].map(val_map).values)

    X_train.append(train_df[num_cols].values)
    X_valid.append(valid_df[num_cols].values)
    X_test.append(test_df[num_cols].values)

    return X_train, X_valid, X_test


def train_predict(train_df, test_df, params, model_name=None):
    if model_name == None:
        #model_name = 'l1_nn_%s'%datetime.now().strftime('%m%d%H%M')
        model_name = 'l1_nn'
    log = Logger(os.path.join('log', '%s.log'%model_name))

    np.random.seed(41)
    random.seed(41)
    tf.set_random_seed(41)

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

    train_test_df = pd.concat([train_df[cols], test_df[cols]], axis=0)
    emb_cols = []
    num_cols = []
    for col in cols:
        if ('cat' in col or 'bin' in col) and not 'made' in col and len(np.unique(train_test_df[col])) >= 3:
            emb_cols.append(col)
        else:
            num_cols.append(col)

    log.info('Embedded features:')
    for col in emb_cols:
        log.info('- %s' %col)
    log.info('\n')
    log.info('Numerical features:')
    for col in num_cols:
        log.info('- %s' %col)
    log.info('\n')

    X = train_df[cols].values
    y = train_df['target'].values
    X_test = test_df[cols].values

    for col in num_cols:
        ss = StandardScaler()
        ss.fit(train_test_df[col].values.reshape(-1, 1))
        train_df[col] = ss.transform(train_df[col].values.reshape(-1, 1))[:, 0]
        test_df[col] = ss.transform(test_df[col].values.reshape(-1, 1))[:, 0]

    prob_train = np.zeros(len(X))
    prob_test = np.zeros(len(X_test))

    kfold = 5
    scores = []
    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=41)
    for i, (train_ind, valid_ind) in enumerate(skf.split(X, y)):
        y_train, y_valid = y[train_ind], y[valid_ind]

        X_train, X_valid, X_test = convert(train_df, test_df, train_ind, valid_ind, emb_cols, num_cols)

        model = build_model(params, train_df, test_df, emb_cols, num_cols)

        model.summary()

        model.compile(optimizer=Adam(lr=params['lr']),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        callbacks = [
            roc_auc_callback(training_data=(X_train, y_train), validation_data=(X_valid, y_valid)),
            EarlyStopping(monitor='norm_gini_val', min_delta=0, patience=10, mode='max'),
            ReduceLROnPlateau(monitor='norm_gini_val', factor=0.1, patience=2, mode='max', epsilon=0.0001, min_lr=1e-4, verbose=1),
            ModelCheckpoint(monitor='norm_gini_val', filepath='processed/tmp.hdf5', mode='max', verbose=1, save_best_only=True)]

        model.fit_generator(generator(X_train, y_train, params['batch_size']),
            steps_per_epoch=len(y_train)*4//params['batch_size'],
            epochs=params['n_epoch'],
            verbose=1,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks)

        model.load_weights('processed/tmp.hdf5')

        prob = model.predict_proba(X_valid, batch_size=4096)[:, 0]
        prob_train[valid_ind] = prob
        score = gini_norm(prob, y_valid)
        scores.append(score)
        log.info('- Fold %d/%d score: %f' %(i + 1, kfold, score))

        prob = model.predict_proba(X_test, batch_size=4096)[:, 0]
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
    train_df, test_df = fillna(train_df, test_df, all_feature=True)
    train_df, test_df = drop_calc(train_df, test_df)


    # This prints out (rows, columns) in each dataframe
    print('Train shape: %s'%str(train_df.shape))
    print('Test shape: %s'%str(test_df.shape))

    params = {
        'n_epoch': 10000,
        'batch_size': 64,
        'lr': 1e-3,
        'l1_out': 64,
        'l2_out': 32,
        'l1_drop': 0.3,
        'l2_drop': 0.3}

    train_predict(train_df, test_df, params)


if __name__ == '__main__':
    main()
