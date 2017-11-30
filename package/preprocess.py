import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE

from package.util import log_info


def fillna(train_df, test_df, all_feature=False):
    log_info('Fill NA (cat only)')

    cols = [c for c in train_df.columns if c not in ['id','target']]
    train_test_df = pd.concat([train_df[cols], test_df[cols]])

    cat_cols = []   # category val
    bin_cols = []   # binary val
    num_cols = []   # numeric val
    for s in cols:
        if 'cat' in s:
            cat_cols.append(s)
        elif 'bin' in s:
            bin_cols.append(s)
        else:
            num_cols.append(s)

    for cat_col in cat_cols:
        '''
        train_df[cat_bin_col] = train_df[cat_bin_col].replace(-1,
            train_test_df[cat_bin_col][train_test_df[cat_bin_col] != -1].mode()[0])
        test_df[cat_bin_col] = test_df[cat_bin_col].replace(-1,
            train_test_df[cat_bin_col][train_test_df[cat_bin_col] != -1].mode()[0])
        '''
        train_df[cat_col] = train_df[cat_col].replace(-1,
            train_test_df[cat_col].mode()[0])
        test_df[cat_col] = test_df[cat_col].replace(-1,
            train_test_df[cat_col].mode()[0])

    if all_feature:
        '''
        for bin_col in bin_cols:
            train_df[bin_col] = train_df[bin_col].replace(-1,
                train_test_df[bin_col][train_test_df[bin_col] != -1].mode()[0])
            test_df[bin_col] = test_df[bin_col].replace(-1,
                train_test_df[bin_col][train_test_df[bin_col] != -1].mode()[0])
        '''

        for num_col in num_cols:
            train_df[num_col] = train_df[num_col].replace(-1,
                train_test_df[num_col].median())
            test_df[num_col] = test_df[num_col].replace(-1,
                train_test_df[num_col].median())

    return train_df, test_df


def dummy_encoding(train_df, test_df):
    log_info('Dummy encoding')

    cols = [c for c in train_df.columns if c not in ['id','target']]
    cat_cols = []   # category val
    for s in cols:
        if 'cat' in s and not 'made' in s:
            cat_cols.append(s)

    train_df = pd.get_dummies(train_df, columns=cat_cols)
    test_df = pd.get_dummies(test_df, columns=cat_cols)

    return train_df, test_df


def count_encoding(train_df, test_df, replace=True):
    log_info('Count encoding')

    cols = [c for c in train_df.columns if c not in ['id','target']]
    train_test_df = pd.concat([train_df[cols], test_df[cols]])

    cat_cols = []   # category val
    for s in cols:
        if 'cat' in s and not 'made' in s:
            cat_cols.append(s)

    for cat_col in cat_cols:
        cats = train_test_df[cat_col].unique()
        for cat in cats:
            if replace:
                train_df[cat_col] = train_df[cat_col].replace(cat,
                    np.sum(train_test_df[cat_col] == cat) / len(train_test_df))
                test_df[cat_col] = test_df[cat_col].replace(cat,
                    np.sum(train_test_df[cat_col] == cat) / len(train_test_df))
            else:
                train_df['made_count_'+cat_col] = train_df[cat_col].replace(cat,
                    np.sum(train_test_df[cat_col] == cat) / len(train_test_df))
                test_df['made_count_'+cat_col] = test_df[cat_col].replace(cat,
                    np.sum(train_test_df[cat_col] == cat) / len(train_test_df))

    return train_df, test_df


def likelihood_encoding(train_df, test_df, fillna=False):
    log_info('Likelihood encoding')

    df_ = df.copy()

    cat_cols = []
    for s in list(df_.iloc[:, 2:].columns):
        if 'cat' in s:
            cat_cols.append(s)

    for cat_col in cat_cols:
        tmp = df_[cat_col]
        cats = tmp.unique()
        if fillna:
            if -1 in cats:
                tmp = tmp.replace(-1, tmp.mode()[0])
                cats = tmp.unique()
        for cat in cats:
            tmp = tmp.replace(cat, np.mean(train_df[train_df[cat_col] == cat]['target']))
        df_[cat_col] = tmp

    return df_


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encoding_(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=200,
                  smoothing=10,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def target_encoding(train_df, test_df, replace=True):
    log_info('Target encoding')

    cols = [c for c in train_df.columns if c not in ['id','target']]

    cat_cols = []   # category val
    for s in cols:
        if 'cat' in s and not 'made' in s:
            cat_cols.append(s)

    for cat_col in cat_cols:
        if replace:
            train_df[cat_col], test_df[cat_col] = target_encoding_(train_df[cat_col],
                                                test_df[cat_col],
                                                train_df['target'])
        else:
            train_df['made_target_avg_'+cat_col], test_df['made_target_avg_'+cat_col] \
                = target_encoding_(train_df[cat_col],
                                    test_df[cat_col],
                                    train_df['target'])

    return train_df, test_df


def over_sampling(X, y):
    log_info('over_sampling')

    smote = SMOTE(random_state=41)
    X, y = smote.fit_sample(X, y)

    return X, y


def drop_calc(train_df, test_df):
    log_info('drop_calc')

    cols = [c for c in train_df.columns if c not in ['id','target']]

    calc_cols = []
    for s in cols:
        if 'calc' in s and not 'made' in s:
            calc_cols.append(s)

    train_df = train_df.drop(calc_cols, axis=1)
    test_df = test_df.drop(calc_cols, axis=1)

    return train_df, test_df


def drop_cat(train_df, test_df):
    log_info('drop_cat')

    cols = [c for c in train_df.columns if c not in ['id','target']]

    calc_cols = []
    for s in cols:
        if 'cat' in s and not 'made' in s:
            calc_cols.append(s)

    train_df = train_df.drop(calc_cols, axis=1)
    test_df = test_df.drop(calc_cols, axis=1)

    return train_df, test_df
