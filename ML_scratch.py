"""
Machine learning and statistics
This file is meant to be a testing ground for functions in development
"""

# evaluate an lasso regression model on the dataset
# grid search hyperparameters for lasso regression
from numpy import arange
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
import numpy as np
from configs import Config
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pingouin as pg
from sklearn.preprocessing import StandardScaler

def make_target_and_data(dataset):
    """

    :param dataset: dataframe
    :return: groups(target), data
    """
    groups = dataset['Group'].unique()
    counter = 0
    assignments = np.array(dataset['Group'])
    for group in groups:
        assignments[assignments == group] = counter
        counter += 1
    data = dataset.select_dtypes(['number']).dropna(axis=1)
    return groups, assignments.astype('int'), data

def split_pre_post(df, pretitle, posttitle):
    pre_data = df[df['Time_Point']==pretitle]
    post_data = df[df['Time_Point']==posttitle]
    return pre_data, post_data

def diff_frames(frame_1, frame_2):
    groups = frame_1['Group']
    df_1 = frame_1.select_dtypes(['number']).dropna(axis=1)
    df_2 = frame_2.select_dtypes(['number']).dropna(axis=1)
    index = df_1.index
    df_2.index=index
    diff = df_2 - df_1
    diff['Group'] = groups
    return diff

def lasso(data):
    X, y = data[:, :-1], data[:, -1]
    # define model
    model = Lasso(alpha=1.0)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
    return scores

def lasso_pre_split(X,y):
    # define model
    model = Lasso(alpha=0.99)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
    return scores

def grid_search(X,y):
    # define model
    model = Lasso()
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['alpha'] = arange(0, 1, 0.01)
    # define search
    search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X, y)
    # summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)

def hyper_param(X,y):
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define model
    model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
    # fit model
    model.fit(X, y)
    # summarize chosen configuration
    print('alpha: %f' % model.alpha_)
    return(model)

def smart_append(dict, key, addition):
    #print(f"key: {key} \n keys:{dict.keys()}")
    if key in dict.keys():
        dict[key].append(addition)
        #print(f"append: {dict[key]}")
    else:
        #print(f"{key} added")
        dict[key] = [addition]
        #print(f"add {dict[key]}")
    #return dict


def get_coefficients(model, orient='columns'):
    features = model.feature_names_in_
    if 'components_' in dir(model):
        components = model.components_
    elif 'coef_'in dir(model):
        components = model.coef_
    else:
        print('Model not supported')
        return
    holding_dict = {}
    depth = len(components.shape)
    #print(depth)
    if depth > 1:
        for item in components:
            zipped_list = zip(features, item)
            list_list = [x for x in zipped_list]
            for pair in list_list:
                #print(f"p1: {pair[0]}, p2: {pair[1]}")
                smart_append(holding_dict, pair[0], pair[1])
                #print(f"{holding_dict[pair[0]]}")
    elif depth == 1:
        zipped_list = zip(features, components)
        list_list = [x for x in zipped_list]
        for pair in list_list:
            # print(f"p1: {pair[0]}, p2: {pair[1]}")
            smart_append(holding_dict, pair[0], pair[1])
            # print(f"{holding_dict[pair[0]]}")
    else:
        print("Error: invalid depth for get_coefficient function")

    component_list = []
    i = 1
    while i <= len(components):
        component_list.append(f"Factor_{i}")
        i+=1

    #return holding_dict
    return pd.DataFrame(columns=component_list).from_dict(holding_dict, orient=orient)

def LARS(X, y):
    return linear_model.Lars().fit(X, y)


def output_lasso(dataframe):
    X = dataframe.drop(['Group'], axis=1)
    X = X.select_dtypes(['number']).dropna(axis=1)
    true_X = X.drop(['NumberOfRunsUsedForCalculatingTrialStatistics', 'von_Frey', 'tweezer'], axis=1)
    y_vF = X['von_Frey']
    y_tw = X['tweezer']

    hp_vF = hyper_param(true_X, y_vF)
    name = hp_vF.feature_names_in_
    coef = hp_vF.coef_
    zp = zip(name, coef)
    dct = {p[0]: p[1] for p in zp}
    dct
    df_vF = pd.DataFrame.from_dict(dct, orient='index')
    #df_vF.to_excel('lasso_catwalk_coef_von_Frey_pain.xlsx')

    hp_tw = hyper_param(true_X, y_tw)
    name = hp_tw.feature_names_in_
    coef = hp_tw.coef_
    zp = zip(name, coef)
    dct = {p[0]: p[1] for p in zp}
    dct
    df_tw = pd.DataFrame.from_dict(dct, orient='index')
    #df_tw.to_excel('lasso_catwalk_coef_tweezer_pain.xlsx')

    return df_vF, df_tw

def lasso_by_group(dataframe):
    groups = np.unique(dataframe['Group'])
    numeric = dataframe.select_dtypes(['number']).dropna(axis=1)
    numeric['Group'] = np.array(dataframe['Group'])
    for treatment in groups:
        selected_data = numeric[numeric['Group'] == treatment]
        df_vF, df_tw = output_lasso(selected_data)
        df_vF.index.name=f'LASSO - {treatment} - von Frey'
        df_vF.columns=['von Frey']
        df_vF.to_excel(f'{treatment}_lasso_catwalk_coef_von_Frey_pain.xlsx')
        df_tw.index.name = f'LASSO - {treatment} - Tweezer'
        df_tw.columns = ['tweezer']
        df_tw.to_excel(f'{treatment}_lasso_catwalk_coef_tweezer_pain.xlsx')


    # lars = linear_model.Lars().fit(X, y)
    # c = get_coefficients(lars, orient='index')

def output_lars(dataframe):
    X = dataframe.drop(['Group'], axis=1)
    X = X.select_dtypes(['number']).dropna(axis=1)
    true_X = X.drop(['NumberOfRunsUsedForCalculatingTrialStatistics', 'von_Frey', 'tweezer'], axis=1)
    y_vF = X['von_Frey']
    y_tw = X['tweezer']

    vF_lars = LARS(true_X, y_vF)
    tw_lars = LARS(true_X, y_tw)

    df_vF = get_coefficients(vF_lars, orient='index')
    df_tw = get_coefficients(tw_lars, orient='index')

    return df_vF, df_tw

def lars_by_group(dataframe):
    groups = np.unique(dataframe['Group'])
    for treatment in groups:
        selected_data = dataframe[dataframe['Group'] == treatment]
        #print(f'{treatment} \n {selected_data[["von_Frey", "tweezer"]]}')
        df_vF, df_tw = output_lars(selected_data)
        df_vF.index.name=f'LARS - {treatment} - von Frey'
        df_vF.columns=['von Frey']
        df_vF.to_excel(f'{treatment}_LARS_catwalk_coef_von_Frey_pain.xlsx')
        df_tw.index.name = f'LARS - {treatment} - Tweezer'
        df_tw.columns = ['tweezer']
        df_tw.to_excel(f'{treatment}_lasso_LARS_coef_tweezer_pain.xlsx')



def standard_dataframe(dataframe):
    """
    uses the StandardScaler function to create a standardized dataframe, stripping out any non-numeric data
    the group assignments are retained
    :param dataframe: dataframe to be standardized
    :return: standardized dataframe
    """
    scale = StandardScaler()
    numeric = dataframe.select_dtypes(['number']).dropna(axis=1)
    std = scale.fit_transform(numeric)
    std_df = pd.DataFrame(data=std, columns=numeric.columns)
    std_df['Group'] = np.array(dataframe['Group'])
    return std_df


if __name__ == '__main__':
    from heading_manager import headingManager
    from sort_data import Sorter
    from data_reader import dataReader
    from configs import Config
    cfg = Config('configs.ini')
    data = dataReader(cfg.full_data)
    srt = Sorter(data.dataset)
    tmp_data = data.dataset
    tmp_data[tmp_data == '-'] = np.nan
    clean_data = tmp_data.dropna(axis=1)
    clean = srt.remove_SD_columns(data.dataset)
    b, p = split_pre_post(data.dataset[clean], 'baseline', '24h post treatment')
    diff = diff_frames(b,p)
    # load the dataset
    #url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
    #dataframe = read_csv(url, header=None)
    #data = dataframe.values
    p_cat_only = p.drop(['NumberOfRunsUsedForCalculatingTrialStatistics', 'von_Frey', 'tweezer'], axis=1)
    p_group = np.array(p['Group'])
    #target_names, y, X = make_target_and_data(p)
    #lasso = lasso_pre_split(X,y)
    #grid_search(X,y)
    #hp_model = hyper_param(X,y)
    #lasso_by_group(diff)
    #lars_by_group(diff)
    #lasso_by_group(p)
    #p_vf, p_tw = output_lasso(p)
    std_p = standard_dataframe(p)
    #lasso_by_group(std_p)
    #s_vf, s_tw = output_lasso(std_p)



"""
    X_process = p.drop(['NumberOfRunsUsedForCalculatingTrialStatistics', 'von_Frey', 'tweezer'], axis=1)
    X_cfa = X_process[X_process['Group']=='cfa']
    X_cfa = X_cfa.select_dtypes(['number']).dropna(axis=1)
    y_cfa = p[p['Group']=='cfa'][['von_Frey', 'tweezer']]
    lars_cfa = LARS(X_cfa,y_cfa)

    X_ctrl = X_process[X_process['Group']=='ctrl']
    X_ctrl = X_ctrl.select_dtypes(['number']).dropna(axis=1)
    y_ctrl = p[p['Group']=='ctrl'][['von_Frey', 'tweezer']]
    lars_ctrl = LARS(X_ctrl,y_ctrl)

    X_acid = X_process[X_process['Group']=='acid']
    X_acid = X_acid.select_dtypes(['number']).dropna(axis=1)
    y_acid = p[p['Group']=='acid'][['von_Frey', 'tweezer']]
    lars_acid = LARS(X_acid,y_acid)

    df_cfa = get_coefficients(lars_cfa, orient='index')
    df_ctrl = get_coefficients(lars_ctrl, orient='index')
    df_acid = get_coefficients(lars_acid, orient='index')

    df_cfa.to_excel('cfa.xlsx')
    df_acid.to_excel('acid.xlsx')
    df_ctrl.to_excel('ctrl.xlsx')
"""





"""
    Notes
    name = hp_vF.feature_names_in_
    coef = hp_vF.coef_
    zp = zip(name, coef)
    dct = {p[0] : p[1] for p in zp}
    dct
    df = pd.DataFrame.from_dict(dct, orient='index')
    df.to_excel('lasso_catwalk_coef_von_Frey_pain.xlsx")
    
    name = hp_tw.feature_names_in_
    coef = hp_tw.coef_
    zp = zip(name, coef)
    dct = {p[0] : p[1] for p in zp}
    dct
    df = pd.DataFrame.from_dict(dct, orient='index')
    df.to_excel('lasso_catwalk_coef_tweezer_pain.xlsx")
    """
"""
    hp_vF = hyper_param(X, y_vF)
    name = hp_vF.feature_names_in_
    coef = hp_vF.coef_
    zp = zip(name, coef)
    dct = {p[0]: p[1] for p in zp}
    dct
    df = pd.DataFrame.from_dict(dct, orient='index')
    df.to_excel('lasso_catwalk_coef_von_Frey_pain.xlsx')

    hp_tw = hyper_param(X, y_tw)
    name = hp_tw.feature_names_in_
    coef = hp_tw.coef_
    zp = zip(name, coef)
    dct = {p[0]: p[1] for p in zp}
    dct
    df = pd.DataFrame.from_dict(dct, orient='index')
    df.to_excel('lasso_catwalk_coef_tweezer_pain.xlsx')
"""


"""
    If you are using LASSO for feature selection, you usually employ cross-validation for selection of a lambda value
     based on your metric of interest (e.g., accuracy, logloss, etc.). Then you keep all covariates not set to zero
      based on the selected lambda value. You don't need to utilize an arbitrary percentage # feature to keep, since
       some of those may not be informative
"""