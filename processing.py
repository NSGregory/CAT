import pandas as pd
from sort_data import Sorter
from data_reader import dataReader
from configs import Config
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt

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
    print(depth)
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
        print("Error")

    component_list = []
    i = 1
    while i <= len(components):
        component_list.append(f"Factor_{i}")
        i+=1

    #return holding_dict
    return pd.DataFrame(columns=component_list).from_dict(holding_dict, orient=orient)

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

def pull_data():
    cfg = Config('configs.ini')
    data = dataReader(cfg.full_data)
    srt = Sorter(data.dataset)
    clean = srt.remove_SD_columns(data.dataset)
    full_data = data.dataset[clean]
    return full_data

def diff_frames(frame_1, frame_2):
    groups = frame_1['Group']
    df_1 = frame_1.select_dtypes(['number']).dropna(axis=1)
    df_2 = frame_2.select_dtypes(['number']).dropna(axis=1)
    index = df_1.index
    df_2.index=index
    diff = df_2 - df_1
    diff['Group'] = groups
    return diff

def split_pre_post(df, pretitle, posttitle):
    pre_data = df[df['Time_Point']==pretitle]
    post_data = df[df['Time_Point']==posttitle]
    return pre_data, post_data

def to_numeric(dataframe):
    numeric_data = dataframe.select_dtypes(['number']).dropna(axis=1)
    numeric_data['Group'] = np.array(dataframe['Group'])
    return numeric_data

def normal_test(dataframe):
    #first remove non_numeric columns
    numeric = dataframe.select_dtypes(['number']).dropna(axis=1)
    features = numeric.columns
    feature_dict = {}
    for feature in features:
        feature_dict[feature] = stats.normaltest(numeric[feature]).pvalue
    return pd.DataFrame().from_dict(feature_dict, orient='index')

def split_by_normality(df):
    """

    :param dataframe: Requires that the dataframe be the product of the normal_test function
    :return: 1. dataframe that is not normallly distribuited
             2. dataframe that is normally distributed
    """
    non_normal = df[df < 0.05].dropna(axis=0)
    normal = df[df >= 0.05].dropna(axis=0)
    return non_normal, normal

def hist(df):
    """
    Cycles through a list of histograms by pressing a key
    :param df: collection of data you want to make histograms of
    :return:
    """
    numeric = df.select_dtypes(['number']).dropna(axis=1)
    keys = numeric.columns
    print(f"{len(keys)}: {keys}")
    i = 0
    keypress = "!"
    plt.ion()
    while keypress != "q":
        plt.figure()
        hist = sns.histplot(data=numeric, x=keys[i])
        plt.plot()
        plt.show()
        plt.pause(0.05)
        i += 1
        if i >= len(keys):
            break
        keypress = input("q to quit, any key to continue")
        plt.close()




if __name__ == "__main__":
    full_data = pull_data()
    pr_data, pt_data = split_pre_post(full_data, 'baseline', '24h post treatment')
    pr_data = to_numeric(pr_data)
    pt_data = to_numeric(pt_data)
    pr_normal = normal_test(pr_data)
    pt_normal = normal_test(pt_data)
    rej, normal = split_by_normality(pt_normal)
    r,n = split_by_normality(pr_normal)
    #hist(pt_data[rej.index])
