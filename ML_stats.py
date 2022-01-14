"""
Machine learning and statistics
This file is intended to the be the destination for stable functions and methods

"""

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

def make_target_and_data(dataset, verbose=False):
    """

    :param dataset: dataframe
    :return: groups(target), data
    """
    groups = dataset['Group'].unique()
    counter = 0
    assignments = np.array(dataset['Group'])

    for group in groups:
        assignments[assignments == group] = counter
        if verbose:
            print(f"{group} -> {counter}")
        counter += 1
    data = dataset.select_dtypes(['number']).dropna(axis=1)
    return groups, assignments.astype('int'), data

def PCA_v_LDA(dataset, save=False):

    target_names, y, X = make_target_and_data(dataset, verbose=True)
    #print(y,'\n',X,'\n', target_names)

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    title = "PCA of Dataset"
    plt.title(title)
    if save:
        safe_name = safe_filename(title)
        plt.savefig("Timepoints " + safe_name + ".png")
        plt.close()

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    title = "LDA of dataset"
    plt.title(title)

    if save:
        safe_name = safe_filename(title)
        plt.savefig("Timepoints " + safe_name + ".png")
        plt.close()
    else:
        plt.show()

    return pca, lda, X_r, X_r2


def safe_filename(self, filename):
    new_name = filename
    illegal_chars = ['/', '@', '#']
    for char in illegal_chars:
        new_name = new_name.replace(char, '_')
    return new_name

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

def z_frame(dataframe):
    group = dataframe['Group']
    df = dataframe.select_dtypes(['number']).dropna(axis=1)
    df = df.apply(stats.zscore)
    df['Group'] = group
    return df

def heatmaps(dataframe):
    groups = list(dataframe['Group'].unique())
    num_columns = 2
    num_rows = (len(groups) / num_columns).__ceil__()
    fig, axes = plt.subplots(num_rows,
                             num_columns,
                             figsize=(18, 10),
                             sharey=False)  # syntax note: (2,3) is 2 rows with 3 columns
    x_place = 0
    y_place = 0
    for group in groups:
        select_frame = dataframe[dataframe['Group'] == group].select_dtypes(['number']).dropna(axis=1)
        plot = sns.heatmap(select_frame,
                           ax=axes[y_place, x_place])
        x_place += 1
        if x_place > (num_columns - 1):
            y_place += 1
            x_place = 0
        plt.setp(plot.get_xticklabels(), rotation=25)
        plot.set_title(f"{group}")

    plt.tight_layout()

def flatten_display_dict(display_dict, target_dataframe):
    """Display dict is a specific product of the sort_data library.  It makes a dict with multiple
    layers of nesting.  This will reduce the nesting to a single layer list.
    In the final step it will eliminate any members of the flat list that are not present in the target dataframe"""

    flat = [display_dict[key] for key in display_dict.keys()]
    tmp = []
    for nested_list in flat:
        for item in nested_list:
            tmp.append(item)
    tmp_2 = []
    for nested_list in tmp:
        for item in nested_list:
            tmp_2.append(item)

    tmp_3 = [x for x in tmp_2 if x in target_dataframe.columns]
    return tmp_3

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


def yield_pairwise_tukey_across_dataframe(target_dataframe, target_columns, grouping_column="Group"):
    sig = {}
    tukey = {}
    numeric_only_frame = target_dataframe[target_columns]
    non_zero_list = [item for item in target_columns if sum(numeric_only_frame[item]) != 0 ]
    for item in non_zero_list:
        print(item)

        anova_result = pg.anova(target_dataframe, item, between=grouping_column)

        # result is a nested list
        # index values [0][4] currently point to p-value
        p_value = anova_result.values[0][4]
        if p_value < 0.05:
            sig[item] = p_value
            # pht: "post hoc tukey"
            pht = pg.pairwise_tukey(target_dataframe, item, between=grouping_column)
            i = 0
            smart_append(tukey, 'Parameter', item)
            smart_append(tukey, 'p-value', p_value)

            while i < len(pht):
                # column indices ('A', 'B', 'p-tukey') are specified by the pg.pairwise_tukey function
                col = pht['A'][i] + " vs " + pht['B'][i]
                val = pht['p-tukey'][i]
                smart_append(tukey, col, val)
                i += 1
            # print(pht)

    ## make it into a more useful dataframe format
    # sig_df = pd.DataFrame().from_dict(sig, orient='index') # deprecated by more informative tukey_df
    tukey_df = pd.DataFrame().from_dict(tukey, orient='columns')
    return tukey_df

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

def pca_weighted_dataframe(pca, dataframe):
    pca = pca
    df = dataframe
    pc = get_coefficients(pca)
    pc1, pc2 = pc.iloc[0], pc.iloc[1]
    pc1_column = []
    pc2_column = []
    for sample in df.iloc:
        weighted_pc1 = sample * pc1
        pc1_column.append(weighted_pc1.sum())
        weighted_pc2 = sample * pc2
        pc2_column.append(weighted_pc2.sum())

    data = {'pc1': pc1_column,
            'pc2': pc2_column}

    return pd.DataFrame(data)



if __name__ == '__main__':
    from heading_manager import headingManager
    from sort_data import Sorter
    from data_reader import dataReader
    from configs import Config
    cfg = Config('configs.ini')
    data = dataReader(cfg.full_data)
    srt = Sorter(data.dataset)
    display_dict = srt.paw_dict
    tmp_data = data.dataset
    tmp_data[tmp_data == '-'] = np.nan
    clean_data = tmp_data.dropna(axis=1)
    #graph = displayData(clean_data)
    #graph.preview_plots(display_dict)
    #graph.visualize_assignments(display_dict)
    #pca, lda, X_r, X_r2 = PCA_v_LDA(data.dataset)

    ##make some heatmaps for an overview
    clean = srt.remove_SD_columns(data.dataset)
    b, p = split_pre_post(data.dataset[clean], 'baseline', '24h post treatment')
    #b, p = split_pre_post(data.dataset[clean], 'Baseline', 'During drug')
    diff = diff_frames(b,p)
    z_diff = z_frame(diff)
    #heatmaps(z_diff)
    #pca, lda, X_r, X_r2 = PCA_v_LDA(diff)

    valid_columns = np.unique(flatten_display_dict(display_dict, diff))
    #tukey_df = yield_pairwise_tukey_across_dataframe(diff, valid_columns)
    #parameters = list(tukey_df['Parameter'])
    #parameters.append("Group")
    #narrowed_df = diff[parameters]

    #reduced = diff.drop(['von_Frey', 'tweezer'], axis=1)
    #pca, lda, X_r, X_r2 = PCA_v_LDA(narrowed_df)

    catwalk_data_only = diff.drop(['NumberOfRunsUsedForCalculatingTrialStatistics', 'von_Frey', 'tweezer'], axis=1)
    cdo_col = catwalk_data_only.columns
    cdo_col = cdo_col[0]
    cdo_tukey = yield_pairwise_tukey_across_dataframe(diff, cdo_col)
    #pca, lda, X_r, X_r2 = PCA_v_LDA(catwalk_data_only)
    #tukey_df.to_excel('key_parameters.xlsx')

    #pc_frame = get_coefficients(pca)
    #pc_out = get_coefficients(pca, orient='index')
    #lda_frame = get_coefficients(lda)
    #lda_out = get_coefficients(lda, orient='index')

    #pca_w = pca_weighted_dataframe(pca, catwalk_data_only)
    #pca_w['Group']=catwalk_data_only['Group']
    #pca, lda_2, X_r, X_r2 = PCA_v_LDA(pca_w)

    #pca_to_lda_out = get_coefficients(lda_2, orient='index')

    #pc_out.to_excel('PCA_catwalk_pain.xlsx')
    #lda_out.to_excel('LDA_raw_data_catwalk_pain.xlsx')
    #pca_to_lda_out.to_excel('PCA_to_LDA_catwalk_pain.xlsx')









    """
    Notes:
    # How to make correlogram
    cor = diff.corr()
    keep = cor[abs(cor) >= 0.75]
    matrix = np.triu(keep)
    sns.heatmap(keep, mask=matrix, cmap="coolwarm")
    
    # How to test dataframe wide anova
    ## Start by getting valid columns from the dictionary
    ## This produces a nested list of multiple levels
    flat = [display_dict[key] for key in display_dict.keys()]
    tmp = []
    for list in flat:
        for item in list:
          tmp.append(item)
    tmp_2 = []
    for list in tmp:
        for item in list:
            tmp_2.append(item)
    
    ## The nested lists are now a simple list of strings
    ## Confirm that each entry in the list of strings is a 
    ## valid column in the dataframe to be used
    tmp_3 = [x for x in tmp_2 if x in diff.columns]
    def smart_append(dict, key, addition):
        if key in dict.keys():
            dict[key].append(addition)
        else:
          dict[key] = [addition] 
    ## Do the stats and assign significant values to the sig hash table
    sig = {}
    tukey = {}
    for item in tmp_3:
        a = pg.anova(diff, item, between="Group")
        #print(a)
        p_value = a.values[0][4]
        if p_value < 0.05:
            sig[item] = p_value
            print(f"{item} is sig")
            pht = pg.pairwise_tukey(diff, item, between="Group")
            i = 0
            dct = {}
            #dct['Parameter'] = item
            #dct['p-value'] = p_value
            smart_append(tukey, 'Parameter', item)
            smart_append(tukey, 'p-value', p_value)

            while i < len(pht):
                col = pht['A'][i] + " vs " + pht['B'][i]
                val = pht['p-tukey'][i]
                print(f'{col} / {val}')
                smart_append(tukey, col, val)
                i += 1            
            print(pht)
    
    ## make it into a more useful dataframe format        
    sig_df = pd.DataFrame().from_dict(sig, orient='index')
    tukey_df = pd.DataFrame().from_dict(tukey, orient='index')
    
    lt = []
    pc1 = pc.iloc[0]
    for x in reduced.iloc:
        v = x*pc1
        lt.append(v.sum())
    
    rt = []
    pc2 = pc.iloc[1]
        for x in reduced.iloc:
        v = x*pc2
        rt.append(v.sum())
    """