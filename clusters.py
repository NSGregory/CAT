"""
Initial implementation of cluster analysis

It is not currently generalized.  Many variables will need
to be set manually if not using the original dataset.

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

def make_clean_data(dataframe):
    remove_columns = ['NumberOfRunsUsedForCalculatingTrialStatistics', 'von_Frey', 'tweezer']

    # remove SD columns
    srt = Sorter(dataframe)
    trimmed_set = srt.remove_SD_columns(dataframe)

    # drop empty placeholders
    tmp_data = dataframe[trimmed_set]
    tmp_data[tmp_data == '-'] == np.nan
    tmp_data = tmp_data.dropna(axis=1)

    #remove experiment specific columns
    experiment_specifc_data = tmp_data.drop(remove_columns, axis=1)
    group_labels = np.array(experiment_specifc_data['Group'])

    #keep only numbers
    numeric_data = experiment_specifc_data.select_dtypes(['number']).dropna(axis=1)

    return experiment_specifc_data, numeric_data, group_labels

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

def run_PCA(dataset, graph=False, save=False):
    target_names, y, X = make_target_and_data(dataset, verbose=False)
    #print(y,'\n',X,'\n', target_names)

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    if graph:
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

    return pca, X_r

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

def PCA_then_kmean_cluster(dataframe, graph=False, save=False):
    ptx_pca, ptx_X_r = run_PCA(dataframe, graph=graph, save=save)
    pcaw_ptx = pca_weighted_dataframe(ptx_pca, post_tx)
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(pcaw_ptx)
    X = pcaw_ptx.values

    if graph:
        plt.figure()
        colors = ["navy", "turquoise", "darkorange"]
        lw = 2
        cluster_name = [0, 1, 2]
        for color, i, cluster_name in zip(colors, [0, 1, 2], cluster_name):
            plt.scatter(
                X[pred_y == i, 0], X[pred_y == i, 1], color=color, alpha=0.8, lw=lw, label=cluster_name
            )
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        title = "Cluster Assignments of Dataset"
        plt.title(title)
        if save:
            safe_name = safe_filename(title)
            plt.savefig("Timepoints " + safe_name + ".png")
            plt.close()

    return kmeans, pred_y

def PCA_then_agglom_cluster(dataframe, graph=False, save=False, DT=1):
    ptx_pca, ptx_X_r = run_PCA(dataframe, graph=False, save=False)
    pcaw_ptx = pca_weighted_dataframe(ptx_pca, post_tx)
    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=DT,compute_full_tree=True).fit(pcaw_ptx)
    pred_y = cluster.labels_
    X = pcaw_ptx.values

    if graph:
        plt.figure()
        colors = ["navy", "turquoise", "darkorange"]
        lw = 2
        cluster_name = [0, 1, 2]
        for color, i, cluster_name in zip(colors, [0, 1, 2], cluster_name):
            plt.scatter(
                X[pred_y == i, 0], X[pred_y == i, 1], color=color, alpha=0.8, lw=lw, label=cluster_name
            )
        #plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        title = f"Cluster Assignments of Dataset\nDT: {DT}"
        plt.title(title)
        if save:
            safe_name = safe_filename(title)
            plt.savefig("Timepoints " + safe_name + ".png")
            plt.close()

    return cluster, pred_y

def safe_filename(self, filename):
    new_name = filename
    illegal_chars = ['/', '@', '#']
    for char in illegal_chars:
        new_name = new_name.replace(char, '_')
    return new_name

if __name__ == '__main__':
    from data_reader import dataReader
    from configs import Config
    from sort_data import Sorter
    cfg = Config('configs.ini')
    data = dataReader(cfg.full_data)
    full_data, numeric_only, group_label = make_clean_data(data.dataset)
    baseline, post_tx = split_pre_post(full_data, 'baseline', '24h post treatment')
    save = False
    graph = False
    dataframe = post_tx
    ptx_pca, ptx_X_r = run_PCA(dataframe, graph=graph, save=save)
    pcaw_ptx = pca_weighted_dataframe(ptx_pca, post_tx)
    cluster = AgglomerativeClustering().fit(pcaw_ptx)

    #PCA_then_kmean_cluster(post_tx, graph=True)
    c = 50
    while c <= 1000:
        cluster, pred_y = PCA_then_agglom_cluster(post_tx, graph=True, DT=c)
        c += 50






