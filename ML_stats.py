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


import pingouin as pg
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from matplotlib.colors import ListedColormap
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
#from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


from sklearn.feature_selection import SelectFromModel
#classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import time

#understand the data
import shap
import eli5

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#kernel = 1.0 * RBF(1.0)
#gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X, y)
classifier_list = [LinearDiscriminantAnalysis(), MLPClassifier(max_iter=100000), KNeighborsClassifier(), SVC(),
                   GaussianProcessClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
                   AdaBoostClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis()]

classifier_names = ["LDA", "MLP Classifier", "Nearest Neighbor", "Support Vector", "Gaussian Process", "Decision Tree",
                    "Random Forest", "Ada Boost", "Gaussian NB", "Quadratic Discriminant"]

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
    #pc3, pc4 = pc.iloc[2], pc.iloc[3]
    #pc5, pc6, pc7, pc8, pc9, pc10 = pc.iloc[4], pc.iloc[5], pc.iloc[6], pc.iloc[7], pc.iloc[8], pc.iloc[9]
    pc1_column = []
    pc2_column = []
    pc3_column = []
    pc4_column = []
    pc5_column = []
    pc6_column = []
    pc7_column = []
    pc8_column = []
    pc9_column = []
    pc10_column = []

    for sample in df.iloc:
        weighted_pc1 = sample * pc1
        pc1_column.append(weighted_pc1.sum())
        weighted_pc2 = sample * pc2
        pc2_column.append(weighted_pc2.sum())
        # weighted_pc3 = sample * pc3
        # pc3_column.append(weighted_pc3.sum())
        # weighted_pc4 = sample * pc4
        # pc4_column.append(weighted_pc4.sum())
        # weighted_pc5 = sample * pc5
        # pc5_column.append(weighted_pc5.sum())
        # weighted_pc6 = sample * pc6
        # pc6_column.append(weighted_pc6.sum())
        # weighted_pc7 = sample * pc7
        # pc7_column.append(weighted_pc7.sum())
        # weighted_pc8 = sample * pc8
        # pc8_column.append(weighted_pc8.sum())
        # weighted_pc9 = sample * pc9
        # pc9_column.append(weighted_pc9.sum())
        # weighted_pc10 = sample * pc10
        # pc10_column.append(weighted_pc10.sum())

    data = {'pc1': pc1_column,
            'pc2': pc2_column
            # 'pc3': pc3_column,
            # 'pc4': pc4_column,
            # 'pc5': pc5_column,
            # 'pc6': pc6_column,
            # 'pc7': pc7_column,
            # 'pc8': pc8_column,
            # 'pc9': pc9_column,
            # 'pc10': pc10_column
            }

    return pd.DataFrame.from_dict(data)

def run_PCA(dataset, graph=False, save=False, components=2):
    target_names, y, X = make_target_and_data(dataset, verbose=False)
    #print(y,'\n',X,'\n', target_names)

    pca = PCA(n_components=components)
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

def run_LDA(dataset, graph=False, save=False):
    target_names, y, X = make_target_and_data(dataset, verbose=False)
    #print(y,'\n',X,'\n', target_names)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)
    colors = ["darkred", "darkorange", "navy"]
    if graph:
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        clf = make_pipeline(StandardScaler(), SVC())
        X_train, X_test, y_train, y_test = train_test_split(
            X_r2, y, test_size=0.4, random_state=42
        )
        clf.fit(X_train, y_train)
        # score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(
            clf, X_r2, cmap=plt.cm.RdYlBu, alpha=0.8, ax=ax, eps=0.5
        )

        #title = "LDA of dataset"
        #plt.title(title)
        plt.xlabel(f"Explained variance: {str(lda.explained_variance_ratio_[0]).split('.')[1][0:2]}%")
        plt.ylabel(f"Explained Variance: {str(lda.explained_variance_ratio_[1]).split('.')[1][0:2]}%")
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            ax.scatter(
                X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
            )
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        if save:
            safe_name = safe_filename(title)
            plt.savefig("Timepoints " + safe_name + ".png")
            plt.close()
        else:
            plt.show()

    return lda, X_r2

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

def random_forest_MDI(X,y,df=0, upper_limit=0, use_df=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, test_size=0.33)
    feature_names = [f"feature {i}" for i in range(X.shape[1])]
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    if use_df:
        rd_forest_importances = pd.Series(importances, index=df.columns[0:upper_limit])  # 185 vs 138
    else:
        rd_forest_importances = pd.Series(importances, index=feature_names)
    sorted_importances = rd_forest_importances.sort_values(ascending=False)
    fig, ax = plt.subplots()
    sorted_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    return sorted_importances

def analyze_rf_MDI(rd_forest_importances, threshold, df):
    keep = rd_forest_importances[rd_forest_importances >= threshold]
    keep = keep.dropna(axis=0)
    k_index = list(keep.index)
    keep_X = df[k_index] #without group
    k_index.append('Group')
    keep_df = df[k_index]
    pca, lda, Xr, Xr2 = PCA_v_LDA(keep_df)
    return pca, lda, Xr, Xr2

def random_forest_permutation_importances(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, test_size=0.33)
    start_time = time.time()
    result = permutation_importance(
        forest, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
    return forest_importances

def class_name(obj):
    return str(obj.__class__).split('.')[-1].split("'")[0]

@ignore_warnings(category=ConvergenceWarning)
def cross_val_permutation(cross_val_models, datasets, score_types):
    '''
    Assesses the quality of a range of models using leave-one-out analysis
    :param cross_val_models: a list of model objects to compare
    :param datasets: a list of dataset tuples (X, y) where X is the dataset without labels and y is the labels
    :param score_type: a list of strings to denote which scoring method(s) to use; list can be found in the manual
    for scikit-learn
    :return:
    '''
    i = 1
    for dataset in datasets:
        print(f"Dataset: {i}")
        i += 1
        X = dataset[0]
        y = dataset[1]
        for model in cross_val_models:
            for score in score_types:
                model_name = class_name(model)
                try:
                    result = np.mean(cross_val_score(model, X, y, scoring=score, cv=LeaveOneOut(), n_jobs=-1))
                    print(f"{model_name} - {score}: {result}")
                except:
                    print(f"There was an error with: {model}, {score} in this dataset")
        for score in score_types:
            kernel = 1.0 * RBF(1.0)
            gpc = GaussianProcessClassifier(kernel=kernel,random_state = 0).fit(X, y)
            result = np.mean(cross_val_score(gpc, X, y, scoring=score, cv=LeaveOneOut(), n_jobs=-1))
            print(f"GaussianProcessClassifier - {score}: {result}")
    return

def graph_classifiers(datasets, classifiers, names):
    figure = plt.figure(figsize=((len(datasets)*len(classifiers)), len(classifiers)))
    i = 1
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        print(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        print(X_train, X_test, y_train, y_test)
        print(X[:, :-1])
        x_min, x_max = X[:, :-1].min() - 0.5, X[:, :-1].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        # just plot the dataset first
        cm = plt.cm.RdYlBu
        cm_train = ListedColormap(["#961400","#966900", "#0000FF" ])
        cm_test = ListedColormap(["#FF0000","#ffea00", "#0000FF" ])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_train, marker="+", edgecolors="k")
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_test, marker = "o", alpha=0.4, edgecolors="k"
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            DecisionBoundaryDisplay.from_estimator(
                clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )
            eli5.explain_weights(clf)
            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_train, marker = "+", edgecolors="k"
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0],
                X_test[:, 1],
                c=y_test,
                cmap=cm_test,
                marker = "o",
                edgecolors="k",
                alpha=0.6,
            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                x_max - 0.3,
                y_min + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1

    plt.tight_layout()
    plt.show()

def shap_waterfall_plot(model, X, y, feature_names):
    '''
    This function will create a waterfall plot for a given model and dataset
    :param model: the model to use
    :param X: the dataset
    :param y: the labels
    :param feature_names: the names of the features
    :return: a waterfall plot
    '''
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.plots.waterfall(shap_values[0], max_display=10, feature_names=feature_names, show=False)
    plt.show()
def main():
    #import needed modules
    #do this here because this file itself may be a module some day and we don't want to always import these
    from sort_data import Sorter
    from data_reader import dataReader
    from configs import Config

    #read in data bast on configuration settings
    cfg = Config('configs.ini')
    data = dataReader(cfg.full_data)
    srt = Sorter(data.dataset)

    #process the data so it is useable by the classifiers
    tmp_data = data.dataset
    tmp_data[tmp_data == '-'] = np.nan
    clean_data = tmp_data.dropna(axis=1)
    clean = srt.remove_SD_columns(data.dataset)

    #split the data into pre and post treatment time points; this is done for each dataset
    baseline, post_treatment = split_pre_post(data.dataset[clean], 'baseline', '24h post treatment')

    #get the difference between the two time points
    diff = diff_frames(baseline, post_treatment)

    #get the z-scores of the datasets
    z_diff = z_frame(diff)
    z_baseline = z_frame(baseline)
    z_post = z_frame(post_treatment)

    #strip out the extra columns that are not part of the catwalk analysis
    cdo_post_treatment_data = post_treatment.drop(['NumberOfRunsUsedForCalculatingTrialStatistics',
                                                   'von_Frey', 'tweezer'], axis=1)
    cdo_baseline_data = baseline.drop(['NumberOfRunsUsedForCalculatingTrialStatistics',
                                                   'von_Frey', 'tweezer'], axis=1)

    #strip out the extra columns but keep the behavior data; this will skew the results of classification
    #but can be used for linear regression of behavior
    behavior_post_treatment_data = post_treatment.drop(['NumberOfRunsUsedForCalculatingTrialStatistics'], axis=1)
    behavior_baseline_data = baseline.drop(['NumberOfRunsUsedForCalculatingTrialStatistics'], axis=1)

    #standardize the data
    std_post_treatment_data = standard_dataframe(cdo_post_treatment_data)
    std_baseline_data = standard_dataframe(cdo_baseline_data)
    std_behavior_post_treatment_data = standard_dataframe(behavior_post_treatment_data)
    std_behavior_baseline_data = standard_dataframe(behavior_baseline_data)


    return

if __name__ == '__main__':
    from sort_data import Sorter
    from data_reader import dataReader
    from configs import Config
    from sklearn import datasets
    cfg = Config('configs.ini')
    #cfg.pierre_2 = cfg.get_cfg('Filepaths', 'pierre_2')
    data = dataReader(cfg.full_data)
    #data = dataReader(cfg.pierre_2)
    srt = Sorter(data.dataset)
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
    #b, p = split_pre_post(data.dataset[clean], 'Baseline', 'Drugday')
    diff = diff_frames(b,p)
    z_diff = z_frame(diff)
    z_post = z_frame(p)
    #heatmaps(z_diff)
    #pca, lda, X_r, X_r2 = PCA_v_LDA(p)
    #valid_columns = p.select_dtypes(['number']).dropna(axis=1).columns
    #tukey_df = yield_pairwise_tukey_across_dataframe(diff, valid_columns)
    #parameters = list(tukey_df['Parameter'])
    #parameters.append("Group")
    #narrowed_df = diff[parameters]

    #reduced = diff.drop(['von_Frey', 'tweezer'], axis=1)
    #pca, lda, X_r, X_r2 = PCA_v_LDA(narrowed_df)

    catwalk_data_only = p.drop(['NumberOfRunsUsedForCalculatingTrialStatistics', 'von_Frey', 'tweezer'], axis=1)
    #pca, lda, X_r, X_r2 = PCA_v_LDA(catwalk_data_only)
    #p_pca = pca_weighted_dataframe(pca, catwalk_data_only)
    #catwalk_data_only = p.drop(['NumberOfRunsUsedForCalculatingTrialStatistics'], axis=1)
    std_cdo = standard_dataframe(catwalk_data_only)
    pca, lda, X_r, X_r2 = PCA_v_LDA(std_cdo)
    pca_w_df = pca_weighted_dataframe(pca, std_cdo)
    # grp, y, X = make_target_and_data(std_cdo)
    # p_pca['Group'] = y
    #run_LDA(p_pca, graph=True)

    #p_2 = standard_dataframe(p)
    p_1 = std_cdo
    p_2 = p_1
    p_pca, p_lda, p_xr,  p_xr2 = PCA_v_LDA(p_1)
    p_grp, p_y, p_X = make_target_and_data(p_1)
    rmfdi = random_forest_MDI(p_X, p_y, upper_limit=185, df=p_2, use_df=True)
    rf_pca, rf_lda, rf_xr, rf_xr2 = analyze_rf_MDI(rmfdi, 0.02, p_2)
    p_pca_w = pca_weighted_dataframe(rf_pca, p_2[rf_lda.feature_names_in_])
    p_pca_w['Group'] = p_y
    #run_LDA(p_pca_w, graph=True)
    run_LDA(std_cdo, graph=False)

    baseline_rmfdi = p[rf_lda.feature_names_in_]
    baseline_rmfdi['Group'] = p_y
    baseline_rmfdi = standard_dataframe(baseline_rmfdi)
    final_pca, final_lda, final_X_r, final_X_r2 = PCA_v_LDA(baseline_rmfdi)
    pw_bl_rmfdi = pca_weighted_dataframe(final_pca, baseline_rmfdi)

    ppw = p_pca_w.drop(columns=['Group'])
    brm = baseline_rmfdi.drop(columns=['Group'])
    #print("BRM")
    #cross_val_permutation(classifier_list, [(brm, p_y)], ['accuracy'])
    #print("PPW")
    #cross_val_permutation(classifier_list, [(ppw, p_y)], ['accuracy'])
    #datasets = [(X_r, p_y), (X_r2, p_y), (final_X_r, p_y), (final_X_r2, p_y)]
    #graph_classifiers(datasets, classifier_list, classifier_names)
    #cross_val_permutation(classifier_list, datasets, ['accuracy'] )

    # iris = datasets.load_iris()
    # X_i = iris.data[:, :2]  # we only take the first two features.
    # y_i = iris.target
    # df_i = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # df_i['Group'] = iris.target
    # i_p, i_l, i_xr, i_xr2 = PCA_v_LDA(df_i)
    # bigX_i = iris.data
    # rfmdi = random_forest_MDI(bigX_i, y_i, df=df_i, upper_limit=4, use_df=True)
    # p,l, xr,xr2 = analyze_rf_MDI(rfmdi,0.3, df_i)
    # df_i_w = pca_weighted_dataframe(p, df_i)
    # df_i_w['Group'] = df_i['Group']
    # run_LDA(df_i_w, graph=True)

    #random_forest_MDI()

    #rd_forest_importances.plot.bar()



    #cdo_col = catwalk_data_only.columns
    #cdo_col = cdo_col[0]
    #cdo_tukey = yield_pairwise_tukey_across_dataframe(diff, cdo_col)
    #pca, lda, X_r, X_r2 = PCA_v_LDA(std_cdo)
    #tukey_df.to_excel('key_parameters.xlsx')

    #pc_frame = get_coefficients(pca)
    #pc_out = get_coefficients(pca, orient='index')
    #lda_frame = get_coefficients(lda)
    #lda_out = get_coefficients(lda, orient='index')

    #pca_w = pca_weighted_dataframe(pca, std_cdo)
    #pca_w['Group']=catwalk_data_only['Group']
    #pca_2, lda_2, X_r, X_r2 = PCA_v_LDA(pca_w)

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

"""
pd.Series(sel.estimator_.feature_importances_.ravel()).hist()

"""