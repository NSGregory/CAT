
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




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



from MLdataPackage import dataPackage

class modelMaker:

    def __init__(self):
        '''
        Make sure the config file for the project is updated with the correct filepaths
        and the correct behavior labels.
        '''
        #pull data into the class
        self.package= dataPackage()
        #behavior data is a dictionary of dataframes
        self.behaviors = self.package.behaviors
        self.catwalk = self.package.catwalk_data()
        self.catwalk_std = self.package.catwalk_data(transform='standardize')

        #generate standard outputs
        self.groups, self.y, self.X = self.make_target_and_data(self.catwalk, verbose=False)
        self.pca, self.X_r_pca = self.pca_(self.catwalk, graph=False, save=False, components=2)
        self.lda, self.X_r_lda = self.LDA_(self.catwalk, graph=False, save=False)

        #set up classifiers for iteration
        self.classifier_list = [LinearDiscriminantAnalysis(), MLPClassifier(max_iter=100000), KNeighborsClassifier(), SVC(),
                           GaussianProcessClassifier(), DecisionTreeClassifier(), RandomForestClassifier(),
                           AdaBoostClassifier(), GaussianNB(), QuadraticDiscriminantAnalysis()]

        self.classifier_names = ["LDA", "MLP Classifier", "Nearest Neighbor", "Support Vector", "Gaussian Process",
                            "Decision Tree",
                            "Random Forest", "Ada Boost", "Gaussian NB", "Quadratic Discriminant"]
    def make_target_and_data(self, dataset, verbose=False):
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

    def safe_filename(self, filename):
        new_name = filename
        illegal_chars = ['/', '@', '#']
        for char in illegal_chars:
            new_name = new_name.replace(char, '_')
        return new_name

    def pca_(self, dataset, graph=False, save=False, components=2):
        '''

        :param dataset: data to be decomposed
        :param graph: boolean, whether or not to graph the results
        :param save: boolean, whether or not to save the graph
        :param components: number of components to be returned
        :return: pca, X_r
        '''
        target_names, y, X = self.make_target_and_data(dataset, verbose=False)
        # print(y,'\n',X,'\n', target_names)

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
                safe_name = self.safe_filename(title)
                plt.savefig("Timepoints " + safe_name + ".png")
                plt.close()

        return pca, X_r

    def LDA_(self, dataset, graph=False, save=False):
        target_names, y, X = self.make_target_and_data(dataset, verbose=False)
        # print(y,'\n',X,'\n', target_names)

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

            # title = "LDA of dataset"
            # plt.title(title)
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

    def class_name(self, obj):
        return str(obj.__class__).split('.')[-1].split("'")[0]
    @ignore_warnings(category=ConvergenceWarning)
    def cross_val_permutation(self, cross_val_models, datasets, score_types):
        '''
        Assesses the quality of a range of classifer models using leave-one-out analysis
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
                    model_name = self.class_name(model)
                    try:
                        result = np.mean(cross_val_score(model, X, y, scoring=score, cv=LeaveOneOut(), n_jobs=-1))
                        print(f"{model_name} - {score}: {result}")
                    except:
                        print(f"There was an error with: {model}, {score} in this dataset")
            for score in score_types:
                kernel = 1.0 * RBF(1.0)
                gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X, y)
                result = np.mean(cross_val_score(gpc, X, y, scoring=score, cv=LeaveOneOut(), n_jobs=-1))
                print(f"GaussianProcessClassifier - {score}: {result}")
        return

    def graph_classifiers(self, datasets, classifiers, names):
        figure = plt.figure(figsize=((len(datasets) * len(classifiers)), len(classifiers)))
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
            cm_train = ListedColormap(["#961400", "#966900", "#0000FF"])
            cm_test = ListedColormap(["#FF0000", "#ffea00", "#0000FF"])
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            if ds_cnt == 0:
                ax.set_title("Input data")
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_train, marker="+", edgecolors="k")
            # Plot the testing points
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_test, marker="o", alpha=0.4, edgecolors="k"
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
                #eli5.explain_weights(clf)
                # Plot the training points
                ax.scatter(
                    X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_train, marker="+", edgecolors="k"
                )
                # Plot the testing points
                ax.scatter(
                    X_test[:, 0],
                    X_test[:, 1],
                    c=y_test,
                    cmap=cm_test,
                    marker="o",
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
        return

    def test_classifiers(self):
        self.cross_val_permutation(self.classifier_list, [(self.X, self.y)], ['accuracy'])
        self.graph_classifiers([(self.X_r_pca, self.y),(self.X_r_lda, self.y)], self.classifier_list, self.classifier_names)
        return




if __name__ == '__main__':
    package = dataPackage()
    behavior = package.behavioral_data()
    catwalk = package.catwalk_data()

    m = modelMaker()
    m.test_classifiers()
