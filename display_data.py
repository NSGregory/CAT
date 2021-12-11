"""For visualizing pandas dataframes
   This instance is for the GroupAlign script.  Will use settings from GroupAlign to make a series of subplots
   of each of the values of interest after the samples have been assigned to groups."""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy import stats
import numpy as np
from configs import Config

class displayData:

    def __init__(self, data):
        #dataset to initialize with
        #candidate for removal? currently using pickle files created elsewhere
        self.data = data

        #get configurations, predominantly filepaths
        configs = Config('config.ini')
        self.file_list = configs.file_list
        self.pickled_data_file = configs.pickled_data
        self.decomposed_data_file = configs.decomposed_data
        self.headings_file = configs.headings

        #data
        self.pickled_data = self.open_pickle_file(self.pickled_data_file)
        self.decomposed_data = self.open_pickle_file(self.decomposed_data_file)
        self.headings = self.open_pickle_file(self.headings_file)

        #derived variables
        self.sets_to_graph = self.make_set_to_graph()

    def scatter_from_group(self, sets_to_graph):
        """Assumes that the input is a dataframe object, potentially with multiple entries"""
        graph_data = self.data
        sets_to_graph = sets_to_graph
        for set in sets_to_graph:
            df = graph_data
            sns.set_theme(style="ticks", color_codes=True)
            num_columns = 3
            num_rows = (len(sets_to_graph)/num_columns).__ceil__() #rows*columns must be >= number of subplots
            fig, axes = plt.subplots(num_rows,
                                     num_columns,
                                     figsize=(15, 8),
                                     sharey=False) #syntax note: (2,3) is 2 rows with 3 columns
            fig.suptitle(set)
            x_place = 0
            y_place = 0
            ax_i = 0 #iterating through axes
            for subplot in sets_to_graph:

                if df[subplot].max() < 1:
                    range = 1 #to cover percentages
                else:
                    range = (df[subplot].max()*1.2).__ceil__() #make the upper limit on y axis slightly larger than max value

                plot = sns.boxplot(x="Group",
                                   ax=axes[y_place, x_place],
                            y=subplot,
                            #kind="box",
                            data=df)
                plot = sns.stripplot(x="Group",
                              y=subplot,
                              ax=axes[y_place, x_place],
                              alpha=0.7,
                              jitter=0.2,
                              color='k',
                              data=df)
                #plot.set(title=subplot)   #can't just plot all columns because each dataset will have some non-numeric columns
                plot.set_ylim(0,range)

                x_place += 1
                if x_place > (num_columns-1):
                    y_place += 1
                    x_place = 0

    def scatter_from_dict(self, dict_to_graph, x_val, y_val):
        graph_data = dict_to_graph
        num_columns = 3
        num_rows = (len(graph_data.keys())/num_columns).__ceil__() #rows*columns must be >= number of subplots
        fig, axes = plt.subplots(num_rows,
                                 num_columns,
                                 figsize=(15, 8),
                                 sharey=False) #syntax note: (2,3) is 2 rows with 3 columns
        x_place = 0
        y_place = 0

        for set in graph_data.keys():
            df = graph_data[set]
            df = self.clean_dataframe(df, x_val, y_val)
            if df[y_val].max() < 1:
                range = 1 #to cover percentages
            else:
                range = (df[y_val].max()*1.2).__ceil__() #make the upper limit on y axis slightly larger than max value
            slope, intercept, r_value, pv, se = stats.linregress(df[x_val], df[y_val])
            plot = sns.scatterplot(data=graph_data[set],
                            x=x_val,
                            y=y_val,
                            ax=axes[y_place, x_place]
                            )
            plot.set_ylim(0,range)
            plot.set_title(set)


            x_place += 1
            if x_place > (num_columns - 1):
                y_place += 1
                x_place = 0

        plt.tight_layout()

    def regplot_from_dict(self, dict_to_graph, x_val, y_val, save=False):
        graph_data = dict_to_graph
        num_columns = 3
        num_rows = (len(graph_data.keys())/num_columns).__ceil__() #rows*columns must be >= number of subplots
        fig, axes = plt.subplots(num_rows,
                                 num_columns,
                                 figsize=(15, 8),
                                 sharey=False) #syntax note: (2,3) is 2 rows with 3 columns
        x_place = 0
        y_place = 0

        for set in graph_data.keys():
            df = graph_data[set]
            #df = self.clean_dataframe(df, x_val, y_val)
            if df[y_val].max() < 1:
                range = 1 #to cover percentages
            else:
                range = (df[y_val].max()*1.2).__ceil__() #make the upper limit on y axis slightly larger than max value
            mask = ~np.isnan(df[x_val]) & ~np.isnan(df[y_val])
            slope, intercept, r_value, pv, se = stats.linregress(df[x_val][mask], df[y_val][mask])
            print(f"slope: {slope}\nintercept: {intercept}\nr_value:{r_value}")
            plot = sns.regplot(data=df,
                            x=x_val,
                            y=y_val,
                            fit_reg=True,
                            ci=None,
                            label=f"y={slope}x+{intercept}\nR^2={r_value}",
                            ax=axes[y_place, x_place]
                            )
            plot.set_ylim(0,range)
            plot.set_title(set)
            plot.legend(loc='best')


            x_place += 1
            if x_place > (num_columns - 1):
                y_place += 1
                x_place = 0

        plt.tight_layout()
        if save:
            plt.savefig("Regplot "+x_val+" by "+y_val+".png")
            plt.close()
        return


    def boxplot_from_dict(self, dict_to_graph): # not functional yet
        """Assumes that the input is a dict object, potentially with multiple entries"""
        graph_data = dict_to_graph
        columns_to_graph = dict_to_graph.keys()
        for set in graph_data.keys():
            df = graph_data[set]
            df = self.clean_dataframe(df)
            sns.set_theme(style="ticks", color_codes=True)
            num_columns = 3
            num_rows = (len(columns_to_graph)/num_columns).__ceil__() #rows*columns must be >= number of subplots
            fig, axes = plt.subplots(num_rows,
                                     num_columns,
                                     figsize=(15, 8),
                                     sharey=False) #syntax note: (2,3) is 2 rows with 3 columns
            fig.suptitle(set)
            x_place = 0
            y_place = 0
            ax_i = 0 #iterating through axes
            for subplot in columns_to_graph:

                if df[subplot].max() < 1:
                    range = 1 #to cover percentages
                else:
                    range = (df[subplot].max()*1.2).__ceil__() #make the upper limit on y axis slightly larger than max value

                plot = sns.boxplot(x="Assignments",
                                   ax=axes[y_place, x_place],
                            y=subplot,
                            #kind="box",
                            data=df)
                plot = sns.stripplot(x="Assignments",
                              y=subplot,
                              ax=axes[y_place, x_place],
                              alpha=0.7,
                              jitter=0.2,
                              color='k',
                              data=df)
                #plot.set(title=subplot)   #can't just plot all columns because each dataset will have some non-numeric columns
                plot.set_ylim(0,range)

                x_place += 1
                if x_place > (num_columns-1):
                    y_place += 1
                    x_place = 0

    def timeline_by_group(self, points_to_graph):
        """Input is the columns from the dataframe that form the points you want to graph.  Treatment groups
        are denoted by hue.

        Have to take a subsection of the total dataset (defined by points_to_graph) and then use the melt function
        to convert the column titles into variables."""

        data=self.data
        timeline = points_to_graph
        data_to_keep = ['Group'] + timeline
        line_data = data[data_to_keep]
        df = line_data.melt("Group")
        df.dropna(how='any') #drop missing data
        graphing_data = df[df.value != 999] # 999 appears to be a placeholder in the dataset
        plt.figure()
        sns.lineplot(data=graphing_data, x='variable', y='value', hue='Group', ci='sd')
        return
    def scatterplot(self, x_val, y_val, hue=None, save=False):
        data = self.data
        plt.figure()
        sns_plot = sns.scatterplot(data=data, x=x_val, y=y_val, hue=hue)
        if save:
            plt.savefig("Scatter "+x_val+" by "+y_val+".png")
            plt.close()
        return

    def regplot(self, x_val, y_val, save=False):
        data = self.data
        plt.figure()
        sns_plot = sns.regplot(data=data, x=x_val, y=y_val)
        if save:
            plt.savefig("Regplot "+x_val+" by "+y_val+".png")
            plt.close()
        return

    def boxscatter(self, x_val, y_val, save=False):
        df = self.data
        plt.figure()
        plot = sns.boxplot(x=x_val,
                           y=y_val,
                           # kind="box",
                           data=df)
        plot = sns.stripplot(x=x_val,
                             y=y_val,
                             alpha=0.7,
                             jitter=0.2,
                             color='k',
                             data=df)
        if save:
            plt.savefig("Box " + x_val + " by " + y_val + ".png")
            plt.close()

    def get_headers_from_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as handle:
            self.header_dict = pickle.load(handle)
            return self.header_dict

    def clean_dataframe(self, dataframe, x_val, y_val):
        """remove data cells that would prevent graphing the data"""
        df = dataframe
        vals = [x_val, y_val]
        for val in vals:
            series = pd.to_numeric(df[val], errors='coerce')
            clean_series = series.dropna(how='any')
            df[val]=clean_series
            df = df[df[val] !=999] #in this dataset 999 is used to indicate an error
        return df

    def output_full_stack(self, dc_data, parameter):
        """ dc_data is the decomposed dataset by group"""
        for set in self.sets_to_graph:
            a = {x: self.clean_dataframe(dc_data[x], parameter, set) for x in fc_data.keys()}
            self.regplot_from_dict(a, parameter, set, save=True)
        return

    def make_set_to_graph(self):
        #self.get_headers_from_pickle(self.headings_file)
        sums = [key + " sum" for key in self.headings.keys()]
        avgs = [key + " avg" for key in self.headings.keys()]
        cpt_times = ['CPT 30s', 'CPT 60s', 'CPT 90s', 'CPT 120s']
        vas_times = ['VAS10', 'VAS20', 'VAS30', 'VAS40', 'VAS50',
                     'VAS60', 'VAS70', 'VAS80', 'VAS90', 'VAS100'
                     ]

        return sums + ['SHS Total', 'SHS Heat Subscale', 'SHS Cold Subscale', 'CPT Stop Time (s)'] +cpt_times + vas_times

    def open_pickle_file(self, filename):
        with open(filename, 'rb') as handle:
            return pickle.load(handle)




def open_pickle_file(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)



if __name__ == '__main__':
    from rich.traceback import install
    install(show_locals=True)
    #import ipdb
    #ipdb.set_trace()
    data = pd.read_pickle('C:/Users/gregoryn/Dropbox (Personal)/Science/Projects/Choir/Grit/merged_df.pickle')
    fc_data = open_pickle_file('C:/Users/gregoryn/Dropbox (Personal)/Science/Projects/Choir/Grit/df_by_groups.pickle')
    headings_pickle = 'C:/Users/gregoryn/Dropbox (Personal)/Science/Projects/Choir/Grit/title_headings.pickle'
    groups = ['SHS Total', 'SHS Heat Subscale', 'SHS Cold Subscale','CPT Stop Time (s)']
    graph = displayData(data)
    graph.get_headers_from_pickle(headings_pickle)
    sums = [key+" sum" for key in graph.header_dict.keys() ]
    avgs = [key +" avg" for key in graph.header_dict.keys()]
    #graph.scatter_from_group(sums)
    #graph.scatter_from_group(groups)
    cpt_times = ['CPT 30s', 'CPT 60s', 'CPT 90s', 'CPT 120s']
    vas_times = ['VAS10',	'VAS20',	'VAS30',	'VAS40',	'VAS50',
                 'VAS60',	'VAS70',	'VAS80',	'VAS90',	'VAS100'
                 ]
    #graph.timeline_by_group(cpt_times)
    #graph.timeline_by_group(vas_times)

    time = ['Group'] + vas_times
    vas_dataset = data[time]
    st = 'CPT Stop Time (s)'
    # for survey in sums:
    #     graph.scatterplot('grit sum', survey, hue='Group', save=True)
    #
    # for survey in sums:
    #     graph.regplot('grit sum', survey, save=True)
    #excelWkst.save(out_filename)
    #base = importr("base")
    #anticlust = importr("anticlust")


    # unclear why self.clean_dataframe works better when separated out from the graphing function
    sets_to_graph = sums + ['SHS Total', 'SHS Heat Subscale', 'SHS Cold Subscale', 'CPT Stop Time (s)'] +cpt_times + vas_times
    #sets_to_graph = [st]
    for set in sets_to_graph:
        a = {x: graph.clean_dataframe(fc_data[x], 'bdi sum', set) for x in fc_data.keys()}
        graph.regplot_from_dict(a, 'bdi sum', set, save=True)

"""
useful snippets

df[label].replace('\D+','', regex=True, inplace=True) - remove any non digits in the column and replace with empty str
df_big[pd.to_numeric(df_big['id'], errors='coerce').notnull()

"""