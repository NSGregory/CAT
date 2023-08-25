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
        self.data = data

    def preview_plots(self, dict_to_graph, save=False):
        """
        for looking at the data before you break it down into groups
        :param dict_to_graph:
        :param save:
        :return:
        """
        figures = dict_to_graph.keys()
        for figure in figures:
            #plt.figure()
            subplot_list = dict_to_graph[figure]
            sns.set_theme(style="ticks", color_codes=True)
            num_columns = 3
            num_rows = (len(subplot_list) / num_columns).__ceil__()  # rows*columns must be >= number of subplots
            fig, axes = plt.subplots(num_rows,
                                     num_columns,
                                     figsize=(15, 8),
                                     sharey=False)  # syntax note: (2,3) is 2 rows with 3 columns
            fig.suptitle(figure)
            x_place = 0
            y_place = 0
            if num_rows > 1:
                for subplot in subplot_list:
                    df = self.data[subplot]
                    range = self.set_range(subplot)
                    plot = sns.boxplot(
                                       ax=axes[y_place, x_place],
                                       data=df)
                    plot = sns.stripplot(
                                         ax=axes[y_place, x_place],
                                     alpha=0.7,
                                     jitter=0.2,
                                     color='k',
                                     data=df)
                    plot.set_ylim(0, range)
                    plt.setp(plot.get_xticklabels(), rotation=15)

                    x_place += 1
                    if x_place >= (num_columns):
                        y_place += 1
                        x_place = 0
            else:
                for subplot in subplot_list:
                    df = self.data[subplot]
                    range = self.set_range(subplot)
                    plot = sns.boxplot(ax=axes[x_place],
                                       data=df)
                    plot = sns.stripplot(ax=axes[x_place],
                                     alpha=0.7,
                                     jitter=0.2,
                                     color='k',
                                     data=df)
                    plot.set_ylim(0, range)
                    plt.setp(plot.get_xticklabels(), rotation=15)
                    x_place += 1

            plt.tight_layout()
            if save:
                plt.savefig("Box " + figure + ".png")
                plt.close()


    def set_range(self, item):
        df = self.data
        output = 0
        #print(item)
        #print(df[item].max())
        if type(df[item].max()) == str or type(df[item].max()) == chr:
            #print(df[item].max())
            print("Something bad happened, a non-number was passed to set_range")
            pass
        elif df[item].max() < 1:
            range = df[item].max() * 1.2
        else:
            range = (df[item].max() * 1.2).__ceil__()  # make the upper limit on y axis slightly larger than max value


        return range

    def select_valid_keys(self, key_list):
        df = self.data
        keys = df.keys()
        valid = [x for x in key_list if x in keys]
        return valid



    def boxplot_from_dict(self, dict_to_graph): # not functional yet
        """
        Assumes that the input is a dict object, potentially with multiple entries.
        Each key will reference a list of parameters.  This list should be used to select pertinent columns from
        the dataframe.
        """
        graph_parameters = dict_to_graph
        parameter_subgroups = dict_to_graph.keys()
        df = self.data
        for subgroup in parameter_subgroups:
            parameter_list = graph_parameters[subgroup]
            df = self.boxplot_clean_dataframe(df, y_val=parameter_list)
            sns.set_theme(style="ticks", color_codes=True)
            num_columns = 3
            num_rows = (len(parameter_subgroups)/num_columns).__ceil__() #rows*columns must be >= number of subplots
            fig, axes = plt.subplots(num_rows,
                                     num_columns,
                                     figsize=(15, 8),
                                     sharey=False) #syntax note: (2,3) is 2 rows with 3 columns
            fig.suptitle(set)
            x_place = 0
            y_place = 0
            ax_i = 0 #iterating through axes
            for subplot in parameter_list:
                print(df[subplot])
                if df[subplot].max() < 1:
                    range = 1 #to cover percentages
                else:
                    range = (df[subplot].max()*1.2).__ceil__() #make the upper limit on y axis slightly larger than max value

                plot = sns.boxplot(y=subplot,
                                   ax=axes[y_place, x_place],
                            #kind="box",
                            data=df)
                plot = sns.stripplot(y=subplot,
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

    def visualize_assignments(self, parameter_dict, save=False):
        parameters = parameter_dict.keys()
        for parameter in parameters:
            # plt.figure()
            plot_list = parameter_dict[parameter] #l1
            for subplot_list in plot_list: #l2
                verified_subplot_list = self.select_valid_keys(subplot_list)
                print(verified_subplot_list)
                sns.set_theme(style="ticks", color_codes=True)
                num_columns = 3
                num_rows = (len(verified_subplot_list) / num_columns).__ceil__()  # rows*columns must be >= number of subplots
                fig, axes = plt.subplots(num_rows,
                                         num_columns,
                                         figsize=(15, 8),
                                         sharey=False)  # syntax note: (2,3) is 2 rows with 3 columns
                fig.suptitle(parameter)
                x_place = 0
                y_place = 0
                if num_rows > 1:
                    for subplot in verified_subplot_list: #l3
                        df = self.data
                        range = self.set_range(subplot)
                        plot = sns.boxplot(x='Group',
                                           y=subplot,
                            ax=axes[y_place, x_place],
                            data=df)
                        plot = sns.stripplot(x='Group',
                                           y=subplot,
                            ax=axes[y_place, x_place],
                            alpha=0.7,
                            jitter=0.2,
                            color='k',
                            data=df)
                        plot.set_ylim(0, range)
                        plt.setp(plot.get_xticklabels(), rotation=15)
                        x_place += 1
                        if x_place >= (num_columns):
                            y_place += 1
                            x_place = 0
                else:
                    for subplot in subplot_list:
                        df = self.data
                        range = self.set_range(subplot)
                        plot = sns.boxplot(x='Group',
                                           y=subplot,
                                            ax=axes[x_place],
                                           data=df)
                        plot = sns.stripplot(x='Group',
                                           y=subplot,
                                            ax=axes[x_place],
                                             alpha=0.7,
                                             jitter=0.2,
                                             color='k',
                                             data=df)
                        plot.set_ylim(0, range)
                        plt.setp(plot.get_xticklabels(), rotation=15)
                        x_place += 1

                plt.tight_layout()
                if save:
                    safe_name = self.safe_filename(subplot)
                    plt.savefig("Anticlustered " + safe_name + ".png")
                    plt.close()

    def visualize_timepoints(self, parameter_dict, save=False):
        print(parameter_dict)
        parameters = parameter_dict.keys()
        df = self.data
        df = self.drop_noldus_null(df)
        group_count = len(df['Group'].unique())
        for parameter in parameters:
            # plt.figure()
            plot_list = parameter_dict[parameter] #l1
            for subplot_list in plot_list: #l2
                verified_subplot_list = self.select_valid_keys(subplot_list)
                print(verified_subplot_list)
                sns.set_theme(style="ticks", color_codes=True)
                num_columns = 3
                num_rows = (len(verified_subplot_list) / num_columns).__ceil__()  # rows*columns must be >= number of subplots
                fig, axes = plt.subplots(num_rows,
                                         num_columns,
                                         figsize=(15, 8),
                                         sharey=False)  # syntax note: (2,3) is 2 rows with 3 columns
                fig.suptitle(parameter)
                x_place = 0
                y_place = 0
                if num_rows > 1:
                    for subplot in verified_subplot_list: #l3
                        range = self.set_range(subplot)
                        plot = sns.pointplot(x='Time_Point',
                                           y=subplot,
                                           hue='Group',

                            ax=axes[y_place, x_place],
                            data=df,
                            legend=False)
                        plot.get_legend().remove()
                        plot = sns.stripplot(x='Time_Point',
                                           y=subplot,
                                             hue='Group',
                            ax=axes[y_place, x_place],
                            alpha=0.7,
                            jitter=0.2,

                            data=df)
                        plot.get_legend().remove()
                        plot.set_ylim(0, range)
                        plt.setp(plot.get_xticklabels(), rotation=15)
                        x_place += 1
                        if x_place >= (num_columns):
                            y_place += 1
                            x_place = 0
                else:
                    for subplot in subplot_list:
                        #df = self.data
                        range = self.set_range(subplot)
                        plot = sns.pointplot(x='Time_Point',
                                           y=subplot,
                                           hue='Group',

                                            ax=axes[x_place],
                                             legend=False,
                                           data=df)
                        plot.get_legend().remove()
                        plot = sns.stripplot(x='Time_Point',
                                           y=subplot,
                                             hue='Group',
                                            ax=axes[x_place],
                                             alpha=0.7,
                                             jitter=0.2,

                                             data=df)

                        plot.set_ylim(0, range)
                        plot.get_legend().remove()
                        plt.setp(plot.get_xticklabels(), rotation=15)
                        x_place += 1
                h,l = plot.get_legend_handles_labels()
                fig.legend(h[0:group_count], l[0:group_count], loc='upper right', ncol=group_count)
                plt.tight_layout()
                if save:
                    safe_name = self.safe_filename(subplot)
                    plt.savefig("Timepoints " + safe_name + ".png")
                    plt.close()

    def safe_filename(self, filename):
        new_name = filename
        illegal_chars = ['/', '@', '#']
        for char in illegal_chars:
            new_name = new_name.replace(char, '_')
        return new_name

    def regplot_clean_dataframe(self, dataframe, x_val=None, y_val=None):
        """remove data cells that would prevent graphing the data"""
        df = dataframe
        vals = [x_val, y_val]
        for val in vals:
            series = pd.to_numeric(df[val], errors='coerce')
            clean_series = series.dropna(how='any')
            df[val]=clean_series
            df = df[df[val] !=999] #in this dataset 999 is used to indicate an error
        return df

    def drop_noldus_null(self, dataframe):
        df = dataframe
        output=pd.DataFrame()
        for col in df.columns:
            df = df[df[col] != '-']  # '-' is the indication for no data
            output[col] = df[col]
        return output.dropna(axis=1)

    def boxplot_clean_dataframe(self, dataframe, y_val=None):
        df = dataframe
        vals = y_val
        output = pd.DataFrame()
        for val in vals:
            series = pd.to_numeric(df[val], errors='coerce')
            clean_series = series.dropna(how='any')
            print(clean_series)
            df[val] = clean_series
            df = df[df[val] != 999]  # in this dataset 999 is used to indicate an error
            df = df[df[val] != '-']  # '-' is the indication for no data
            output[val] = df[val]
        return output

    def output_full_stack(self, dc_data, parameter, type='box'):
        if type == 'box':
            pass
        elif type == 'reg':
            for set in self.sets_to_graph:
                a = {x: self.clean_dataframe(dc_data[x], parameter, set) for x in fc_data.keys()}
                self.regplot_from_dict(a, parameter, set, save=True)
        return

    def open_pickle_file(self, filename):
        with open(filename, 'rb') as handle:
            return pickle.load(handle)




def open_pickle_file(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)



if __name__ == '__main__':
    from heading_manager import headingManager
    from sort_data import Sorter
    from data_reader import dataReader
    from configs import Config
    cfg = Config('configs.ini')
    data = dataReader(cfg.full_data)
    srt = Sorter(data.dataset)
    display_dict = srt.full_display_dict()
    tmp_data = data.dataset
    tmp_data[tmp_data == '-'] = np.nan
    clean_data = tmp_data.dropna(axis=1)
    graph = displayData(clean_data)
    #graph.preview_plots(display_dict)
    #graph.visualize_assignments(display_dict)
    graph.visualize_timepoints(display_dict, save=True)

