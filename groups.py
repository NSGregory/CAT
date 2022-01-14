"""Takes a dataReader object and uses it to assign subjects to groups as determined by the parameters of that object
   Expected parameters to be supplied in the excel document from the end user:

   Grouped  -- TRUE / FALSE : do you have to keep the subject from a group together
   Var_range -- float : if multiple factors are being considered, how much variation in the primary factor are you
                        willing to tolerate; allows for tuning of secondary factors if multiple groups produce
                        acceptable variation in the primary factor
   Data_tabs -- csv string : which tabs should data be pulled from
   Vals -- csv string : Values that should be used to determine groupings.  Listed in order of priority
   Group_Names  -- csv string : Names of the final groups the subjects should be assigned to

   Output is a dictionary of dataframes enumerating the assignments.  If a single excel tab is analyzed, it will still
   be in the form of a dictionary.  This is to preserve the column name pairing with assignments for later use.
"""
import matplotlib.pyplot as plt
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

class assignGroups:

    def __init__(self, dataset):
        self.data = dataset

    def clean(self, items):
        """removes unwanted whitespace across a comma separated string and converts to a list
           allows for some typos and variation in end user input"""
        items = items.split(',')
        tmp = []
        for x in items:
            tmp.append(x.strip())
        return tmp

    def assign_datasets(self, datasets, vals, treatment_groups, tabs):
        """Main function of the class. Iterates through available datasets"""
        group_assignments={}
        for tab in tabs:
            group_assignments[tab] = (self.anticluster(datasets[tab], vals, treatment_groups))
        return group_assignments

    def anticluster(self, dataset, vals, groups):
        """Requires robjects library.  Uses anticlustering to assign groups.  Returns a dataframe with each row
        containing an integer that corresponds to the group it should be assigned to.

        dataset: the dataset you want to analyze and assign groups to
        vals: the values used in the analysis, must be the column headers in the datset
        groups: the group names that the data will be assigned to; as a list"""

        #subset of the original dataframe that will be used in analysis
        #any missing data will break the anticluster function
        evaluated_subset = dataset[vals]

        #eliminate non-numeric data
        numeric_subset = evaluated_subset.select_dtypes(['number']).dropna()



        #converts pandas dataframe into an r object needed for the anticluster function
        robject_dataframe = self.pandas_to_robject_dataframe(numeric_subset)

        #create anticlustering object
        anticlust = importr("anticlust")

        #dataset, number of subsets to divide into, objective, method
        #yields an R object that can't be used in python, needs to be converted
        group_assignments = anticlust.anticlustering(robject_dataframe,
                                                     len(groups),
                                                     objective="variance",
                                                     method="exchange")
        #group assignments are given as numbers
        unformatted_output = pd.DataFrame(group_assignments)

        formatted_output = self.format_anticluster_result(unformatted_output, groups)

        return formatted_output

    def format_anticluster_result(self, input, groups):
        """Converts numeric values into the actual group name provided by user.
        Input: the dataframe containing the assignment values
        Groups: the list of names corresponding to the groups

        Yields a single column dataframe with the user defined group names"""

        index = 0
        for name in groups:
            index +=1
            input.replace(index,name, inplace=True)
        input.columns=['Assignments']
        return input

    def pandas_to_robject_dataframe(self, pd_dataframe):
        with localconverter(ro.default_converter + pandas2ri.converter):
            robject_dataframe = ro.conversion.py2rpy(pd_dataframe)
        return robject_dataframe

    def merge_assignment_with_df(self, assignments, dataframes):
        """Iterates through the dict using df.join() function
           Returns a dictionary"""
        merged_dict={}
        for tab in assignments.keys():
            merged_entry=dataframes[tab].join(assignments[tab])
            merged_dict[tab]=merged_entry
        return merged_dict


# snippets
# this will break if any of the values are not numbers (NaN, text, etc)
# will produce a list
#result = anticlust.anticlustering(rdf, k=4, objective="variance", method="exchange")
# can convert list to DF
#formatted_df = pd.DataFrame(result)

# need to convert the panadas DataFrame to something R can use
#with localconverter(ro.default_converter + pandas2ri.converter):
#    r_from_pd_df = ro.conversion.py2rpy(arr)

#repr = ro.RObjectMixin.r_repr("function(x) round(colMeans(x),2))")

if __name__=='__main__':
    from data_reader import dataReader
    from heading_manager import headingManager
    from sort_data  import Sorter
    from configs import Config
    from display_data import displayData
    import seaborn as sns
    import scipy.stats as stats
    cfg = Config('configs.ini')
    data = dataReader(cfg.full_data).dataset
    groups = assignGroups(data)
    hd = headingManager()
    srt = Sorter(data)
    paw_dict = srt.paw_dict
    paw_list = [paw_dict[key] for key in paw_dict.keys()]
    paw_list = srt.remove_SD_columns(paw_list)
    tx = ['cfa', 'acid', 'ctrl']
    tmp = []
    for sublist in paw_list:
        for subsublist in sublist:
            for item in subsublist:
                tmp.append(item)

    print(tmp)
    grp = groups.anticluster(data,tmp,tx)
    assign_df = data
    g_df = assign_df.select_dtypes(['number']).dropna(axis=1)
    z_df = g_df.apply(stats.zscore)
    g_df['Group'] = grp['Assignments']
    graph = displayData(g_df)
    #graph.preview_plots(paw_dict)
    #graph.visualize_assignments(paw_dict, save=True)



    # this portion for creating a heatmap
    # z_df is a conversion of the dataframe to a z score
    # stats.zscore requires only numbers be present
    reduced_columns = srt.remove_SD_columns(z_df.columns)
    z_df = z_df[reduced_columns]
    hm_a = z_df[g_df['Group']=='acid'].dropna(axis=1)
    hm_c = z_df[g_df['Group']=='ctrl'].dropna(axis=1)
    hm_f = z_df[g_df['Group']=='cfa'].dropna(axis=1)

    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.title('acid')
    sns.heatmap(data=hm_a, vmin=-4, vmax=4)

    plt.subplot(132)
    plt.title('cfa')
    sns.heatmap(data=hm_f, vmin=-4, vmax=4)

    plt.subplot(133)
    plt.title('ctrl')
    sns.heatmap(data=hm_c, vmin=-4, vmax=4)

    plt.tight_layout()


    #base = importr("base")
    #anticlust = importr("anticlust")