from heading_manager import headingManager
from data_reader import dataReader
from configs import Config


class Sorter():
    def __init__(self, data):
        self.configs = Config('configs.ini')
        self.heading = headingManager()
        self.data = data
        self.paw_dict = self.paw_sort()

    def paw_sort(self):
        """This function depends on the headingManager library's index based approach.  Changes to that library or the
        base index of headings could make this function behave in unpredictable ways.

        Using the index approach, this function performs an analysis of all four paws using the following coordinates:
        nd_left_front
        nd_right_front
        nd_left_hind
        nd_right_hind
        """
        parameter_list = ['nd_left_front', 'nd_right_front', 'nd_left_hind', 'nd_right_hind']
        superlist = self.data.columns

        #sub parameters
        #todo: consider moving these to configs
        paw_print_parameters = ['PrintLength_(mm)_Mean', 'PrintWidth_(mm)_Mean', 'PrintArea_(mm²)_Mean']
        stand_parameters = ['Stand_(s)_Mean','StandIndex_Mean']
        bodyspeed_parameters = ['BodySpeed_(mm/s)_Mean','BodySpeedVariation_(%)_Mean']
        contact_parameters = ['MaxContactAt_(%)_Mean','MaxContactArea_(mm²)_Mean', 'MaxContactMaxIntensity_Mean',
            'MaxContactMeanIntensity_Mean']
        stride_parameters = ['Swing_(s)_Mean', 'SwingSpeed_(mm/s)_Mean', 'StrideLength_(mm)_Mean', 'StepCycle_(s)_Mean']
        stance_parameters = ['SingleStance_(s)_Mean', 'InitialDualStance_(s)_Mean', 'TerminalDualStance_(s)_Mean']
        max_intensity_parameters = ['MaxIntensityAt_(%)_Mean', 'MaxIntensity_Mean','MeanIntensity_Mean']
        all_paw_parameters = [paw_print_parameters, stance_parameters, bodyspeed_parameters, contact_parameters,
                              stride_parameters, stand_parameters, max_intensity_parameters]

        paw_dict = {'paw_print': self.subparameter(superlist, paw_print_parameters),
                         'stand': self.subparameter(superlist, stand_parameters),
                         'bodyspeed': self.subparameter(superlist, bodyspeed_parameters),
                         'contact' : self.subparameter(superlist, contact_parameters),
                         'stride' : self.subparameter(superlist, stride_parameters),
                         'stance' : self.subparameter(superlist, stance_parameters),
                         'max_intensity' : self.subparameter(superlist, max_intensity_parameters)
                    }


        return paw_dict

    def parameter_sort(self, parameter):
        pass

    def dataframe_by_parameter(self, parameter):
        data = self.data.dataset
        parameter = parameter
        columns = self.heading.parameter_to_column_header(parameter)
        return data[columns]

    def subparameter(self, super_list, subparameter_list, flat=False):
        output = []
        for parameter in subparameter_list:
            matching_items = [entry for entry in super_list if parameter in entry]
            if not flat:
                output.append(matching_items)
            #can make this a flat list so its easier to compose the dataframe
            if flat:
            #keeping list format to see if it helps with previewing
                for item in matching_items:
                    output.append(item)
        return output


    def subparameter_with_depth(self, super_list, subparameter_list):
        pass

    def remove_SD_columns(self, untreated_list ):
        processed_list = []
        for entry in untreated_list:
            if "_SD" not in entry:
                processed_list.append(entry)
        return processed_list

    def clean_data(self, untreated_data):
        raw_df = untreated_data
        pass



if '__main__' != __name__:
    pass
else:
    configs = Config('configs.ini')
    heading = headingManager()
    filename = configs.sample_data
    dataset = dataReader(filename).dataset
    analysis = Sorter(dataset)
    pdict = analysis.paw_sort()

    paw_print_parameters = ['PrintLength_(mm)_Mean', 'PrintWidth_(mm)_Mean', 'PrintArea_(mm²)_Mean']
    stand_parameters = ['Stand_(s)_Mean','StandIndex_Mean']
    bodyspeed_parameters = ['BodySpeed_(mm/s)_Mean','BodySpeedVariation_(%)_Mean']
    contact_parameters = ['MaxContactAt_(%)_Mean','MaxContactArea_(mm²)_Mean', 'MaxContactMaxIntensity_Mean',
        'MaxContactMeanIntensity_Mean']
    stride_parameters = ['Swing_(s)_Mean', 'SwingSpeed_(mm/s)_Mean', 'StrideLength_(mm)_Mean', 'StepCycle_(s)_Mean']
    stance_parameters = ['SingleStance_(s)_Mean', 'InitialDualStance_(s)_Mean', 'TerminalDualStance_(s)_Mean']
    max_intensity_parameters = ['MaxIntensityAt_(%)_Mean', 'MaxIntensity_Mean','MeanIntensity_Mean']
    all_paw_parameters = [paw_print_parameters, stance_parameters, bodyspeed_parameters, contact_parameters,
                              stride_parameters, stand_parameters, max_intensity_parameters]
    cols = dataset.columns
    sp = analysis.subparameter(cols,stride_parameters)