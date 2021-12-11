from heading_manager import headingManager
from data_reader import dataReader
from configs import Config


class Analyzer():
    def __init__(self, data):
        self.configs = Config('configs.ini')
        self.heading = headingManager()
        self.data = data

    def paw_analysis(self):
        """This function depends on the headingManager library's index based approach.  Changes to that library or the
        base index of headings could make this function behave in unpredictable ways.

        Using the index approach, this function performs an analysis of all four paws using the following coordinates:
        nd_left_front
        nd_right_front
        nd_left_hind
        nd_right_hind
        """
        parameter_list = ['nd_left_front', 'nd_right_front', 'nd_left_hind', 'nd_right_hind']
        df = self.dataframe_by_parameter(parameter_list)

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


        return df  # temporary to allow for interactive iteration

    def dataframe_by_parameter(self, parameter):
        data = self.data.dataset
        parameter = parameter
        columns = self.heading.parameter_to_column_header(parameter)
        return data[columns]

    def subparameter(self, super_list, subparameter_list):
        output = []
        for parameter in subparameter_list:
            matching_items = [entry for entry in super_list if parameter in entry]
            #keep this a flat list so its easier to compose the dataframe
            for item in matching_items:
                output.append(item)
        return output


if '__main__' != __name__:
    pass
else:
    configs = Config('configs.ini')
    heading = headingManager()
    filename = configs.sample_data
    dataset = dataReader(filename)
    analysis = Analyzer(dataset)
    df = analysis.paw_analysis()

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
    cols = df.columns
    sp = analysis.subparameter(cols,stride_parameters)