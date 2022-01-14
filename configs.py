"""Pulls configuration data from configs.ini"""

from configparser import ConfigParser
import os
import ast

class Config:
    """Designed for getting lab config data"""
    def __init__(self, filename):
        #filenames
        self.filename = filename
        self.sample_data = self.get_cfg('Filepaths','sample_data')
        self.full_data = self.get_cfg('Filepaths', 'full_data')
        self.headings = self.get_cfg('Filepaths','header_pickle')
        self.headings_csv = self.get_cfg('Filepaths','header_excel')
        self.index_headings_csv = self.get_cfg('Filepaths', 'index_header_excel')

        #coordinates
        self.coordinates = self.get_options('Coordinates')
        self.nd_couplings = self.get_cfg('Coordinates', 'nd_couplings')
        self.nd_phase_dispersions = self.get_cfg('Coordinates', 'nd_phase_dispersions')
        self.nd_step_sequence = self.get_cfg('Coordinates', 'nd_step_sequence')
        self.nd_left_hind = self.get_cfg('Coordinates', 'nd_left_hind')
        self.nd_left_front = self.get_cfg('Coordinates', 'nd_left_front')
        self.nd_right_hind = self.get_cfg('Coordinates', 'nd_right_hind')
        self.nd_right_front = self.get_cfg('Coordinates', 'nd_right_front')
        self.nd_other_stats = self.get_cfg('Coordinates', 'nd_other_stats')
        self.nd_print_positions = self.get_cfg('Coordinates', 'nd_print_positions')
        self.nd_bos_mean = self.get_cfg('Coordinates', 'nd_bos_mean')
        self.nd_run_values = self.get_cfg('Coordinates', 'nd_run_values')
        self.nd_admin = self.get_cfg('Coordinates', 'nd_admin')

        #parameters for anticlustering
        self.anticluster_parameters = []

    def get_cfg(self, group, config):
        if os.path.isfile(self.filename):
            parser = ConfigParser()
            parser.read(self.filename)
            file_list = ast.literal_eval(parser.get(group, config))
            return file_list
        else:
            try:
                # explicitly define the filepath for when it is made into an executable
                bundle_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = bundle_dir+"/"+self.filename
                parser = ConfigParser()
                parser.read(full_path)
                file_list = ast.literal_eval(parser.get(group, config))
                return file_list
            except:
                print("Config file not found")
    def Files(self, filename):
        if os.path.isfile(filename):
            parser = ConfigParser()
            parser.read(filename)
            file_list = ast.literal_eval(parser.get('Filepaths', 'files'))
            return file_list
        else:
            try:
                # explicitly define the filepath for when it is made into an executable
                bundle_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = bundle_dir+"/"+filename
                parser = ConfigParser()
                parser.read(full_path)
                file_list = ast.literal_eval(parser.get('Filepaths', 'files'))
                return file_list
            except:
                print("Config file not found")

    def get_options(self, category):
        if os.path.isfile(self.filename):
            parser = ConfigParser()
            parser.read(self.filename)
            options = parser.options(category)
            return options
        else:
            try:
                # explicitly define the filepath for when it is made into an executable
                bundle_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = bundle_dir+"/"+self.filename
                parser = ConfigParser()
                parser.read(full_path)
                options = parser.options(category)
                return options
            except:
                print("Config file not found")







#for testing



if __name__ == '__main__':
    lab = Config('configs.ini')
    print(lab.headings_csv)
