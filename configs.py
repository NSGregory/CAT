"""Pulls configuration data from configs.ini"""

from configparser import ConfigParser
import os
import ast

class Config:
    """Designed for getting lab config data"""
    def __init__(self, filename):
        self.filename = filename
        self.file_list = self.Files(filename)
        self.pickled_data =  self.get_cfg('pickled_data')
        self.decomposed_data = self.get_cfg('decomposed_data')
        self.headings = self.get_cfg('headings')
        self.headings_csv = self.get_cfg('headings_csv')

    def get_cfg(self, config):
        if os.path.isfile(self.filename):
            parser = ConfigParser()
            parser.read(self.filename)
            file_list = ast.literal_eval(parser.get('Filepaths', config))
            return file_list
        else:
            try:
                # explicitly define the filepath for when it is made into an executable
                bundle_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = bundle_dir+"/"+self.filename
                parser = ConfigParser()
                parser.read(full_path)
                file_list = ast.literal_eval(parser.get('Filepaths', config))
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







#for testing



if __name__ == '__main__':
    lab = Config('config.ini')
    print(lab.file_list)
