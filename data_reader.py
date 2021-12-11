"""Module for taking data from an excel sheet and processing it into useful pandas dataFrames"""

import pandas as pd

class dataReader:

    def __init__(self, filename):
        self.filename = filename
        self.header_skip = 0
        self.dataset = self.get_dataset()
        self.key_list = list(self.dataset.keys())

    def get_dataset(self):
        """Gets a dataframe from the given file.  If a list of files is provided, it will concatenate the dataframes.
        This will assume that the two files have the same structure, which VSC excels will."""

        if type(self.filename) == str: #if the self.filename is a single entry (a string)
            return pd.read_excel(self.filename, skiprows=self.header_skip, engine='openpyxl')

        elif type(self.filename) == list: #if self.filename is a list
            data_frame_dict = {}
            for file in self.filename:
                print(file)
                if self.extension(file) == 'xlsx' or self.extension(file) == "xls":
                    data_frame_dict[self.get_filename(file)] = pd.read_excel(file, skiprows=self.header_skip, engine='openpyxl')
                elif self.extension(file) == 'csv':
                    # some of the csv files include non-standard characters
                    # using this encoding should address that
                    data_frame_dict[self.get_filename(file)] = pd.read_csv(file, engine='python',encoding='cp1252')
                else:
                    print("Unrecognized file extension")
                    print(f"Skipping: {file}")
                    pass
            return data_frame_dict

        else:
            print("Something unexpected happened.  Make sure you entered either a string or a list.")
            exit

    def extension(self, filename):
        """Splits at a '.' and returns the second item in the list
        This makes the assumption that the filename only contains one '.'
        and that it denotes the file extension"""
        return filename.split('.')[1]

    def get_filename(self, filepath):
        """ expect something like C:/file/path/to/file.xls"""
        return filepath.split('/')[-1]






if __name__ =='__main__':
    from heading_manager import headingManager
    heading = headingManager()
    slice =heading.fetch_coord_slice(heading.coordinates.coordinates[0])
    from configs import Config
    lab = Config('configs.ini')
    data = dataReader(lab.sample_data)




