import pandas as pd


class DataClean(object):
    def __init__(self, data_file_path):
        self._data_file_path = data_file_path

    def read_data_from_file(self):
        try:
            self._data = pd.read_csv(self._data_file_path, ',')
            updated_att_names = list()
            for att_name in self._data.columns:
                updated_att_names.append(att_name.strip())
            self._data.columns = updated_att_names
            return self._data
        except:
            print("Error: Read data of file %s" % self._data_file_path)
            exit(1)

    def remove_abnormal_data(self):
        pass

    def strip_space_data(self, data):
        data = data.applymap(lambda x: x.strip() if type(x) is str else x)
        return data

    def get_atts_from_file(self):
        pass
