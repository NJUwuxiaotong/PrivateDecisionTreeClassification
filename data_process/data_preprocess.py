import pandas as pd

from common import constants as const


class DataPreProcess(object):
    def __init__(self, data_file_path):
        self._data_file_path = data_file_path
        self._data = None
        self._attributes = None
        self._att_types = None
        self._att_values = None
        self._data_shape = None

    def read_data_from_file(self, start_pos=0, end_pos=-1):
        try:
            print("It is noted that our system just supports \'csv\' files")
            self._data = pd.read_csv(self._data_file_path, ',')
            r_n = self._data.shape[0]
        except Exception as e:
            print("Error: Read data of file %s" % self._data_file_path)
            print("Reason: %s" % e)
            exit(1)

        if end_pos == -1 or end_pos > r_n:
            print("INFO: choose data from %s to %s" % (start_pos, r_n))
        else:
            print("INFO: choose data from %s to %s" % (start_pos, end_pos))

        self._data = self._data[start_pos:end_pos]
        updated_att_names = list()
        for att_name in self._data.columns:
            updated_att_names.append(att_name.strip())
        # remove space of each value
        self._data = self._data.applymap(
            lambda x: x.strip() if type(x) is str else x)
        self._data_shape = self._data.shape
        self._data.columns = updated_att_names
        self._attributes = self._data.columns
        self._att_types = self._data.dtypes
        self.get_value_range_of_attributes()

    def get_value_range_of_attributes(self):
        self._att_values = dict()
        for attribute in self._attributes:
            if self.attribute_types[attribute] == const.DFRAME_INT64:
                self._att_values[attribute] = \
                    {"max": self._data[attribute].max(),
                     "min": self._data[attribute].min()}
            else:
                self._att_values[attribute] = \
                    self._data[attribute].drop_duplicates(keep="first").values

    def show_statistical_info(self):
        print("------------------ DATA STATISTICS -------------------")
        self._data.info()
        print("----------------------------------------------------")
        print('--------------- Values of Attributes -----------------')
        for attribute in self._attributes:
            print("ATT: %s, Values: %s" %
                  (attribute, self._att_values[attribute]))
        print("------------------ END STATISTICS --------------------")

    def check_abnormal_data(self):
        print("INFO: There is no null data")
        print("WARNING: There are abnormal data. They are:")
        print("ATT occupation: [\'?\']")
        print("ATT native-country: [\'?\']")

    def process_abnormal_data(self):
        self.remove_abnormal_data()
        self._data_shape = self._data.shape
        self.get_value_range_of_attributes()
        print("---> The data has been cleaned.")
        self.show_statistical_info()

    def remove_abnormal_data(self):
        abnormal_index = [True] * self._data_shape[0]
        for att in self._attributes:
            abnormal_index &= (self._data[att].map(str) != "?")
        self._data = self._data[abnormal_index]

    def remove_null_data(self):
        pass

    @property
    def data(self):
        return self._data

    @property
    def attributes(self):
        return self._attributes

    @property
    def attribute_types(self):
        return self._att_types

    @property
    def attribute_values(self):
        return self._att_values

    @property
    def data_shape(self):
        return self._data_shape
