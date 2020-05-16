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

    def read_data_from_file(self):
        try:
            self._data = pd.read_csv(self._data_file_path, ',')[:1000]
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
        except Exception as e:
            print("Error: Read data of file %s" % self._data_file_path)
            print("Reason: %s" % e)
            exit(1)

        self._att_values = dict()
        for attribute in self._attributes:
            if self.attribute_types[attribute] == const.DFRAME_INT64:
                self._att_values[attribute] = \
                    {"max": self._data[attribute].max(),
                     "min": self._data[attribute].min()}
            else:
                self._att_values[attribute] = \
                    self._data[attribute].drop_duplicates(keep="first").values

    def check_abnormal_data(self):
        pass

    def show_statistical_info(self):
        print(self._data.info())
        print('--------------- Values of Attributes -----------------')
        for attribute in self._attributes:
            print("ATT: %s, Values: %s" %
                  (attribute, self._att_values[attribute]))

    def remove_abnormal_data(self):
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
