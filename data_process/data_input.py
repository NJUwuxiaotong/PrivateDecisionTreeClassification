import os

from common import constants as const
from data_process.data_preprocess import DataPreProcess


class DataInput(object):
    """
    Function: get data from file. Besides, it also gets other information,
    including attributes and their values.
    """
    def __init__(self, dataset_name):
        self._dataset_name = dataset_name
        self._dataset_absolute_path = \
            const.DATASET_PATH + self._dataset_name + const.DATASET_SUFFIX
        self._check_file_path()
        self._data = None                # DataFrame
        self._attributes = None          # key except of class
        self._attribute_values = None    # (key, value) except of class
        self._attribute_types = None      # attribute type except of class
        self._attribute_num = 0          # except of class
        self._class_attribute = None     # only one class name
        self._class_label = list()       # class value
        self._range_of_int64_attributes = None
        self.data_process = DataPreProcess(self._dataset_absolute_path)

    def _check_file_path(self):
        print("Info: The absolute path of input file is %s" %
              self._dataset_absolute_path)
        if not os.path.exists(self._dataset_absolute_path):
            print("Error: Dataset [%s] doesn't exist." % self._dataset_name)
            exit(1)

        """
        attribute_type_file_path = \
            const.DATASET_PATH + self._dataset_name \
            + const.ATTRIBUTE_TYPE_FILE_SUFFIX
        if not os.path.exists(attribute_type_file_path):
            print("Error: Attribute type file for [%s] doesn't exist." %
                  self._dataset_name)
            exit(1)
        """

    def read_data(self):
        self.data_process.read_data_from_file()
        self.data_process.process_abnormal_data()
        self._data = self.data_process.data
        self._attributes = self.data_process.attributes[:-1]
        self._attributes = self._attributes.tolist()
        self._attribute_num = len(self._attributes)
        self._class_attribute = self.data_process.attributes[-1]
        self._attribute_values = self.data_process.attribute_values
        self._class_label = self._attribute_values.pop(
            self._class_attribute)
        self._attribute_types = self.data_process.attribute_types
        self._attribute_types.pop(self._class_attribute)
        self.get_range_of_int64_attributes()

    def get_range_of_int64_attributes(self):
        self._range_of_int64_attributes = dict()
        for att in self._attributes:
            if self._attribute_types[att] == const.DFRAME_INT64:
                values = self._data[att].values
                max_value = values.max()
                min_value = values.min()
                self._range_of_int64_attributes[att] = [min_value * 0.75,
                                                        max_value * (1+0.25)]
