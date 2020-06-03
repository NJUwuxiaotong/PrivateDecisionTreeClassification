import os

from common import constants as const
from data_process.data_preprocess import DataPreProcess


class DecisionTree(object):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, is_private=False, privacy_value=1):
        self._dataset_name = dataset_name
        self._check_file_path()
        self._training_data = None       # type: DataFrame
        self._training_data_shape = None
        self._test_data = None
        self._test_data_shape = None
        self._attributes = None          # (key, value) except of class
        self._attribute_values = None
        self._attribute_type = None      # attribute type except of class
        self._attribute_num = 0          # except of class
        self.class_att = None            # only one class name
        self.class_att_value = list()    # class value
        self.training_per = training_per
        self.test_per = test_per
        self.is_private = is_private
        self.privacy_value_per_node = privacy_value
        self._check_parameters()
        self.training_num = None
        self.test_num = None
        dataset_absolute_path = \
            const.DATASET_PATH + self._dataset_name + const.DATASET_SUFFIX
        self.data_process = DataPreProcess(dataset_absolute_path)
        self._read_data()
        self.root_node = None
        if tree_depth is None:
            self._tree_depth = int(self._attribute_num/2)
        else:
            self._tree_depth = tree_depth
        self.unit_space = "\t"
        self.training_time = 0.0
        self.test_time = 0.0

    def _check_file_path(self):
        dataset_absolute_path = const.DATASET_PATH + self._dataset_name \
                                + const.DATASET_SUFFIX
        attribute_type_file_path = \
            const.DATASET_PATH + self._dataset_name \
            + const.ATTRIBUTE_TYPE_FILE_SUFFIX

        print(dataset_absolute_path)
        if not os.path.exists(dataset_absolute_path):
            print(self._dataset_name)
            print("Error: Dataset [%s] doesn't exist." % self._dataset_name)
            exit(1)

        if not os.path.exists(attribute_type_file_path):
            print(attribute_type_file_path)
            print("Error: Attribute type file for [%s] doesn't exist." %
                  self._dataset_name)
            exit(1)

    def _check_parameters(self):
        if self.training_per + self.test_per > 1:
            print("Error: The percentage (%s, %s) of data in training and "
                  "test exceeds 1" % (self.training_per, self.test_per))
            exit(1)

    def _read_data(self):
        self.data_process.read_data_from_file()
        self._training_data = self.data_process.data
        self._training_data_shape = self.data_process.data_shape
        self.training_num = \
            int(self._training_data_shape[0] * self.training_per)
        self.test_num = int(self._training_data_shape[0] * self.test_per)
        self._test_data = self._training_data[-1*self.test_num:]
        self._training_data = self._training_data[:self.training_num]
        self._training_data_shape = tuple([self.training_num,
                                           self._training_data_shape[1]])
        self._test_data_shape = tuple([self.test_num,
                                       self._training_data_shape[1]])
        self._attributes = self.data_process.attributes[:-1]
        self._attribute_num = len(self._attributes)
        self.class_att = self.data_process.attributes[-1]
        self._attribute_values = self.data_process.attribute_values
        self.class_att_value = self._attribute_values.pop(
            self.class_att)
        self._attribute_type = self.data_process.attribute_types
        self._attribute_type.pop(self.class_att)
