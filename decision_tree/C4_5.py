import math
import os
import pandas as pd

from common import constants as const
from data_process.data_preprocess import DataPreProcess
from decision_tree.C4_5_node import NonLeafNode, LeafNode


class C45(object):
    def __init__(self, dataset_name, tree_depth=None):
        self._dataset_name = dataset_name
        self._check_file_path()
        self._data = None
        self._data_shape = None
        self._attributes = None   # (key, value) except of class
        self._attribute_values = None
        self._attribute_type = None    # attribute type except of class
        self._attribute_num = 0        # except of class
        self.att_class = None          # only one class name
        self.att_class_value = list()    # class value
        self._read_data()
        self.root_node = None
        if tree_depth is None:
            self._tree_depth = int(self._attribute_num/2)
        else:
            self._tree_depth = tree_depth
        self.unit_space = "\t"

    def _check_file_path(self):
        dataset_absolute_path = const.DATASET_PATH + self._dataset_name \
                                + const.DATASET_SUFFIX
        attribute_type_file_path = const.DATASET_PATH + self._dataset_name \
                                   + const.ATTRIBUTE_TYPE_FILE_SUFFIX

        if not os.path.exists(dataset_absolute_path):
            print(self._dataset_name)
            print("Error: Dataset [%s] doesn't exist." % self._dataset_name)
            exit(1)

        if not os.path.exists(attribute_type_file_path):
            print(attribute_type_file_path)
            print("Error: Attribute type file for [%s] doesn't exist." %
                  self._dataset_name)
            exit(1)

    def _read_data(self):
        dataset_absolute_path = const.DATASET_PATH + self._dataset_name \
                                + const.DATASET_SUFFIX
        data_process = DataPreProcess(dataset_absolute_path)
        data_process.read_data_from_file()
        self._data = data_process.data
        self._data_shape = data_process.data_shape
        self._attributes = data_process.attributes[:-1]
        self._attribute_num = len(self._attributes)
        self.att_class = data_process.attributes[-1]
        self._attribute_values = data_process.attribute_values
        self.att_class_value = self._attribute_values.pop(
            self.att_class)
        self._attribute_type = data_process.attribute_types
        self._attribute_type.pop(self.att_class)

    def _information(self, d_usage):
        information = 0
        total_num = sum(d_usage)
        for class_value in self.att_class_value:
            sub_usage = d_usage & (self._data[self.att_class] == class_value)
            sub_num = sum(sub_usage)
            if sub_num != 0:
                p = sub_num/total_num
                information -= p * math.log2(p)
        return information

    def _information_after_division(self, d_usage, att):
        """
        param d_usage:
        param att:
        return [gain_info, overcomes, usages]
        """
        total_num = sum(d_usage)
        overcomes = list()
        usages = list()
        if self._attribute_type[att] == const.DFRAME_INT64:
            can_info = 1000000
            can_threshold = 0
            unique_values = self._data[att][d_usage].drop_duplicates(
                keep='first').values
            unique_values.sort()

            if len(unique_values) == 1:
                return self._information(d_usage), \
                       "=="+str(unique_values[0]), \
                       [d_usage, None]

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i+1])/2
                l_usage = d_usage & (self._data[att] < threshold)
                r_usage = d_usage & (self._data[att] >= threshold)
                p_info = (sum(l_usage)*self._information(l_usage) +
                          sum(r_usage)*self._information(r_usage))/total_num
                if p_info < can_info:
                    can_info = p_info
                    can_threshold = threshold
            return can_info, \
                   ["<<"+str(can_threshold), ">="+str(can_threshold)], \
                   [l_usage, r_usage]
        else:
            gain_info = 0
            for value in self._attribute_values[att]:
                sub_usage = d_usage & (self._data[att] == value)
                sub_num = sum(sub_usage)
                if sub_num != 0:
                    gain_info += sub_num/total_num*self._information(sub_usage)
                    overcomes.append(value)
                    usages.append(sub_usage)
            return gain_info, overcomes, usages

    def _select_split_att(self, d_usage):
        split_att = None
        sub_usages = None
        overcomes = None

        gain_info = self._information(d_usage)
        info_gain = 10000
        for att in self._attributes:
            can_info_gain, can_overcomes, can_sub_usages = \
                self._information_after_division(d_usage, att)

            if can_info_gain < info_gain:
                info_gain = can_info_gain
                split_att = att
                sub_usages = can_sub_usages
                overcomes = can_overcomes
        gain_info -= info_gain
        return info_gain, split_att, overcomes, sub_usages

    def set_leaf_node(self, usage):
        leaf_node = LeafNode(is_leaf=True)
        for label in self.att_class_value:
            sub_usage = self._data[usage][self.att_class] == label
            leaf_node.add_class_result(label, sum(sub_usage))
        return leaf_node

    def check_leaf_same_class(self, d_usage):
        total_num = sum(d_usage)
        for label in self.att_class_value:
            sub_usage = d_usage & (self._data[self.att_class] == label)
            sub_num = sum(sub_usage)
            if 0 < sub_num < total_num:
                return False
        return True

    def construct_tree(self):
        data_len = len(self._data)
        data_usage = [True] * data_len
        current_depth = self._tree_depth
        self.construct_sub_tree(None, current_depth, None, data_usage)

    def construct_sub_tree(self, parent_node, max_depth, outcome, d_usage):
        if max_depth == 0:
            return

        info_gain, split_att, outcomes, sub_usages = \
            self._select_split_att(d_usage)

        # current node is leaf
        if info_gain == 0 or max_depth == 1 or \
                self.check_leaf_same_class(d_usage):
            # no parent node
            leaf_node = self.set_leaf_node(d_usage)
            if not parent_node:
                self.root_node = leaf_node
                self.root_node.set_parent_node(None)
                return
            else:
                parent_node.add_sub_node(outcome, leaf_node)
                return

        # current node is non-leaf
        non_leaf_node = NonLeafNode(is_leaf=False, att_name=split_att)
        if not parent_node:
            self.root_node = non_leaf_node
        else:
            parent_node.add_sub_node(outcome, non_leaf_node)

        num = 0
        for sub_outcome in outcomes:
            self.construct_sub_tree(non_leaf_node, max_depth-1, sub_outcome,
                                    sub_usages[num])
            num += 1

    def show_C4_5(self, current_node, outcome, t_space):
        if not current_node:
            return
        if current_node.is_leaf:
            print("%s%s --> %s" % (t_space, outcome,
                                   current_node.class_results))
        else:
            print("%s%s --> %s" % (t_space, outcome, current_node.att_name))
            for sub_outcome, sub_node in current_node.sub_nodes.items():
                self.show_C4_5(sub_node, sub_outcome, t_space+self.unit_space)