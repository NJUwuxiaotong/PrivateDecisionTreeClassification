import math
import os
import pandas as pd
import random
import time

from common import constants as const
from data_process.data_preprocess import DataPreProcess
from decision_tree.C4_5_node import NonLeafNode, LeafNode


class C45(object):
    def __init__(self, dataset_name, training_per=0.7, tree_depth=None):
        self._dataset_name = dataset_name
        self._check_file_path()
        self._training_data = None
        self._training_data_shape = None
        self._test_data = None
        self._test_data_shape = None
        self._attributes = None   # (key, value) except of class
        self._attribute_values = None
        self._attribute_type = None    # attribute type except of class
        self._attribute_num = 0        # except of class
        self.att_class = None          # only one class name
        self.att_class_value = list()    # class value
        self.training_per = training_per
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

    def _read_data(self):
        self.data_process.read_data_from_file()
        self._training_data = self.data_process.data
        self._training_data_shape = self.data_process.data_shape
        training_num = int(self._training_data_shape[0] * self.training_per)
        self._test_data = self._training_data[training_num:]
        self._training_data = self._training_data[:training_num]
        self._training_data_shape = tuple([training_num,
                                           self._training_data_shape[1]])
        self._test_data_shape = tuple([len(self._test_data),
                                       self._training_data_shape[1]])
        self._attributes = self.data_process.attributes[:-1]
        self._attribute_num = len(self._attributes)
        self.att_class = self.data_process.attributes[-1]
        self._attribute_values = self.data_process.attribute_values
        self.att_class_value = self._attribute_values.pop(
            self.att_class)
        self._attribute_type = self.data_process.attribute_types
        self._attribute_type.pop(self.att_class)

    def _information(self, d_usage):
        information = 0
        total_num = sum(d_usage)
        for class_value in self.att_class_value:
            sub_usage = d_usage & (self._training_data[self.att_class] == class_value)
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
            can_l_usage = None
            can_r_usage = None
            unique_values = self._training_data[att][d_usage].drop_duplicates(
                keep='first').values
            unique_values.sort()

            if len(unique_values) == 1:
                return self._information(d_usage), \
                       "=="+str(unique_values[0]), \
                       [d_usage, None]

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i+1])/2
                l_usage = d_usage & (self._training_data[att] < threshold)
                r_usage = d_usage & (self._training_data[att] >= threshold)
                p_info = (sum(l_usage)*self._information(l_usage) +
                          sum(r_usage)*self._information(r_usage))/total_num
                if p_info < can_info:
                    can_info = p_info
                    can_threshold = threshold
                    can_l_usage = l_usage
                    can_r_usage = r_usage
            return can_info, \
                   ["<<"+str(can_threshold), ">="+str(can_threshold)], \
                   [can_l_usage, can_r_usage]
        else:
            gain_info = 0
            for value in self._attribute_values[att]:
                sub_usage = d_usage & (self._training_data[att] == value)
                sub_num = sum(sub_usage)
                if sub_num != 0:
                    gain_info += sub_num/total_num*self._information(sub_usage)
                    overcomes.append(value)
                    usages.append(sub_usage)
            return gain_info, overcomes, usages

    def get_continuous_one_by_one(self, att, d_usage):
        pass

    def get_continuous_median(self, att, d_usage):
        unique_values = self._training_data[att][d_usage].drop_duplicates(
            keep='first').values
        threshold = unique_values.median()
        return threshold

    def get_continuous_mean(self, att, d_usage):
        unique_values = self._training_data[att][d_usage].drop_duplicates(
            keep='first').values
        threshold = unique_values.mean()
        return threshold

    def get_random_continuous(self, att, d_usage):
        unique_values = self._training_data[att][d_usage].drop_duplicates(
            keep='first').values
        max_v = unique_values.max()
        min_v = unique_values.min()
        random.random()
        return

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
            sub_usage = self._training_data[usage][self.att_class] == label
            leaf_node.add_class_result(label, sum(sub_usage))
        return leaf_node

    def check_leaf_same_class(self, d_usage):
        total_num = sum(d_usage)
        for label in self.att_class_value:
            sub_usage = d_usage & (self._training_data[self.att_class] == label)
            sub_num = sum(sub_usage)
            if 0 < sub_num < total_num:
                return False
        return True

    def construct_tree(self):
        data_len = len(self._training_data)
        data_usage = [True] * data_len
        current_depth = self._tree_depth
        print("Start to construct decision tree ......")
        training_start_time = time.time()
        self.construct_sub_tree(None, current_depth, None, data_usage)
        training_end_time = time.time()
        self.training_time = training_end_time - training_start_time
        print("End to construct decision tree.")

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
            print("NEW LEAF NODE: <%s>" % (leaf_node))
            if not parent_node:
                self.root_node = leaf_node
                self.root_node.set_parent_node(None)
                return
            else:
                parent_node.add_sub_node(outcome, leaf_node)
                return

        # current node is non-leaf
        non_leaf_node = NonLeafNode(is_leaf=False, att_name=split_att)
        print("New NON-LEAF NODE: <%s>" % (split_att))
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

    def get_random_class_label(self):
        label_num = len(self.att_class_value)
        random_n = random.randint(0, label_num-1)
        return self.att_class_value[random_n]

    def get_class_label_of_record(self, record):
        """
        :param
        :return
        """
        node = self.root_node
        while True:
            if node.is_leaf:
                return node.get_most_frequent_class()

            att = node.att_name
            value = record[att].values[0]
            sub_nodes = node.sub_nodes
            if self._attribute_type[att] == const.DFRAME_INT64:
                node_keys = list(sub_nodes.keys())
                l_bp = node_keys[0]
                r_bp = node_keys[1]
                bp = float(l_bp[2:])
                if value < bp:
                    if l_bp[:2] == "<<":
                        node = node.sub_nodes[l_bp]
                    else:
                        node = node.sub_nodes[r_bp]
                else:
                    if l_bp[:2] == "<<":
                        node = node.sub_nodes[r_bp]
                    else:
                        node = node.sub_nodes[l_bp]
            else:
                if value not in sub_nodes.keys():
                    return self.get_random_class_label()
                else:
                    node = node.sub_nodes[value]

    def list_class_labels_of_test_data(self):
        class_labels = []
        for i in range(self._test_data_shape[0]):
            record = self._test_data[i:i+1]
            class_label = self.get_class_label_of_record(record)
            class_labels.append(class_label)
        return class_labels

    def get_test_results(self):
        test_start_time = time.time()
        test_class_labels = self.list_class_labels_of_test_data()
        test_end_time = time.time()
        self.test_time = test_end_time - test_start_time
        right_class_labels = self._test_data[self.att_class]
        right_ratio = sum(test_class_labels == right_class_labels)/len(
            test_class_labels)

        print("------------------- Result Statistics ----------------------")
        print("Training Time: %.5s" % self.training_time)
        print("Test Time:     %.5s" % self.test_time)
        print("Test Results:  %.5s" % right_ratio)
        print("------------------------------------------------------------")
