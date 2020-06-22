import abc
import copy
import math
import random
import time
import os

from common import constants as const
from data_process.data_preprocess import DataPreProcess
from decision_tree.decision_tree_node import NonLeafNode, LeafNode


class DecisionTree(object):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None):
        self._dataset_name = dataset_name
        self._check_file_path()
        self._training_data = None       # type: DataFrame
        self._training_data_shape = None
        self._test_data = None
        self._test_data_shape = None
        self._attributes = None          # key except of class
        self._attribute_values = None    # (key, value) except of class
        self._attribute_type = None      # attribute type except of class
        self._attribute_num = 0          # except of class
        self.class_att = None            # only one class name
        self.class_att_value = list()    # class value
        self.training_per = training_per
        self.test_per = test_per
        self._check_parameters()
        self.training_num = None
        self.test_num = None
        self.range_of_int64_attributes = None
        dataset_absolute_path = \
            const.DATASET_PATH + self._dataset_name + const.DATASET_SUFFIX
        self.data_process = DataPreProcess(dataset_absolute_path)
        self._read_data()
        self.get_range_of_int64_attributes()

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

    def get_range_of_int64_attributes(self):
        self.range_of_int64_attributes = dict()
        for att in self._attributes:
            if self._attribute_type[att] == const.DFRAME_INT64:
                values = self._training_data[att].values
                max_value = values.max()
                min_value = values.min()
                self.range_of_int64_attributes[att] = [min_value*0.75,
                                                       max_value*(1+0.25)]

    def _information_entropy(self, d_usage):
        total_num = sum(d_usage)
        if total_num == 0:
            return 0

        infor_entropy = 0
        for class_value in self.class_att_value:
            sub_usage = d_usage & \
                        (self._training_data[self.class_att] == class_value)
            sub_num = sum(sub_usage)
            if sub_num > 0:
                p = sub_num / total_num
                infor_entropy -= p * math.log2(p)
        return infor_entropy

    def construct_tree(self):
        data_usage = [True] * self.training_num
        current_depth = copy.deepcopy(self._tree_depth)
        print("Start to construct decision tree ......")
        training_start_time = time.time()
        att_candidates = copy.deepcopy(self._attributes.values)
        self.construct_sub_tree(None, data_usage, att_candidates,
                                current_depth, None)
        training_end_time = time.time()
        self.training_time = training_end_time - training_start_time
        print("End to construct decision tree.")

    @abc.abstractmethod
    def _information_metric(self, d_usage, att, *params):
        return None, None, None

    def get_left_right_usage(self, att, d_usage, threshold):
        l_usage = d_usage & (self._training_data[att] < threshold)
        r_usage = d_usage & (self._training_data[att] >= threshold)
        return l_usage, r_usage

    def _select_split_attribute(self, d_usage, candidate_attributes, *params):
        split_attribute = None
        sub_usages = None
        overcomes = None
        info_gain = None
        for att in candidate_attributes:
            can_info_gain, can_overcomes, can_sub_usages = \
                self._information_metric(d_usage, att, params[0])
            if info_gain is None or can_info_gain > info_gain:
                info_gain = can_info_gain
                split_attribute = att
                sub_usages = can_sub_usages
                overcomes = can_overcomes
        return info_gain, split_attribute, overcomes, sub_usages

    def construct_sub_tree(self, parent_node, d_usage, candidate_attributes,
                           max_depth, outcome):
        candidate_attribute_num = len(candidate_attributes)
        info_gain, split_attribute, outcomes, sub_usages = \
            self._select_split_attribute(d_usage, candidate_attributes, None)

        # leaf node
        if candidate_attribute_num == 0 or max_depth == 1 or info_gain == 0 \
                or self.check_leaf_same_class(d_usage):
            # no parent node
            leaf_node = self.set_leaf_node(d_usage)
            print("NEW LEAF NODE: <%s>" % leaf_node)
            if not parent_node:
                self.root_node = leaf_node
                self.root_node.set_parent_node(None)
                return
            else:
                parent_node.add_sub_node(outcome, leaf_node)
                return

        # current node is non-leaf
        non_leaf_node = NonLeafNode(is_leaf=False, att_name=split_attribute)
        print("New NON-LEAF NODE: <%s>" % split_attribute)
        if not parent_node:
            self.root_node = non_leaf_node
        else:
            parent_node.add_sub_node(outcome, non_leaf_node)

        num = 0
        for sub_outcome in outcomes:
            sub_candidate_attributes = copy.deepcopy(candidate_attributes)
            sub_candidate_attributes = sub_candidate_attributes[
                sub_candidate_attributes != split_attribute]
            self.construct_sub_tree(non_leaf_node, sub_usages[num],
                                    sub_candidate_attributes, max_depth - 1,
                                    sub_outcome)
            num += 1

    def set_leaf_node(self, usage):
        leaf_node = LeafNode(is_leaf=True)
        for class_key in self.class_att_value:
            c_usage = usage & (self._training_data[self.class_att] == class_key)
            c_num = sum(c_usage)
            if c_num > 0:
                leaf_node.add_class_value(class_key, c_num)
        leaf_node.set_class_result(self.get_most_frequent_class(usage))
        return leaf_node

    def get_most_frequent_class(self, usage):
        chosen_label = None
        result = None
        for label in self.class_att_value:
            r = usage & (self._training_data[self.class_att] == label)
            if (result is None) or (result < sum(r)):
                result = sum(r)
                chosen_label = label
        return chosen_label

    def check_leaf_same_class(self, d_usage):
        total_num = sum(d_usage)
        for label in self.class_att_value:
            sub_usage = d_usage & (self._training_data[self.class_att] == label)
            sub_num = sum(sub_usage)
            if 0 < sub_num < total_num:
                return False
        return True

    def show_structure_of_decision_tree(self, current_node, outcome, t_space):
        if not outcome:
            outcome = "ROOT NODE"
        if not current_node:
            return
        if current_node.is_leaf:
            print("%s%s --> %s/%s" % (t_space, outcome,
                                      current_node.class_results,
                                      current_node.class_values))
        else:
            print("%s%s --> %s" % (t_space, outcome, current_node.att_name))
            for sub_outcome, sub_node in current_node.sub_nodes.items():
                self.show_structure_of_decision_tree(
                    sub_node, sub_outcome, t_space + self.unit_space)

    def show_statistics_of_decision_tree(self, current_node, statistics):
        if not current_node:
            return
        if current_node.is_leaf:
            for class_key, result in current_node.class_values.items():
                if class_key not in statistics.keys():
                    statistics[class_key] = result
                else:
                    statistics[class_key] += result
        else:
            for sub_outcome, sub_node in current_node.sub_nodes.items():
                self.show_statistics_of_decision_tree(
                    sub_node, statistics)

    def get_random_class_label(self):
        label_num = len(self.class_att_value)
        random_n = random.randint(0, label_num-1)
        return self.class_att_value[random_n]

    def get_class_label_of_record(self, record):
        node = self.root_node
        while True:
            if node.is_leaf:
                return node.class_results

            att = node.att_name
            value = record[att].values[0]
            sub_nodes = node.sub_nodes
            if self._attribute_type[att] == const.DFRAME_INT64:
                node_keys = list(sub_nodes.keys())

                if len(node_keys) == 0:
                    return self.get_random_class_label()

                if len(node_keys) == 1:
                    bp = float(node_keys[0][2:])
                    if (value < bp and node_keys[0][:2] == "<<") or \
                            (value > bp and node_keys[0][:2] == ">="):
                        node = node.sub_nodes[node_keys[0]]
                    else:
                        return self.get_random_class_label()

                if len(node_keys) == 2:
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
        right_class_labels = self._test_data[self.class_att]
        right_ratio = sum(test_class_labels == right_class_labels)/len(
            test_class_labels)

        print("----------- Result Statistics -----------")
        print("| Training Time | %-.10s            |" % self.training_time)
        print("| Test Time     | %-.10s            |" % self.test_time)
        print("| Test Results  | %-.10s            |" % right_ratio)
        print("-----------------------------------------")
