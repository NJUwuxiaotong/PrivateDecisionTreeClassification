import abc
import copy
import math
import random
import time

from common import constants as const
from decision_tree.data_initialization import DataInitialization
from decision_tree.single_decision_tree_classifier.decision_tree_node import NonLeafNode, LeafNode


class DecisionTree(DataInitialization):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None):
        super(DecisionTree, self).__init__(dataset_name, training_per, test_per)

        self.root_node = None
        if tree_depth is None:
            self._tree_depth = int(self._attribute_num/2)
        else:
            self._tree_depth = tree_depth

        # representation format
        self.unit_space = "\t"
        self.training_time = 0.0
        self.test_time = 0.0

    def _information_entropy(self, d_usage):
        total_num = sum(d_usage)
        if total_num == 0:
            return 0

        infor_entropy = 0
        for class_value in self._class_label:
            sub_usage = d_usage & (
                    self._training_data[self._class_attribute] == class_value)
            sub_num = sum(sub_usage)
            if sub_num > 0:
                p = sub_num / total_num
                infor_entropy -= p * math.log2(p)
        return infor_entropy

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

    def construct_tree(self):
        data_usage = [True] * self._training_num
        current_depth = copy.deepcopy(self._tree_depth)
        print("Start to construct decision tree ......")
        training_start_time = time.time()
        att_candidates = copy.deepcopy(self._attributes.values)
        self.construct_sub_tree(None, data_usage, att_candidates,
                                current_depth, None)
        training_end_time = time.time()
        self.training_time = training_end_time - training_start_time
        print("End to construct decision tree.")

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
        for class_key in self._class_label:
            c_usage = usage & (
                    self._training_data[self._class_attribute] == class_key)
            c_num = sum(c_usage)
            if c_num > 0:
                leaf_node.add_class_value(class_key, c_num)
        leaf_node.set_class_result(self.get_most_frequent_class(usage))
        return leaf_node

    def get_most_frequent_class(self, usage):
        chosen_label = None
        result = None
        for label in self._class_label:
            r = usage & (self._training_data[self._class_attribute] == label)
            if (result is None) or (result < sum(r)):
                result = sum(r)
                chosen_label = label
        return chosen_label

    def check_leaf_same_class(self, d_usage):
        total_num = sum(d_usage)
        for label in self._class_label:
            sub_usage = d_usage & (
                    self._training_data[self._class_attribute] == label)
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
        label_num = len(self._class_label)
        random_n = random.randint(0, label_num-1)
        return self._class_label[random_n]

    def get_class_label_of_record(self, record):
        node = self.root_node
        while True:
            if node.is_leaf:
                return node.class_results

            att = node.att_name
            value = record[att].values[0]
            sub_nodes = node.sub_nodes
            if self._attribute_types[att] == const.DFRAME_INT64:
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
        right_class_labels = self._test_data[self._class_attribute]
        right_ratio = sum(test_class_labels == right_class_labels)/len(
            test_class_labels)

        print("----------- Result Statistics -----------")
        print("| Training Time | %-.10s            |" % self.training_time)
        print("| Test Time     | %-.10s            |" % self.test_time)
        print("| Test Results  | %-.10s            |" % right_ratio)
        print("-----------------------------------------")
