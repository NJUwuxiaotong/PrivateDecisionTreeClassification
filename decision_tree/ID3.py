import copy
import math
import numpy as np
import random
import time

from common import constants as const
from decision_tree.ID3_node import NonLeafNode, LeafNode
from decision_tree.decision_tree import DecisionTree


class ID3(DecisionTree):
    """
    Literature: J. Ross Quinlan. C4.5: Programs for machine learning.
    """
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, is_private=False, privacy_value=1):
        super(ID3, self).__init__(dataset_name, training_per, test_per,
                                  tree_depth, is_private, privacy_value)

    def _information(self, d_usage, total_num, is_privacy=False,
                     privacy_value=None):
        information = 0
        # total_num = sum(d_usage)
        for class_value in self.class_att_value:
            sub_usage = d_usage & \
                        (self._training_data[self.class_att] == class_value)
            # sub_num = sum(sub_usage)
            sub_num = self.get_num_of_records(
                sub_usage, is_privacy, privacy_value)
            if sub_num > 0 and total_num > 0:
                p = sub_num/total_num
                information -= p * math.log2(p)
        return information

    def _information_after_division(self, d_usage, att, is_private=False,
                                    privacy_value=None):
        """
        param d_usage:
        param att:
        return [gain_info, overcomes, usages]
        """
        total_num = sum(d_usage)
        overcomes = list()
        usages = list()
        if self._attribute_type[att] == const.DFRAME_INT64:
            p_info, threshold, usages = \
                self.get_split_value_of_int64(
                    att, d_usage, is_private, privacy_value)
            return p_info, \
                   ["<<"+str(threshold), ">="+str(threshold)], usages
        else:
            gain_info = 0
            for value in self._attribute_values[att]:
                sub_usage = d_usage & (self._training_data[att] == value)
                # sub_num = sum(sub_usage)
                sub_num = self.get_num_of_records(
                    sub_usage, is_private, privacy_value)
                if sub_num > 0:
                    gain_info += sub_num / total_num * self._information(
                        sub_usage, sub_num, is_private, privacy_value)
                    overcomes.append(value)
                    usages.append(sub_usage)
            return gain_info, overcomes, usages

    def get_num_of_records(self, d_usage, is_privacy=False,
                           privacy_value=None):
        if not is_privacy:
            return sum(d_usage)
        else:
            return sum(d_usage) + self.noisy(privacy_value)

    def noisy(self, privacy_value):
        """
        Function: provide an interface to add the noisy for privacy
        preservation
        """
        return 0.0

    def get_threshold_by_1b1(self, att, d_usage):
        can_info = 1000000
        can_threshold = 0
        total_num = sum(d_usage)
        unique_values = self._training_data[att][d_usage].drop_duplicates(
            keep='first').values
        unique_values.sort()

        if len(unique_values) == 1:
            return unique_values[0]

        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            l_usage, r_usage = self.get_left_right_usage(
                att, d_usage, threshold)
            p_info = \
                (sum(l_usage) * self._information(l_usage, sum(l_usage)) +
                 sum(r_usage) * self._information(r_usage, sum(r_usage))) / \
                total_num
            if p_info < can_info:
                can_info = p_info
                can_threshold = threshold
        return can_threshold

    def get_left_right_usage(self, att, d_usage, threshold):
        l_usage = d_usage & (self._training_data[att] < threshold)
        r_usage = d_usage & (self._training_data[att] >= threshold)
        return l_usage, r_usage

    def get_split_value_of_int64(self, att, d_usage, is_privacy=False,
                                 privacy_value=None, split_type="random"):
        """
        split type: random, median, mean, complex
        """
        #total_num = sum(d_usage)
        total_num = self.get_num_of_records(d_usage, is_privacy)
        unique_values = self._training_data[att][d_usage].drop_duplicates(
            keep='first').values
        if split_type == "random":
            max_v = unique_values.max()
            min_v = unique_values.min()
            threshold = min_v + random.random()*(max_v - min_v)

        elif split_type == "mean":
            threshold = unique_values.mean()
        elif split_type == "median":
            threshold = np.median(unique_values)
        elif split_type == "complex":
            threshold = self.get_threshold_by_1b1(att, d_usage)
        else:
            print("Error: Chosen split method %s is not in "
                  "[random, mean, median, complex]" % split_type)
            exit(1)

        l_usage, r_usage = self.get_left_right_usage(att, d_usage, threshold)
        l_num = self.get_num_of_records(l_usage, is_privacy, privacy_value)
        r_num = self.get_num_of_records(r_usage, is_privacy, privacy_value)
        p_info = \
            (l_num * self._information(
                l_usage, l_num, is_privacy, privacy_value) +
             r_num * self._information(
                        r_usage, r_num, is_privacy, privacy_value)) / total_num
        return p_info, threshold, [l_usage, r_usage]

    def _select_split_att(self, d_usage, candidate_atts):
        split_att = None
        sub_usages = None
        overcomes = None

        gain_info = 0
        info_gain = 10000
        privacy_value = self.privacy_value_per_node/(2*len(candidate_atts))
        for att in candidate_atts:
            can_info_gain, can_overcomes, can_sub_usages = \
                self._information_after_division(
                    d_usage, att, self.is_private, privacy_value)

            if can_info_gain < info_gain:
                info_gain = can_info_gain
                split_att = att
                sub_usages = can_sub_usages
                overcomes = can_overcomes
        gain_info -= info_gain
        return info_gain, split_att, overcomes, sub_usages

    def set_leaf_node(self, usage):
        leaf_node = LeafNode(is_leaf=True)
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

    def construct_tree(self):
        data_len = len(self._training_data)
        data_usage = [True] * data_len
        current_depth = self._tree_depth
        print("Start to construct decision tree ......")
        training_start_time = time.time()
        att_candidates = copy.deepcopy(self._attributes.values)
        self.construct_sub_tree(None, data_usage, att_candidates,
                                current_depth, None)
        training_end_time = time.time()
        self.training_time = training_end_time - training_start_time
        print("End to construct decision tree.")

    def check_conditions_of_leaf_node(self, info_gain, max_depth,
                                      candidate_atts, usage, *params):
        return info_gain == 0 or max_depth == 1 or len(candidate_atts) == 0 \
               or self.check_leaf_same_class(usage)

    def construct_sub_tree(self, parent_node, d_usage, candidate_atts,
                           max_depth, outcome):
        """
        parent_node: parent node
        d_usage: current data
        candidate_atts: attributes that can be selected
        max_depth: the depth of decision tree
        outcome: attribute value
        """
        if max_depth == 0 or sum(d_usage) == 0:
            return

        record_num = sum(d_usage)
        info = self._information(d_usage, record_num, is_privacy=False)
        info_gain, split_att, outcomes, sub_usages = \
            self._select_split_att(d_usage, candidate_atts)
        info_gain = info + info_gain

        # current node is leaf
        if self.check_conditions_of_leaf_node(info_gain, max_depth,
                                              candidate_atts, d_usage):
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
            sub_candidate_atts = copy.deepcopy(candidate_atts)
            sub_candidate_atts = sub_candidate_atts[
                sub_candidate_atts != split_att]
            self.construct_sub_tree(non_leaf_node, sub_usages[num],
                                    sub_candidate_atts, max_depth-1,
                                    sub_outcome)
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

    def prune(self):
        pass

    def get_random_class_label(self):
        label_num = len(self.class_att_value)
        random_n = random.randint(0, label_num-1)
        return self.class_att_value[random_n]

    def get_class_label_of_record(self, record):
        """
        :param
        :return
        """
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
        print("| Training Time | %.5s                 |" % self.training_time)
        print("| Test Time     | %.5s                 |" % self.test_time)
        print("| Test Results  | %.5s                 |" % right_ratio)
        print("-----------------------------------------")
