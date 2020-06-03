import copy
import numpy as np
import random

from common import constants as const
from decision_tree.decision_tree_node import NonLeafNode
from decision_tree.decision_tree import DecisionTree


class ID3(DecisionTree):
    """
    Literature: J. Ross Quinlan. C4.5: Programs for machine learning.
    """
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None):
        super(ID3, self).__init__(dataset_name, training_per, test_per,
                                  tree_depth)

    def get_information_of_discrete_attribute(self, d_usage, att):
        total_num = sum(d_usage)
        info_entropy_after_split = 0
        overcomes = list()
        usages = []
        for value in self._attribute_values[att]:
            sub_usage = d_usage & (self._training_data[att] == value)
            sub_num = sum(sub_usage)
            if sub_num > 0:
                info_entropy_after_split += \
                    sub_num / total_num * self._information_entropy(sub_usage)
                overcomes.append(value)
                usages.append(sub_usage)
        return info_entropy_after_split, overcomes, usages

    def get_information_of_int64_attribute(self, att, d_usage,
                                           split_type="mean"):
        """
        split type: random, median, mean, complex
        """
        total_num = sum(d_usage)
        unique_values = self._training_data[att][d_usage].drop_duplicates(
            keep='first').values
        if split_type == "random":
            max_v = unique_values.max()
            min_v = unique_values.min()
            threshold = min_v + random.random() * (max_v - min_v)
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
        l_num = sum(l_usage)
        r_num = sum(r_usage)
        p_info = (l_num * self._information_entropy(l_usage) +
                  r_num * self._information_entropy(r_usage)) / total_num
        return p_info, ["<<"+str(threshold),
                        ">="+str(threshold)], [l_usage, r_usage]

    def get_threshold_by_1b1(self, att, d_usage):
        can_info = 100000
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
            l_num = sum(l_usage)
            r_num = sum(r_usage)
            p_info = \
                (l_num * self._information_entropy(l_usage) +
                 r_num * self._information_entropy(r_usage)) / total_num
            if p_info < can_info:
                can_info = p_info
                can_threshold = threshold
        return can_threshold

    def _generate_information_of_specified_attribute(self, d_usage, att):
        """
        param d_usage: data
        param att: attribute
        return [info, overcomes, usages]
        """
        if self._attribute_type[att] == const.DFRAME_INT64:
            info_entropy_after_split, overcomes, usages = \
                self.get_information_of_int64_attribute(att, d_usage)
            return info_entropy_after_split, overcomes, usages
        else:
            info_entropy_after_split, overcomes, usages = \
                self.get_information_of_discrete_attribute(d_usage, att)
            return info_entropy_after_split, overcomes, usages

    def get_left_right_usage(self, att, d_usage, threshold):
        l_usage = d_usage & (self._training_data[att] < threshold)
        r_usage = d_usage & (self._training_data[att] >= threshold)
        return l_usage, r_usage

    def _information_gain(self, d_usage, att):
        info_entropy_before_split = self._information_entropy(d_usage)
        info_entropy_after_split, overcomes, sub_usages = \
            self._generate_information_of_specified_attribute(d_usage, att)
        info_gain = info_entropy_before_split - info_entropy_after_split
        return info_gain, overcomes, sub_usages

    def _select_split_attribute(self, d_usage, candidate_attributes):
        split_attribute = None
        sub_usages = None
        overcomes = None
        info_gain = 0
        for att in candidate_attributes:
            can_info_gain, can_overcomes, can_sub_usages = \
                self._information_gain(d_usage, att)

            if can_info_gain > info_gain:
                info_gain = can_info_gain
                split_attribute = att
                sub_usages = can_sub_usages
                overcomes = can_overcomes
        return info_gain, split_attribute, overcomes, sub_usages

    def construct_sub_tree(self, parent_node, d_usage, candidate_attributes,
                           max_depth, outcome):
        candidate_attribute_num = len(candidate_attributes)
        info_gain, split_attribute, outcomes, sub_usages = \
            self._select_split_attribute(d_usage, candidate_attributes)

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
