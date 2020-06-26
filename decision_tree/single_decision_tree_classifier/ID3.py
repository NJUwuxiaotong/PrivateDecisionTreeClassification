import numpy as np
import random

from common import constants as const
from decision_tree.single_decision_tree_classifier.decision_tree import DecisionTree


class ID3(DecisionTree):
    """
    Literature: J. Ross Quinlan. C4.5: Programs for machine learning.
    """
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None):
        super(ID3, self).__init__(dataset_name, training_per, test_per,
                                  tree_depth)

    def get_information_of_discrete_attribute(self, d_usage, att, *params):
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

    def get_split_value_of_int64_attribute(self, att, d_usage,
                                           split_type="mean", *params):
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
        return threshold

    def get_information_of_int64_attribute(self, att, d_usage,
                                           split_type="mean", *params):
        """
        split type: random, median, mean, complex
        """
        threshold = self.get_split_value_of_int64_attribute(
            att, d_usage, split_type, params[0])
        total_num = sum(d_usage)
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

    def _generate_information_of_specified_attribute(self, d_usage, att,
                                                     *params):
        """
        param d_usage: data
        param att: attribute
        return [info, overcomes, usages]
        """
        if self._attribute_type[att] == const.DFRAME_INT64:
            info_entropy_after_split, overcomes, usages = \
                self.get_information_of_int64_attribute(
                    att, d_usage, "mean", params[0])
            return info_entropy_after_split, overcomes, usages
        else:
            info_entropy_after_split, overcomes, usages = \
                self.get_information_of_discrete_attribute(d_usage, att,
                                                           params[0])
            return info_entropy_after_split, overcomes, usages

    def _information_gain(self, d_usage, att, *params):
        info_entropy_before_split = self._information_entropy(d_usage)
        info_entropy_after_split, overcomes, sub_usages = \
            self._generate_information_of_specified_attribute(
                d_usage, att, params[0])
        info_gain = info_entropy_before_split - info_entropy_after_split
        return info_gain, overcomes, sub_usages

    def _information_metric(self, d_usage, att, *params):
        info_gain, overcomes, sub_usages = self._information_gain(
            d_usage, att, params[0])
        return info_gain, overcomes, sub_usages
