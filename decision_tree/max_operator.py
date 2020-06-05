import numpy as np
import random

from decision_tree.ID3 import ID3


class MaxOperator(ID3):
    def _information_gain(self, d_usage, att, *params):
        max_operator, overcomes, sub_usages = \
            self._generate_information_of_specified_attribute(
                d_usage, att, params[0])
        return max_operator, overcomes, sub_usages

    def get_information_of_int64_attribute(self, att, d_usage,
                                           split_type="mean", *params):
        """
        split type: random, median, mean, complex
        """
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

        max_operator = 0
        l_usage, r_usage = self.get_left_right_usage(att, d_usage, threshold)
        l_num = sum(l_usage)
        r_num = sum(r_usage)

        if l_num > 0:
            max_value = self._information_entropy(l_usage)
            max_operator += max_value
        if r_num > 0:
            max_value = self._information_entropy(r_usage)
            max_operator += max_value
        return max_operator, ["<<"+str(threshold),
                              ">="+str(threshold)], [l_usage, r_usage]

    def get_information_of_discrete_attribute(self, d_usage, att, *params):
        max_operator = 0
        overcomes = list()
        usages = []
        for value in self._attribute_values[att]:
            sub_usage = d_usage & (self._training_data[att] == value)
            sub_num = sum(sub_usage)
            if sub_num > 0:
                max_value = self._information_entropy(sub_usage)
                max_operator += max_value
                overcomes.append(value)
                usages.append(sub_usage)
        return max_operator, overcomes, usages

    def _information_entropy(self, d_usage):
        max_value = None
        for class_value in self.class_att_value:
            sub_usage = d_usage & \
                        (self._training_data[self.class_att] == class_value)
            sub_num = sum(sub_usage)
            if max_value is None or max_value < sub_num:
                max_value = sub_num
        return max_value
