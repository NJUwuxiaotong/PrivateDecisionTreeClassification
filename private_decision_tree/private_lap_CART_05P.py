import math
import numpy as np
import random

from private_decision_tree.private_lap_ID3_05P import PrivateLapID305P


class PrivateLapCART05P(PrivateLapID305P):
    def _select_split_attribute(self, d_usage, candidate_attributes, *params):
        split_attribute = None
        sub_usages = None
        overcomes = None
        gini_index = None
        for att in candidate_attributes:
            can_gini_index, can_overcomes, can_sub_usages = \
                self._generate_information_of_specified_attribute(
                    d_usage, att, params[0])

            if gini_index is None or can_gini_index > gini_index:
                gini_index = can_gini_index
                split_attribute = att
                sub_usages = can_sub_usages
                overcomes = can_overcomes
        return gini_index, split_attribute, overcomes, sub_usages

    def _information_entropy(self, d_usage, *params):
        privacy_value = params[0]["privacy_value"]
        total_num = params[0]["total_num"]
        if total_num == 0:
            return 0

        gini_index = 1
        for class_value in self.class_att_value:
            c_usage = d_usage & (
                    self._training_data[self.class_att] == class_value)
            c_num = sum(c_usage) + self.noisy(privacy_value)
            p = c_num / total_num
            gini_index -= p * p
        return gini_index

    def get_information_of_discrete_attribute(self, d_usage, att, *params):
        overcomes = list()
        usages = []
        gini_index = 0
        privacy_value = params[0]["privacy_value"]
        for value in self._attribute_values[att]:
            sub_usage = d_usage & (self._training_data[att] == value)
            sub_num = sum(sub_usage) + self.noisy(privacy_value)
            if sub_num <= 0:
                continue
            params[0]["total_num"] = sub_num
            info_value = self._information_entropy(sub_usage, params[0])
            overcomes.append(value)
            usages.append(sub_usage)
            info_value = -1 * (1 + info_value) * sub_num
            gini_index += info_value
        return gini_index, overcomes, usages

    def get_information_of_int64_attribute(self, att, d_usage,
                                           split_type="mean", *params):
        """
        split type: random, median, mean, complex
        """
        unique_values = self._training_data[att][d_usage]\
            .drop_duplicates(keep='first').values
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

        gini_index = 0
        privacy_value = params[0]["privacy_value"]
        l_usage, r_usage = self.get_left_right_usage(att, d_usage, threshold)
        l_num = sum(l_usage) + self.noisy(privacy_value)
        r_num = sum(r_usage) + self.noisy(privacy_value)

        p_info = 0
        if l_num > 0:
            p_info += self._information_entropy(
                l_usage, {"privacy_value": privacy_value, "total_num": l_num})
            gini_index -= l_num * (1+p_info)

        if r_num > 0:
            p_info += self._information_entropy(
                r_usage, {"privacy_value": privacy_value, "total_num": r_num})
            gini_index -= r_num * (1 + p_info)
        return p_info, ["<<"+str(threshold),
                        ">="+str(threshold)], [l_usage, r_usage]
