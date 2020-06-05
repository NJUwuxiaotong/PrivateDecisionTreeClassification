import math
import numpy as np
import random

from private_decision_tree.private_lap_ID3_05P import PrivateLapID305P


class PrivateLapC4505P(PrivateLapID305P):
    def _select_split_attribute(self, d_usage, candidate_attributes, *params):
        gain_ratio = None
        split_attribute = None
        sub_usages = None
        overcomes = None
        privacy_value = params[0]["privacy_value"]

        total_num = 0
        while total_num <= 0:
            total_num = sum(d_usage) + self.noisy(privacy_value)

        params[0]["num_in_att"] = total_num
        for att in candidate_attributes:
            can_gain_ratio, can_overcomes, can_sub_usages = \
                self._generate_information_of_specified_attribute(
                    d_usage, att, params[0])

            if gain_ratio is None or gain_ratio < can_gain_ratio:
                gain_ratio = can_gain_ratio
                split_attribute = att
                sub_usages = can_sub_usages
                overcomes = can_overcomes
        return gain_ratio, split_attribute, overcomes, sub_usages

    def _information_entropy(self, d_usage, *params):
        privacy_value = params[0]["privacy_value"]
        total_num = params[0]["total_num"]
        info_value = 0
        for class_value in self.class_att_value:
            c_usage = d_usage & (
                    self._training_data[self.class_att] == class_value)
            c_num = sum(c_usage) + self.noisy(privacy_value)
            if c_num > 0:
                p = c_num/total_num
                info_value -= p * math.log2(p)
        return info_value

    def get_information_of_discrete_attribute(self, d_usage, att, *params):
        info_value = 0
        overcomes = list()
        usages = []
        information_value = 0
        privacy_value = params[0]["privacy_value"]
        num_of_att = params[0]["num_in_att"]
        for value in self._attribute_values[att]:
            sub_usage = d_usage & (self._training_data[att] == value)
            sub_num = sum(sub_usage) + self.noisy(privacy_value)
            if sub_num <= 0:
                continue
            params[0]["total_num"] = sub_num
            info_value += self._information_entropy(sub_usage, params[0])
            overcomes.append(value)
            usages.append(sub_usage)
            p = sub_num/num_of_att
            information_value -= p*math.log2(p)
        info_value /= information_value
        return info_value, overcomes, usages

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

        num_of_att = params[0]["num_in_att"]
        privacy_value = params[0]["privacy_value"]
        information_value = 0
        l_usage, r_usage = self.get_left_right_usage(att, d_usage, threshold)
        l_num = sum(l_usage) + self.noisy(privacy_value)
        r_num = sum(r_usage) + self.noisy(privacy_value)

        p_info = 0
        if l_num > 0:
            p_info += self._information_entropy(
                l_usage, {"privacy_value": privacy_value, "total_num": l_num})
            p = l_num/num_of_att
            information_value -= p * math.log2(p)
        if r_num > 0:
            p_info += self._information_entropy(
                r_usage, {"privacy_value": privacy_value, "total_num": r_num})
            p = r_num/num_of_att
            information_value -= p * math.log2(p)
        p_info /= information_value
        return p_info, ["<<"+str(threshold),
                        ">="+str(threshold)], [l_usage, r_usage]

    def compute_information_value(self):
        pass

    def _split_different_type_of_nodes(self, nodes):
        non_leaf_nodes = list()
        leaf_nodes = list()

        for node in nodes:
            if node.is_leaf:
                leaf_nodes.append(node)
            else:
                non_leaf_nodes.append(node)
        return non_leaf_nodes, leaf_nodes

    def get_deep_non_leaf_nodes(self, node):
        """
        return
        """
        if node.is_leaf:
            return []

        deep_non_leaf_nodes = list()
        current_nodes = list()
        next_nodes = list()
        current_nodes.append(node)

        while current_nodes:
            for node in current_nodes:
                sub_nodes = node.sub_nodes.values()
                non_leaf_nodes, leaf_nodes = \
                    self._split_different_type_of_nodes(sub_nodes)
                if not non_leaf_nodes:           # all leaf nodes
                    deep_non_leaf_nodes.append(node)
                else:                            # there are non-leaf nodes
                    next_nodes.append(non_leaf_nodes)

            current_nodes = next_nodes
            next_nodes = []
        return deep_non_leaf_nodes

    def _check_whether_to_prune(self, node):
        return False

    def prune(self, node):
        while True:
            deep_non_leaf_nodes = self.get_deep_non_leaf_nodes(node)
            merged_nodes = list()
            for node in deep_non_leaf_nodes:
                if self._check_whether_to_prune(node):
                    merged_nodes.append(node)

            if not merged_nodes:
                break
