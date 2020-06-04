import copy
import math
import numpy as np
import random
import time

from common import constants as const
from decision_tree.decision_tree_node import NonLeafNode, LeafNode
from decision_tree.ID3 import ID3
from pub_lib import pub_functions


class PrivateLapID305P(ID3):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, privacy_budget=1):
        super(PrivateLapID305P, self).__init__(
            dataset_name, training_per, test_per, tree_depth)
        self.privacy_budget = privacy_budget
        self.privacy_budget_per_node = self.privacy_budget/(2 * tree_depth)

    def noisy(self, privacy_value):
        """
        Function: provide an interface to add the noisy for privacy
        preservation.
        It chooses Laplace mechanism.
                Probability density function:
        Pr(x|l) = 1/(2*l)*exp^{-|x|/l}, in which l=sensitivity/privacy_value
        So, the probability distributed function:
            F(x) =  exp^{x/l}/2               x<0
                    1-exp{-x/l}/2             x>0
        """
        return pub_functions.generate_random_value_from_Laplace(
            -3, 3, privacy_value)

    def get_num_with_noisy(self, usage, privacy_value):
        u_num = sum(usage)
        return u_num + self.noisy(privacy_value)

    def get_max_num_of_att_values(self, attributes, usage):
        max_num = 0
        for att in attributes:
            if self._attribute_type[att] == const.DFRAME_INT64:
                if max_num < 2:
                    max_num = 2
            else:
                unique_values = self._training_data[att][usage].\
                    drop_duplicates(keep='first').values
                v_num = len(unique_values)
                if max_num < v_num:
                    max_num = v_num
        return max_num

    def check_termination_condition(self, record_num, max_att_num):
        condition = record_num / (max_att_num * len(self._attribute_values)) \
                    < math.sqrt(2) / self.privacy_budget_per_node
        return condition

    def set_leaf_node(self, usage):
        chosen_class = None
        max_class_num = None
        leaf_node = LeafNode(is_leaf=True)
        for class_key in self.class_att_value:
            c_usage = usage & (self._training_data[self.class_att] == class_key)
            c_num = sum(c_usage) + self.noisy(self.privacy_budget_per_node)
            leaf_node.add_class_value(class_key, c_num)
            if max_class_num is None or max_class_num < c_num:
                chosen_class = class_key
                max_class_num = c_num
        leaf_node.set_class_result(chosen_class)
        return leaf_node

    def _information_gain(self, d_usage, att, *params):
        info_gain, overcomes, sub_usages = \
            self._generate_information_of_specified_attribute(
                d_usage, att, params[0])
        return info_gain, overcomes, sub_usages

    def _information_entropy(self, d_usage, *params):
        privacy_value = params[0]["privacy_value"]
        total_num = params[0]["total_num"]
        info_value = 0
        for class_value in self.class_att_value:
            c_usage = d_usage & (
                    self._training_data[self.class_att] == class_value)
            c_num = sum(c_usage) + self.noisy(privacy_value)
            if c_num > 0:
                info_value += c_num * math.log2(c_num / total_num)
        return info_value

    def get_information_of_discrete_attribute(self, d_usage, att, *params):
        info_value = 0
        overcomes = list()
        usages = []
        privacy_value = params[0]["privacy_value"]
        for value in self._attribute_values[att]:
            sub_usage = d_usage & (self._training_data[att] == value)
            sub_num = sum(sub_usage) + self.noisy(privacy_value)
            if sub_num <= 0:
                continue
            params[0]["total_num"] = sub_num
            info_value += self._information_entropy(sub_usage, params[0])
            overcomes.append(value)
            usages.append(sub_usage)
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

        privacy_value = params[0]["privacy_value"]
        l_usage, r_usage = self.get_left_right_usage(att, d_usage, threshold)
        l_num = sum(l_usage) + self.noisy(privacy_value)
        r_num = sum(r_usage) + self.noisy(privacy_value)

        p_info = 0
        if l_num > 0:
            p_info += self._information_entropy(
                l_usage, {"privacy_value": privacy_value, "total_num": l_num})
        if r_num > 0:
            p_info += self._information_entropy(
                r_usage, {"privacy_value": privacy_value, "total_num": r_num})
        return p_info, ["<<"+str(threshold),
                        ">="+str(threshold)], [l_usage, r_usage]

    def construct_sub_tree(self, parent_node, d_usage, candidate_attributes,
                           max_depth, outcome):
        candidate_attribute_num = len(candidate_attributes)
        total_num_with_noisy = self.get_num_with_noisy(
            d_usage, self.privacy_budget_per_node)
        att_max_num = self.get_max_num_of_att_values(candidate_attributes,
                                                     d_usage)

        # leaf node
        if candidate_attribute_num == 0 or max_depth == 1 \
                or self.check_termination_condition(total_num_with_noisy,
                                                    att_max_num):
            leaf_node = self.set_leaf_node(d_usage)
            print("NEW LEAF NODE: <%s>" % leaf_node)
            # no parent node
            if not parent_node:
                self.root_node = leaf_node
                self.root_node.set_parent_node(None)
                return
            else:
                parent_node.add_sub_node(outcome, leaf_node)
                return

        # current node is non-leaf
        sub_privacy_value = \
            self.privacy_budget_per_node/candidate_attribute_num
        info_gain, split_attribute, outcomes, sub_usages = \
            self._select_split_attribute(
                d_usage, candidate_attributes,
                {"privacy_value": sub_privacy_value})

        non_leaf_node = \
            NonLeafNode(is_leaf=False, att_name=split_attribute)
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
