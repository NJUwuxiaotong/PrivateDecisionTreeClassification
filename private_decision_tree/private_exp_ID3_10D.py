import copy
import math

from common import constants as const
from decision_tree.ID3 import ID3
from decision_tree.decision_tree_node import NonLeafNode, LeafNode
from pub_lib import pub_functions


class PrivateID310D(ID3):
    """
    Literature: Arik Friedman and Assaf Schuster. Data Mining with
    Differential Privacy. in Proceedings of KDD, 2010.
    """
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, privacy_budget=1):
        super(PrivateID310D, self).__init__(
            dataset_name, training_per=training_per, test_per=test_per,
            tree_depth=tree_depth)
        self.privacy_value_per_node = privacy_budget / (2 * self._tree_depth)
        self.sensitivity = math.log(self.training_num+1) + 1/math.log(2)
        self.num_of_ranges_for_int64 = 5

    def check_conditions_of_leaf_node(self, info_gain, max_depth,
                                      candidate_attributes, usage, *params):
        att_max_num = params[0]["att_max_num"]
        record_num = params[0]["record_num"]
        threshold = record_num/(att_max_num*len(self.class_att_value))
        print("Threshold: %s" % threshold)
        return info_gain == 0 or max_depth == 1 \
               or len(candidate_attributes) == 0 \
               or self.check_leaf_same_class(usage) or \
               threshold < math.sqrt(2) / self.privacy_value_per_node

    def noisy(self, sensitivity, privacy_value):
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
            -3, 3, sensitivity/privacy_value)

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

    def get_most_frequent_class(self, usage):
        chosen_label = None
        result = None
        for label in self.class_att_value:
            r = usage & (self._training_data[self.class_att] == label)
            r_num = sum(r)
            noisy_value = self.noisy(self.privacy_value_per_node)
            if (result is None) or (result < r_num + noisy_value):
                result = r_num + noisy_value
                chosen_label = label
        return chosen_label



    def divide_int64_attribute(self, att):
        attribute_range = self.range_of_int64_attributes[att]
        exp_value_list = list()
        for i in range(self.num_of_ranges_for_int64):
            exp_value = 0

        chosen_index = pub_functions.generate_random_value_from_exponential(
            exp_value_list)
        threshold = 0
        return threshold


    def expMech(self, d_usage, candidate_atts, privacy_value):
        """
        func: information function name
        """
        infos = list()
        overcomes = dict()
        for att in candidate_atts:
            if self._attribute_type[att] == const.DFRAME_INT64:
                pass
            else:
                info, overcome = self.information_gain(d_usage, att)
                info = privacy_value * info / (
                        2 * self.information_gain_sensitivity())
                infos.append(info)
                overcomes[att] = overcome
        chosen_index = pub_functions.generate_random_value_from_exponential(
            infos)
        chosen_att = candidate_atts[chosen_index]
        chosen_overcome = overcomes[chosen_att]
        print(chosen_overcome)
        exit(1)
        return chosen_att, list(overcomes[chosen_att].keys()), \
               list(overcomes[chosen_att].values())

    def information_gain_for_int64_attribute(self, d_usage, value_ranges):
        info = 0
        overcomes = dict()
        if self._attribute_type[att] == const.DFRAME_INT64:
            pass
        else:
            unique_values = self._training_data[att][d_usage].drop_duplicates(
               keep='first').values
            for value in unique_values:
                v_usage = d_usage & (self._training_data[att] == value)
                v_num = sum(v_usage)
                if v_num == 0:
                    continue

                overcomes[value] = v_usage
                for class_value in self.class_att_value:
                    c_usage = v_usage & (
                            self._training_data[self.class_att] == class_value)
                    c_num = sum(c_usage)
                    if c_num > 0:
                        p = c_num / v_num
                        info -= p * math.log2(p)
        return info, overcomes

    def information_gain(self, d_usage, att):
        info = 0
        overcomes = dict()
        if self._attribute_type[att] == const.DFRAME_INT64:
            pass
        else:
            unique_values = self._training_data[att][d_usage].drop_duplicates(
               keep='first').values
            for value in unique_values:
                v_usage = d_usage & (self._training_data[att] == value)
                v_num = sum(v_usage)
                if v_num == 0:
                    continue

                overcomes[value] = v_usage
                for class_value in self.class_att_value:
                    c_usage = v_usage & (
                            self._training_data[self.class_att] == class_value)
                    c_num = sum(c_usage)
                    if c_num > 0:
                        p = c_num / v_num
                        info -= p * math.log2(p)
        return info, overcomes

    def information_gain_sensitivity(self):
        return math.log2(self._training_data_shape[0] + 1) + 1/math.log(2)

    def gini_index(self, d_usage, att):
        """
        used in CART algorithm.
        """
        g_info = 0
        overcomes = dict()
        unique_values = self._training_data[att][d_usage].drop_duplicates(
            keep='first').values
        for value in unique_values:
            info = 0
            v_usage = d_usage & (self._training_data[att] == value)
            v_num = sum(v_usage)
            if v_num == 0:
                continue

            overcomes[value] = v_usage
            for class_value in self.class_att_value:
                c_usage = v_usage & \
                          (self._training_data[self.class_att] == class_value)
                c_num = sum(c_usage)
                if c_num > 0:
                    p = c_num / v_num
                    info += p * p
            g_info -= v_num * (1 - info)
        return g_info, overcomes

    def gini_index_sensitivity(self):
        return 2

    def max_operator(self, d_usage, att):
        info = 0
        overcomes = dict()
        unique_values = self._training_data[att][d_usage].drop_duplicates(
            keep='first').values
        for value in unique_values:
            v_usage = d_usage & (self._training_data[att] == value)
            for class_value in self.class_att_value:
                c_usage = v_usage \
                          & (self._training_data[self.class_att] == class_value)
                c_num = sum(c_usage)
                if c_num > info:
                    info = c_num
        return info, overcomes

    def max_operator_sensitivity(self):
        return 1

    def get_num_with_noisy(self, usage, sensivity, privacy_value):
        u_num = sum(usage)
        return u_num + self.noisy(sensivity, sensivity/privacy_value)

    def check_termination_condition(self, record_num, max_att_num):
        condition = record_num / (max_att_num * len(self._attribute_values)) \
                    < math.sqrt(2) / self.privacy_value_per_node
        return condition

    def construct_sub_tree(self, parent_node, d_usage, candidate_attributes,
                           max_depth, outcome):
        candidate_attribute_num = len(candidate_attributes)
        total_num_with_noisy = self.get_num_with_noisy(
            d_usage, 1, self.privacy_value_per_node)
        att_max_num = self.get_max_num_of_att_values(candidate_attributes,
                                                     d_usage)
        # current node is leaf
        if candidate_attribute_num == 0 or max_depth == 1 \
                or self.check_termination_condition(total_num_with_noisy,
                                                    att_max_num):
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
        split_att, outcomes, sub_usages = \
            self.expMech(d_usage, candidate_attributes,
                         self.privacy_value_per_node)
        non_leaf_node = NonLeafNode(is_leaf=False, att_name=split_att)
        print("New NON-LEAF NODE: <%s>" % split_att)
        if not parent_node:
            self.root_node = non_leaf_node
        else:
            parent_node.add_sub_node(outcome, non_leaf_node)

        num = 0
        for sub_outcome in outcomes:
            sub_candidate_atts = copy.deepcopy(candidate_attributes)
            sub_candidate_atts = sub_candidate_atts[
                sub_candidate_atts != split_att]
            self.construct_sub_tree(non_leaf_node, sub_usages[num],
                                    sub_candidate_atts, max_depth-1,
                                    sub_outcome)
            num += 1
