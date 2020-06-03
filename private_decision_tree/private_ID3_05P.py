import copy
import math
import time

from common import constants as const
from decision_tree.ID3 import ID3
from decision_tree.decision_tree_node import NonLeafNode, LeafNode
from pub_lib import pub_functions


class PrivateID3_05P(ID3):
    """
    Literature: Avrim Blum, Cynthia Dwork, Frank McSherry, and Kobbi Nissim.
    Practical privacy: the SuLQ framework. in ACM SIGMOD-SIGACT-SIGART Symposium
    on Principles of Database Systems. ACM, 128-138, 2005.
    """
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, privacy_value=1):
        super(PrivateID3_05P, self).__init__(
            dataset_name, training_per=training_per, test_per=test_per,
            tree_depth=tree_depth, is_private=True,
            privacy_value=privacy_value)
        self.privacy_value_per_node = \
            privacy_value / (2 * self._tree_depth)
        self.sensitivity = 1
        self.privacy_parameter = self.sensitivity/self.privacy_value_per_node

    def check_conditions_of_leaf_node(self, info_gain, max_depth,
                                      candidate_atts, usage, *params):
        att_max_num = params[0]["att_max_num"]
        record_num = params[0]["record_num"]
        threshold = record_num/(att_max_num*len(self.class_att_value))
        print("Threshold: %s" % threshold)
        return info_gain == 0 or max_depth == 1 or len(candidate_atts) == 0 \
               or self.check_leaf_same_class(usage) or \
               threshold < math.sqrt(2)/self.privacy_value_per_node

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
            -3, 3, self.privacy_parameter)

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

    def construct_sub_tree(self, parent_node, d_usage, candidate_attributes,
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

        att_max_num = self.get_max_num_of_att_values(candidate_attributes, d_usage)
        record_num = self.get_num_of_records(d_usage, True,
                                             self.privacy_value_per_node)

        info = self._information_entropy(d_usage, record_num, is_privacy=False)
        info_gain, split_att, outcomes, sub_usages = \
            self._select_split_attribute(d_usage, candidate_attributes)
        info_gain = info + info_gain

        # current node is leaf
        if self.check_conditions_of_leaf_node(
                info_gain, max_depth, candidate_attributes, d_usage,
                {"att_max_num": att_max_num, "record_num": record_num}):
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
            sub_candidate_atts = copy.deepcopy(candidate_attributes)
            sub_candidate_atts = sub_candidate_atts[
                sub_candidate_atts != split_att]
            self.construct_sub_tree(non_leaf_node, sub_usages[num],
                                    sub_candidate_atts, max_depth-1,
                                    sub_outcome)
            num += 1
