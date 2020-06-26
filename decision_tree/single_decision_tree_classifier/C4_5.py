import math

from decision_tree.single_decision_tree_classifier.ID3 import ID3


class C45(ID3):
    def _select_split_attribute(self, d_usage, candidate_attributes, *params):
        gain_ratio = 0
        split_attribute = None
        sub_usages = None
        overcomes = None

        total_num = sum(d_usage)
        for att in candidate_attributes:
            information_value = 0
            can_info_gain, can_overcomes, can_sub_usages = \
                self._generate_information_of_specified_attribute(
                    d_usage, att, params[0])

            for sub_usage in can_sub_usages:
                p = sum(sub_usage)/total_num
                if p > 0:
                    information_value -= p * math.log2(p)
            if information_value == 0:
                return 0, att, can_overcomes, can_sub_usages

            can_gain_ratio = can_info_gain/information_value
            if gain_ratio < can_gain_ratio:
                gain_ratio = can_gain_ratio
                split_attribute = att
                sub_usages = can_sub_usages
                overcomes = can_overcomes
        return gain_ratio, split_attribute, overcomes, sub_usages

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
