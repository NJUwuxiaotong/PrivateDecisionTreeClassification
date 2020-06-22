import copy
import random

from common import constants as const
from decision_tree.decision_tree import DecisionTree
from decision_tree.decision_tree_node import NonLeafNode, LeafNode


class RandomDecisionTree(DecisionTree):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None):
        super(RandomDecisionTree, self).__init__(dataset_name, training_per,
                                                 test_per, tree_depth)
        if tree_depth >= self._attribute_num:
            print("Error: The depth of random decision tree is greater than "
                  "the number of attributes, e.g., %s > %s." %
                  (tree_depth, self._attribute_num))
            exit(1)
        self.candidate_attributes = None
        self._leaf_nodes = list()

    def generate_candidate_attributes(self):
        self.candidate_attributes = []
        attributes = copy.deepcopy(self._attributes)
        attributes = attributes.tolist()
        attribute_num = len(attributes)
        print("Info: Start to randomly select %s attributes" % self._tree_depth)
        for i in range(self._tree_depth - 1):
            attribute_index = random.randint(0, attribute_num - 1)
            chosen_attribute = attributes.pop(attribute_index)
            self.candidate_attributes.append(chosen_attribute)
            attribute_num -= 1
        print("Info: randomly select attributes: %s" %
              self.candidate_attributes)

    def construct_tree(self):
        print("Info: Start to construct random decision tree ...")
        print("Info: Construct non-leaf nodes ...")
        self.root_node = self.construct_sub_trees(
            None, self.candidate_attributes)
        print("Info: End non-leaf nodes")
        self.update_statistics()
        print("Info: End to construct random decision tree.")

    def construct_sub_trees(self, parent_node, candidate_attributes):
        if not candidate_attributes:
            leaf_node = LeafNode(parent_node, is_leaf=True)
            return leaf_node
        else:
            candidate_attribute_num = len(candidate_attributes)
            chosen_index = random.randint(0, candidate_attribute_num - 1)
            chosen_attribute = candidate_attributes.pop(chosen_index)
            non_leaf_node = NonLeafNode(parent_node, is_leaf=False,
                                        att_name=chosen_attribute)
            if self._attribute_type[chosen_attribute] == const.DFRAME_INT64:
                attribute_range = \
                    self.range_of_int64_attributes[chosen_attribute]
                max_v = attribute_range[0]
                min_v = attribute_range[1]
                threshold = min_v + random.random() * (max_v - min_v)
                new_candidate_attributes = copy.deepcopy(
                    candidate_attributes)
                sub_node = self.construct_sub_trees(
                    non_leaf_node, new_candidate_attributes)
                sub_node.set_parent_index("<<"+str(threshold))
                non_leaf_node.add_sub_node("<<"+str(threshold), sub_node)
                new_candidate_attributes = copy.deepcopy(
                    candidate_attributes)
                sub_node = self.construct_sub_trees(
                    non_leaf_node, new_candidate_attributes)
                sub_node.set_parent_index(">="+str(threshold))
                non_leaf_node.add_sub_node(">="+str(threshold), sub_node)
            else:
                attribute_values = self._attribute_values[chosen_attribute]
                for attribute_value in attribute_values:
                    new_candidate_attributes = copy.deepcopy(
                        candidate_attributes)
                    sub_node = self.construct_sub_trees(
                        non_leaf_node, new_candidate_attributes)
                    sub_node.set_parent_index(attribute_value)
                    non_leaf_node.add_sub_node(attribute_value, sub_node)
        return non_leaf_node

    def update_statistics(self):
        print("Info: Start to add training data")
        for i in range(self._training_data_shape[0]):
            record = self._training_data[i:i+1]
            self.add_instance(self.root_node, record)
        print("Info: End to add training data")

    def add_instance(self, node, record):
        if not node.is_leaf:
            # find sub-node of current node
            current_attribute = node.att_name
            record_value = record[current_attribute]
            r_value = record_value.values[0]
            if self._attribute_type[current_attribute] == const.DFRAME_INT64:
                for threshold, sub_node in node.sub_nodes.items():
                    prefix = threshold[:2]
                    value = float(threshold[2:])
                    if (prefix == "<<" and r_value < value) or \
                            (prefix == ">=" and r_value >= value):
                        self.add_instance(sub_node, record)
            else:
                sub_nodes = node.sub_nodes
                sub_node_keys = sub_nodes.keys()
                if r_value in sub_node_keys:
                    self.add_instance(sub_nodes[r_value], record)
        else:
            class_value = record[self.class_att].values[0]
            node.increment_class_value(class_value)

    def get_leaf_nodes(self, current_node):
        if not current_node:
            return
        if current_node.is_leaf:
            self._leaf_nodes.append(current_node)
        else:
            for sub_outcome, sub_node in current_node.sub_nodes.items():
                self.get_leaf_nodes(sub_node)

    def prune_random_decision_tree(self):
        self.get_leaf_nodes(self.root_node)
        for leaf_node in self._leaf_nodes:
            self._prune_random_decision_tree(leaf_node)

    def _prune_random_decision_tree(self, current_node):
        if (current_node.is_leaf and not current_node.class_values) or \
                (not current_node.is_leaf and not current_node.sub_nodes):
            parent_node = current_node.parent_node
            parent_index = current_node.parent_index
            parent_node.sub_nodes.pop(parent_index)
            # prune parent node
            if parent_node is not None:
                self._prune_random_decision_tree(parent_node)

    def _information_metric(self, d_usage, att, *params):
        pass
