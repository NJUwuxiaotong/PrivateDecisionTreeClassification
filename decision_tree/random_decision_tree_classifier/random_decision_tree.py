import copy
import random

from common import constants as const
from decision_tree.single_decision_tree_classifier.decision_tree_node \
    import NonLeafNode, LeafNode


class RandomDecisionTree(object):
    def __init__(self, attribute_types, attribute_values,
                 range_of_int64_attributes, class_att):
        self._attribute_type = attribute_types
        self._attribute_values = attribute_values
        self._class_att = class_att
        self.range_of_int64_attributes = range_of_int64_attributes
        self.root_node = None
        self.candidate_attributes = []
        self._leaf_nodes = list()

    def generate_candidate_attributes(self, attributes, tree_depth):
        attributes = copy.deepcopy(attributes)
        # int64_attributes = list(self.range_of_int64_attributes.keys())
        # discrete_attributes = list(set(attributes) - set(int64_attributes))
        # attributes = discrete_attributes
        attribute_num = len(attributes)
        print("Info: Start to randomly select %s attributes" % tree_depth)
        for i in range(tree_depth - 1):
            attribute_index = random.randint(0, attribute_num - 1)
            chosen_attribute = attributes.pop(attribute_index)
            self.candidate_attributes.append(chosen_attribute)
            attribute_num -= 1
        print("Info: randomly select attributes: %s" %
              self.candidate_attributes)

    def construct_tree_structure(self):
        print("Info: Start to construct random decision tree ...")
        print("Info: Construct non-leaf nodes ...")
        self.root_node = self.construct_sub_trees(
            None, self.candidate_attributes)

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
                # threshold = min_v + random.random() * (max_v - min_v)
                threshold = (min_v + max_v)/2
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

    def update_statistics(self, data, record_num):
        print("Info: Start to add training data")
        for i in range(record_num):
            record = data[i:i+1]
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
            class_value = record[self._class_att].values[0]
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

    def get_class_label_of_record(self, record):
        node = self.root_node
        while True:
            if node.is_leaf:
                return node.class_values

            att = node.att_name
            value = record[att].values[0]
            sub_nodes = node.sub_nodes
            if self._attribute_type[att] == const.DFRAME_INT64:
                node_keys = list(sub_nodes.keys())

                if len(node_keys) == 0:
                    return None

                if len(node_keys) == 1:
                    bp = float(node_keys[0][2:])
                    if (value < bp and node_keys[0][:2] == "<<") or \
                            (value > bp and node_keys[0][:2] == ">="):
                        node = node.sub_nodes[node_keys[0]]
                    else:
                        return None

                if len(node_keys) == 2:
                    l_bp = node_keys[0]
                    r_bp = node_keys[1]
                    bp = float(l_bp[2:])

                    if value < bp:
                        if l_bp[:2] == "<<":
                            node = node.sub_nodes[l_bp]
                        else:
                            node = node.sub_nodes[r_bp]
                    else:
                        if l_bp[:2] == "<<":
                            node = node.sub_nodes[r_bp]
                        else:
                            node = node.sub_nodes[l_bp]
            else:
                if value not in sub_nodes.keys():
                    return None
                else:
                    node = node.sub_nodes[value]

    def test_training_records(self, data, record_num):
        num = 0
        for i in range(record_num):
            record = data[i:i + 1]
            class_value = self.get_class_label_of_record(record)
            if class_value is None:
                pass
            else:
                num += 1
        print("RESULT: %s " % num)

    def test_one_test_records(self, record):
        print(record)
        print(self.get_class_label_of_record(record))

    def _information_metric(self, d_usage, att, *params):
        pass
