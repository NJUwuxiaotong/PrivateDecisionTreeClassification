from pub_lib import pub_functions as pf
from decision_tree.random_decision_tree_classifier.random_decision_tree \
    import RandomDecisionTree


def add_noisy_for_random_decision_tree(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args)
        leaf_nodes = self.get_leaf_nodes(self.root_node)
        for leaf_node in leaf_nodes:
            for class_label, class_value in leaf_node.class_values.items():
                leaf_node.class_values[class_label] += \
                    pf.generate_random_value_from_Laplace(
                        self.noisy_lower, self.noisy_upper, self.privacy_budget)
    return wrapper


class PrivateRandomDecisionTree(RandomDecisionTree):
    def __init__(self, attribute_types, attribute_values,
                 range_of_int64_attributes, class_att, privacy_budget,
                 noisy_boundary):
        super(PrivateRandomDecisionTree, self).__init__(
            attribute_types, attribute_values, range_of_int64_attributes,
            class_att)
        self.privacy_budget = privacy_budget
        self.noisy_upper = noisy_boundary
        self.noisy_lower = -1 * noisy_boundary

    @add_noisy_for_random_decision_tree
    def update_statistics(self, data, record_num):
        super(PrivateRandomDecisionTree, self).update_statistics(
            data, record_num)

    def prune_random_decision_tree(self):
        pass
