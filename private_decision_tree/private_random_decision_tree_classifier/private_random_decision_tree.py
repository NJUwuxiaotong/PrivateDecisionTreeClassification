from pub_lib import pub_functions as pf
from decision_tree.random_decision_tree_classifier.random_decision_tree \
    import RandomDecisionTree


def add_noisy_for_random_decision_tree(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args)
        self.get_leaf_nodes(self.root_node)
        for leaf_node in self._leaf_nodes:
            for class_label, class_value in leaf_node.class_values.items():
                noisy = pf.generate_random_value_from_Laplace(
                        self.noisy_lower, self.noisy_upper, self.privacy_budget)
                leaf_node.class_values[class_label] += noisy
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

    #@add_noisy_for_random_decision_tree
    def update_statistics(self, data, record_num):
        # super(PrivateRandomDecisionTree, self).update_statistics(
        #    data, record_num)
        print("Info: Start to add training data")
        for i in range(record_num):
            record = data[i:i+1]
            self.add_instance(self.root_node, record)
        print("Info: End to add training data")

        self._leaf_nodes = []
        self.get_leaf_nodes(self.root_node)
        for leaf_node in self._leaf_nodes:
            for class_label, class_value in leaf_node.class_values.items():
                noisy = pf.generate_random_value_from_Laplace(
                        self.noisy_lower, self.noisy_upper, self.privacy_budget)
                leaf_node.class_values[class_label] += noisy

        self._leaf_nodes = []
        self.get_leaf_nodes(self.root_node)
        #import pdb; pdb.set_trace()

    def prune_random_decision_tree(self):
        pass
