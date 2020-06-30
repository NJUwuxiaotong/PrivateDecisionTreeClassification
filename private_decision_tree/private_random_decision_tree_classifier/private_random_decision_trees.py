
from private_decision_tree.private_random_decision_tree_classifier.\
    private_random_decision_tree import PrivateRandomDecisionTree
from decision_tree.random_decision_tree_classifier.random_decision_trees \
    import RandomDecisionTrees


class PrivateRandomDecisionTrees(RandomDecisionTrees):
    def __init__(self, dataset_name, privacy_budget, training_per=0.7,
                 test_per=0.3, tree_depth=None, rdt_num=10, noisy_boundary=30):
        super(PrivateRandomDecisionTrees, self).__init__(
            dataset_name, training_per, test_per, tree_depth, rdt_num)
        self.privacy_budget = 1/privacy_budget
        self.noisy_boundary = noisy_boundary

    def construct_random_forest(self):
        for i in range(self.rdt_num):
            random_decision_tree = PrivateRandomDecisionTree(
                self._attribute_types, self._attribute_values,
                self._range_of_int64_attributes, self._class_attribute,
                self.privacy_budget, self.noisy_boundary)
            random_decision_tree.generate_candidate_attributes(
                self._attributes, self._tree_depth)
            random_decision_tree.construct_tree_structure()
            random_decision_tree.update_statistics(
                self._training_data, self._training_num)
            # random_decision_tree.prune_random_decision_tree()
            self.random_decision_trees.append(random_decision_tree)
