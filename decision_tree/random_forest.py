from decision_tree.decision_tree import DecisionTree

from decision_tree.random_decision_tree import RandomDecisionTree


class RandomForest(DecisionTree):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, rdt_num=10):
        super(RandomForest, self).__init__(
            dataset_name, training_per, test_per, tree_depth)
        self.rdt_num = rdt_num
        self.random_decision_trees = list()

    def construct_random_forest(self):
        for i in range(self.rdt_num):
            random_decision_tree = RandomDecisionTree(
                self._dataset_name, self.training_per, self.test_per,
                self._tree_depth)
            random_decision_tree.generate_candidate_attributes()
            random_decision_tree.construct_tree()
            random_decision_tree.prune_random_decision_tree()
            self.random_decision_trees.append(random_decision_tree)


