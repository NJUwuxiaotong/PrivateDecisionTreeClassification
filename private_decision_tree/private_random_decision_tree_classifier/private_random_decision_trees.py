from decision_tree.random_decision_tree_classifier.random_decision_trees \
    import RandomDecisionTrees


class PrivateRandomDecisionTrees(RandomDecisionTrees):
    def __init__(self, dataset_name, privacy_budget, training_per=0.7,
                 test_per=0.3, tree_depth=None, rdt_num=10):
        super(PrivateRandomDecisionTrees, self).__init__(
            dataset_name, training_per, test_per, tree_depth, rdt_num)
        self.privacy_budget = privacy_budget
