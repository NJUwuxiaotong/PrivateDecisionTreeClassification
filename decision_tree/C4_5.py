from decision_tree.decision_tree import DecisionTree


class C4_5(DecisionTree):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, is_private=False, privacy_value=1):
        super(C4_5, self).__init__(dataset_name, training_per, test_per,
                                   tree_depth, is_private, privacy_value)
