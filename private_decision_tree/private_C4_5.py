from decision_tree.C4_5 import C4_5


class PrivateC4_5(C4_5):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, privacy_parameter=1):
        super(C4_5, self).__init__(dataset_name, training_per, test_per,
                                   tree_depth)
        self.privacy_parameter = privacy_parameter
        self.privacy_per_level = privacy_parameter/self._tree_depth

    def get_attribute(self):
        pass

    def laplace_mechanism(self):
        pass

    def exponential_mechanism(self):
        pass
