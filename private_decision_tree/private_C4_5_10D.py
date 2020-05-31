from decision_tree.C4_5 import C4_5
from pub_lib import pub_functions


class PrivateC4_5_10D(C4_5):
    """
    Literature: Arik Friedman and Assaf Schuster. Data Mining with
    Differential Privacy. in Proceedings of KDD, 2010.
    """
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, privacy_value=1):
        super(PrivateC4_5_10D, self).__init__(
            dataset_name, training_per=training_per, test_per=test_per,
            tree_depth=tree_depth, is_private=True,
            privacy_value=privacy_value)
