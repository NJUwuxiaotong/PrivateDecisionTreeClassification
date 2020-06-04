import os

from common import constants as const
from data_process.data_preprocess import DataPreProcess
from decision_tree.decision_tree import DecisionTree


class PrivateDecisionTree(DecisionTree):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, privacy_budget=None):
        super(PrivateDecisionTree, self).__init__(dataset_name, training_per,
                                                  test_per, tree_depth)
        self.privacy_budget = privacy_budget

    def construct_sub_tree(self, parent_node, d_usage, candidate_attributes,
                           max_depth, outcome):
        pass
