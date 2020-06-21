import random

from decision_tree.decision_tree import DecisionTree


class RandomDecisionTree(DecisionTree):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None):
        super(RandomDecisionTree, self).__init__(dataset_name, training_per,
                                                 test_per, tree_depth)
        if tree_depth >= self._attribute_num:
            print("Error: The depth of random decision tree is greater than "
                  "the number of attributes, e.g., %s > %s." %
                  (tree_depth, self._attribute_num))
            exit(1)
        self.candidate_attributes = None

    def generate_candidate_attributes(self):
        self.candidate_attributes = []
        attributes = list(self._attributes.keys())
        attribute_num = len(attributes)
        print("Info: Start to randomly select %s attributes" % self._tree_depth)
        for i in range(self._tree_depth):
            attribute_index = random.randint(0, attribute_num - 1)
            chosen_attribute = attributes.pop(attribute_index)
            self.candidate_attributes.append(chosen_attribute)
            attribute_num -= 1

    def construct_tree(self):
        pass

    def construct_sub_trees(self, candidate_attributes):
        pass

    def _information_metric(self, d_usage, att, *params):
        pass
