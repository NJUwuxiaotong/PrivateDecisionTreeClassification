from private_decision_tree.private_random_decision_tree_classifier\
    .private_random_decision_trees import PrivateRandomDecisionTrees


# dataset: adult, nursery
rf = PrivateRandomDecisionTrees(
    'nursery', 1, training_per=0.9, test_per=0.1, rdt_num=20)
statistics = dict()

rf.construct_random_forest()
rf.get_test_results()
