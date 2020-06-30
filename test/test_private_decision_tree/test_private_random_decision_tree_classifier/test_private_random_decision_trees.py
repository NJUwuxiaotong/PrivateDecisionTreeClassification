from private_decision_tree.private_random_decision_tree_classifier\
    .private_random_decision_trees import PrivateRandomDecisionTrees


# dataset: adult, nursery
rf = PrivateRandomDecisionTrees(
    'nursery', 0.01, training_per=0.8, test_per=0.2, rdt_num=20)
statistics = dict()

rf.construct_random_forest()

#rf.random_decision_trees[0].test_training_records(
#    rf._training_data, rf._training_num)

rf.get_test_results()
