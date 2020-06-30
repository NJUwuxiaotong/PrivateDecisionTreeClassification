from decision_tree.random_decision_tree_classifier.random_decision_trees \
    import RandomDecisionTrees


# dataset: adult, nursery, agaricus-lepiota
rf = RandomDecisionTrees('agaricus-lepiota', training_per=0.5, test_per=0.1,
                         rdt_num=20)
statistics = dict()

rf.construct_random_forest()

#rf.random_decision_trees[0].test_training_records(
#    rf._training_data, rf._training_num)

rf.get_test_results()
