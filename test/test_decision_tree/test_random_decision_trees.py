from decision_tree.random_decision_tree_classifier.random_decision_trees import RandomDecisionTrees


# dataset: adult, nursery
rf = RandomDecisionTrees('nursery', training_per=0.9, test_per=0.1,
                         rdt_num=20)
statistics = dict()

rf.construct_random_forest()
rf.get_test_results()
