from decision_tree.random_forest import RandomForest


rf = RandomForest('adult', training_per=0.7, test_per=0.2, tree_depth=7,
                  rdt_num=10)
statistics = dict()

rf.construct_random_forest()
rf.test_one_test_record()
rf.get_test_results()

exit(1)
rf.get_test_results()
