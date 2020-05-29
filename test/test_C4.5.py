from decision_tree.C4_5 import C4_5


c45 = C4_5('adult', training_per=0.7, test_per= 0.3, tree_depth=7)
c45.construct_tree()
c45.show_C4_5(c45.root_node, None, "")
c45.get_test_results()
