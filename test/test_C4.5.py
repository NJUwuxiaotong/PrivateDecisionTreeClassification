from decision_tree.C4_5 import C45


c45 = C45('adult', training_per=0.001, tree_depth=5)
c45.construct_tree()
c45.show_C4_5(c45.root_node, None, "")
right_ratio = c45.get_test_results()
print(right_ratio)
