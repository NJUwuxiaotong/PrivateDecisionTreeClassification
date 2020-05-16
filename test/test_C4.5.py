from decision_tree.C4_5 import C45


c45 = C45('adult', tree_depth=5)
c45.construct_tree()
c45.show_C4_5(c45.root_node, None, "")
