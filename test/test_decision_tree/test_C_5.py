from decision_tree.C4_5 import C45


c45 = C45('adult', training_per=0.3, test_per=0.3, tree_depth=7)
c45.construct_tree()

print("\n------------ C4.5 DECISION STRUCTURE ------------------")
c45.show_structure_of_decision_tree(c45.root_node, None, "")
print("------------------------------------------------------\n")

c45.get_test_results()
