from decision_tree.max_operator import MaxOperator


id3 = MaxOperator('adult', training_per=0.1, test_per=0.3, tree_depth=5)
id3.construct_tree()

print("\n------------ ID3 DECISION STRUCTURE ------------------")
id3.show_structure_of_decision_tree(id3.root_node, None, "")
print("------------------------------------------------------\n")

id3.get_test_results()
