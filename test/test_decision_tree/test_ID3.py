from decision_tree.single_decision_tree_classifier.ID3 import ID3


id3 = ID3('adult', training_per=0.5, test_per=0.2, tree_depth=5)
id3.construct_tree()

print("\n------------ ID3 DECISION STRUCTURE ------------------")
id3.show_structure_of_decision_tree(id3.root_node, None, "")
print("------------------------------------------------------\n")

id3.get_test_results()
