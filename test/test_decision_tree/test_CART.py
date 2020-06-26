from decision_tree.single_decision_tree_classifier.CART import CART


cart = CART('adult', training_per=0.1, test_per=0.3, tree_depth=5)
cart.construct_tree()

print("\n------------ ID3 DECISION STRUCTURE ------------------")
cart.show_structure_of_decision_tree(cart.root_node, None, "")
print("------------------------------------------------------\n")

cart.get_test_results()
