from private_decision_tree.private_single_decision_tree_classifier.\
    private_exp_CART_10D import PrivateCART10D


private_cart = PrivateCART10D(
    'adult', training_per=0.4, test_per=0.3, tree_depth=7, privacy_budget=10)
private_cart.construct_tree()
private_cart.show_structure_of_decision_tree(private_cart.root_node, None, "")
private_cart.get_test_results()
