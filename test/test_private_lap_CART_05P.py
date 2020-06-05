from private_decision_tree.private_lap_CART_05P import PrivateLapCART05P


private_cart = PrivateLapCART05P(
    'adult', training_per=0.4, test_per=0.3, tree_depth=7, privacy_budget=20)
private_cart.construct_tree()
private_cart.show_structure_of_decision_tree(private_cart.root_node, None, "")
private_cart.get_test_results()
