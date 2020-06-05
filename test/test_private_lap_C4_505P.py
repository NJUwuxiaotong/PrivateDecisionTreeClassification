from private_decision_tree.private_lap_C4_5_05P import PrivateLapC4505P


private_c45 = PrivateLapC4505P(
    'adult', training_per=0.4, test_per=0.3, tree_depth=7, privacy_budget=20)
private_c45.construct_tree()
private_c45.show_structure_of_decision_tree(private_c45.root_node, None, "")
private_c45.get_test_results()
