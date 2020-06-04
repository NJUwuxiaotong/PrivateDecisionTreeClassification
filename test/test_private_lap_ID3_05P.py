from private_decision_tree.private_lap_ID3_05P import PrivateLapID305P


private_id3 = PrivateLapID305P(
    'adult', training_per=0.4, test_per=0.3, tree_depth=7, privacy_budget=20)
private_id3.construct_tree()
private_id3.show_structure_of_decision_tree(private_id3.root_node, None, "")
private_id3.get_test_results()
