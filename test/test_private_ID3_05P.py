from private_decision_tree.private_ID3_05P import PrivateID3_05P



private_c45 = PrivateID3_05P('adult', training_per=0.2, test_per= 0.3,
                             tree_depth=7, privacy_value=20)
private_c45.construct_tree()
private_c45.show_structure_of_decision_tree(private_c45.root_node, None, "")
private_c45.get_test_results()
