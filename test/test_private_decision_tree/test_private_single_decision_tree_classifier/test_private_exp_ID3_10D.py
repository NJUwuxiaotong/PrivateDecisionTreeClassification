from private_decision_tree.private_single_decision_tree_classifier.\
    private_exp_ID3_10D import PrivateID310D

private_id3 = PrivateID310D(
    'adult', training_per=0.4, test_per=0.3, tree_depth=7, privacy_budget=10)
private_id3.construct_tree()
private_id3.show_structure_of_decision_tree(private_id3.root_node, None, "")
private_id3.get_test_results()
