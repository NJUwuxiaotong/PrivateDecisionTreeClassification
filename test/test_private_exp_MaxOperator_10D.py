from private_decision_tree.private_exp_MaxOperator_10D \
    import PrivateMaxOperator10D


private_max_operator = PrivateMaxOperator10D(
    'adult', training_per=0.4, test_per=0.3, tree_depth=7, privacy_budget=10)
private_max_operator.construct_tree()
private_max_operator.show_structure_of_decision_tree(private_max_operator.root_node, None, "")
private_max_operator.get_test_results()
