from private_decision_tree.private_C4_5_05P import PrivateC4_5_05P



private_c45 = PrivateC4_5_05P('adult', training_per=0.1, test_per= 0.3,
                              tree_depth=7, privacy_value=1)
private_c45.construct_tree()
private_c45.show_C4_5(private_c45.root_node, None, "")
private_c45.get_test_results()
