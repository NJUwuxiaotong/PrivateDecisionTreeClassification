from decision_tree.random_decision_tree import RandomDecisionTree


rdt = RandomDecisionTree('adult', training_per=0.5, test_per=0.2, tree_depth=3)
statistics = dict()
rdt.generate_candidate_attributes()
rdt.construct_tree()
rdt.show_statistics_of_decision_tree(rdt.root_node, statistics)

print("\n------------ RANDOM DECISION STRUCTURE ------------------")
rdt.show_structure_of_decision_tree(rdt.root_node, None, "")
print("-----------------------------------------------------------")
print("Check statistics vs training data: %s-%s " %
      (sum(statistics.values()), rdt.training_num))
print("---------------------------------------------------------\n")

rdt.prune_random_decision_tree()
rdt.show_structure_of_decision_tree(rdt.root_node, None, "")
rdt.show_statistics_of_decision_tree(rdt.root_node, statistics)
print("Check statistics vs training data: %s-%s " %
      (sum(statistics.values()), rdt.training_num))

rdt.test_training_records()
rdt.test_one_test_records()

exit(1)
rdt.get_test_results()
