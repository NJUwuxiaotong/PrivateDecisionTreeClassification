from decision_tree.decision_tree import DecisionTree

from decision_tree.random_decision_tree import RandomDecisionTree


class RandomForest(DecisionTree):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, rdt_num=10):
        super(RandomForest, self).__init__(
            dataset_name, training_per, test_per, tree_depth)
        self.rdt_num = rdt_num
        self.random_decision_trees = list()

    def construct_random_forest(self):
        for i in range(self.rdt_num):
            random_decision_tree = RandomDecisionTree(
                self._dataset_name, self.training_per, self.test_per,
                self._tree_depth)
            random_decision_tree.generate_candidate_attributes()
            random_decision_tree.construct_tree()
            random_decision_tree.prune_random_decision_tree()
            self.random_decision_trees.append(random_decision_tree)

    def check_class_value_of_record(self, record):
        class_value_statistics = dict()
        for i in range(self.rdt_num):
            class_value = self.random_decision_trees[i]\
                .get_class_label_of_record(record)
            if class_value is None:
                continue
            for key_class, value in class_value.items():
                if key_class in class_value_statistics.keys():
                    class_value_statistics[key_class] += value
                else:
                    class_value_statistics[key_class] = value

        values = class_value_statistics.values()
        value_sum = sum(values)
        results = dict()
        for key_class, value in class_value_statistics.items():
            results[key_class] = value/value_sum
        return results

    def test_one_test_record(self):
        record = self._test_data[0:1]
        print(record)
        print(self.check_class_value_of_record(record))

    def predict_class_value_of_record_by_max_probability(self, record):
        predicted_probabilities = self.check_class_value_of_record(record)
        if not predicted_probabilities:
            return self.get_random_class_label()

        chosen_class = None
        max_probability = None
        for class_key, class_value in predicted_probabilities.items():
            if max_probability is None or max_probability < class_value:
                chosen_class = class_key
                max_probability = class_value
        return chosen_class

    def predict_class_value_of_record_by_vote(self, record):
        class_value_statistics = dict()
        for i in range(self.rdt_num):
            class_value = self.random_decision_trees[i]\
                .get_class_label_of_record(record)
            if class_value is None:
                continue
            chosen_class = None
            max_num = None
            for key_class, value in class_value.items():
                if max_num is None or max_num < value:
                    chosen_class = key_class
                    max_num = value
            if chosen_class in class_value_statistics.keys():
                class_value_statistics[chosen_class] += 1
            else:
                class_value_statistics[chosen_class] = 1

        chosen_class = None
        max_num = None
        for key_class, value in class_value_statistics.items():
            if max_num is None or max_num < value:
                chosen_class = key_class
                max_num = value
        return chosen_class

    def predict_class_value_of_record(self, record):
        return self.predict_class_value_of_record_by_vote(record)

    def get_test_results(self):
        prediction_accuracy = 0
        for i in range(self._test_data_shape[0]):
            record = self._test_data[i:i+1]
            class_value = record[self.class_att].values[0]
            predicted_class_value = self.predict_class_value_of_record(record)
            if class_value == predicted_class_value:
                prediction_accuracy += 1

        prediction_accuracy /= self._test_data_shape[0]
        print("Prediction Accuracy: %s" % prediction_accuracy)
