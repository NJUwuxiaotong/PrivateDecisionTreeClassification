from decision_tree.ID3 import ID3


class CART(ID3):
    def _information_entropy(self, d_usage):
        total_num = sum(d_usage)
        if total_num == 0:
            return 0

        gini_index = 1
        for class_value in self.class_att_value:
            c_usage = d_usage & (
                    self._training_data[self.class_att] == class_value)
            c_num = sum(c_usage)
            p = c_num / total_num
            gini_index -= p * p
        return gini_index

    def _select_split_attribute(self, d_usage, candidate_attributes, *params):
        split_attribute = None
        sub_usages = None
        overcomes = None
        gini_index = None
        for att in candidate_attributes:
            can_gini_index, can_overcomes, can_sub_usages = \
                self._generate_information_of_specified_attribute(
                    d_usage, att, params[0])

            if gini_index is None or can_gini_index > gini_index:
                gini_index = can_gini_index
                split_attribute = att
                sub_usages = can_sub_usages
                overcomes = can_overcomes
        return gini_index, split_attribute, overcomes, sub_usages
