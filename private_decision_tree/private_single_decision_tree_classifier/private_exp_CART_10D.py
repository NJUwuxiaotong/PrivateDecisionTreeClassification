from common import constants as const
from private_decision_tree.private_single_decision_tree_classifier.private_exp_ID3_10D import PrivateID310D


class PrivateCART10D(PrivateID310D):
    def compute_updated_information_gain(self, d_usage):
        gini_index = 0
        total_num = sum(d_usage)
        if total_num == 0:
            return 0
        for class_value in self.class_att_value:
            c_usage = d_usage & (
                    self._training_data[self.class_att] == class_value)
            c_num = sum(c_usage)
            if c_num > 0:
                p = c_num / total_num
                gini_index -= p * p
        gini_index = 1 + gini_index
        return gini_index

    def information_gain(self, d_usage, att, split_value=None):
        info = 0
        overcomes = dict()
        if self._attribute_type[att] == const.DFRAME_INT64:
            l_usage = d_usage & (self._training_data[att] < split_value)
            r_usage = d_usage & (self._training_data[att] >= split_value)
            l_num = sum(l_usage)
            r_num = sum(r_usage)
            info = \
                -1 * l_num * self.compute_updated_information_gain(l_usage) - \
                r_num * self.compute_updated_information_gain(r_usage)
            overcomes = {"<<" + str(split_value): l_usage,
                         ">=" + str(split_value): r_usage}
        else:
            unique_values = self._training_data[att][d_usage].drop_duplicates(
                keep='first').values
            for value in unique_values:
                v_usage = d_usage & (self._training_data[att] == value)
                v_num = sum(v_usage)
                if v_num == 0:
                    continue
                overcomes[value] = v_usage
                info -= v_num * (1 + self.compute_updated_information_gain(
                    v_usage))
        return info, overcomes
