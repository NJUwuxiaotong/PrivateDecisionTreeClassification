from common import constants as const
from private_decision_tree.private_exp_ID3_10D import PrivateID310D


class PrivateMaxOperator10D(PrivateID310D):
    def compute_updated_information_gain(self, d_usage):
        max_operator = None
        for class_value in self.class_att_value:
            c_usage = d_usage & (
                    self._training_data[self.class_att] == class_value)
            c_num = sum(c_usage)
            if max_operator is None or max_operator < c_num:
                max_operator = c_num
        return max_operator

    def information_gain(self, d_usage, att, split_value=None):
        info = 0
        overcomes = dict()
        if self._attribute_type[att] == const.DFRAME_INT64:
            l_usage = d_usage & (self._training_data[att] < split_value)
            r_usage = d_usage & (self._training_data[att] >= split_value)
            info = \
                self.compute_updated_information_gain(l_usage) + \
                self.compute_updated_information_gain(r_usage)
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
                info += self.compute_updated_information_gain(v_usage)
        return info, overcomes
