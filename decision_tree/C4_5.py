import math

from decision_tree.ID3 import ID3


class C45(ID3):
    def _select_split_attribute(self, d_usage, candidate_attributes):
        gain_ratio = 0
        split_attribute = None
        sub_usages = None
        overcomes = None

        total_num = sum(d_usage)
        for att in candidate_attributes:
            information_value = 0
            can_info_gain, can_overcomes, can_sub_usages = \
                self._generate_information_of_specified_attribute(d_usage, att)

            for sub_usage in can_sub_usages:
                p = sum(sub_usage)/total_num
                if p > 0:
                    information_value -= p * math.log2(p)
            if information_value == 0:
                return 0, att, can_overcomes, can_sub_usages

            can_gain_ratio = can_info_gain/information_value
            if gain_ratio < can_gain_ratio:
                gain_ratio = can_gain_ratio
                split_attribute = att
                sub_usages = can_sub_usages
                overcomes = can_overcomes
        return gain_ratio, split_attribute, overcomes, sub_usages

    def prune(self):
        pass
