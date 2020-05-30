import math

from decision_tree.C4_5 import C4_5


class PrivateC4_5_05P(C4_5):
    """
    Literature: Avrim Blum, Cynthia Dwork, Frank McSherry, and Kobbi Nissim.
    Practical privacy: the SuLQ framework. in ACM SIGMOD-SIGACT-SIGART Symposium
    on Principles of Database Systems. ACM, 128-138, 2005.
    """
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, privacy_value=1):
        super(C4_5, self).__init__(dataset_name, training_per, test_per,
                                   tree_depth, True, privacy_value)
        self.privacy_value_per_query = \
            privacy_value / 2 / self._tree_depth / self._attribute_num
        self.sensitivity = 1
        self.privacy_parameter = self.sensitivity/self.privacy_value_per_query

    def noisy(self, privacy_value):
        """
        Function: provide an interface to add the noisy for privacy
        preservation
        """
        return 0.0

    def laplace_mechanism(self, privacy_value):
        p = 1/(2*self.privacy_parameter)*math.exp(-|x|/self.privacy_parameter)
