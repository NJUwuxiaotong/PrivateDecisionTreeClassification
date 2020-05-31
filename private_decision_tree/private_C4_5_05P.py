from decision_tree.C4_5 import C4_5
from pub_lib import pub_functions

class PrivateC4_5_05P(C4_5):
    """
    Literature: Avrim Blum, Cynthia Dwork, Frank McSherry, and Kobbi Nissim.
    Practical privacy: the SuLQ framework. in ACM SIGMOD-SIGACT-SIGART Symposium
    on Principles of Database Systems. ACM, 128-138, 2005.
    """
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3,
                 tree_depth=None, privacy_value=1):
        super(PrivateC4_5_05P, self).__init__(
            dataset_name, training_per=training_per, test_per=test_per,
            tree_depth=tree_depth, is_private=True,
            privacy_value=privacy_value)
        self.privacy_value_per_query = \
            privacy_value / (2 * self._tree_depth * self._attribute_num)
        self.sensitivity = 1
        self.privacy_parameter = self.sensitivity/self.privacy_value_per_query

    def noisy(self, privacy_value):
        """
        Function: provide an interface to add the noisy for privacy
        preservation.
        It chooses Laplace mechanism.
                Probability density function:
        Pr(x|l) = 1/(2*l)*exp^{-|x|/l}, in which l=sensitivity/privacy_value
        So, the probability distributed function:
            F(x) =  exp^{x/l}/2               x<0
                    1-exp{-x/l}/2             x>0
        """
        return pub_functions.generate_random_value_from_Laplace(
            0, 10, self.privacy_parameter)
