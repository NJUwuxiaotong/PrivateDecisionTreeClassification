class Node(object):
    def __init__(self, parent_node=None, is_leaf=None):
        # parent node of the node
        self._parent_node = parent_node
        # check whether the node is leaf
        self._is_leaf = is_leaf

    def set_parent_node(self, p_node):
        self._parent_node = p_node

    def set_is_leaf(self, is_leaf):
        self._is_leaf = is_leaf

    @property
    def parent_node(self):
        return self._parent_node

    @property
    def is_leaf(self):
        return self._is_leaf


class NonLeafNode(Node):
    def __init__(self, parent_node=None, is_leaf=None, att_name=None,
                 sub_nodes=None):
        super(NonLeafNode, self).__init__(parent_node, is_leaf)
        self._att_name = att_name
        self._sub_nodes = sub_nodes

    def set_att_name(self, att_name):
        self._att_name = att_name

    def add_sub_node(self, outcome, sub_node):
        # if the type of attribute is continuous, the outcome is string
        # with prefix "<<" or ">="
        if not self._sub_nodes:
            self._sub_nodes = dict()
        self._sub_nodes[outcome] = sub_node

    @property
    def att_name(self):
        return self._att_name

    @property
    def sub_nodes(self):
        return self._sub_nodes


class LeafNode(Node):
    def __init__(self, parent_node=None, is_leaf=None, class_results=None):
        super(LeafNode, self).__init__(parent_node, is_leaf)
        self._class_results = class_results

    def add_class_result(self, class_key, class_num):
        if not self._class_results:
            self._class_results = dict()
        self._class_results[class_key] = class_num

    def get_most_frequent_class(self):
        chosen_label = None
        result = None
        for class_label, value in self._class_results.items():
            if result is None:
                result = value
                chosen_label = class_label
            elif result < value:
                result = value
                chosen_label = class_label
        return chosen_label

    @property
    def class_results(self):
        return self._class_results
