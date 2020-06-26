class Node(object):
    def __init__(self, parent_node=None, is_leaf=None, parent_index=None):
        # parent node of the node
        self._parent_node = parent_node
        # check whether the node is leaf
        self._is_leaf = is_leaf
        # parent index
        self._parent_index = parent_index

    def set_parent_node(self, p_node):
        self._parent_node = p_node

    def set_is_leaf(self, is_leaf):
        self._is_leaf = is_leaf

    def set_parent_index(self, parent_index):
        self._parent_index = parent_index

    @property
    def parent_node(self):
        return self._parent_node

    @property
    def is_leaf(self):
        return self._is_leaf

    @property
    def parent_index(self):
        return self._parent_index


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
    def __init__(self, parent_node=None, is_leaf=None, class_result=None):
        """
        class_result: output class
        """
        super(LeafNode, self).__init__(parent_node, is_leaf)
        self._class_result = class_result
        self._class_values = dict()

    def set_class_result(self, class_key):
        self._class_result = class_key

    def add_class_value(self, class_key, class_value):
        if class_key in self._class_values.keys():
            print("WARNING: class key [%s] has been in class values"
                  % class_key)
        self._class_values[class_key] = class_value

    def increment_class_value(self, class_key):
        if class_key in self._class_values.keys():
            self._class_values[class_key] += 1
        else:
            self._class_values[class_key] = 1

    @property
    def class_values(self):
        return self._class_values

    @property
    def class_results(self):
        return self._class_result
