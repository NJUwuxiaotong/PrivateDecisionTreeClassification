from decision_tree.data_initialization import DataInitialization


data = DataInitialization("adult", 0.7, 0.3)
data.initial_data()

print("ATTRIBUTES:      %s" % data._attributes)
print("ATTRIBUTE NUM:   %s" % data._attribute_num)
print("ATTRIBUTE VALUE: %s" % data._attribute_values)
print("ATTRIBUTE TYPE:  %s" % data._attribute_types)

print("CLASS ATT:       %s" % data._class_attribute)
print("CLASS VALUE: \n  %s" % data._class_label)

print("NUM OF TRANING SET: %s" % data._training_num)
print("NUM OF TEST SET:    %s" % data._test_num)
