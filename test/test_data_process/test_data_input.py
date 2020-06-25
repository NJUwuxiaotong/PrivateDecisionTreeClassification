from data_process.data_input import DataInput


data_input = DataInput("adult")
data_input.read_data()
print("Data shape:       %s" % list(data_input._data.shape))
print("Attribute number: %s" % data_input._attribute_num)
print("Attribute:        %s" % data_input._attributes)
print("Attribute type: \n%s" % data_input._attribute_types)
