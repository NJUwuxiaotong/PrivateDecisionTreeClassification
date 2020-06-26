from data_process.data_preprocess import DataPreProcess


# dataset: adult, nursery,
data_path = "F:\\workspace\\PrivateDecisionTreeClassification\\da" \
            "tasets\\nursery.data"

data_pro = DataPreProcess(data_path)
data_pro.read_data_from_file()

print(data_pro.data_shape)

data_pro.show_statistical_info()

data_pro.process_abnormal_data()
