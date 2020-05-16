from data_process.data_preprocess import DataPreProcess


data_path = "D:\\workspace\\projects\\PrivateDecisionTreeClassification\\da" \
            "tasets\\adult.data"

data_pro = DataPreProcess(data_path)
data_pro.read_data_from_file()

data_pro.show_statistical_info()