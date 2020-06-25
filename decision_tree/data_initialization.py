from data_process.data_input import DataInput


class DataInitialization(DataInput):
    def __init__(self, dataset_name, training_per=0.7, test_per=0.3):
        super(DataInitialization, self).__init__(dataset_name)

        self._training_data = None        # DataFrame
        self._training_data_shape = None
        self._test_data = None
        self._test_data_shape = None
        self._training_per = training_per
        self._test_per = test_per
        self._check_parameters()
        self._training_num = None
        self._test_num = None

    def initial_data(self):
        # input data from file
        self.read_data()
        data_shape = self._data.shape

        # training data
        self._training_num = int(data_shape[0] * self._training_per)
        self._training_data = self._data[:self._training_num]
        self._training_data_shape = self._training_data.shape

        # test data
        self._test_num = data_shape[0] - self._training_num
        self._test_data = self._data[-1*self._test_num:]
        self._test_data_shape = self._test_data.shape

    def _check_parameters(self):
        if self._training_per + self._test_per > 1:
            print("Error: The percentage (%s, %s) of data in training and "
                  "test exceeds 1" % (self._training_per, self._test_per))
            exit(1)
