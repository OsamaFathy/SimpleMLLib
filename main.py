import numpy as np
import math


class Regresssion:
    def __init__(self, data_set_examples, dataset_output, thetas=[]):
        self.data_set_examples = self.__convert_to_np_array__(data_set_examples)
        self.dataset_output = self.__convert_to_np_array__(dataset_output)
        if len(thetas) > 0 and len(thetas) != len(data_set_examples[0]) + 1:
            raise ValueError("Size of theta is not compatible with the number of features in the examples")
        self.thetas = self.__convert_to_np_array__(thetas)
        self.__prepare__()

    def __convert_to_np_array__(self, arr):
        ret = np.array(arr)
        return ret.astype(float)

    def __add_dimention__(self, arr):
        return np.array([arr])

    def __add_ones__(self, data):
        return np.insert(data, 0, np.ones([1, data.shape[0]]), axis=1)

    def __prepare__(self):
        # if not already given, initialize thetas with zeros.
        examples = self.data_set_examples.shape[0]
        if self.thetas.size == 0:
            self.thetas = np.zeros([self.data_set_examples[0].size + 1])
        self.data_set_examples = self.__add_ones__(self.data_set_examples)

    def __normlize__(self, data):
        mn = np.min(data, 0)
        mx = np.max(data, 0)
        mean = np.mean(data, 0)
        for i in range(data.shape[0]):
            for j in range(1, data.shape[1]):
                data[i][j] = (data[i][j] - mean[j]) / (mx[j] - mn[j])
        return mn, mx, mean

    def __normailze_elemnet__(self, element, mn, mx, mean):
        return (element - mean) / (mx - mn)

    def __denormalize_element__(self, element, mn, mx, mean):
        return element * (mx - mn) + mean


class LogisticRegression(Regresssion):
    def __init__(self, data_set_examples, dataset_output, thetas=[]):
        super().__init__(data_set_examples, dataset_output, thetas)

    def __calculate_cost__(self, output, correct, lmbda):
        output = self.__convert_to_np_array__(output)
        correct = self.__convert_to_np_array__(correct)
        sum = 0
        for x, y in zip(output, correct):
            if y == 1:
                sum += -math.log2(x) / 2
            else:
                sum += -math.log2(1 - x) / 2
        sum += lmbda * np.sum(self.thetas * self.thetas)
        return sum / len(output)

    # todo add option to gradually decrease the learning rate
    def __gradient_descent__(self, a, iterations, lmbda):
        self.feat_range_mn, self.feat_range_mx, self.feat_range_mean = self.__normlize__(self.data_set_examples)
        for k in range(iterations):
            output = self.predict(self.data_set_examples)
            diff = output - self.dataset_output
            for i in range(self.thetas.size):
                self.thetas[i] -= a / self.thetas.size * (
                            np.sum(diff * self.data_set_examples[:, i]) + lmbda * np.sum(self.thetas))

    def predict(self, inp, add_ones=False, normalize=False):
        inp = self.__convert_to_np_array__(inp)
        if inp.ndim == 1:
            inp = self.__add_dimention__(inp)
        if add_ones:
            inp = self.__add_ones__(inp)

        if normalize:
            for i in range(inp.shape[0]):
                for j in range(1, inp.shape[1]):
                    inp[i][j] = self.__normailze_elemnet__(inp[i][j], self.feat_range_mn[j], self.feat_range_mx[j],
                                                           self.feat_range_mean[j])

        inp = np.transpose(inp)
        if inp.shape[0] != self.thetas.size:
            raise ValueError("Number of features in the input is not correct")
        return 1 / (1 + 1 / np.exp(self.thetas @ inp))

    def train(self, a=0.001, iterations=100000):
        self.__gradient_descent__(a, iterations, 0.001)


class LinearRegression(Regresssion):
    def __init__(self, data_set_examples, dataset_output, thetas=[]):
        super().__init__(data_set_examples, dataset_output, thetas)

    def __calculate_cost__(self, output, correct, lmbda):
        output = self.__convert_to_np_array__(output)
        correct = self.__convert_to_np_array__(correct)
        sum = np.sum(np.square((output - correct)))
        sum += lmbda * np.sum(self.thetas * self.thetas)
        return sum / (2. * len(output))

    # todo add option to gradually decrease the learning rate
    def __gradient_descent__(self, a, iterations, lmbda):
        self.feat_range_mn, self.feat_range_mx, self.feat_range_mean = self.__normlize__(self.data_set_examples)
        for k in range(iterations):
            output = self.predict(self.data_set_examples)
            diff = output - self.dataset_output
            for i in range(self.thetas.size):
                self.thetas[i] -= a / self.thetas.size * (
                        np.sum(diff * self.data_set_examples[:, i]) + lmbda * np.sum(self.thetas))

    def train(self, a=0.001, iterations=100000):
        self.__gradient_descent__(a, iterations, 0.001)

    def predict(self, inp, add_ones=False, normalize=False):
        inp = self.__convert_to_np_array__(inp)
        if inp.ndim == 1:
            inp = self.__add_dimention__(inp)
        if add_ones:
            inp = self.__add_ones__(inp)

        if normalize:
            for i in range(inp.shape[0]):
                for j in range(1, inp.shape[1]):
                    inp[i][j] = self.__normailze_elemnet__(inp[i][j], self.feat_range_mn[j], self.feat_range_mx[j],
                                                           self.feat_range_mean[j])

        inp = np.transpose(inp)
        if inp.shape[0] != self.thetas.size:
            raise ValueError("Number of features in the input is not correct")

        return self.thetas @ inp


LR = LinearRegression([(1, 2), (1, 3), (2, 4), (5, 2), (10, 2)], [3, 4, 6, 7, 12], [1, 2, 3])
LR.train(0.001, 100000)
print(LR.predict([5, 2], True, True))

LLR = LogisticRegression([(1, 2), (1, 3), (2, 4), (5, 2), (10, 2)], [0, 0, 1, 1, 1], [1, 1, 1])
LLR.train(0.03, 100000)
print(LLR.predict([5, 3], True, True))
