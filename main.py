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


class LogisticRegression(Regresssion):
    def __init__(self, data_set_examples, dataset_output, thetas=[]):
        super().__init__(data_set_examples, dataset_output, thetas)

    def __calculate_cost__(self, output, correct):
        output = self.__convert_to_np_array__(output)
        correct = self.__convert_to_np_array__(correct)
        sum = 0
        for x, y in zip(output, correct):
            if y == 1:
                sum += -math.log2(x) / 2
            else:
                sum += -math.log2(1 - x) / 2
        return sum / len(output)

    def predict(self, inp, add_ones=False):
        inp = self.__convert_to_np_array__(inp)
        if inp.ndim == 1:
            inp = self.__add_dimention__(inp)
        if add_ones:
            inp = self.__add_ones__(inp)

        inp = np.transpose(inp)
        if inp.shape[0] != self.thetas.size:
            raise ValueError("Number of features in the input is not correct")
        return 1 / (1 + 1/np.exp(self.thetas @ inp))

    # todo add option to gradually decrease the learning rate
    def __gradient_descent__(self, a, iterations):
        for k in range(iterations):
            output = self.predict(self.data_set_examples)
            diff = output - self.dataset_output
            for i in range(self.thetas.size):
                self.thetas[i] -= a * np.sum(diff * self.data_set_examples[:, i])

    def train(self, a=0.001, iterations=100000):
        self.__gradient_descent__(a, iterations)


class LinearRegression(Regresssion):
    def __init__(self, data_set_examples, dataset_output, thetas=[]):
        super().__init__(data_set_examples, dataset_output, thetas)

    def __calculate_cost__(self, output, correct):
        output = self.__convert_to_np_array__(output)
        correct = self.__convert_to_np_array__(correct)
        sum = np.sum(np.square((output - correct)))
        return sum / (2. * len(output))

    def predict(self, inp, add_ones=False):
        inp = self.__convert_to_np_array__(inp)
        if inp.ndim == 1:
            inp = self.__add_dimention__(inp)
        if add_ones:
            inp = self.__add_ones__(inp)

        inp = np.transpose(inp)
        if inp.shape[0] != self.thetas.size:
            raise ValueError("Number of features in the input is not correct")
        return self.thetas @ inp

    # todo add option to gradually decrease the learning rate
    def __gradient_descent__(self, a, iterations):
        for k in range(iterations):
            output = self.predict(self.data_set_examples)
            diff = output - self.dataset_output
            for i in range(self.thetas.size):
                self.thetas[i] -= a / self.thetas.size * np.sum(diff * self.data_set_examples[:, i])

    def train(self, a=0.0001, iterations=10000):
        self.__gradient_descent__(a, iterations)


LR = LinearRegression([(1, 2), (1, 3), (2, 4), (5, 2), (10, 2)], [3, 4, 6, 7, 12], [1, 2, 3])
LR.train()
print(LR.__calculate_cost__([1, 2], [1, 3]))
print(LR.predict([5, 2], True))

LLR = LogisticRegression([(1, 2), (1, 3), (2, 4), (5, 2), (10, 2)], [0, 0, 1, 1, 1], [1, 1, 1])
LLR.train()
print(LLR.predict([5, 2], True))


