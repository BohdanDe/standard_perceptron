import math
import pickle
import random

from losses import softmax_cross_entropy_gradient
from optimizers import SGD


def softmax(values):
    values_copy = values.copy()
    r = []
    max_v = max(values)
    for index, _ in enumerate(values):
        values_copy[index]-=max_v
    exponents = []
    for i in values_copy:
        exponents.append(math.exp(i))
    sum = 0 
    for i in exponents:
        sum+=i
    for i in exponents:
        r.append(i/sum)
    return r

def relu(values):
    r = []
    for i in values:
        if i < 0:
            r.append(0)
        else:
            r.append(i)
    return r

def relu_der(values):
    r = []
    for i in values:
        if i < 0:
            r.append(0)
        else:
            r.append(1)
    return r

class Dense():
    def __init__(self, weights=None, bias=None, activation_function=None, size=None, input_size=None):
        self.weights = weights
        self.bias = bias
        self.input = input
        self.activation_function = activation_function
        self.result = []
        self.size = size
        self.input_size = input_size
        self.l_index = 0

    def build(self, size_of_prev):
        if self.input_size is None:
            size_of_prev = size_of_prev
        limit = math.sqrt(6/size_of_prev)
        temp_weights = []
        for i in range(size_of_prev):
            temp_neuron = []
            for j in range(self.size):
                #temp_neuron.append(random.uniform(-1,1))
                temp_neuron.append(random.uniform(-limit, limit))
            temp_weights.append(temp_neuron)
        self.weights = temp_weights
        self.bias = [0] * self.size

    def multiply(self, _input, return_z=False):
        self.input = _input
        self.result = []
        for i in range (len(self.weights[0])):
            amount_of_inputs = len(self.weights)
            #print(amount_of_inputs)
            temp_r = 0
            for j in range(amount_of_inputs):
                #print(f"{self.input[j]}*{self.weights[j][i]}={self.input[j]*self.weights[j][i]}")
                temp_r+=self.input[j]*self.weights[j][i]
            self.result.append(temp_r)
        for index, i in enumerate(self.bias):
            self.result[index]+=i
        to_ret = None
        if (self.activation_function == "relu"):
            to_ret = relu(self.result)
        elif (self.activation_function == "softmax"):
            to_ret = softmax(self.result)
        if return_z:
            return to_ret, self.result
        return to_ret

    def get_delta_softmax(self, y_pred, y_true):
        result = softmax_cross_entropy_gradient(y_pred, y_true)
        return result

    def backward(self, output):
        if self.activation_function == "softmax":
            error = softmax_cross_entropy_gradient()

    def summary(self):
        params = 0
        for i in self.weights:
            params += len(i)
        params += len(self.bias)
        return params

class Sequential():
    def __init__(self, layers: list):
        self.layers = layers
        for index, layer in enumerate(layers):
            layer.l_index = index
            if layer.weights and layer.bias is not None:
                continue

            if index == 0:
                layer.build(layer.input_size)
            else:
                layer.build(layers[index-1].size)
    
    def predict(self, input, each_layer_log=False):
        layers_outputs = {}
        to_pass = input
        for i in self.layers:
            to_pass = i.multiply(to_pass)
            if each_layer_log:
                layers_outputs[f"Layer {i.l_index}"] = to_pass
        if each_layer_log:
            print(layers_outputs)
        return to_pass

    def summary(self):
        params = 0
        for n, i in enumerate(self.layers):
            layer_params = i.summary()
            params += layer_params
            print(f"Layer {i.l_index}, params: {layer_params}, activation: {i.activation_function}")
        print(f"Total params: {params}")

    def step(self, input, correct, lr):
        layers_outputs = {}
        layers_outputs[-1]= input
        layers_z = {}
        layers_z[-1] = input

        to_pass = input

        for i in self.layers:
            a, z = i.multiply(to_pass, True)
            to_pass = a
            layers_outputs[i.l_index] = a
            layers_z[i.l_index] = z

        deltas = {}

        dW = {}
        db = {}
        for i in reversed(self.layers):
            if i.activation_function == "softmax":
                w = self.layers[-1].get_delta_softmax(to_pass, correct)
                b = []
                for index, bias in enumerate(i.bias):
                    b.append(bias - w[index])
                deltas[i.l_index] = self.layers[-1].get_delta_softmax(to_pass, correct) #{"w": w, "b": b}
            elif i.activation_function == "relu":
                curr_layer_deltas = []
                z = relu_der(layers_z[i.l_index])
                next_layer_w = self.layers[i.l_index+1].weights
                for index, row in enumerate(next_layer_w):
                    sum = 0
                    for index2, value in enumerate(row):
                        sum += value * deltas[i.l_index+1][index2]
                    curr_layer_deltas.append(sum*z[index])
                deltas[i.l_index] = curr_layer_deltas

                # for j in self.layers[i.l_index+1].weights:
                #     sum = 0
                #     for index, delta in enumerate(deltas[i.l_index+1]):
                #         sum+=j[index]*delta*z[index]
                #         #print(f"{j[index]}*{delta}*{z[index]}")
                #     curr_layer_deltas.append(sum)
                # deltas[i.l_index] = curr_layer_deltas

            dW[i.l_index] = [[0.0 for _ in range(len(i.weights[0]))] for _ in range(len(i.weights))] # заглушка для градієнтів, просто нулі
        for i in reversed(self.layers):
            a = layers_outputs[i.l_index-1]
            for index1, inp in enumerate(a):
                for index2, delta in enumerate(deltas[i.l_index]):
                    dW[i.l_index][index1][index2] = inp*delta
        db = deltas
        #print(dW)
        #print(db)

        #update weights and biases using SGD

        for i in self.layers:
            new_w, new_b = SGD(i.weights, i.bias, dW[i.l_index], db[i.l_index], lr)
            i.weights = new_w
            i.bias = new_b

    def save(self, filename):
        # 1. Clean up the junk so we don't save the last input data
        # Think of this like taking out the trash before locking the house
        for layer in self.layers:
            if hasattr(layer, 'input'):
                layer.input = None
            if hasattr(layer, 'result'):
                layer.result = []

        # 2. Freeze it
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        # 3. Thaw it out
        with open(filename, 'rb') as f:
            return pickle.load(f)