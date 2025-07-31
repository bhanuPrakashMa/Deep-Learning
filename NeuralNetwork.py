import numpy as np
from Layers.Base import BaseLayer
import copy


class NeuralNetwork():
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.current_label_tensor = None

    
    def forward(self):
        input_data, self.current_label_tensor = self.data_layer.next()
        for lay in range(len(self.layers)):
            input_data = self.layers[lay].forward(input_data)
        output_tensor = self.loss_layer.forward(input_data, self.current_label_tensor)
        return output_tensor


    def backward(self):
        error_data = self.loss_layer.backward(self.current_label_tensor)
        for lay in reversed(self.layers):
            error_data = lay.backward(error_data)


    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)


    def train(self, iterations):
        for _ in range(iterations):
            error_val = self.forward()
            self.loss.append(error_val)
            self.backward()

    def test(self,input_tensor):
        output = input_tensor
        for lay in self.layers:
            output = lay.forward(output)
        return output





