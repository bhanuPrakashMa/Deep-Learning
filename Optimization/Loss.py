from Layers.Base import BaseLayer
import sys
import numpy as np

class CrossEntropyLoss():


    def __init__(self):
        self.input_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.input_tensor = prediction_tensor
        input_tensor = prediction_tensor[np.nonzero(label_tensor)] + sys.float_info.epsilon
        output_tensor = -np.sum(np.log(input_tensor))

        return output_tensor
    

    def backward(self, label_tensor):
        output_tensor = - 1 * ( label_tensor / (self.input_tensor + sys.float_info.epsilon))
        return output_tensor