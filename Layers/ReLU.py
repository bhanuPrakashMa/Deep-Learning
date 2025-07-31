from Layers.Base import BaseLayer
import numpy as np


class ReLU(BaseLayer):


    def __init__(self):
        super().__init__()
        self.input_tensor = []


    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = np.maximum(0, self.input_tensor)
        return output_tensor
    
    def backward(self, error_tensor):
        output = error_tensor * (np.where(self.input_tensor > 0, 1, 0))
        return output
    