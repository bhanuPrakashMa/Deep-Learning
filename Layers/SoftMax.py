from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()

        self.output_tensor = None


    def forward(self, input_tensor):
        input_tensor = input_tensor - np.max(input_tensor, axis=1, keepdims = True)
        input_tensor = np.exp(input_tensor)
        self.output_tensor = input_tensor / np.sum(input_tensor, axis=1, keepdims=True)

        return self.output_tensor
    

    def backward(self, error_tensor):
        gradient = error_tensor - np.sum(error_tensor * self.output_tensor, axis =1, keepdims=True)
        gradient = self.output_tensor * gradient

        return gradient