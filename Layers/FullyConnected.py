from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    
    def __init__(self, input_size, output_size):

        super().__init__()
        self.input_size = input_size + 1
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (self.input_size, self.output_size))
        self._gradient_weights = None
        self._optimizer = None

    
    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        self.input_tensor = np.hstack((input_tensor, np.ones((batch_size, 1))))
        output_tensor = np.dot(self.input_tensor, self.weights)
        return output_tensor
    

    def backward(self, error_tensor):
        loss_grad_input = np.dot (error_tensor, self.weights.T)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        output_error_to_previous_layer = loss_grad_input[:, :-1] # idi change cheyaniki chudu
        return output_error_to_previous_layer
    

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, learning_rate):
        self._optimizer = learning_rate
        return
    

    @property
    def gradient_weights(self):
        return self._gradient_weights
    

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights
        return



