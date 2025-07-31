class Sgd:    
    
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        weights = weight_tensor
        updated_weights = weights - self.learning_rate * gradient_tensor
        return updated_weights
        