import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.baises = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.baises
    
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Softmax:
    def forward(self, inputs):
        
            
X = np.array([1.0, 2.0, -1.0])

layer1 = Layer_Dense(3, 2)
activation1 = Activation_ReLU()

layer1.forward(X)

activation1.forward(layer1.output)

print('Activated Output:\n', activation1.output)