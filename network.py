import numpy as np

class Network:
    def __init__(self, input_size: int, hidden_size: int):
        self.weights1 = np.random.uniform(size=(input_size, hidden_size))
        self.weights2 = np.random.uniform(size=(input_size, hidden_size))

    def forward(self, x: np.ndarray):
        h : np.ndarray = x.T.dot( self.weights1 )
        v = h.dot(self.weights2.T) 
        return self.softmax(v)
    
    def softmax(self, x):
        exponent = np.exp(x)
        return exponent / np.sum(exponent)

if __name__ == "__main__":
    vocab_size = 10
    x = np.zeros(shape=( 10, 1 ))
    x[0] = 1
    network = Network(vocab_size, hidden_size=5)
    y = network.forward(x)
    # print(y)
    print(y.shape)
    print(network.softmax(y))