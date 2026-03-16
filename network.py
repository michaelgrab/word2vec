import numpy as np

class Network:
    def __init__(self, input_size: int, hidden_size: int):
        self.weights1 = np.random.uniform(size=(input_size, hidden_size))
        self.weights2 = np.random.uniform(size=(input_size, hidden_size))

    def forward(self, x: np.ndarray):
        h : np.ndarray = x.T.dot( self.weights1 )
        v = h.dot(self.weights2.T) 
        return v

if __name__ == "__main__":
    vocab_size = 10
    x = np.zeros(shape=( 10, 1 ))
    x[0] = 1
    netwrok = Network(vocab_size, hidden_size=5)
    y = netwrok.forward(x)
    print(y)
    print(y.shape)