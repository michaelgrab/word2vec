import numpy as np
from data import generate_training_data, generate_dictionary_data, get_file_data, get_cxt_word_columns
import random
import matplotlib.pyplot as plt

class Network:
    def __init__(self, input_size: int, hidden_size: int, learning_rate: float=0.01):
        self.weights1 = np.random.uniform(-0.5, 0.5, size=(input_size, hidden_size))
        self.weights2 = np.random.uniform(-0.5, 0.5, size=(hidden_size, input_size))
        self.learning_rate = learning_rate
        self.vocab_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: np.ndarray):
        h = np.dot(self.weights1.T, x)
        u = np.dot(self.weights2.T, h)
        y_c = self.softmax(u)
        return y_c, h
    
    def softmax(self, x):
        exponent = np.exp(x - np.max(x))    # add softmax normalization
        return exponent / np.sum(exponent)
    
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray):
        return -np.sum(y_true * np.log(y_pred + 1e-15))
    
    def backwards(self, y: np.ndarray, t: np.ndarray, h: np.ndarray, x: np.ndarray):
        error =  y - t
        
        gradient2 = np.outer(h, error)

        gradient1 = np.outer(x, np.dot(self.weights2, error) )

        # gradient descent step
        self.weights1 -= self.learning_rate * gradient1
        self.weights2 -= self.learning_rate * gradient2 

    
if __name__ == "__main__":

    # text = ['Best way to success is through hardwork and persistence']

    text = get_file_data(stop_word_removal='yes')
    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictionary_data(text)
    window_size = 2
    training_data,_ = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'no')
    

    net = Network(vocab_size, hidden_size=10)

    print(f"vocabulary length {vocab_size}")
    # training loop implementation
    losses = []
    avg_losses = []
    epochs = 10
    for epoch in range(epochs):
        # randomize the order of the samples
        # np.random.shuffle(training_data)
        for sample in training_data:
            x, y_true = sample
            y_pred, hidden_vec = net.forward(x)
            loss = net.loss(y_pred, y_true)
            net.backwards(y_pred, y_true, hidden_vec, x)

            # logging
            print(f"epoch: {epoch}, loss: {loss}")
            losses.append(loss)
            avg_losses.append( np.average(losses) )
        epoch +=1

    x = list(range(len(losses)))
    y = losses
    plt.scatter(x, y, marker=".")
    plt.plot(x, avg_losses, color="r")
    plt.show()
