import numpy as np
from data import generate_training_data, generate_dictinoary_data, get_file_data
import random
import matplotlib.pyplot as plt

class Network:
    def __init__(self, input_size: int, hidden_size: int, learning_rate: float=0.01):
        # Xavier/Glorot initialization
        self.weights1 = np.random.uniform(-0.5, 0.5, size=(input_size, hidden_size))
        self.weights2 = np.random.uniform(-0.5, 0.5, size=(hidden_size, input_size))
        self.learning_rate = learning_rate
        self.vocab_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: np.ndarray):
        h : np.ndarray = x.T.dot( self.weights1 )
        v = self.weights2.T.dot(h.T)
        return v.flatten(), self.softmax(v.T).flatten(), h
    
    def softmax(self, x):
        exponent = np.exp(x - np.max(x))    # add softmax normalization
        return exponent / np.sum(exponent)
    
    def loss(self, logits: np.ndarray, ground_truth: np.ndarray):
        C = np.sum(ground_truth)
        correct_word_score = np.sum(np.multiply(logits, ground_truth))
        m = np.max(logits)                                      # softmax normalization
        log_term = m + np.log( np.sum(np.exp( logits - m )) )
        normalization_term = C * log_term
        return -correct_word_score + normalization_term
    
    def backwards(self, y: np.ndarray, t: np.ndarray, h: np.ndarray, x: np.ndarray):
        C = np.sum(t)
        error = C * y - t
        gradient2 = np.outer(h, error)

        gradient1 = np.outer(x, np.dot(self.weights2, error) )

        # gradient decent step
        self.weights1 = self.weights1 - (self.learning_rate * gradient1)
        self.weights2 = self.weights2 - (self.learning_rate * gradient2) 

    
if __name__ == "__main__":

    # text = ['Best way to success is through hardwork and persistence']

    text = get_file_data(stop_word_removal='yes')
    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictinoary_data(text)
    window_size = 2
    training_data,_ = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'no')
    
    # randomize the order of the samples
    np.random.shuffle(training_data)

    net = Network(vocab_size, hidden_size=5)

    print(f"vocabulary length {vocab_size}")
    # training loop implementation
    losses = []
    avg_losses = []
    epochs = 100
    for epoch in range(epochs):
        sample = random.sample(training_data, k=1)[0]
        x, label = sample
        u, y, hidden_vec = net.forward(x)
        loss = net.loss(u, label)
        net.backwards(y, label, hidden_vec, x)

        # logging
        # print(f"epoch: {epoch}, loss: {loss}")
        losses.append(loss)
        avg_losses.append( np.average(losses) )
        epoch +=1

    x = list(range(epochs))
    y = losses
    plt.scatter(x, y, marker=".")
    plt.plot(x, avg_losses, color="r")
    plt.show()
