import numpy as np
import random
import matplotlib.pyplot as plt
import time

from data import generate_training_data, generate_dictionary_data

class SkipGramNetwork:
    def __init__(self, input_size: int, hidden_size: int, learning_rate: float=0.01):
        self.weights1 = np.random.uniform(-0.5, 0.5, size=(input_size, hidden_size))
        self.weights2 = np.random.uniform(-0.5, 0.5, size=(hidden_size, input_size))
        self.learning_rate = learning_rate
        self.vocab_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x: np.ndarray):
        h = np.dot(x.T, self.weights1)
        u = np.dot(h.T, self.weights2)
        y_c = self.softmax(u)
        return y_c, h, u
    
    def softmax(self, x):
        exponent = np.exp(x - np.max(x))    # add softmax normalization
        return exponent / np.sum(exponent)
    
    # why is the loss function returning extremely low values?
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

def train_skip_gram(training_data, vocab_size, epochs=500, print_freq=5):
    """
        print_freq: print epoch loss every x epochs 
    """    
    net = SkipGramNetwork(vocab_size, hidden_size=20)

    print("*" * 20)
    print("Starting training skip-gram")
    print(f"vocabulary size: {vocab_size}")
    print(f"hidden vector size {net.hidden_size}")
    print(f"training data size {len(training_data)}")
    print("*" * 20)
    # training loop implementation
    start_time = time.time()
    losses = []
    epoch_losses = []
    avg_losses = []
    cumulative_avg = []
    for epoch in range(epochs):
        # randomize the order of the samples
        # np.random.shuffle(training_data)
        for sample in training_data:
            target, context = sample
            y_pred, hidden_vec, u = net.forward(target)
            loss = net.loss(u, context)
            net.backwards(y_pred, context, hidden_vec, target)
            # loss = calculate_loss()

            # logging
            epoch_losses.append(loss)
            losses.append(loss)
        avg_losses.append( np.average(epoch_losses) )
        cumulative_avg.append( np.average( losses ) )
        epoch_losses = []
        if epoch % print_freq == 0:
            print(f"epoch: {epoch}, loss: {loss}")
        epoch +=1
    end_time = time.time()    
    print("*" * 20)
    print(f"training time {end_time - start_time}")
    print("*" * 20)
    x = list(range(len(avg_losses)))
    # y = losses
    # plt.scatter(x, y, marker=".")
    plt.plot(x, avg_losses, color="r", label="running average")
    plt.plot(x, cumulative_avg, color="b", label="cumulative average")
    plt.xlabel("training epochs")
    plt.title("Training loss Skip-Gram")
    plt.show()

if __name__ == "__main__":

    text = ['Best way to is hardwork and persistence persistence persistence']

    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictionary_data(text)
    window_size = 2

    training_data,_,_ = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'no', single_positive_cxt=False)
    train_skip_gram(training_data, vocab_size, epochs=100)