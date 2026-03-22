import numpy as np
import random
import matplotlib.pyplot as plt
import time

from data import generate_training_data, generate_dictionary_data

class NegativeSamplingNetwork:
    def __init__(self, vocab_size, hidden_size, K, learning_rate, alpha=3/4):
        """
            K is the number of negative words that will be sampled
        """
        self.v_input = np.random.uniform(-0.5, 0.5, size=(vocab_size, hidden_size))  
        self.v_output = np.random.uniform(-0.5, 0.5, size=(hidden_size, vocab_size))
        self.K = K   
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.hidden_size = hidden_size

    def forward(self, x):
        """
          x is the target word

        """
        h = np.dot(x.T, self.v_input)
        c_output = np.dot(h.T, self.v_output) 
        return c_output, h
    
    def sigmoid(self, x):
        sig = np.zeros_like(x, dtype=float)
        
        pos_mask = (x >= 0)
        neg_mask = ~pos_mask
        
        sig[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        
        exp_x_neg = np.exp(x[neg_mask])
        sig[neg_mask] = exp_x_neg / (1 + exp_x_neg)
        
        return sig

    def get_unigram_dist(self, occurrences, corpus_length):
        vocab_size = len(occurrences)
        dist = np.zeros(vocab_size)
        for i, n in enumerate(occurrences):
            dist[i] = n / corpus_length
        return dist
    
    def get_noisy_dist(self, occurrences, corpus_length):
        # distribuiton from which negative sampling is performed
        unigram = self.get_unigram_dist(occurrences, corpus_length)
        noisy_dist = unigram**self.alpha
        z = sum(noisy_dist)
        noisy_dist_norm = noisy_dist / z
        return noisy_dist_norm
    
    def sample_neg(self, dist):
        samples = np.random.choice( len(dist), size=self.K, p=dist, replace=False )
        return samples
    
    def backwards(self, logits, neg_ind, h, cxt, x):
        pos_idx = np.argmax(cxt)
        center_idx = np.argmax(x)
        indicies = np.concatenate(([pos_idx], neg_ind))
        # calculate sigmoid from the logits 
        sigma = self.sigmoid(logits)

        # select just the elements coresponding to the positive word and the negative words
        selected_sigma = sigma[indicies]
        # the vector is of size (K+1)

        # t[positive_word] = 1, t[negative_word] = 0
        t = np.zeros(self.K + 1)
        t[0] = 1
        # calculate the error
        error = selected_sigma - t

        # selecting weights from v_output
        # positive word and negative words
        selected_v_output = self.v_output.T[indicies]
        # the matrix size will be (K+1, N)

        # select weights from v_input that will be updated
        # the weights are of the center word
        current_weight_input = self.v_input[center_idx]
        # the vector of size (N)

        # gradient decent of v_input
        grad_v_input = np.dot( error, selected_v_output )
        updated_weight_input = current_weight_input - (self.learning_rate * grad_v_input)
        self.v_input[center_idx] = updated_weight_input

        # calculating gradients for v_output
        current_weights_output = self.v_output[:, indicies]
        grad_v_ouput = np.outer( h, error )
        updated_weights_output = current_weights_output - (self.learning_rate * grad_v_ouput)
        self.v_output[:, indicies] = updated_weights_output

    def loss(self, logits, neg_ind, cxt):
        cxt_ind = np.argmax(cxt)
        negative_part = self.sigmoid(-logits[neg_ind])
        return - np.log(self.sigmoid(logits[cxt_ind])) - np.sum( np.log(negative_part) )

def train_negative_sampling(training_data, occurrences, length_of_corpus, vocab_size, epochs=500, print_freq=5):    
    """
        print_freq: print epoch loss every x epochs 
    """
    net = NegativeSamplingNetwork(vocab_size, 5, 4, 0.01)
    noisy_dist = net.get_noisy_dist(occurrences, length_of_corpus)
    print("*" * 20)
    print("Starting training of skip gram with negative sampling")
    print(f"vocabulary size: {vocab_size}")
    print(f"number of negative samples {net.K}")
    print(f"hidden vector size {net.hidden_size}")
    print(f"training data size {len(training_data)}")
    print("*" * 20)

    start_time = time.time()
    epoch = 0
    running_losses = []
    epoch_losses = []
    for epoch in range(epochs):
        for sample in training_data:
            x, cxt = sample
            neg_indicies = net.sample_neg(noisy_dist)
            logits, h = net.forward(x)
            # c_neg = c_output[neg_indicies]
            
            # pos = np.dot(cxt, c_output)
            net.backwards(logits, neg_indicies, h, cxt, x)
            loss = net.loss(logits, neg_indicies, cxt)
            epoch_losses.append(loss)
        running_losses.append(np.average(epoch_losses))
        epoch_losses = []
        if epoch % print_freq == 0:
            print(f"epoch: {epoch}, loss: {loss}")
        epoch +=1
    end_time = time.time()    
    print("*" * 20)
    print(f"training time {end_time - start_time}")
    print("*" * 20)
    x = list(range(len(running_losses)))
    plt.plot(x, running_losses, color="r")
    plt.title("Skip gram with negative sampling training loss")
    plt.xlabel("training epochs")
    plt.ylabel("aeverage epoch loss")
    plt.show()
    
if __name__ == "__main__":
    text = ['Best way to is hardwork and persistence persistence persistence']

    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictionary_data(text)
    window_size = 2

    training_data,_, occurrences = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'no', single_positive_cxt=True)
    train_negative_sampling(training_data, occurrences, length_of_corpus, vocab_size, epochs=50)
