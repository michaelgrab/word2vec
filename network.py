import numpy as np
from data import generate_training_data, generate_dictionary_data, get_file_data
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

class SkipGramNegativeSampling:
    def __init__(self, vocab_size, hidden_size, neg_sample_num, learning_rate, alpha=3/4):
        self.v_input = np.random.uniform(-0.5, 0.5, size=(vocab_size, hidden_size))  
        self.v_output = np.random.uniform(-0.5, 0.5, size=(hidden_size, vocab_size))
        self.neg_sample_num = neg_sample_num  
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
        return 1 + (1 + np.exp(-x))

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
        samples = np.random.choice( len(dist), size=self.neg_sample_num, p=dist, replace=False )
        return samples
    
    def backwards(self, logits, neg_ind, h, cxt):
        t = np.zeros(self.neg_sample_num + 1)
        t[0] = 1
        logits_pos = np.dot(logits, cxt)
        logits_neg = logits[neg_ind]
        logits_out = np.concatenate((logits_pos, logits_neg))

        row_pos = np.dot(cxt, self.v_output)
        row_neg = self.v_output[:][neg_ind]
        rows = np.concatenate((row_pos, row_neg)) 
        
        grad_v_input_pos = np.dot( (self.sigmoid(logits_out) - t), rows)
        pass

    def backwards_alternative(self, logits, neg_ind, h, cxt):
        c_pos = np.dot(self.v_output, cxt)
        W_neg = self.v_output[:, neg_ind]
        grad_v_input = (self.sigmoid( np.dot(logits, cxt)) - 1) * c_pos
        neg_logits = logits[neg_ind]
        for u, c_neg in zip(W_neg.T, neg_logits):
            grad_v_input_neg = self.sigmoid(u) * c_neg
            grad_v_input+= grad_v_input_neg

        cxt_ind = np.argmax(cxt)
        weigths_in = self.v_input[cxt_ind]
        new_weights_in = weigths_in - (self.learning_rate * grad_v_input)
        self.v_input[cxt_ind] = new_weights_in    

def train_skip_gram():
    text = get_file_data(stop_word_removal='yes')
    
    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictionary_data(text)
    window_size = 2
    training_data,_,_ = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'no', single_positive_cxt=False)

    net = Network(vocab_size, hidden_size=20)

    print(f"vocabulary length {vocab_size}")
    # training loop implementation
    losses = []
    epoch_losses = []
    avg_losses = []
    cumulative_avg = []
    epochs = 500
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
        print(f"epoch: {epoch}, loss: {loss}")
        epoch +=1

    x = list(range(len(avg_losses)))
    # y = losses
    # plt.scatter(x, y, marker=".")
    plt.plot(x, avg_losses, color="r")
    plt.plot(x, cumulative_avg, color="b")
    plt.show()

def train_negative_sampling():    
    text = ['Best way to success is through hardwork and persistence persistence persistence persistence']

    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictionary_data(text)
    window_size = 2
    training_data,_, occurrences = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'no', single_positive_cxt=True)

    net = SkipGramNegativeSampling(vocab_size, 5, 4, 0.01)
    noisy_dist = net.get_noisy_dist(occurrences, length_of_corpus)

    sample = random.sample(training_data, k=1)[0]

    print(f"vocabulary size: {vocab_size}")
    print(f"number of negative samples {net.neg_sample_num}")
    print(f"hidden vector size {net.hidden_size}")

    x, cxt = sample
    neg_indicies = net.sample_neg(noisy_dist)
    logits, h = net.forward(x)
    # c_neg = c_output[neg_indicies]
    
    # pos = np.dot(cxt, c_output)
    net.backwards_alternative(logits, neg_indicies, h, cxt)
    pass

if __name__ == "__main__":
    train_negative_sampling()
    # text = ['Best way to success is through hardwork and persistence']
    # text = get_file_data(stop_word_removal='yes', filepath='dataset/text8.txt')
