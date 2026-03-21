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

    def forward(self, x):
        """
          x is the target word

        """
        h = np.dot(x.T, self.v_input)
        c_output = np.dot(h.T, self.v_output) 
        return c_output, h
    
    def sigmoid(self, x):
        pass

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
                  
#---------------------------------
# added for reference
def calculate_error(y_pred,context_words):
    
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}
    
    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update ( {index : 'yes'} )
        
    number_of_1_in_context_vector = len(index_of_1_in_context_words)
    
    for i,value in enumerate(y_pred):
        
        if index_of_1_in_context_words.get(i) != None:
            total_error[i]= (value-1) + ( (number_of_1_in_context_vector -1) * value)
        else:
            total_error[i]= (number_of_1_in_context_vector * value)
            
            
    return  np.array(total_error)

def backward_prop(weight_inp_hidden,weight_hidden_output,total_error, hidden_layer, target_word_vector,learning_rate):
    
    dl_weight_inp_hidden = np.outer(target_word_vector, np.dot(weight_hidden_output, total_error.T))
    dl_weight_hidden_output = np.outer(hidden_layer, total_error)
    
    # Update weights
    weight_inp_hidden = weight_inp_hidden - (learning_rate * dl_weight_inp_hidden)
    weight_hidden_output = weight_hidden_output - (learning_rate * dl_weight_hidden_output)
    
    return weight_inp_hidden,weight_hidden_output

def calculate_loss(u,ctx):
    
    sum_1 = 0
    for index in np.where(ctx==1)[0]:
        sum_1 = sum_1 + u[index]
    
    sum_1 = -sum_1
    sum_2 = len(np.where(ctx==1)[0]) * np.log(np.sum(np.exp(u)))
    
    total_loss = sum_1 + sum_2
    return total_loss
# ----------------------------------------------------------------------

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
    print(f"number of ")
    x, c_pos = sample
    negatives = net.sample_neg(noisy_dist)
    h, c_output = net.forward(x)
    h
    c_output

if __name__ == "__main__":
    train_negative_sampling()
    # text = ['Best way to success is through hardwork and persistence']
    # text = get_file_data(stop_word_removal='yes', filepath='dataset/text8.txt')
