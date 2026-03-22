# data preparation
# https://github.com/rahul1728jha/Word2Vec_Implementation/blob/master/Word_2_Vec.ipynb

import re
import numpy as np
import random

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 

# if set to True, the one hot encodings will contain multiple 1's
# otherwise it will be usual single 1 per vector
MULTI_ONE_VEC = True

def get_file_data(stop_word_removal='no', filepath='dataset/jef_archer.txt', max_lines=100):
    file_contents = []
    with open(filepath) as f:
        file_contents = f.read()
    text = []
    for val in file_contents.split('.'):
        sent = re.findall("[A-Za-z]+", val)
        line = ''
        for words in sent:
            
            if stop_word_removal == 'yes': 
                if len(words) > 1 and words not in stop_words:
                    line = line + ' ' + words
            else:
                if len(words) > 1 :
                    line = line + ' ' + words
        text.append(line)
        if len(text) > max_lines:
            break
    return text

def generate_dictionary_data(text):
    word_to_index= dict()
    index_to_word = dict()
    corpus = []
    count = 0
    vocab_size = 0
    
    for row in text:
        for word in row.split():
            word = word.lower()
            corpus.append(word)
            if word_to_index.get(word) == None:
                word_to_index.update ( {word : count})
                index_to_word.update ( {count : word })
                count  += 1
    vocab_size = len(word_to_index)
    length_of_corpus = len(corpus)
    
    return word_to_index,index_to_word,corpus,vocab_size,length_of_corpus

def get_one_hot_vectors(target_word,context_words,vocab_size,word_to_index):
    
    #Create an array of size = vocab_size filled with zeros
    trgt_word_vector = np.zeros(vocab_size)
    
    #Get the index of the target_word according to the dictionary word_to_index. 
    #If target_word = best, the index according to the dictionary word_to_index is 0. 
    #So the one hot vector will be [1, 0, 0, 0, 0, 0, 0, 0, 0]
    index_of_word_dictionary = word_to_index.get(target_word) 
    
    #Set the index to 1
    trgt_word_vector[index_of_word_dictionary] = 1
    
    #Repeat same steps for context_words but in a loop
    ctxt_word_vector = np.zeros(vocab_size)
    
    
    for word in context_words:
        index_of_word_dictionary = word_to_index.get(word) 
        ctxt_word_vector[index_of_word_dictionary] = 1
        
    return trgt_word_vector,ctxt_word_vector

def one_hot_vector(word, vocab_size, word_to_index):
    word_vec = np.zeros(vocab_size)
    index = word_to_index.get(word)
    word_vec[index] = 1
    return word_vec

def generate_samples(trgt_word, cxt_word, vocab_size, word_to_index):
    trgt_one_hot = one_hot_vector(trgt_word, vocab_size, word_to_index) 
    samples = []
    for c_w in cxt_word:
        cxt_word_one_hot = one_hot_vector(c_w, vocab_size, word_to_index)
        sample = trgt_one_hot, cxt_word_one_hot
        samples.append(sample)
    return samples    


#Note : Below comments for trgt_word_index, ctxt_word_index are with the above sample text for understanding the code flow

def generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,sample=None, single_context = True):
    """
        returns:
            training data
                pairs of one hot encoded center word and multi hot encoded context words
                or
                pairs of one hot encoded center word and one hot encoded context word
            training sample words
                list of pairs of center word and list of context words
            occurences
                array where each index corresponds to the number the word occurres in the corpus    
    """
    training_data =  []
    training_sample_words =  []
    occurrences = np.zeros(vocab_size)
    for i,word in enumerate(corpus):
        # print(f"loading word {i}")

        index_target_word = i
        target_word = word
        context_words = []
        occurrences[word_to_index.get(word)] += 1

        #when target word is the first word
        if i == 0:  

            # trgt_word_index:(0), ctxt_word_index:(1,2)
            context_words = [corpus[x] for x in range(i + 1 , window_size + 1)] 


        #when target word is the last word
        elif i == len(corpus)-1:

            # trgt_word_index:(9), ctxt_word_index:(8,7), length_of_corpus = 10
            context_words = [corpus[x] for x in range(length_of_corpus - 2 ,length_of_corpus -2 - window_size  , -1 )]

        #When target word is the middle word
        else:

            #Before the middle target word
            before_target_word_index = index_target_word - 1
            for x in range(before_target_word_index, before_target_word_index - window_size , -1):
                if x >=0:
                    context_words.extend([corpus[x]])

            #After the middle target word
            after_target_word_index = index_target_word + 1
            for x in range(after_target_word_index, after_target_word_index + window_size):
                if x < len(corpus):
                    context_words.extend([corpus[x]])

        if single_context:
            samples = generate_samples(trgt_word=target_word,
                cxt_word=context_words, vocab_size=vocab_size, word_to_index=word_to_index)
            training_data.extend(samples)   
        else:
            trgt_word_vector,ctxt_word_vector = get_one_hot_vectors(target_word,context_words,vocab_size,word_to_index)
            sample = trgt_word_vector, ctxt_word_vector
            training_data.append(sample)
        
        if sample is not None:
            training_sample_words.append([target_word,context_words])   
        
    return training_data,training_sample_words,occurrences

def get_cxt_word_columns(cxt_word_vector: np.ndarray):
    columns = []
    vocab_size = len(cxt_word_vector)
    for i in range(vocab_size):
        if cxt_word_vector[i] == 0:
            continue
        v = np.zeros_like(cxt_word_vector)
        v[i] = 1
        columns.append(v)
    return columns



if(__name__ == "__main__"):
    # text = ['Best way to success is through hardwork and persistence']
    # text = get_file_data(stop_word_removal='yes', filepath='dataset/shakespeare.txt')
    text = get_file_data(stop_word_removal='yes')
    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictionary_data(text)
    print('Number of unique words:' , vocab_size)
    print('index_to_word : ', random.sample(  list( word_to_index.items() ), 3) )
    # print('word_to_index : ',word_to_index)
    print('index_to_word : ', random.sample(  list( index_to_word.items() ), 3) )
    # print('corpus:',corpus)
    print('Length of corpus :',length_of_corpus)

    window_size = 2
    training_data,training_sample_words,_ = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'yes')

    # for i in range( 3 ):
    #     print('*' * 50)
    #     print('Target word: %s . Target vector: %s ' %(training_sample_words[i][0],training_data[i][0]))
    #     print('Context word:%s . Context  vector: %s ' %(training_sample_words[i][1],training_data[i][1]))

    print('Target word: %s . Target vector: %s ' %(training_sample_words[0][0],training_data[0][0]))
    print('Context word:%s . Context  vector: %s ' %(training_sample_words[0][1],training_data[0][1]))

    print("one hot encodings for every column")
    columns = get_cxt_word_columns(training_data[0][1])
    for column in columns:
        print(column)    