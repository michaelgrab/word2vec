import numpy as np
from negative_sampling import train_negative_sampling
from data import generate_training_data, generate_dictionary_data
from data import get_file_data
import random

import kagglehub
import os

# useful for evaluating model quality
def get_most_similar(word, net, word_to_index, index_to_word, top_n=5):
    if word not in word_to_index:
        return f"Word '{word}' not in vocabulary."

    word_idx = word_to_index[word]
    
    v_w = net.v_input[word_idx]

    similarities = []
    
    v_w_norm = np.linalg.norm(v_w)

    for i in range(net.vocab_size):
        v_other = net.v_input[i]
        v_other_norm = np.linalg.norm(v_other)
        
        if v_w_norm == 0 or v_other_norm == 0:
            similarity = 0
        else:
            similarity = np.dot(v_w, v_other) / (v_w_norm * v_other_norm)
        
        similarities.append(similarity)

    similarities = np.array(similarities)
    
    top_indices = np.argsort(similarities)[-(top_n + 1):][::-1]

    results = []
    for idx in top_indices:
        sim_word = index_to_word[idx]
        if sim_word != word: 
            results.append((sim_word, similarities[idx]))
            
    return results[:top_n]

if __name__ == "__main__":
    # text = ['Best way to is hardwork and persistence persistence persistence']
    path = kagglehub.dataset_download("muhammedfathi/game-of-thrones-book-files")
    files = os.listdir(path)
    print(files)
    text = get_file_data(stop_word_removal='yes', filepath=os.path.join(path, files[0]))

    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictionary_data(text)
    window_size = 2

    training_data,sample_words, occurences = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'yes', single_context=True)
    net = train_negative_sampling(training_data, occurences, length_of_corpus, vocab_size, epochs=50, plot_chart=False, print_log=True)    
    for _ in range(10):
        target_word, context_words = random.sample(sample_words, k=1)[0]

        print(f"\nWords most similar to '{target_word}':")
        similar_words = get_most_similar(target_word, net, word_to_index, index_to_word, top_n=5)
        
        for word, score in similar_words:
            print(f"{word}: {score:.4f}")  