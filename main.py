from skip_gram import train_skip_gram
from negative_sampling import train_negative_sampling
from data import get_file_data
from data import generate_training_data, generate_dictionary_data, get_file_data

def main():
    text = get_file_data(stop_word_removal='yes')
    # text = ['Best way to is hardwork and persistence persistence persistence']

    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = generate_dictionary_data(text)
    window_size = 2

    # training of the skip gram model
    # display line chart at the end
    training_data,_,_ = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'no', single_positive_cxt=False)
    train_skip_gram(training_data, vocab_size, epochs=50)

    # train skip gram with negative sampling
    training_data,_, occurrences = generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,'no', single_positive_cxt=True)
    train_negative_sampling(training_data, occurrences, length_of_corpus, vocab_size, epochs=50)

if __name__ == "__main__":
    main()
