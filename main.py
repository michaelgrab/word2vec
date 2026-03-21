from network import train_skip_gram
from data import get_file_data

def main():
    # text = get_file_data(stop_word_removal='yes')
    text = ['Best way to is hardwork and persistence persistence persistence']

    # training of the skip gram model
    # display line chart at the end
    train_skip_gram(text)

if __name__ == "__main__":
    main()
