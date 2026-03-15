import numpy as np

def get_text_dataset(path_to_file):
    with open(path_to_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    return np.array(lines)

text_ds = get_text_dataset('dataset/shakespeare.txt')

for i in range(0):
    print(text_ds[i])