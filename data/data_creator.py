from os import listdir
from os.path import isfile, join
import os

def data_create_Rotten_Tomatoes():
    data = []
    labels = []
    cwd = os.getcwd()
    data_dir = join(cwd, "data/rt-polaritydata")
    files = [('rt-polarity.neg', 0), ('rt-polarity.pos', 1)]
    for file, label in files:
        with open(join(data_dir, file), errors = 'ignore') as curr_file:
            for line in curr_file:
                data += [line.strip()]
                labels += [label]
    return data, labels

def data_create_SNLI():
    data = dict()
    labels = dict()
    cwd = os.getcwd()
    data_dir = join(cwd, "data/SNLI")
    files = ['snli_1.0_train.txt', 'snli_1.0_dev.txt', 'snli_1.0_test.txt']
    for file in files:
        df = pd.read_csv(join(data_dir, file), sep = '	')
        premises = df['sentence1'].to_numpy(dtype = str)
        hypothesises = df['sentence2'].to_numpy(dtype = str)
        golden_labels = df['gold_label'].to_numpy(dtype = str)
        data[file] = [first + ', ' + second for first, second in zip(premises, hypothesises)]
        labels[file] = golden_labels
    return data, labels
