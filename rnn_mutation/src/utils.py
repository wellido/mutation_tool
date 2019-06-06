from __future__ import print_function
from keras.datasets import imdb
from keras.models import Model
from keras.preprocessing import sequence
from functools import reduce
import re
import tarfile
from itertools import combinations
import random
import numpy as np
import csv

from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split(r'(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return (pad_sequences(xs, maxlen=story_maxlen),
            pad_sequences(xqs, maxlen=query_maxlen), np.array(ys))


def data_preprocess(data_type):
    """

    :param data_type:
    :return:
    """
    if data_type == "imdb":
        max_features = 20000
        maxlen = 80
        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        print('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        return (x_train, y_train), (x_test, y_test)
    else:
        try:
            path = get_file('babi-tasks-v1-2.tar.gz',
                            origin='https://s3.amazonaws.com/text-datasets/'
                                   'babi_tasks_1-20_v1-2.tar.gz')
        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
                  '.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise

        # Default QA1 with 1000 samples
        # challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
        # QA1 with 10,000 samples
        # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
        # QA2 with 1000 samples
        # challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
        # QA2 with 10,000 samples
        challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
        with tarfile.open(path) as tar:
            train = get_stories(tar.extractfile(challenge.format('train')))
            test = get_stories(tar.extractfile(challenge.format('test')))

        vocab = set()
        for story, q, answer in train + test:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        story_maxlen = max(map(len, (x for x, _, _ in train + test)))
        query_maxlen = max(map(len, (x for _, x, _ in train + test)))

        x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
        tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)
        return (x, xq, y), (tx, txq, ty)


def distance_calculate(ori_result, mu_result):
    """

    :param ori_result:
    :param mu_result:
    :return:
    """
    distance_result = 0
    flatten_ori = ori_result.flatten()
    flatten_mutant = mu_result.flatten()
    for i in range(len(flatten_ori)):
        if flatten_ori[i] != 0:
            # distance_result += abs(flatten_mutant[i] - flatten_ori[i]) / flatten_ori[i]
            distance_result += abs(flatten_mutant[i] - flatten_ori[i])
    return distance_result


def imdb_word2vec(string):
    """

    :param string:
    :return:
    """
    maxlen = 80
    string_to_vec = imdb.get_word_index()
    string_to_vec = {k: (v + 3) for k, v in string_to_vec.items()}
    string_to_vec["PAD"] = 0
    string_to_vec["START"] = 1
    string_to_vec["UNK"] = 2
    string_vector = []
    str_list = string.split(" ")
    for word in str_list:
        string_vector.append(string_to_vec[word])
    string_vector = sequence.pad_sequences([string_vector], maxlen=maxlen)
    final_vector = string_vector[0]
    return final_vector


def imdb_vec2word(data):
    """

    :param data:
    :return:
    """
    string_to_vec = imdb.get_word_index()
    string_to_vec = {k: (v + 3) for k, v in string_to_vec.items()}
    string_to_vec["PAD"] = 0
    string_to_vec["START"] = 1
    string_to_vec["UNK"] = 2
    id_to_word = {value: key for key, value in string_to_vec.items()}
    word = ' '.join(id_to_word[id] for id in data)
    return word


def embedding_input(model, layer_index, x_test, time_step, save_path=None):
    """

    :param model:
    :param layer_index:
    :param x_test:
    :param time_step:
    :param save_path
    :return:
    """
    new_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
    embedding_result = new_model.predict(x_test.reshape(1, time_step))
    if save_path:
        np.savez(save_path, embedding_x=embedding_result)
    return embedding_result


def choose_attack_position(all_position, position_number, select_number):
    """

    :param all_position:
    :param position_number:
    :param select_number:
    :return:
    """
    combination = [c for c in combinations(range(all_position), position_number)]
    comb_len = len(combination)
    # index_select = np.random.choice(comb_len, select_number, replace=False)
    index_select = random.sample(range(0, comb_len - 1), select_number)
    return_list = [combination[x] for x in index_select]
    return_list = [list(x) for x in return_list]
    return return_list


def read_csv(csv_path):
    """

    :param csv_path:
    :return:
    """
    # 读取csv至列表
    csvFile = open(csv_path, 'r')
    reader = csv.reader(csvFile)
    result = []
    for item in reader:
        result.append(item)
    csvFile.close()
    result = np.array(result)
    return result


def model_QC(model, x_test, y_test, batch_size):
    """

    :param model:
    :param x_test:
    :param y_test:
    :param batch_size:
    :return:
    """

    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return score, acc
