import bz2
import gzip
import json
import os
import sys
import io
import re
import random
import csv
import numpy as np
import torch
from pypinyin import pinyin, Style
from tqdm import tqdm

from utils import processing, clean_str

csv.field_size_limit(sys.maxsize)

def read_weibo_corpus(path,is_adv=False, adv_dict=None, shuffle=False,noise_type=None):
    """
    weibo2018
    """
    data = []
    labels = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            
            if noise_type=='add':
                if seniment=='1':
                    content = content + '我心情很好。'
                if seniment=='0':
                    content = content + '你情绪低落。'
            content = processing(content, is_adv=is_adv, adv_dict=adv_dict)
            data.append(content.split(" "))
            labels.append(int(seniment))
    
    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]
      
    return data, labels

def read_tnews_corpus(label_path, data_path, is_adv=False, adv_dict=None, shuffle=False,
                      noise_type=None):
    """
    加载tnews
    """
    label2idx = {}
    with open(label_path, "r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            # print(item, item.get('label'))
            label2idx[item['label']] = len(label2idx)

    data = []
    labels = []
    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            content_pre = item['sentence']
            # print(content_pre)
            if noise_type == 'add':
                keyword = item['keywords']
                if keyword == '':
                    content_pre = item['sentence']
                else:
                    content_pre = content_pre+' '+keyword.split(',')[0]
                # print(content_pre)
            
            content = processing(content_pre, is_adv=is_adv, adv_dict=adv_dict)
            label = label2idx[item['label']]
            data.append(content.split(" "))
            labels.append(label)
    
    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]
      
    return data, labels

def read_corpus(path, csvf=False , clean=True, MR=True,
                encoding='utf8', shuffle=False, lower=True,
                del_stop=False,
                is_adv=False, adv_dict={},
                noise_type=None):
    data = []
    labels = []
    if not csvf:
        with open(path, encoding=encoding) as fin:
            for line in fin:
                if MR:
                    text, sep, label = line.partition('\t')
                    label = int(label)
                    if noise_type=='add':
                        if label == 0:
                            text = text +' '+'This work is spectacular'
                        else:
                            text = text +' '+'This work is bad'
                else:
                    label, sep, text = line.partition(',')
                    label = int(label) - 1
                if clean:
                    text = clean_str(text.strip(), del_stop=del_stop) if clean else text.strip()
                  
                if is_adv:
                    text = text.split(" ")
                    ret = [ ]
                    for wd in text:
                        if wd in adv_dict:
                            continue
                        else:
                            ret.append(wd)
                    text = " ".join(ret)
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())
    else:
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                if 'ag' in path:
                    if noise_type == 'add':
                        text = line[2]
                    else:
                        text = line[1] + " " + line[2]
                    # lent = len(text.split(" "))
                    label = int(line[0])-1
                else:
                    text = line[0]
                    label = int(line[1])
                  
                if clean:
                    text = clean_str(text.strip(), del_stop=del_stop) if clean else text.strip()
                if is_adv:
                    text = text.split(" ")
                    ret = [ ]
                    for wd in text:
                        if wd in adv_dict:
                            continue
                        else:
                            ret.append(wd)
                    text = " ".join(ret)
                if lower:
                    text = text.lower()
                labels.append(label)
                data.append(text.split())

    if shuffle:
        perm = list(range(len(data)))
        random.shuffle(perm)
        data = [data[i] for i in perm]
        labels = [labels[i] for i in perm]

    return data, labels
    
def read_MR(path, seed=1234):
    file_path = os.path.join(path, "rt-polarity.all")
    data, labels = read_corpus(file_path, encoding='latin-1')
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_SUBJ(path, seed=1234):
    file_path = os.path.join(path, "subj.all")
    data, labels = read_corpus(file_path, encoding='latin-1')
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_CR(path, seed=1234):
    file_path = os.path.join(path, "custrev.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_MPQA(path, seed=1234):
    file_path = os.path.join(path, "mpqa.all")
    data, labels = read_corpus(file_path)
    random.seed(seed)
    perm = list(range(len(data)))
    random.shuffle(perm)
    data = [ data[i] for i in perm ]
    labels = [ labels[i] for i in perm ]
    return data, labels

def read_TREC(path, seed=1234):
    train_path = os.path.join(path, "TREC.train.all")
    test_path = os.path.join(path, "TREC.test.all")
    train_x, train_y = read_corpus(train_path, TREC=True, encoding='latin-1')
    test_x, test_y = read_corpus(test_path, TREC=True, encoding='latin-1')
    random.seed(seed)
    perm = list(range(len(train_x)))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]
    return train_x, train_y, test_x, test_y

def read_SST(path, seed=1234):
    train_path = os.path.join(path, "stsa.binary.phrases.train")
    valid_path = os.path.join(path, "stsa.binary.dev")
    test_path = os.path.join(path, "stsa.binary.test")
    train_x, train_y = read_corpus(train_path, False)
    valid_x, valid_y = read_corpus(valid_path, False)
    test_x, test_y = read_corpus(test_path, False)
    random.seed(seed)
    perm = list(range(len(train_x)))
    random.shuffle(perm)
    train_x = [ train_x[i] for i in perm ]
    train_y = [ train_y[i] for i in perm ]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def cv_split(data, labels, nfold, test_id):
    assert (nfold > 1) and (test_id >= 0) and (test_id < nfold)
    lst_x = [ x for i, x in enumerate(data) if i%nfold != test_id ]
    lst_y = [ y for i, y in enumerate(labels) if i%nfold != test_id ]
    test_x = [ x for i, x in enumerate(data) if i%nfold == test_id ]
    test_y = [ y for i, y in enumerate(labels) if i%nfold == test_id ]
    perm = list(range(len(lst_x)))
    random.shuffle(perm)
    M = int(len(lst_x)*0.9)
    train_x = [ lst_x[i] for i in perm[:M] ]
    train_y = [ lst_y[i] for i in perm[:M] ]
    valid_x = [ lst_x[i] for i in perm[M:] ]
    valid_y = [ lst_y[i] for i in perm[M:] ]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

def cv_split2(data, labels, nfold, valid_id):
    assert (nfold > 1) and (valid_id >= 0) and (valid_id < nfold)
    train_x = [ x for i, x in enumerate(data) if i%nfold != valid_id ]
    train_y = [ y for i, y in enumerate(labels) if i%nfold != valid_id ]
    valid_x = [ x for i, x in enumerate(data) if i%nfold == valid_id ]
    valid_y = [ y for i, y in enumerate(labels) if i%nfold == valid_id ]
    return train_x, train_y, valid_x, valid_y

def pad(sequences, pad_token='<pad>', pad_left=True):
    ''' input sequences is a list of text sequence [[str]]
        pad each text sequence to the length of the longest
    '''
    max_len = max(5,max(len(seq) for seq in sequences))
    if pad_left:
        return [ [pad_token]*(max_len-len(seq)) + seq for seq in sequences ]
    return [ seq + [pad_token]*(max_len-len(seq)) for seq in sequences ]


def create_one_batch(x, y, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda(), torch.LongTensor(y).cuda()


def create_one_batch_x(x, map2id, oov='<oov>'):
    oov_id = map2id[oov]
    x = pad(x)
    length = len(x[0])
    batch_size = len(x)
    x = [ map2id.get(w, oov_id) for seq in x for w in seq ]
    
    x = torch.LongTensor(x)
    assert x.size(0) == length*batch_size
    return x.view(batch_size, length).t().contiguous().cuda()


# shuffle training examples and create mini-batches
def create_batches(x, y, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]
    y = [ y[i] for i in lst ]

    sum_len = 0.
    for ii in x:
        sum_len += len(ii)
    batches_x = [ ]
    batches_y = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx, by = create_one_batch(x[i*size:(i+1)*size], y[i*size:(i+1)*size], map2id)
        batches_x.append(bx)
        batches_y.append(by)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]
        batches_y = [ batches_y[i] for i in perm ]

    sys.stdout.write("{} batches, avg sent len: {:.1f}\n".format(
        nbatch, sum_len/len(x)
    ))

    return batches_x, batches_y


# shuffle training examples and create mini-batches
def create_batches_x(x, batch_size, map2id, perm=None, sort=False):

    lst = perm or range(len(x))

    # sort sequences based on their length; necessary for SST
    if sort:
        lst = sorted(lst, key=lambda i: len(x[i]))

    x = [ x[i] for i in lst ]

    sum_len = 0.0
    batches_x = [ ]
    size = batch_size
    nbatch = (len(x)-1) // size + 1
    for i in range(nbatch):
        bx = create_one_batch_x(x[i*size:(i+1)*size], map2id)
        sum_len += len(bx)
        batches_x.append(bx)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_x = [ batches_x[i] for i in perm ]

    # sys.stdout.write("{} batches, avg len: {:.1f}\n".format(
    #     nbatch, sum_len/nbatch
    # ))

    return batches_x


def load_embedding_npz(path):
    data = np.load(path)
    return [ w.decode('utf8') for w in data['words'] ], data['vals']

def load_embedding_txt(path):
    print('load word embeddings ...')
    if path.endswith(".gz"):
        file_open = gzip.open
        open_mode = "rb"
    elif path.endswith(".bz2"):
        file_open = bz2.open
        open_mode = "rb"
    else:
        file_open = open
        open_mode = "r"
    words = [ ]
    vals = [ ]
    real_words={}
    with file_open(path, open_mode) as fin:
        fin.readline()
        for line in tqdm(fin):
            line = line.rstrip()
            if line:
                if open_mode=='r':
                    parts = line.split(' ')
                else:
                    parts = line.decode('utf8').strip().split(' ')
                word = parts[0]
                val_list = parts[1:]
                
                if word not in real_words:
                    real_words[word]=1
                else:
                    print('duplicate word:', word)
                    continue
                words.append(word)
                vals += [ float(x) for x in val_list ]
    return words, np.asarray(vals).reshape(len(words), -1)

def load_embedding(path):
    if path.endswith(".npz"):
        return load_embedding_npz(path)
    else:
        return load_embedding_txt(path)


if __name__ == '__main__':
    load_embedding("data/sgns.weibo.word.bz2")