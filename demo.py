# import tensorflow_hub as hub
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# embed = hub.load("tf_hub_cache/large-5")
# embeddings = embed([
#     "The quick brown fox jumps over the lazy dog.",
#     "I am a sentence for which I would like to get its embedding"])
#
# print(embeddings)


# from collections import Counter
#
# import numpy as np
# import time
# import random
#
# def fast_choice(options, probs):
#     x = random.random()#np.random.rand()
#     cum = 0
#     for i, p in enumerate(probs):
#         cum += p
#         if x < cum:
#             return options[i]
#     return options[-1]
#
# def choice(options,probs):
#     x = np.random.rand()
#     cum = 0
#     for i,p in enumerate(probs):
#         cum += p
#         if x < cum:
#             break
#     return options[i]
#
#
# options = ['a', 'b', 'c', 'd']+['a', 'b', 'c', 'd']+['a', 'b', 'c', 'd']+['a', 'b', 'c', 'd']+['a', 'b', 'c', 'd']
# probs = [1/20 for i in range(20)]
# runs = 10000
#
#
# now = time.time()
# temp = []
# for i in range(runs):
#     op = choice(options, probs)
#     temp.append(op)
# temp = Counter(temp)
# for op, x in temp.items():
#     print(op, x/runs)
# print(time.time()-now)
#
# print("")
# now = time.time()
# temp = []
# for i in range(runs):
#     op = np.random.choice(options, p=probs)
#     temp.append(op)
# temp = Counter(temp)
# for op, x in temp.items():
#     print(op, x/runs)
# print(time.time()-now)
#
# now = time.time()
# for i in range(runs):
#     # np.random.uniform()
#     random.uniform(0.0, 1.0)
# print('uniform():', time.time()-now)
#
#
# from scipy.special import softmax
#
# ret=softmax([0.1,0.2,0.7,1.0])
# print(ret)
import pickle

import gensim
from tqdm import tqdm
from dataloader import read_weibo_corpus, read_tnews_corpus

word2vecModel = gensim.models.KeyedVectors.load_word2vec_format("data/sohu.news.word2vec.bin", binary=True)

# for widx in word2vecModel.index_to_key:
#
#   idx = word2vecModel.index_to_key.index(widx)
#   embed = word2vecModel.get_vector(widx)
#
#   print(widx, idx, embed)

# dir_name = "data/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt"
# word2vecModel = gensim.models.KeyedVectors.load_word2vec_format(dir_name, binary=False)

weibo2018_x1, _ = read_weibo_corpus('data/weibo2018/train.txt', shuffle=True)
weibo2018_x2, _ = read_weibo_corpus('data/weibo2018/test.txt', shuffle=False)

tnews_x1, _ = read_tnews_corpus(data_path="data/tnews/train.json",
                                           label_path="data/tnews/labels.json",
                                           shuffle=True)

tnews_x2, _ = read_tnews_corpus(data_path="data/tnews/dev.json",
                                           label_path="data/tnews/labels.json",
                                           shuffle=True)

sim_dict = {}
word2idx = {}

non_dict_wd = 0

#weibo2018_x1, weibo2018_x2, tnews_x1, tnews_x2
for data_list in [tnews_x2]:
  for text in tqdm(data_list):
    for wd in text:
      if wd not in word2idx:
        try:
          word2vecModel.get_vector(wd)
          word2idx[wd] = len(word2idx)
        except:
          continue

print(len(word2idx))

#weibo2018_x1, weibo2018_x2, tnews_x1, tnews_x2
for data_list in [tnews_x2]:
  for text in tqdm(data_list):
    for wd in text:
      if wd not in word2idx:
        continue
      s_idx = word2idx[wd]
      if s_idx in sim_dict:
        continue
        
      sim_tmp = [(1.0, s_idx)]
      try:
        ret = word2vecModel.most_similar(wd, topn=1000)
        
        for item in ret:
          e_wd, score = item[0], item[1]
          
          if e_wd not in word2idx:
            # word2idx[e_wd] = len(word2idx)
            continue
          
          e_idx = word2idx[e_wd]
          if s_idx % 1000 == 0:
            print(wd, s_idx, e_wd, e_idx, score)
          sim_tmp.append((score, e_idx))
          if len(sim_tmp) ==50:
            break
      except:
        non_dict_wd+=1
        continue
        # print('not in dictionary...')
      sim_dict[s_idx] = list(sim_tmp)

print(len(word2idx))
print(non_dict_wd)
print(len(sim_dict))

mat = []
idx2word = {word2idx[wd]:wd for wd in word2idx}

fout = open('tnews-counter-fitted-vectors.txt', 'w')
for idx in tqdm(range(len(idx2word))):
  wd = idx2word[idx]
  embed = list(word2vecModel.get_vector(wd))
  mat.append(sim_dict[idx])
  strs = ' '.join([str(wd)] + list(map(str, embed)))+'\n'
  fout.write(strs)
  fout.flush()

fout.close()
pickle.dump(mat, open('tnews-mat.txt', 'wb'))
  