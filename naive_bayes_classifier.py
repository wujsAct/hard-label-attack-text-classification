import json
import sys

import pandas as pd

from utils import load_corpus, load_tnews_corpus, \
  load_MR_corpus, load_AG_corpus, stopwords, load_adv_corpus, load_org_adv_corpus
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from checklist_data_gen import ch_tongyi_word_bert, add_typos_bert, add_INV_bert, add_ch_INV_bert

class Naive_Bayesian(object):
  def __init__(self, df_train):
    self.vectorizer = CountVectorizer(token_pattern='\[?\w+\]?',
                                 stop_words=stopwords)
    X_train = self.vectorizer.fit_transform(df_train["words"])
    y_train = df_train["label"]
    self.clf = MultinomialNB()
    self.clf.fit(X_train, y_train)
  
  def pred_test(self, df_test):
    X_test = self.vectorizer.transform(df_test["words"])
    y_test = df_test["label"]
    y_pred = self.clf.predict(X_test)
    
    
    print(metrics.classification_report(y_test, y_pred))
    print("准确率:", metrics.accuracy_score(y_test, y_pred))
    
  def get_stat_info(self):
    idx2vocab = {}
    for key in self.vectorizer.vocabulary_:
      idx2vocab[self.vectorizer.vocabulary_[key]] = key
      
    rets = self.clf.feature_log_prob_
    m, n = np.shape(rets)
    print(m, n)
    ret = {}
    idx = 0
    
   
    for i in range(m):
      dict_tmp = {}
      for j in range(n):
        dict_tmp[j] = rets[i][j]
        
      sorted_ret = sorted(dict_tmp.items(), key=lambda d:d[1], reverse=True)
      print(len(sorted_ret))
      out_num = 100#int(len(sorted_ret)*0.05)
      for item in sorted_ret[:out_num]:
        idx += 1
        print(i, len(idx2vocab), idx, idx2vocab[item[0]])
        ret[idx2vocab[item[0]]] = len(ret)
    return ret

if __name__ == '__main__':
  data_tag = sys.argv[1]
  adv_tag = sys.argv[2]

  adv_dir_path = '/home/LAB/wujs/wujs-old/bug_manage/info_extract/hard-label-attack-main/adv_results'
  if data_tag == 'weibo2018':
    TRAIN_PATH = "data/weibo2018/train.txt"
    TEST_PATH = "data/weibo2018/test.txt"
    
    train_data = load_corpus(TRAIN_PATH)
    test_data = load_corpus(TEST_PATH)
    if adv_tag=='del':
       test_data = load_corpus(TEST_PATH, is_adv=True, adv_dict=json.load(open('data/'+data_tag+'/naive_adv.dict')))
    elif adv_tag == 'typos':
      test_data = ch_tongyi_word_bert(test_data)
    elif adv_tag == 'inv':
      test_data = add_ch_INV_bert(test_data)
    elif adv_tag == 'add':
      test_data = load_corpus(TEST_PATH, noise_type='add')
    elif adv_tag == 'cnn_adv':
      test_data = load_adv_corpus(adv_dir_path + '/weibo2018-cnn-adv-100')
    elif adv_tag == 'lstm_adv':
      test_data = load_adv_corpus(adv_dir_path + '/weibo2018-lstm-adv-100')
    elif adv_tag == 'bert_adv':
      test_data = load_adv_corpus(adv_dir_path + '/weibo2018-bert-adv-100')
    elif adv_tag == 'org_adv':
      test_data = load_org_adv_corpus(adv_dir_path + '/weibo2018-bert-adv-100', test_data)
    elif adv_tag == 'add':
      test_data = load_corpus(TEST_PATH,
                              noise_type='add')
    else:
      print('normal')
  elif data_tag =='tnews':
    TRAIN_PATH = "data/tnews/train.json"
    TEST_PATH = "data/tnews/dev.json"
    LABEL_PATH ="data/tnews/labels.json"
  
    train_data = load_tnews_corpus(data_path=TRAIN_PATH, label_path=LABEL_PATH)
    test_data = load_tnews_corpus(data_path=TEST_PATH, label_path=LABEL_PATH)
    if adv_tag == 'del':
      test_data = load_tnews_corpus(data_path=TEST_PATH, label_path=LABEL_PATH,
                                    is_adv=True,
                                    adv_dict=json.load(open('data/'+data_tag+'/naive_adv.dict')))
    elif adv_tag == 'typos':
      test_data = ch_tongyi_word_bert(test_data)
    elif adv_tag == 'inv':
      test_data = add_ch_INV_bert(test_data)
    elif adv_tag == 'add':
      test_data = load_tnews_corpus(data_path=TEST_PATH, label_path=LABEL_PATH,
                                    noise_type='add')
    elif adv_tag == 'cnn_adv':
      test_data = load_adv_corpus(adv_dir_path + '/tnews-cnn-adv-10')
    elif adv_tag == 'lstm_adv':
      test_data = load_adv_corpus(adv_dir_path + '/tnews-lstm-adv-10')
    elif adv_tag == 'bert_adv':
      test_data = load_adv_corpus(adv_dir_path + '/tnews-bert-adv-10')
    elif adv_tag == 'org_adv':
      test_data = load_org_adv_corpus(adv_dir_path + '/tnews-bert-adv-10')
    elif adv_tag =='add':
      test_data = load_tnews_corpus(data_path=TEST_PATH, label_path=LABEL_PATH,
                                    noise_type='add')
    else:
      print('normal')
  elif data_tag == 'mr_all':
    TRAIN_PATH="data/mr_all/train.csv"
    TEST_PATH="data/mr_all/dev.csv"
    train_data = load_MR_corpus(TRAIN_PATH)
    test_data = load_MR_corpus(TEST_PATH)
    if adv_tag == 'del':
      test_data = load_MR_corpus(TEST_PATH, is_adv=True, adv_dict=json.load(open('data/'+data_tag+'/naive_adv.dict')))
    elif adv_tag =='typos':
      test_data = add_typos_bert(test_data)
    elif adv_tag == 'inv':
      test_data = add_INV_bert(test_data)
    elif adv_tag =='add':
      test_data = load_MR_corpus(TEST_PATH, noise_type='add')
    elif adv_tag == 'cnn_adv':
      test_data = load_adv_corpus(adv_dir_path + '/mr-cnn-adv-10}')
    elif adv_tag == 'lstm_adv':
      test_data = load_adv_corpus(adv_dir_path + '/mr-lstm-adv-10')
    elif adv_tag == 'bert_adv':
      test_data = load_adv_corpus(adv_dir_path + '/mr-bert-adv-10')
    elif adv_tag == 'org_adv':
      test_data = load_org_adv_corpus(adv_dir_path + '/mr-bert-adv-10')
      
  elif data_tag == 'ag_news_all':
    TRAIN_PATH = "data/ag_news_all/train.csv"
    TEST_PATH = "data/ag_news_all/test.csv"

    train_data = load_AG_corpus(TRAIN_PATH)
    test_data = load_AG_corpus(TEST_PATH)
    if adv_tag == 'del':
      test_data = load_AG_corpus(TEST_PATH,is_adv=True, adv_dict=json.load(open('data/'+data_tag+'/naive_adv.dict')))
    elif adv_tag == 'typos':
      test_data = add_typos_bert(test_data)
    elif adv_tag == 'inv':
      test_data = add_INV_bert(test_data)
    elif adv_tag == 'add':
      test_data = load_AG_corpus(TEST_PATH, noise_type='add')
    elif adv_tag == 'cnn_adv':
      test_data = load_adv_corpus(adv_dir_path+'/ag-cnn-adv-10')
    elif adv_tag == 'lstm_adv':
      test_data = load_adv_corpus(adv_dir_path + '/ag-lstm-adv-10')
    elif adv_tag == 'bert_adv':
      test_data = load_adv_corpus(adv_dir_path + '/ag-bert-adv-10')
      
      
  df_train = pd.DataFrame(train_data, columns=["words", "label"]) #label 0:表示负面，label 1: 表示正面
  df_test = pd.DataFrame(test_data, columns=["words", "label"])
  # print(df_train.head())

  NB_Class = Naive_Bayesian(df_train)
  if adv_tag=='train':
    ret = NB_Class.get_stat_info()
    json.dump(ret, open('data/'+data_tag+'/naive_adv.dict', 'w'))
  
  NB_Class.pred_test(df_test)